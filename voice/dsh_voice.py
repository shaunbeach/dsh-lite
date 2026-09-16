#!/usr/bin/env python3
"""
Voice front end for the dsh harness.

Owns the microphone so the harness does not have to. A wake word opens a window, speech is recorded
until the speaker stops, the clip is transcribed, and the result is sent to ~/.dsh/input.sock as
newline-delimited JSON. The harness only ever sees text.

Four stages, each doing one thing:

    openWakeWord   is anyone talking to me?
    Silero VAD     have they finished?
    MLX Whisper    what did they say?
    classify       is that an instruction, "send it", or "stop"?

Whisper is loaded for the transcription and dropped immediately after. On a 16 GB machine it would
otherwise sit on ~1.6 GB that the language model needs, and loading it from the local cache costs
about a fifth of a second, so there is nothing to gain by keeping it.

Run it alongside a harness that is in /voice mode:

    ~/.dsh/voice-venv/bin/python voice/dsh_voice.py
"""

from __future__ import annotations

import argparse
import collections
import gc
import json
import os
import queue
import re
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

# Whisper weights come from a local cache; the download bars for it are noise in a voice log.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import numpy as np

SAMPLE_RATE = 16_000
#: openWakeWord consumes 80 ms of 16 kHz audio per call.
WAKE_FRAME = 1280
#: Silero v4 scores 32 ms windows.
VAD_FRAME = 512

DEFAULT_SOCKET = Path.home() / ".dsh" / "input.sock"

#: Said alone, these end the turn rather than becoming part of it.
ABORT_PHRASES = {"stop", "stop it", "cancel", "abort", "never mind", "nevermind", "quit that"}
SUBMIT_PHRASES = {"send it", "send", "go", "go ahead", "do it", "run it", "submit", "okay go", "ok go"}

#: Answers to "shall I send that now?". Anything else is taken as more of the instruction, and
#: silence is taken as no, so nothing is ever sent without having been agreed to.
YES_PHRASES = {
    "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "please", "please do", "affirmative",
    "yes send it", "yes please", "send it", "send", "go", "go ahead", "do it", "run it", "submit",
}
NO_PHRASES = {
    "no", "nope", "no thanks", "dont", "dont send it", "do not send it", "dont send",
    "cancel", "stop", "wait", "not yet", "hold on", "never mind", "nevermind", "scratch that",
}


@dataclass
class Config:
    wake_word: str = "hey_jarvis"
    wake_threshold: float = 0.5
    socket_path: Path = DEFAULT_SOCKET
    whisper_repo: str = "mlx-community/whisper-large-v3-turbo"
    #: Speech probability above which Silero is considered to have heard a voice.
    vad_threshold: float = 0.5
    #: Silence that ends an utterance, once speech has started.
    silence_ms: int = 800
    #: How long to wait for speech to begin after the wake word before giving up.
    lead_in_ms: int = 3_000
    #: Hard ceiling on one utterance, so a stuck microphone cannot record forever.
    max_utterance_ms: int = 20_000
    #: Audio kept from before the wake word fired, for speech that runs straight on from it.
    preroll_ms: int = 300
    #: Whether to ask before sending dictated text.
    confirm: bool = True
    confirm_phrase: str = "Shall I send that now?"
    #: How long to wait for an answer. Timing starts when the question finishes, not when it starts.
    confirm_ms: int = 3_000
    #: How many times to ask. More of the instruction can be added instead of answering, and each
    #: addition earns another question; the cap stops that going round forever.
    confirm_rounds: int = 3
    speak: bool = True
    #: What it calls itself, in the acknowledgement and in logs.
    name: str = "Amy"
    ack_phrase: str = "Got it"
    voice_name: str | None = None
    input_device: int | str | None = None
    verbose: bool = False


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


class Speaker:
    """
    Speaks through the macOS `say` command.

    A local acknowledgement is the reason this exists: the harness cannot confirm anything until it
    has the text, so waiting for its reply would leave a silence exactly where the speaker expects
    one. Saying "Got it" the moment speech ends costs nothing and comes from here.

    While it is speaking the microphone is ignored, so the wake word cannot be triggered by its
    own output coming back through the speakers.
    """

    def __init__(self, enabled: bool, voice_name: str | None) -> None:
        self.enabled = enabled
        self.voice_name = voice_name
        self._process: subprocess.Popen | None = None

    @property
    def is_speaking(self) -> bool:
        process = self._process
        return process is not None and process.poll() is None

    def say(self, text: str) -> None:
        if not self.enabled or not text.strip():
            return
        self.stop()
        command = ["say"]
        if self.voice_name:
            command += ["-v", self.voice_name]
        command.append(text)
        try:
            self._process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            self.enabled = False
            log("`say` is unavailable; continuing without speech")

    def stop(self) -> None:
        if self.is_speaking and self._process is not None:
            self._process.terminate()


class HarnessLink:
    """
    The connection to dsh.

    Reconnects on demand rather than at startup, because the socket only exists while a harness is
    in /voice mode: the daemon may well be running before, after, or across several of those.
    """

    def __init__(self, socket_path: Path, speaker: Speaker) -> None:
        self.socket_path = socket_path
        self.speaker = speaker
        self._socket: socket.socket | None = None
        self._reader: threading.Thread | None = None
        self._complained = False

    def _connect(self) -> socket.socket | None:
        if self._socket is not None:
            return self._socket
        if not self.socket_path.exists():
            if not self._complained:
                log(f"no harness at {self.socket_path} — run /voice in dsh")
                self._complained = True
            return None
        try:
            connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            connection.connect(str(self.socket_path))
        except OSError as err:
            if not self._complained:
                log(f"could not reach {self.socket_path}: {err}")
                self._complained = True
            return None

        self._socket = connection
        self._complained = False
        self._reader = threading.Thread(target=self._read_events, args=(connection,), daemon=True)
        self._reader.start()
        log(f"connected to {self.socket_path}")
        return connection

    def _read_events(self, connection: socket.socket) -> None:
        """Speaks what the harness sends back: an acknowledgement is local, but the summary is not."""
        buffer = b""
        try:
            while True:
                chunk = connection.recv(4096)
                if not chunk:
                    break
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    kind = event.get("type")
                    if kind == "done":
                        summary = str(event.get("summary", "")).strip()
                        log(f"harness done: {summary}")
                        self.speaker.say(summary or "Done.")
                    elif kind == "error":
                        log(f"harness rejected a line: {event.get('message')}")
        except OSError:
            pass
        finally:
            self._drop()

    def _drop(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:
                pass
        self._socket = None

    def send(self, payload: dict) -> bool:
        connection = self._connect()
        if connection is None:
            return False
        try:
            connection.sendall((json.dumps(payload) + "\n").encode("utf8"))
            return True
        except OSError as err:
            log(f"lost the harness: {err}")
            self._drop()
            return False


class Transcriber:
    """
    Whisper, held only for as long as a clip takes.

    Loading from the local cache is memory-mapped and takes about two tenths of a second, so the
    saving of keeping it resident is not worth the 1.6 GB it holds on a machine that is also running
    a language model.
    """

    def __init__(self, repo: str, verbose: bool) -> None:
        self.repo = repo
        self.verbose = verbose

    def transcribe(self, audio: np.ndarray) -> str:
        import mlx.core as mx
        import mlx_whisper
        from mlx_whisper.transcribe import ModelHolder

        started = time.time()
        try:
            result = mlx_whisper.transcribe(audio, path_or_hf_repo=self.repo, fp16=True)
            text = str(result.get("text", "")).strip()
        finally:
            ModelHolder.model = None
            ModelHolder.model_path = None
            gc.collect()
            mx.clear_cache()

        if self.verbose:
            held = mx.get_active_memory() / 1e6
            log(f"transcribed {len(audio) / SAMPLE_RATE:.1f}s in {time.time() - started:.1f}s, {held:.1f} MB still held")
        return text


class Endpointer:
    """Silero VAD, tracking whether the speaker has started and whether they have stopped."""

    def __init__(self, model_dir: Path, threshold: float) -> None:
        import onnxruntime

        options = onnxruntime.SessionOptions()
        options.log_severity_level = 4
        self.session = onnxruntime.InferenceSession(
            str(model_dir / "silero_vad.onnx"), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.threshold = threshold
        self.sample_rate = np.array(SAMPLE_RATE, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        # The v4 model carries LSTM state across frames; v5 replaced h/c with a single tensor.
        self.h = np.zeros((2, 1, 64), dtype=np.float32)
        self.c = np.zeros((2, 1, 64), dtype=np.float32)

    def speech_probability(self, frame: np.ndarray) -> float:
        window = frame.astype(np.float32).reshape(1, -1) / 32768.0
        output, self.h, self.c = self.session.run(
            None, {"input": window, "sr": self.sample_rate, "h": self.h, "c": self.c}
        )
        return float(output[0][0])


@dataclass
class Utterance:
    audio: np.ndarray
    seconds: float


class Listener:
    """The microphone, the wake word, and the recording state machine."""

    def __init__(self, config: Config, endpointer: Endpointer, speaker: Speaker) -> None:
        self.config = config
        self.endpointer = endpointer
        self.speaker = speaker
        self.frames: queue.Queue[np.ndarray] = queue.Queue()

        from openwakeword.model import Model

        self.wake = Model(wakeword_models=[config.wake_word], inference_framework="onnx")
        preroll_frames = max(1, (config.preroll_ms * SAMPLE_RATE // 1000) // WAKE_FRAME)
        self.preroll: collections.deque[np.ndarray] = collections.deque(maxlen=preroll_frames)

    def _drain(self) -> None:
        """
        Throws away audio captured while we were talking.

        Frames queue up faster than they are read, so when `say` exits there is still buffered audio
        from while it was speaking. Without dropping it, the tail of a question arrives as the start
        of the answer: "shall I send that now?" followed by "yes" is heard as "that now? Yes."
        """
        try:
            while True:
                self.frames.get_nowait()
        except queue.Empty:
            pass

    def _on_audio(self, indata, _frames, _time, status) -> None:
        if status:
            log(f"audio status: {status}")
        self.frames.put(indata[:, 0].copy())

    def run(self, on_utterance, on_confirm) -> None:
        import sounddevice as sd

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=WAKE_FRAME,
            device=self.config.input_device,
            callback=self._on_audio,
        ):
            spoken_wake = Path(self.config.wake_word).stem.replace("_", " ").split(" v0")[0]
            log(f'{self.config.name} is listening. Say "{spoken_wake}" to wake her.')
            if self.config.name.lower() not in spoken_wake.lower():
                # The name and the trigger are different things, and seeing them side by side
                # without explanation reads as a bug rather than a limitation.
                log(f'  (the wake phrase is a trained model, and there is not one for '
                    f'"{self.config.name}" yet — voice/README.md has how to make one)')
            speaking = False
            while True:
                frame = self.frames.get()

                # Our own speech goes out through the speakers and comes back in; scoring it would
                # let the daemon wake itself up.
                if self.speaker.is_speaking:
                    self.wake.reset()
                    self.preroll.clear()
                    speaking = True
                    continue
                if speaking:
                    speaking = False
                    self._drain()
                    continue

                self.preroll.append(frame)
                scores = self.wake.predict(frame)
                score = max(scores.values()) if scores else 0.0
                if score < self.config.wake_threshold:
                    continue

                log(f"wake ({score:.2f})")
                self._converse(on_utterance, on_confirm)
                self.wake.reset()
                self.preroll.clear()

    def _converse(self, on_utterance, on_confirm) -> None:
        """
        Handles one exchange: an instruction, then a question about whether to send it.

        Asking is what makes voice safe to leave unattended. Transcription mishears, and the harness
        acts without asking, so nothing is sent on a guess: a clear yes sends, a no or silence does
        not, and the words stay in the input line either way to be corrected or sent by hand.
        """
        utterance = self._record(self.config.lead_in_ms)
        if utterance is None:
            return
        if on_utterance(utterance) != "text" or not self.config.confirm:
            return

        for _ in range(self.config.confirm_rounds):
            self.speaker.say(self.config.confirm_phrase)
            # Recording waits out our own question, so the answer window starts when it ends.
            reply = self._record(self.config.confirm_ms)
            if reply is None:
                log("no answer; leaving it unsent")
                return
            if on_confirm(reply) != "more":
                return
        log("asked enough; leaving it unsent")

    def _record(self, lead_in_ms: int) -> Utterance | None:
        """Records until the speaker stops, or gives up if they never start."""
        config = self.config
        self.endpointer.reset()

        collected: list[np.ndarray] = list(self.preroll)
        carry = np.zeros(0, dtype=np.int16)
        started = False
        speaking = False
        speech_ms = 0
        silence_ms = 0
        waited_ms = 0

        while True:
            frame = self.frames.get()

            # The acknowledgement is still coming out of the speakers when the follow-up window
            # opens, and a microphone hears it. Recording through it puts "Got it" into the
            # instruction, so wait it out and start the utterance from scratch afterwards.
            if self.speaker.is_speaking:
                speaking = True
                continue
            if speaking:
                speaking = False
                self._drain()
                self.endpointer.reset()
                collected.clear()
                carry = np.zeros(0, dtype=np.int16)
                started = False
                speech_ms = silence_ms = waited_ms = 0
                continue

            collected.append(frame)
            frame_ms = len(frame) * 1000 // SAMPLE_RATE

            # openWakeWord reads 1280-sample frames and Silero reads 512, so the remainder of each
            # frame is carried into the next.
            carry = np.concatenate([carry, frame])
            voiced = False
            while len(carry) >= VAD_FRAME:
                window, carry = carry[:VAD_FRAME], carry[VAD_FRAME:]
                if self.endpointer.speech_probability(window) >= config.vad_threshold:
                    voiced = True

            if voiced:
                started = True
                speech_ms += frame_ms
                silence_ms = 0
            elif started:
                silence_ms += frame_ms
            else:
                waited_ms += frame_ms

            if started and silence_ms >= config.silence_ms:
                break
            if not started and waited_ms >= lead_in_ms:
                log("nothing said")
                return None
            if speech_ms + silence_ms >= config.max_utterance_ms:
                log("hit the utterance ceiling")
                break

        audio = np.concatenate(collected).astype(np.float32) / 32768.0
        return Utterance(audio=audio, seconds=len(audio) / SAMPLE_RATE)


@dataclass
class Intent:
    """What to do with an utterance: type it, act on it, or type it and then act."""

    action: str
    text: str = ""
    then: str | None = None


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text)).strip().lower()


def classify_confirmation(text: str) -> str:
    """
    Reads an answer to "shall I send that now?".

    Anything that is not a clear yes or no is treated as more of the instruction rather than as an
    answer, so a sentence like "and save it as notes.md" extends what is pending instead of being
    guessed at. Only an explicit yes sends.
    """
    answer = _normalise(text)
    if answer in YES_PHRASES:
        return "yes"
    if answer in NO_PHRASES:
        return "no"
    return "more"


def classify(text: str) -> Intent:
    """
    Decides what an utterance is.

    A control phrase counts when it is the whole utterance, or when it is the final sentence of one:
    "check the weather in Paris. Go." is how the instruction and the send actually get said, in one
    breath with no pause between them for the endpointer to cut on.

    Requiring a sentence of its own is what keeps "tell me where to go" and "go ahead and write the
    file" as dictation. Whisper punctuates, and that punctuation is the only signal available for
    where the instruction stopped.
    """
    whole = _normalise(text)
    if whole in ABORT_PHRASES:
        return Intent("abort")
    if whole in SUBMIT_PHRASES:
        return Intent("submit")

    sentences = re.findall(r"[^.!?]+[.!?]*", text)
    if len(sentences) >= 2:
        last = _normalise(sentences[-1])
        body = "".join(sentences[:-1]).strip()
        if body and last in SUBMIT_PHRASES:
            return Intent("text", body, "submit")
        if body and last in ABORT_PHRASES:
            return Intent("text", body, "abort")

    return Intent("text", text.strip())


def resolve_model_dir() -> Path:
    import openwakeword

    return Path(openwakeword.__file__).parent / "resources" / "models"


def ensure_models(model_dir: Path) -> None:
    if (model_dir / "silero_vad.onnx").exists():
        return
    log("downloading openWakeWord models (first run only)")
    import openwakeword.utils

    openwakeword.utils.download_models()


def parse_args(argv: list[str]) -> Config:
    defaults = Config()
    parser = argparse.ArgumentParser(description="Voice front end for the dsh harness.")
    parser.add_argument("--wake-word", default=defaults.wake_word, help="openWakeWord model name")
    parser.add_argument("--wake-threshold", type=float, default=defaults.wake_threshold)
    parser.add_argument("--socket", type=Path, default=defaults.socket_path)
    parser.add_argument("--whisper", default=defaults.whisper_repo)
    parser.add_argument("--silence-ms", type=int, default=defaults.silence_ms)
    parser.add_argument("--lead-in-ms", type=int, default=defaults.lead_in_ms)
    parser.add_argument("--confirm-ms", type=int, default=defaults.confirm_ms)
    parser.add_argument("--confirm-phrase", default=defaults.confirm_phrase)
    parser.add_argument("--no-confirm", action="store_true", help="send dictation without asking")
    parser.add_argument("--name", default=defaults.name, help="what it calls itself in logs and speech")
    parser.add_argument("--max-utterance-ms", type=int, default=defaults.max_utterance_ms)
    parser.add_argument("--vad-threshold", type=float, default=defaults.vad_threshold)
    parser.add_argument("--voice", dest="voice_name", default=None, help="a `say` voice, e.g. Samantha")
    parser.add_argument("--device", dest="input_device", default=None, help="input device index or name")
    parser.add_argument("--no-speak", action="store_true", help="stay silent; only send text")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    if args.list_devices:
        import sounddevice as sd

        print(sd.query_devices())
        raise SystemExit(0)

    device: int | str | None = args.input_device
    if isinstance(device, str) and device.isdigit():
        device = int(device)

    return Config(
        wake_word=args.wake_word,
        wake_threshold=args.wake_threshold,
        socket_path=args.socket,
        whisper_repo=args.whisper,
        vad_threshold=args.vad_threshold,
        silence_ms=args.silence_ms,
        lead_in_ms=args.lead_in_ms,
        confirm=not args.no_confirm,
        confirm_phrase=args.confirm_phrase,
        confirm_ms=args.confirm_ms,
        name=args.name,
        max_utterance_ms=args.max_utterance_ms,
        speak=not args.no_speak,
        voice_name=args.voice_name,
        input_device=device,
        verbose=args.verbose,
    )


def main(argv: list[str]) -> int:
    config = parse_args(argv)
    model_dir = resolve_model_dir()
    ensure_models(model_dir)

    speaker = Speaker(config.speak, config.voice_name)
    harness = HarnessLink(config.socket_path, speaker)
    transcriber = Transcriber(config.whisper_repo, config.verbose)
    endpointer = Endpointer(model_dir, config.vad_threshold)
    listener = Listener(config, endpointer, speaker)

    def transcribe_or_none(utterance: Utterance) -> str | None:
        text = transcriber.transcribe(utterance.audio)
        if not text:
            log("nothing transcribed")
        return text or None

    def handle(utterance: Utterance) -> str:
        # An acknowledgement only earns its place when nothing else will be said: with confirmation
        # on, the question that follows a couple of seconds later says the same thing and more.
        if not config.confirm:
            speaker.say(config.ack_phrase)

        text = transcribe_or_none(utterance)
        if text is None:
            return "empty"

        intent = classify(text)
        log(f"{intent.action}{'+' + intent.then if intent.then else ''}: {intent.text or text}")

        if intent.action == "abort":
            harness.send({"type": "abort"})
            return "abort"
        if intent.action == "submit":
            harness.send({"type": "submit"})
            return "submit"

        harness.send({"type": "text", "text": intent.text})
        if intent.then == "submit":
            harness.send({"type": "submit"})
            return "submit"
        if intent.then == "abort":
            harness.send({"type": "abort"})
            return "abort"
        return "text"

    def confirm(utterance: Utterance) -> str:
        """Answers the send question, or takes the reply as more of the instruction."""
        text = transcribe_or_none(utterance)
        if text is None:
            return "no"

        answer = classify_confirmation(text)
        log(f"answer ({answer}): {text}")

        if answer == "yes":
            harness.send({"type": "submit"})
        elif answer == "more":
            # Not an answer, so it belongs to the instruction. The harness appends it, and the
            # question comes round again for the longer version.
            harness.send({"type": "text", "text": text})
        return answer

    try:
        listener.run(handle, confirm)
    except KeyboardInterrupt:
        log("stopping")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
