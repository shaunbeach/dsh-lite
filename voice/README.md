# Voice front end

A daemon that listens for a wake word, transcribes what follows, and types it into a running dsh.

It is a separate process on purpose. The harness stays a lean TypeScript program with four
dependencies; the microphone, the wake word, the endpointing and the transcription live here, and
the only thing that crosses between them is text over `~/.dsh/input.sock`.

```
openWakeWord  ->  Silero VAD  ->  MLX Whisper  ->  classify  ->  ~/.dsh/input.sock
 is anyone        have they       what did        instruction,     the harness
 talking?         finished?       they say?       "send it",
                                                  or "stop"?
```

## Setup

```sh
python3 -m venv ~/.dsh/voice-venv
~/.dsh/voice-venv/bin/pip install -r ~/Documents/dsh-lite/voice/requirements.txt
```

The openWakeWord models, including the Silero VAD the daemon uses for endpointing, are downloaded on
first run. Whisper weights come from the Hugging Face cache.

macOS will ask for microphone access the first time. The prompt names the terminal the daemon was
started from, not Python.

## Running

Start dsh, type `/voice` to open the socket, then in another terminal:

```sh
~/Documents/dsh-lite/voice/dsh-voice
```

`dsh-voice` finds the virtual environment and the script relative to itself, so it works from any
directory. Symlink it onto your PATH to shorten that:

```sh
ln -s ~/Documents/dsh-lite/voice/dsh-voice ~/.local/bin/dsh-voice
```

Then it is just `dsh-voice`. Options are passed straight through, for example `dsh-voice -v`.

Say **"hey jarvis"**, then your instruction. The words land in the dsh input line without being
submitted, so a misheard one can be seen and corrected. Say **"send it"** or **"go"** to run it, or
**"stop"** to abort a running turn.

Control phrases only count as commands when they are the whole utterance: *"go ahead and write the
file"* is dictated, not treated as a request to send.

## Memory

Whisper is loaded for each clip and dropped immediately after, which measured 1618 MB held during
transcription and 0.1 MB after. Loading from the local cache is memory-mapped and takes about two
tenths of a second, so keeping it resident would cost 1.6 GB that the language model needs and save
almost nothing. Transcription of a five-second clip takes roughly two seconds.

## Options

| Flag | Default | |
| --- | --- | --- |
| `--wake-word` | `hey_jarvis` | Any openWakeWord model name. |
| `--wake-threshold` | `0.5` | Raise it if the wake word fires on its own. |
| `--silence-ms` | `800` | Silence that ends an utterance. |
| `--lead-in-ms` | `3000` | How long to wait for speech after the wake word. |
| `--whisper` | `mlx-community/whisper-large-v3-turbo` | Any MLX Whisper repo. |
| `--voice` | system voice | A `say` voice, e.g. `Samantha`. |
| `--no-speak` | off | Send text without speaking. |
| `--device` / `--list-devices` | default input | Choose a microphone. |
| `-v` | off | Log timings and memory held after each unload. |

## A custom wake word

`hey_jarvis` is one of openWakeWord's pretrained models, alongside `alexa`, `hey_mycroft` and
`hey_rhasspy`. A word of your own — "Hey Amy" — needs a model trained from synthetic speech through
openWakeWord's training pipeline, which is automated but takes hours rather than minutes. Point
`--wake-word` at the resulting `.onnx` once it exists; nothing else changes.

## Speaking

Acknowledgement is local: the daemon says "Got it" the moment speech ends, because the harness has
nothing to confirm until it has been given the text, and waiting for it would leave a silence
exactly where the speaker expects an answer. The closing summary does come from the harness, which
condenses the model's reply to a sentence or two.

While the daemon is speaking it ignores the microphone, so its own output cannot trigger the wake
word through the speakers.
