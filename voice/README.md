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

Say the wake word, then your instruction. The words land in the dsh input line **without being
submitted**, so a misheard one can be seen and corrected first.

To send it, either press Enter, or just keep talking — after dictation the microphone stays live for
eight seconds, so **"go"** or **"send it"** needs no second wake word:

> "Hey Amy, write me a template for a project readme and save it as notes.md"
> *(the line fills in)*
> "go"

**"stop"** aborts a running turn, and is the Escape key. Control phrases only count when they are the
whole utterance: *"go ahead and write the file"* is dictated, not read as a request to send. The
follow-up window closes after a send or a stop, so the wake word is needed again for the next
instruction; `--follow-up-ms 0` turns it off entirely.

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

## The name, and the wake word

It calls itself **Amy** — in the log line at startup and in anything it speaks — and `--name` changes
that.

The *trigger phrase* is a separate thing, and it is still `hey_jarvis`, because a wake word is a
trained neural model rather than a string to match. openWakeWord ships six: `alexa`, `hey_jarvis`,
`hey_mycroft`, `hey_rhasspy`, `timer` and `weather`. None of them is "Amy".

To actually say "Hey Amy", a model has to be trained on synthetic speech of that phrase. Two routes:

- **openWakeWord's training notebook**, run on a free Colab GPU. Roughly an hour, mostly unattended,
  and nothing is installed on your machine.
- **Locally**, with `openwakeword.train`. It needs torch, torchinfo and torchmetrics — around 2 GB of
  install, competing for the same memory the language model wants, on top of generating tens of
  thousands of training clips.

The notebook is the better trade. Either way the result is one `.onnx` file, and `wakeword_models`
accepts a path, so nothing else changes:

```sh
dsh-voice --wake-word ~/.dsh/wake/hey_amy.onnx
```

Validating the pipeline on `hey_jarvis` first is worth the ten minutes: if something is wrong with
the microphone, the socket or the thresholds, better to find out before spending an hour training.

## Speaking

Acknowledgement is local: the daemon says "Got it" the moment speech ends, because the harness has
nothing to confirm until it has been given the text, and waiting for it would leave a silence
exactly where the speaker expects an answer. The closing summary does come from the harness, which
condenses the model's reply to a sentence or two.

While the daemon is speaking it ignores the microphone, so its own output cannot trigger the wake
word through the speakers.
