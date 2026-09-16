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
submitted**, and Amy asks before sending:

> **"Hey Amy, check the weather in Paris"**
> *(the line fills in, and you can read it)*
> **"Shall I send that now?"**
> **"yes"**

- **"yes"**, "yeah", "sure", "send it", "go" — sends it.
- **"no"**, "not yet", "wait", "don't send it" — leaves it in the line, to correct or send by hand.
- **Silence** — the same as no. She waits three seconds, then goes back to idle.
- **Anything else** is taken as more of the instruction, not as an answer: say *"and save it as
  notes.md"* and it is appended, then she asks again. Three rounds, then she stops asking.

Nothing is ever sent without a clear yes, which is the point: transcription mishears, and the harness
acts without asking a second time.

If the send is already in what you said — *"check the weather in Paris. Go."* — she sends it without
asking, since you have already answered the question. A control phrase counts when it is the whole
utterance or its final sentence, which is what keeps *"tell me where to go"* and *"go ahead and write
the file"* as dictation.

**"stop"**, said on its own after the wake word, aborts a running turn. It is the Escape key.

`--no-confirm` sends dictation straight away, `--confirm-ms` changes the three seconds, and
`--confirm-phrase` changes what she asks.

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
| `--confirm-ms` | `3000` | How long to wait for an answer to "shall I send that now?". |
| `--confirm-phrase` | *"Shall I send that now?"* | What she asks. |
| `--no-confirm` | off | Send dictation without asking. |
| `--name` | `Amy` | What she calls herself. |
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

To actually say "Hey Amy", a model has to be trained on synthetic speech of that phrase. A complete
training config and instructions are in [`training/`](training/README.md) — roughly an hour on a free
Colab GPU, with nothing installed here. The result is one `.onnx` file, and `wakeword_models` accepts
a path, so nothing else changes:

```sh
dsh-voice --wake-word ~/.dsh/wake/hey_amy.onnx
```

Validating the pipeline on `hey_jarvis` first is worth the ten minutes: if something is wrong with
the microphone, the socket or the thresholds, better to find out before spending an hour training.

## Speaking

The send question doubles as the acknowledgement, so there is no separate "Got it" to sit through:
hearing *"shall I send that now?"* confirms both that the words arrived and what is about to happen
with them. With `--no-confirm` there is nothing else to say, so the acknowledgement comes back.

The closing summary comes from the harness, which condenses the model's reply to a sentence or two.

While she is speaking the microphone is ignored, and audio captured during it is thrown away rather
than processed late. Frames queue faster than they are read, so without that the tail of a question
arrives as the start of the answer — *"shall I send that now?"* followed by *"yes"* is heard as
*"that now? Yes."*, which is neither a yes nor a no.
