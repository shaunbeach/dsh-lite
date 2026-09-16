# Training a "hey Amy" wake word

`hey_amy.yaml` is a complete openWakeWord training config for the phrase. What it produces is a
`hey_amy.onnx` of a few hundred kilobytes, which the daemon loads with `--wake-word`.

## What is actually being trained

openWakeWord is three stages, and only the last is trained:

```
audio → melspectrogram.onnx   1.1 MB, frozen
      → embedding_model.onnx  1.3 MB, frozen, a general speech embedding
      → a small classifier    ← this
```

The classifier is a couple of `Linear` and `LayerNorm` layers over a 16×96 embedding. Tens of
thousands of parameters. The frozen stack already knows what speech sounds like; the training
teaches a small head to recognise one pattern in its output, which is why this is an hour rather
than a research project.

## You do not record anything

Positives are synthesised: Piper speaks "hey amy" tens of thousands of times across many voices and
three speaking rates. Negatives are three kinds — phonetic near-misses generated from the target,
the extra phrases listed in the config, and precomputed embeddings of large speech, music and
general-audio corpora that openWakeWord hosts. Everything is then convolved with room impulse
responses and mixed with noise, so the model meets the phrase as a microphone across a room
receives it rather than as clean studio audio.

## Run it in Colab

Open openWakeWord's training notebook straight from GitHub — nothing to download:

**[colab.research.google.com/github/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb](https://colab.research.google.com/github/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb)**

Set the runtime to a GPU (*Runtime → Change runtime type → T4*), then run the cells in order. The
notebook installs the training dependencies, clones `piper-sample-generator`, and downloads the
corpora onto Google's machines.

The notebook does **not** take an uploaded config. It loads openWakeWord's own example and edits it
in Python:

```python
config = yaml.load(open("openwakeword/examples/custom_model.yml", 'r').read(), yaml.Loader)
config["target_phrase"] = ["hey sebastian"]
```

So the way to use `hey_amy.yaml` is to let the notebook own the paths, and take the phrase settings
from it. **After that cell**, add one of your own:

```python
# Amy's phrase settings, over the notebook's downloaded paths.
import yaml, urllib.request

amy = yaml.safe_load(urllib.request.urlopen(
    "https://raw.githubusercontent.com/shaunbeach/dsh-lite/main/voice/training/hey_amy.yaml"
).read())

for key in ("target_phrase", "model_name", "custom_negative_phrases", "n_samples", "n_samples_val"):
    config[key] = amy[key]

print(config["target_phrase"], config["model_name"],
      len(config["custom_negative_phrases"]), "custom negatives")
```

Only those five keys. Everything else in the example config — `feature_data_files`,
`batch_n_per_class`, and every path — describes what the notebook downloaded and where it put it, and
overriding those is how a run dies after the generation step rather than before it. The remaining
values in `hey_amy.yaml` (`layer_size`, `steps`, `max_negative_weight`,
`target_false_positives_per_hour`) already match the example exactly, so there is nothing to change.

Then run the rest of the notebook. Generation and augmentation are most of the wall clock; training
itself is minutes.

Colab is the right place for this. Locally it means torch, torchinfo, torchmetrics and `pronouncing`
— about 2 GB of packages that are not needed to *run* the daemon — plus several GB of corpora, all
competing for the 16 GB that the language model wants.

## Using the result

Download `hey_amy.onnx` and point the daemon at it:

```sh
mkdir -p ~/.dsh/wake
# copy hey_amy.onnx there, then
dsh-voice --wake-word ~/.dsh/wake/hey_amy.onnx
```

The startup line becomes `Amy is listening. Say "hey amy" to wake her.` and the note about there
being no model disappears. Nothing else changes; `wakeword_models` accepts a path.

Make it permanent by adding the flag to a shell alias, or symlinking the model into openwakeword's
own `resources/models` directory so `--wake-word hey_amy` resolves by name.

## Tuning, if it misbehaves

| Symptom | Change |
| --- | --- |
| Fires when nobody said it | Raise `--wake-threshold` first. If that costs too much recall, retrain with a lower `target_false_positives_per_hour` and more `custom_negative_phrases` resembling whatever set it off. |
| Misses the phrase | Lower `--wake-threshold`, watching scores with `dsh-voice -v`. If it is consistently below about 0.3, retrain with more `n_samples` and `augmentation_rounds: 2`. |
| Fires on one particular word | Add that word to `custom_negative_phrases` and retrain. This is the targeted fix, and it is cheap. |

The config leans against false positives on purpose: `max_negative_weight: 1500` and
`target_false_positives_per_hour: 0.2`. Missing a wake word costs a repetition; a false one starts
recording the room and can put words in front of an agent that acts on them.

## Why "hey Amy" and not "Amy"

Short wake words have much higher false-alarm rates — there is less to discriminate on, and a single
common name collides with ordinary speech. Every pretrained model openWakeWord ships is a prefixed
phrase or three distinct syllables: `hey_jarvis`, `hey_mycroft`, `hey_rhasspy`, `alexa`. Two
syllables after a "hey" is a much easier target than one name alone.
