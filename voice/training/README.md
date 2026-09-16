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

Upload **`hey_amy_training.ipynb`** (in this directory) to [colab.research.google.com](https://colab.research.google.com),
set *Runtime → Change runtime type → **T4 GPU***, and run the cells in order.

It is openWakeWord's `automatic_model_training.ipynb` cut down to this wake word: 18 cells instead of
22, with the phrase already configured and the TensorFlow toolchain removed. That toolchain exists
only to emit a `.tflite` next to the ONNX — the daemon loads ONNX — and its three pins
(`tensorflow-cpu==2.8.1`, `tensorflow_probability==0.16.0`, `onnx_tf==1.10.0`) are from 2022 and are
the likeliest thing in the original to fail to install today. The training script calls the tflite
conversion unconditionally at the very end, after the ONNX is already written, so that last step
fails harmlessly and the cell tolerates it.

The notebook pulls the phrase settings from `hey_amy.yaml` in this repository at run time, so editing
that file changes the next training run without touching the notebook.

### If the install cell fails

**`No matching distribution found for piper-phonemize`** means Colab's Python is 3.13 or newer.
`piper-phonemize` builds espeak-ng and its newest Linux wheel is cp312, so there is nothing to
install; the notebook checks the version first and says so rather than failing three cells later.
The fix is a runtime with Python 3.12 or earlier — on Colab, *Runtime → Change runtime type* has
offered older images at times, and the alternative is a local machine with 3.11 or 3.12.

Two things the original notebook gets wrong today, both already fixed here: `piper-sample-generator`
must be pinned to **v2.0.0**, because master has been restructured into a package and no longer
exposes the `generate_samples` module openWakeWord's trainer imports; and openwakeword must be
installed with `--no-deps`, because its dependency list pins `tflite-runtime` and `speexdsp-ns`,
neither of which resolves on a current Colab and neither of which training needs.

Budget 45–60 minutes. Generating the speech is nearly all of it; training itself is minutes.

Colab rather than locally: training wants torch, torchinfo, torchmetrics, speechbrain,
audiomentations and `pronouncing` — about 2 GB that the daemon does not need to run — plus several GB
of corpora, all competing for the 16 GB the language model wants.

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
