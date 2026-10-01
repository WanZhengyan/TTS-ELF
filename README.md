# TTS-ELF

Text-to-speech built on [ELF](https://github.com/lillian039/ELF) (*Embedded Language Flows*), a
flow-matching model originally made for generating **text**. This repo adapts it to generate
**speech** instead, and measures how well that works.

Trained and evaluated on [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) — 24 hours of one
English speaker. Everything here runs on **a single GPU**.

> **Status: the pipeline works end to end, but speech generation does not yet.** The honest
> numbers are in [Results](#results). Read that before investing time.

---

## How it works, in plain terms

Speech is a very long list of numbers — about 24,000 per second. That is far too long to generate
directly. So we use an off-the-shelf **audio codec** (WavTokenizer) that compresses one second of
audio into just **40 whole numbers**, each between 0 and 4095, and can turn those numbers back into
sound. Compression ratio: about 800×.

Our model never touches raw audio. It only has to do this:

```
text  ──▶  [ our model ]  ──▶  40 numbers per second  ──▶  [ codec ]  ──▶  sound
```

The codec is frozen — we never train it. We only train the middle part.

The model works by **starting from pure noise and gradually cleaning it up** into something that
decodes to the right speech, a bit like how image generators work. The text is fed in alongside the
noise at every step, telling it what to say.

One detail worth knowing because it is the main design decision: the text has to be lined up with
the audio timeline. A 5-letter word and a 2-second sound clip have different lengths. We stretch the
text so **every 25-millisecond audio frame carries one character**, by simply repeating each
character over its share of the frames. On LJSpeech that works out to about 2.7 frames per
character.

---

## What you need

| | |
|---|---|
| **GPU** | One, with ~20 GB memory. An H100 is what this was developed on; anything modern works. |
| **Disk** | ~15 GB (dataset 6 GB, preprocessed cache 3.3 GB, codec weights 1.7 GB, rest is small) |
| **Time** | ~15 min setup, ~7 min preprocessing, then 2.5 h for a full training run |
| **OS** | Linux. Slurm scripts are included but optional — every step runs as a plain command. |

If you plan to use the Slurm scripts in `slurm/`, create the output directories first, because Slurm
writes its log file before the script runs:

```bash
mkdir -p logs runs out
```

---

## Setup

### 1. Get the code

```bash
git clone https://github.com/WanZhengyan/TTS-ELF.git
cd TTS-ELF
```

### 2. Make a Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate

# install torch first, matched to your CUDA version
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 3. Get the audio codec

This is a separate project. Clone it and download its weights:

```bash
mkdir -p third_party ckpt
git clone https://github.com/jishengpeng/WavTokenizer.git third_party/WavTokenizer

python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='novateur/WavTokenizer-large-unify-40token',
                filename='wavtokenizer_large_unify_600_24k.ckpt',
                local_dir='ckpt')"
```

### 4. Get LJSpeech

```bash
mkdir -p data && cd data
curl -LO https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -xjf LJSpeech-1.1.tar.bz2
cd ..
```

You should now have `data/LJSpeech-1.1/wavs/` with 13,100 `.wav` files.

---

## Running it

Four steps. Run them in order the first time; after that you can repeat any one on its own.

### Step 1 — Check the codec (1 minute)

Before training anything, measure what the codec itself costs you. This produces one number you
**need** for the next step, so don't skip it.

```bash
python probe_test0.py
```

Prints and saves to `out/test0.json`. The important line:

```
latent_std:  0.26409   <- copy this into step 3 if it differs
```

### Step 2 — Preprocess the dataset (~7 minutes on one GPU)

Runs every clip through the codec once and caches the result, so training doesn't have to redo it.

```bash
python preprocess.py
```

Writes `cache/` (~3.3 GB). You only ever do this once.

### Step 3 — Train (~2.5 hours for 100k steps)

```bash
python train.py \
  --model ELF-XS \
  --steps 100000 \
  --batch 32 \
  --prompt_lo 0 --prompt_hi 0 \
  --out runs/my_run
```

Add `--wandb` to get live curves (needs a wandb account). Progress prints every 50 steps and a
fuller evaluation runs every 5,000.

**Model sizes** — pick with `--model`:

| Name | Parameters | Use for |
|---|---|---|
| `ELF-T` | 6.7 M | Quick smoke test |
| `ELF-XS` | 14.6 M | Default. Same size as Grad-TTS. |
| `ELF-S` | 29.6 M | A bit bigger |
| `ELF-B+` | 118.9 M | For larger datasets, not LJSpeech |

To go faster, try `--model ELF-T --steps 4000 --overfit 100`. That memorises 100 clips in about two
minutes and confirms your setup is correct.

### Step 4 — Listen, and get the numbers

**Make an audio page you can click through:**

```bash
python sample_wav.py --ckpt runs/my_run/ckpt.pt --prompt_frac 0.0
```

Writes `listen.html`. Open it in any browser — the audio is embedded, no server needed. Each clip
gets four tracks so you can hear where quality is lost:

| Track | What it is |
|---|---|
| **Original** | The real recording. The target. |
| **Codec ceiling** | The real recording passed through the codec and back. **Nothing can sound better than this.** |
| **Generated** | What the model produced. |
| **Teacher-forced** | A control: the model's final step is handed the correct answer, skipping the generation part. Tells you whether a problem is in the generation or in the final conversion to numbers. |

**Measure word error rate and naturalness:**

```bash
python eval_metrics.py --ckpt runs/my_run/ckpt.pt --prompt_frac 0.0 --n 100
```

Transcribes the generated audio with Whisper and compares it to the input text. Lower is better.
Also reports UTMOS, an automatic estimate of how natural it sounds (1–5, higher is better).

---

## Results

Measured on 100 held-out clips, text-only input (no audio prompt), `ELF-XS` at 100k steps.

| | Word error rate | Naturalness (UTMOS) |
|---|---|---|
| Real recording | **2.0 %** | 4.39 |
| Codec ceiling | **5.4 %** | 3.91 |
| Teacher-forced | **5.3 %** | 3.84 |
| **Generated** | **99.2 %** | 2.88 |

Reading it:

- Real recordings score 2.0 %, not 0 % — that is the transcriber's own error rate, the floor.
- The codec costs **3.4 points** (2.0 → 5.4). That is the price of 800× compression, and it applies
  no matter how good the model gets.
- **Teacher-forced reaches 5.3 %, matching the ceiling.** This is the one clear success: the part of
  the model that converts its output into codec numbers works, and costs nothing.
- **Generated is 99.2 %** — unintelligible. The generation part does not work yet.

So the failure is isolated to one component, which is useful to know. For reference, the standard
small models on this dataset get around 2.1 % (Grad-TTS, Matcha-TTS), so **the codec ceiling of
5.4 % is already worse than those baselines** — a different codec would be needed to compete even
with a perfect model.

### What we learned along the way

- **The character embeddings learned real phonetics by themselves.** Without being told anything
  about pronunciation, `t` ended up closest to `k`, `p`, `d` (all made by stopping airflow), `s`
  closest to `f`, `z` (all hissing sounds), and vowels grouped away from consonants.
- **The text signal enters the model about 27× weaker than the noise it is competing with**, because
  of a scale mismatch between the two. This is the most likely cause of the generation failure and
  the next thing to fix.
- Training loss going down did **not** mean generation was working. The model learned the easy half
  of the job (cleaning up noise) while never learning the hard half (deciding what to say).

---

## Repo layout

| | |
|---|---|
| `probe_test0.py` | Step 1. Measures the codec before you train on it. |
| `preprocess.py` | Step 2. Caches the dataset. |
| `train.py` | Step 3. Training, sampling and evaluation. |
| `sample_wav.py` | Step 4. Makes the clickable audio page. |
| `eval_metrics.py` | Step 4. Word error rate and naturalness. |
| `diagnose.py` | Pinpoints which component is failing. |
| `elf_tts/` | The model. A near-unmodified copy of ELF — see below. |
| `slurm/` | Optional job scripts for a Slurm cluster. |
| `architecture.html` | **The full technical write-up.** Open in a browser. |
| `ljspeech-probe.html` | Why the experiments are ordered the way they are. |

The two HTML files are the detailed documentation, with diagrams. This README is the short version.

---

## What changed from ELF

Deliberately almost nothing, so that any result is attributable to the change of domain rather than
to a redesign.

- `elf_tts/model.py` is ELF's model file with **one shape change**: its input projection widened
  from 2 blocks to 3, to make room for the text. Plus extra size presets.
- Text joins on the **channel** axis, not by being placed in front of the audio in the sequence.
  This hands the model the text-to-audio alignment instead of making it search for one.
- Text is a small learned character table (40 entries). ELF used a frozen T5 encoder; there is no T5
  here, and no pronunciation dictionary.
- Everything else — the loss, the noise schedule, the sampler, the self-conditioning — is ELF's.

---

## Credits

- [ELF: Embedded Language Flows](https://github.com/lillian039/ELF) — the model this is built on
- [WavTokenizer](https://github.com/jishengpeng/WavTokenizer) — the audio codec
- [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) — the dataset

Both upstream projects are MIT licensed and are **not** vendored here; the setup steps clone them.
