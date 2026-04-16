# Pipeline: run from start to finish

Step-by-step guide for this repo’s scripts. Execute commands from the **project root** (folder containing `run_setup.sh`, `fairseq_/`, `data/`, etc.).

Official Fairseq background:  
https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/unsupervised/README.md

Tested with **PyTorch 2.3.0** and **CUDA 12.3** (adjust to your GPU).

---

## 0. One-time: clone, submodules, permissions

```bash
cd /path/to/wav2vec_unsupervised
chmod +x run_setup.sh run_wav2vec.sh run_gans.sh run_eval.sh \
        setup_functions.sh wav2vec_functions.sh gans_functions.sh eval_functions.sh utils.sh
```

If `fairseq_` is a git submodule:

```bash
git submodule update --init --recursive
```

The submodule URL points at **your fork** (`annaliesenartey/fairseq`) so local changes to Fairseq (for example `wav2vec_u.py`) are **not replaced** by upstream Meta Fairseq when others clone this repo. After cloning, submodules resolve to the commit pinned in this repository.

**Maintainers:** after changing files under `fairseq_/`, commit inside the submodule, push the branch to your fork, then commit the updated submodule pointer in this repo:

```bash
cd fairseq_
git status
git add -p && git commit -m "Describe Fairseq change"
git push origin HEAD:wav2vec-u-custom   # or your branch name
cd ..
git add fairseq_
git commit -m "Bump fairseq_ submodule"
```

---

## 1. Configure paths

`utils.sh` sets `DIR_PATH` (project root). Default: `$HOME/gans_project/wav2vec_unsupervised`. Override:

```bash
export WAV2VEC_UNSUPERVISED_ROOT=/absolute/path/to/wav2vec_unsupervised
```

Optional dataset name (used under `data/clustering/<name>/`; default `librispeech_hf`):

```bash
export WAV2VEC_DATASET_NAME=librispeech_hf
```

You need **`fairseq_/`**, **`pre-trained/wav2vec_vox_new.pt`** (see `MODEL` in `utils.sh`), and the data/LM layout your `prepare_*` steps expect.

---

## 2. Environment setup

```bash
./run_setup.sh
```

Installs system deps, Python venv, PyTorch/Fairseq/KenLM, etc. (see `setup_functions.sh`). On machines where CUDA is already installed, parts of this may be redundant.

---

## 3. Data preparation (`run_wav2vec.sh`)

**Four arguments** (required):

1. Training `.wav` directory  
2. Validation `.wav` directory  
3. Test `.wav` directory  
4. Unlabeled text file (one sentence per line)

Use **16 kHz `.wav`** where possible.

```bash
./run_wav2vec.sh \
  "/path/to/train_wavs" \
  "/path/to/val_wavs" \
  "/path/to/test_wavs" \
  "/path/to/unlabeled_sentences.txt"
```

Runs manifests, VAD, `prepare_audio` / `prepare_text`, clustering features under `data/clustering/$WAV2VEC_DATASET_NAME/`, text under `data/text/`. Logs: `data/logs/<dataset>/pipeline.log`.

---

## 4. GAN training (`run_gans.sh`)

After step 3 completes:

```bash
./run_gans.sh
```

- Implements **`train_gans`** in `gans_functions.sh` (Fairseq Hydra, `w2vu` config).
- Tee log: **`data/results/<dataset>/training1.log`**
- Hydra may also write under **`outputs/<date>/<time>/`** (`hydra_train.log`, `checkpoint_best.pt`, etc.).

**Skip if already done:** `data/checkpoints/<dataset>/progress.checkpoint` can mark `train_gans` completed; remove that line to rerun training.

**Short smoke test (400 updates):**

```bash
SANITY_GAN=1 ./run_gans.sh
```

**Hyperparameters** live in **`gans_functions.sh`** (`train_gans`). Advanced defaults: `fairseq_/examples/wav2vec/unsupervised/config/gan/w2vu.yaml`.

---

## 5. Evaluation (`run_eval.sh`)

Pass checkpoint path **relative to project root**:

```bash
./run_eval.sh outputs/2026-04-12/21-41-55/checkpoint_best.pt
```

Runs `w2vu_generate.py` (Viterbi). Outputs under **`data/transcription_phones/`** (`GANS_OUTPUT_PHONES` in `utils.sh`).

---

## Checklist

| Step | Command |
|------|---------|
| Setup | `./run_setup.sh` |
| Features + text | `./run_wav2vec.sh TRAIN VAL TEST TEXT` |
| Train GAN | `./run_gans.sh` |
| Decode | `./run_eval.sh path/to/checkpoint_best.pt` |

---

## See also

- **`README.md`** — project overview and CUDA notes  
- **`README_PUSH_CONTENTS.md`** — what to track in git vs local-only dirs  
- **`restore_configs_from_outputs.sh`** — replay Hydra overrides from `outputs/<date>/<time>/`
