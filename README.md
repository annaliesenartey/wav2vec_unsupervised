# wav2vec_unsupervised

Scripts for the Fairseq **wav2vec 2.0 unsupervised** pipeline (feature prep, GAN training, decoding). Upstream reference:

https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/unsupervised/README.md

**End-to-end run instructions** (setup → `run_wav2vec.sh` → `run_gans.sh` → `run_eval.sh`) are in **`README_PIPELINE.md`**.

Tested with **PyTorch 2.3.0** and **CUDA 12.3** (match versions to your GPU).

---

## Quick start

1. `chmod +x` the `run_*.sh` and `*_functions.sh` scripts (see `README_PIPELINE.md`).
2. `./run_setup.sh`
3. `./run_wav2vec.sh` with train/val/test wav directories and unlabeled text.
4. `./run_gans.sh`
5. `./run_eval.sh path/to/checkpoint_best.pt` (path relative to repo root)

Details, env vars, and checkpoints: **`README_PIPELINE.md`**.

What to commit vs keep local: **`README_PUSH_CONTENTS.md`**.

---

## System requirements

- Linux (recommended), NVIDIA GPU + CUDA for training, Python venv, Git.

### CUDA

Use a CUDA toolkit compatible with your GPU and PyTorch. This repo was documented with **CUDA 12.3**:  
https://developer.nvidia.com/cuda-12-3-0-download-archive  

Use `hostnamectl` to pick the correct installer for your OS/arch. See **`cuda_installation.txt`** and **`cuda_installation.png`** for one install flow.

---

## Summary

| Doc | Purpose |
|-----|---------|
| **`README_PIPELINE.md`** | Full pipeline from top to bottom |
| **`README_PUSH_CONTENTS.md`** | Repo layout / git-friendly folders |
| **`cuda_installation.txt`** | Example CUDA install commands |
