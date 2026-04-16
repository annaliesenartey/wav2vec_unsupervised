# What This Fork Contains

This repository is a working fork of a wav2vec-U (unsupervised) pipeline.

The fork is intentionally kept small: large datasets, caches, and training outputs are ignored via `.gitignore` and remain local.

## Top-level tracked files

These files are committed and are the main entry points for running the pipeline:

- Shell entry points: `run_setup.sh`, `run_wav2vec.sh`, `run_gans.sh`, `run_eval.sh`
- Helper shells: `setup_functions.sh`, `wav2vec_functions.sh`, `gans_functions.sh`, `eval_functions.sh`, `utils.sh`
- Python: 
- `.py`
- Docs: `README.md`, `cuda_installation.txt`, `cuda_installation.png`
- Tooling: `.python-version`

Approx. size of the tracked top-level files (excluding subfolders): ~228K.

## Included folders

### `fairseq_/` (~86M)

Vendored Fairseq source tree used for training/inference.

What it contains (high level):

- `fairseq_/fairseq/` (~34M): the core Fairseq Python package
- `fairseq_/examples/` (~17M): example projects (includes wav2vec-U unsupervised example code/configs)
- `fairseq_/docs/` (~2.8M): documentation
- `fairseq_/tests/` (~632K): tests
- `fairseq_/alignment_train_cpu_binding*.so` (~9.7M): a compiled extension present in this checkout

Notes:

- This is included so you can run the project without relying on a separately installed Fairseq checkout.

### `kenlm/` (~33M)

KenLM language model tooling used by decoding / LM scoring steps.

What it contains (high level):

- `kenlm/lm/` (~1.1M): LM core sources
- `kenlm/util/` (~840K): utilities
- `kenlm/python/` (~480K): python bindings / wrapper code
- `kenlm/build/` (~24M): local build outputs (present in this checkout)

Notes:

- `kenlm/build/` is currently part of this folder size; if you want the fork smaller, we can add `kenlm/build/` to `.gitignore` and/or remove it before pushing.

## Not pushed (kept local)

These are ignored by the root `.gitignore` by default:

- `data/` (very large): datasets, extracted features, text, etc.
- `.hf_cache/` (very large): Hugging Face cache
- `outputs/`, `multirun/`: hydra runs, logs, checkpoints
- `venv/`: local Python environment
- `pre-trained/`: large model artifacts
- `__pycache__/`, `*.log`, `logs/`: runtime outputs

If you want any of these included in the fork, say which paths and we can adjust the ignore list.