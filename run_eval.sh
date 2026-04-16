#!/bin/bash
# Usage: ./run_eval.sh <checkpoint.pt path relative to DIR_PATH>
# Example: ./run_eval.sh outputs/2026-04-11/15-46-08/checkpoint_best.pt

set -eo pipefail

if [ -z "${1:-}" ]; then
  echo "Usage: $0 <checkpoint.pt path relative to DIR_PATH (see utils.sh)>" >&2
  exit 1
fi

# shellcheck source=eval_functions.sh
source "$(dirname "$0")/eval_functions.sh"

create_dirs
activate_venv
setup_path

transcription_gans_viterbi
