#!/usr/bin/env bash
set -euo pipefail

# Helper: print a stable set of Hydra overrides for a past run,
# so you can rerun training with the same settings even if the submodule/config changes.
#
# Example:
#   ./restore_configs_from_outputs.sh outputs/2026-04-12/02-38-08
#
# Output: prints `fairseq-hydra-train` override args to stdout.

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <outputs/<date>/<time>>" >&2
  exit 2
fi

RUN_DIR="$1"
OVERRIDES="${RUN_DIR%/}/.hydra/overrides.yaml"

if [[ ! -f "$OVERRIDES" ]]; then
  echo "Missing overrides file: $OVERRIDES" >&2
  exit 1
fi

python3 - "$OVERRIDES" <<'PY'
import sys, pathlib
path = pathlib.Path(sys.argv[1])
lines = path.read_text().splitlines()
for ln in lines:
    ln = ln.strip()
    if not ln:
        continue
    # stored as YAML list items: "- key=value"
    if ln.startswith("- "):
        print(ln[2:])
    else:
        print(ln)
PY

