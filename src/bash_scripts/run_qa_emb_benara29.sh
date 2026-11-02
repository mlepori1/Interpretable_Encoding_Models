#!/usr/bin/env bash
# Run directly in an activated environment, or submit with your cluster's
# account/partition/resource arguments. For Slurm, submit from the repo root
# or set PROJECT_DIR explicitly. PYTHON_BIN can select a particular interpreter.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${PROJECT_DIR:-}" ]]; then
    if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        PROJECT_DIR="$SLURM_SUBMIT_DIR"
    else
        PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
    fi
fi

if [[ -z "${PYTHON_BIN:-}" ]]; then
    if command -v python >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python)"
    else
        echo "Activate a Python environment or set PYTHON_BIN to its interpreter." >&2
        exit 1
    fi
fi

QA_EMB_CACHE_DIR="${QA_EMB_CACHE_DIR:-$PROJECT_DIR/features/benara2024/model_cache}"

cd "$PROJECT_DIR"
exec "$PYTHON_BIN" src/featurize_qa_emb.py \
    --cache-dir "$QA_EMB_CACHE_DIR" \
    "$@"
