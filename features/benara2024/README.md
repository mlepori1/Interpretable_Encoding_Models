# QA-Emb-29 reproduction

This directory contains the fixed 29-question vocabulary selected for the
fMRI model reported by Benara et al. (2024), transferred without selecting
questions on this study's brain responses. The existing `qa_emb.csv` is the
matrix used by the repository's `--use_benara2024` regression option.

The generator, saved binary answers, and provenance make this matrix
reproducible without changing the downstream regression code.

## Environment and input checks

From the repository root, use Python 3.10:

```bash
python3.10 -m venv .venv-qa-emb
source .venv-qa-emb/bin/activate
python -m pip install -r requirements-qa-emb.txt
python src/featurize_qa_emb.py --dry-run
```

The dry run verifies the frozen 29 questions and the order of the 200
sentences against all eight `data/processed_csvs_anon/p*/stimset.csv` files.
It neither loads models nor writes outputs. The requirements describe the
tested software environment, not a historical lock for the original inference.

## Rebuild from saved answers on CPU

The three answer matrices are already included. Rebuild in a temporary
directory to preserve the reference artifacts:

```bash
QA_REBUILD_DIR="$(mktemp -d)"
cp -r features/benara2024/raw_answers "$QA_REBUILD_DIR/raw_answers"
python src/featurize_qa_emb.py --aggregate-only --output-dir "$QA_REBUILD_DIR"
python -m pytest -q tests/test_featurize_qa_emb.py
```

This path needs only NumPy, pandas, and pytest. Tests check exact agreement
with the existing feature matrix and its checksums, rescore the saved tokens,
and reject answer caches with mismatched stimuli, questions, prompts, or models.

## Regenerate answers on GPU

Fresh inference requires a CUDA GPU, adequate checkpoint storage, and
Hugging Face access to `mistralai/Mistral-7B-Instruct-v0.2` and
`meta-llama/Meta-Llama-3-8B-Instruct`. Accept any required model terms and
authenticate before running. Use a fresh output directory and the recorded
immutable model revisions:

```bash
QA_REPRO_DIR="$(mktemp -d features/benara2024/reproduction-XXXXXX)"
bash src/bash_scripts/run_qa_emb_benara29.sh \
  --output-dir "$QA_REPRO_DIR" --configs mistral_zero_shot \
  --revision 63a8b081895390a26e140280378bc85ec8bce07a --dtype bfloat16
bash src/bash_scripts/run_qa_emb_benara29.sh \
  --output-dir "$QA_REPRO_DIR" --configs llama_zero_shot llama_five_shot \
  --revision 8afb486c1db24fe5011ec46dfbe5b5dccdb575c2 --dtype bfloat16
```

The launcher uses the active `python`, or `PYTHON_BIN` when explicitly set;
it contains no cluster-specific submission settings. `QA_EMB_CACHE_DIR`
overrides the default untracked cache at `features/benara2024/model_cache`.

The second command also creates the final ensemble. Default batch sizes are
32, 64, and 16, respectively, with greedy one-token decoding and a maximum
input length of 2048 tokens. Resume with the same arguments. Running in the
default output directory reuses the complete saved answers, so use a fresh
directory to test inference. The default `--revision main` is mutable; use
the pinned revisions above for reproduction.

Historical metadata records the resolved model revisions but not package
versions or resolved dtype. The commands explicitly choose bfloat16;
compare regenerated answers with the saved ones rather than assuming
bitwise-identical inference across different hardware/software environments.

## Outputs and interpretation

- `qa_emb.csv` and `qa_emb.pkl`: the unscaled `(200, 29)` feature matrix.
- `questions.csv`: question text and indices into the original candidate pool.
- `stimuli.csv`: row-to-sentence alignment.
- `raw_answers/*.npz`: three binary answer matrices and decoded output tokens.
- `manifest.json`: model revisions, prompts, generation settings, and checksums.

Each feature averages three binary answers: Mistral zero-shot, Llama
zero-shot, and Llama five-shot, giving values 0, 1/3, 2/3, or 1. Following
the authors' released implementation, each evaluation generates one token;
the answer is 1 if the decoded token contains `yes` (case-insensitive), and
0 otherwise. Any downstream scaling must be fit within training folds.

The upstream implementation is
[Interpretable Embeddings](https://github.com/csinva/interpretable-embeddings/tree/786ea8195c2236c0275528f2e0beb0fb2dc4e4d8),
at commit `786ea8195c2236c0275528f2e0beb0fb2dc4e4d8`.
