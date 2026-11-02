#!/usr/bin/env python3
"""Generate the frozen QA-Emb-29 baseline for the paper's sentence stimuli.

This reproduces the feature construction in Benara et al. (2024), without
their dataset-specific feature-selection or encoding-model steps. The released
implementation averages three hard yes/no matrices: Mistral zero-shot, Llama
zero-shot, and Llama five-shot.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import pickle
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "features" / "benara2024"
)
DEFAULT_QUESTIONS_FILE = DEFAULT_OUTPUT_DIR / "questions.csv"
DEFAULT_STIMULUS_FILE = (
    REPO_ROOT / "data" / "processed_csvs_anon" / "p1" / "stimset.csv"
)
DEFAULT_PARTICIPANTS_DIR = REPO_ROOT / "data" / "processed_csvs_anon"

UPSTREAM_REPOSITORY = "https://github.com/csinva/interpretable-embeddings"
UPSTREAM_COMMIT = "786ea8195c2236c0275528f2e0beb0fb2dc4e4d8"
FROZEN_QUESTIONS_SHA256 = (
    "e30f62cc6fc8cdf01a0fd329424e2e75a0dc143af1635c206b4aaf314cd1623a"
)

MISTRAL_ZERO_SHOT_PROMPT = (
    "<s>[INST]'Input text: {example}\n"
    "Question: {question}\n"
    "Answer with yes or no, then give an explanation.[/INST]"
)

LLAMA_ZERO_SHOT_PROMPT = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "You are a concise, helpful assistant.<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "Input text: {example}\n"
    "Question: {question}\n"
    "Answer with yes or no, then give an explanation.<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)

LLAMA_FIVE_SHOT_PROMPT = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "You are a concise, helpful assistant.<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "Input text: and i just kept on laughing because it was so\n"
    "Question: Does the input mention laughter?\n"
    "Answer with Yes or No.<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
    "Yes<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    "Input text: what a crazy day things just kept on happening\n"
    "Question: Is the sentence related to food preparation?\n"
    "Answer with Yes or No.<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
    "No<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    "Input text: i felt like a fly on the wall just waiting for\n"
    "Question: Does the text use a metaphor or figurative language?\n"
    "Answer with Yes or No.<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
    "Yes<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    "Input text: he takes too long in there getting the pans from\n"
    "Question: Is there a reference to sports?\n"
    "Answer with Yes or No.<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
    "No<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    "Input text: was silent and lovely and there was no sound except\n"
    "Question: Is the sentence expressing confusion or uncertainty?\n"
    "Answer with Yes or No.<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
    "No<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    "Input text: {example}\n"
    "Question: {question}\n"
    "Answer with Yes or No.<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)


@dataclass(frozen=True)
class QAConfig:
    name: str
    model_id: str
    prompt: str
    default_batch_size: int


QA_CONFIGS = {
    "mistral_zero_shot": QAConfig(
        name="mistral_zero_shot",
        model_id="mistralai/Mistral-7B-Instruct-v0.2",
        prompt=MISTRAL_ZERO_SHOT_PROMPT,
        default_batch_size=32,
    ),
    "llama_zero_shot": QAConfig(
        name="llama_zero_shot",
        model_id="meta-llama/Meta-Llama-3-8B-Instruct",
        prompt=LLAMA_ZERO_SHOT_PROMPT,
        default_batch_size=64,
    ),
    "llama_five_shot": QAConfig(
        name="llama_five_shot",
        model_id="meta-llama/Meta-Llama-3-8B-Instruct",
        prompt=LLAMA_FIVE_SHOT_PROMPT,
        default_batch_size=16,
    ),
}
ENSEMBLE_CONFIG_NAMES = tuple(QA_CONFIGS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the frozen Benara et al. QA-Emb-29 features."
    )
    parser.add_argument(
        "--stimulus-file",
        type=Path,
        default=DEFAULT_STIMULUS_FILE,
        help="Stimulus CSV whose row order defines the feature-matrix row order.",
    )
    parser.add_argument(
        "--participants-dir",
        type=Path,
        default=DEFAULT_PARTICIPANTS_DIR,
        help="Directory containing p1...p8/stimset.csv for alignment checks.",
    )
    parser.add_argument(
        "--skip-alignment-check",
        action="store_true",
        help="Do not verify that p1...p8 contain identical sentence rows.",
    )
    parser.add_argument(
        "--questions-file",
        type=Path,
        default=DEFAULT_QUESTIONS_FILE,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=(*QA_CONFIGS, "all"),
        default=["all"],
        help="Model/prompt configurations to compute. 'all' is the baseline.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the released-code-compatible per-config batch sizes.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Atomically checkpoint raw answers every N inference batches.",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Hugging Face model revision. Resolved revisions are saved in metadata.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face model cache directory.",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Transformers device_map value (default: auto).",
    )
    parser.add_argument(
        "--max-input-length",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Forbid model downloads and use only the local Hugging Face cache.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Permit 7B/8B inference without CUDA (normally rejected as impractical).",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Build final outputs from complete raw caches without loading models.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the planned work without loading models.",
    )
    return parser.parse_args()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_strings(values: Iterable[str]) -> str:
    payload = json.dumps(
        list(values), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return sha256_bytes(payload)


def score_decoded_token(token: str) -> np.int8:
    """Match the released implementation's hard-answer conversion exactly."""
    return np.int8("yes" in token.lower())


def load_questions(path: Path) -> pd.DataFrame:
    questions = pd.read_csv(path)
    required = {
        "feature_index",
        "feature_name",
        "question",
        "source_importance_normalized",
        "source_candidate_index",
    }
    missing = sorted(required - set(questions.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    if len(questions) != 29:
        raise ValueError(f"Expected 29 frozen questions, found {len(questions)}")
    expected_indices = np.arange(29)
    if not np.array_equal(questions["feature_index"].to_numpy(), expected_indices):
        raise ValueError("feature_index must be the ordered integers 0...28")
    if questions["feature_name"].duplicated().any():
        raise ValueError("feature_name values must be unique")
    if questions["question"].duplicated().any():
        raise ValueError(
            "Question strings must be unique; the two location questions are "
            "similar but intentionally not identical."
        )
    observed_hash = sha256_strings(questions["question"].astype(str).tolist())
    if observed_hash != FROZEN_QUESTIONS_SHA256:
        raise ValueError(
            "Question strings or their order do not match the frozen "
            "Benara et al. QA-Emb-29 vocabulary"
        )
    return questions


def _read_stimulus_file(path: Path) -> pd.DataFrame:
    stimuli = pd.read_csv(path)
    if "sentence" not in stimuli:
        raise ValueError(f"{path} does not contain a 'sentence' column")
    if stimuli["sentence"].isna().any():
        raise ValueError(f"{path} contains missing sentences")
    if stimuli["sentence"].duplicated().any():
        raise ValueError(f"{path} contains duplicate sentences")
    return stimuli


def load_stimuli(
    stimulus_file: Path,
    participants_dir: Path | None,
    check_alignment: bool = True,
) -> pd.DataFrame:
    stimuli = _read_stimulus_file(stimulus_file)
    reference_sentences = stimuli["sentence"].astype(str).tolist()

    if check_alignment:
        if participants_dir is None:
            raise ValueError("participants_dir is required for alignment checks")
        expected_files = [
            participants_dir / f"p{i}" / "stimset.csv" for i in range(1, 9)
        ]
        missing = [str(path) for path in expected_files if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing participant stimulus files: " + ", ".join(missing)
            )
        for path in expected_files:
            participant_sentences = (
                _read_stimulus_file(path)["sentence"].astype(str).tolist()
            )
            if participant_sentences != reference_sentences:
                raise ValueError(
                    f"Sentence rows in {path} do not match {stimulus_file}"
                )
    return stimuli


def resolve_config_names(config_names: Sequence[str]) -> tuple[str, ...]:
    if "all" in config_names:
        if len(config_names) != 1:
            raise ValueError("'all' cannot be combined with named configurations")
        return ENSEMBLE_CONFIG_NAMES
    # Preserve released ensemble order while removing accidental duplicates.
    requested = set(config_names)
    return tuple(name for name in ENSEMBLE_CONFIG_NAMES if name in requested)


def _atomic_save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".npz", prefix=f".{path.stem}.",
            dir=path.parent, delete=False
        ) as handle:
            temp_path = Path(handle.name)
            np.savez_compressed(handle, **arrays)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", dir=path.parent, delete=False
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(value)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _raw_cache_metadata(
    config: QAConfig,
    questions: Sequence[str],
    sentences: Sequence[str],
    revision: str,
) -> dict[str, str]:
    return {
        "config_name": config.name,
        "model_id": config.model_id,
        "requested_revision": revision,
        "prompt_sha256": sha256_strings([config.prompt]),
        "questions_sha256": sha256_strings(questions),
        "sentences_sha256": sha256_strings(sentences),
        "upstream_commit": UPSTREAM_COMMIT,
    }


def save_raw_cache(
    path: Path,
    values: np.ndarray,
    tokens: np.ndarray,
    metadata: dict[str, str],
) -> None:
    _atomic_save_npz(
        path,
        values=values.astype(np.int8, copy=False),
        tokens=tokens.astype("<U128", copy=False),
        metadata=np.array(json.dumps(metadata, sort_keys=True)),
    )


def load_or_initialize_raw_cache(
    path: Path,
    shape: tuple[int, int],
    metadata: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    if not path.exists():
        values = np.full(shape, -1, dtype=np.int8)
        tokens = np.full(shape, "", dtype="<U128")
        return values, tokens, dict(metadata)

    with np.load(path, allow_pickle=False) as cached:
        values = cached["values"].astype(np.int8, copy=True)
        tokens = cached["tokens"].astype("<U128", copy=True)
        saved_metadata = json.loads(str(cached["metadata"].item()))

    if values.shape != shape or tokens.shape != shape:
        raise ValueError(
            f"Raw cache {path} has shape {values.shape}; expected {shape}"
        )
    for key, expected_value in metadata.items():
        if saved_metadata.get(key) != expected_value:
            raise ValueError(
                f"Raw cache {path} metadata mismatch for {key!r}: "
                f"{saved_metadata.get(key)!r} != {expected_value!r}"
            )
    invalid = ~np.isin(values, (-1, 0, 1))
    if invalid.any():
        raise ValueError(f"Raw cache {path} contains values outside -1/0/1")
    return values, tokens, saved_metadata


def _resolve_torch_dtype(torch_module: Any, requested: str) -> Any:
    if requested == "auto":
        if torch_module.cuda.is_available() and torch_module.cuda.is_bf16_supported():
            return torch_module.bfloat16
        if torch_module.cuda.is_available():
            return torch_module.float16
        return torch_module.float32
    return {
        "bfloat16": torch_module.bfloat16,
        "float16": torch_module.float16,
        "float32": torch_module.float32,
    }[requested]


def load_model_and_tokenizer(
    config: QAConfig,
    *,
    revision: str,
    cache_dir: Path | None,
    dtype: str,
    device_map: str,
    local_files_only: bool,
    allow_cpu: bool,
) -> tuple[Any, Any, str | None]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "QA-Emb generation requires torch, transformers, and accelerate. "
            "Install the repository environment first."
        ) from error

    if not torch.cuda.is_available() and not allow_cpu:
        raise RuntimeError(
            "No CUDA GPU is visible. Run this generator on a GPU node, or pass "
            "--allow-cpu only if the very slow 7B/8B CPU run is intentional."
        )

    common_kwargs: dict[str, Any] = {
        "revision": revision,
        "cache_dir": str(cache_dir) if cache_dir is not None else None,
        "local_files_only": local_files_only,
    }
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, **common_kwargs)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        torch_dtype=_resolve_torch_dtype(torch, dtype),
        device_map=device_map,
        low_cpu_mem_usage=True,
        **common_kwargs,
    )
    model.eval()
    resolved_revision = getattr(model.config, "_commit_hash", None)
    return model, tokenizer, resolved_revision


def generate_one_token(
    prompts: Sequence[str],
    model: Any,
    tokenizer: Any,
    max_input_length: int,
) -> list[str]:
    import torch

    inputs = tokenizer(
        list(prompts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_length,
    )
    input_device = model.get_input_embeddings().weight.device
    inputs = {key: value.to(input_device) for key, value in inputs.items()}
    input_width = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = generated[:, input_width:]
    return tokenizer.batch_decode(
        new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )


def run_config(
    config: QAConfig,
    questions: Sequence[str],
    sentences: Sequence[str],
    output_dir: Path,
    args: argparse.Namespace,
) -> Path:
    raw_path = output_dir / "raw_answers" / f"{config.name}.npz"
    expected_metadata = _raw_cache_metadata(
        config, questions, sentences, args.revision
    )
    shape = (len(sentences), len(questions))
    values, tokens, metadata = load_or_initialize_raw_cache(
        raw_path, shape, expected_metadata
    )
    pending_flat = np.flatnonzero(values.reshape(-1) < 0)
    if not len(pending_flat):
        print(f"{config.name}: raw cache is already complete")
        return raw_path

    batch_size = args.batch_size or config.default_batch_size
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.save_every <= 0:
        raise ValueError("--save-every must be positive")

    print(
        f"{config.name}: loading {config.model_id}; "
        f"{len(pending_flat)}/{values.size} answers pending"
    )
    model, tokenizer, resolved_revision = load_model_and_tokenizer(
        config,
        revision=args.revision,
        cache_dir=args.cache_dir,
        dtype=args.dtype,
        device_map=args.device_map,
        local_files_only=args.local_files_only,
        allow_cpu=args.allow_cpu,
    )
    metadata["resolved_revision"] = resolved_revision or "unknown"
    metadata["batch_size"] = str(batch_size)
    metadata["max_input_length"] = str(args.max_input_length)
    metadata["max_new_tokens"] = "1"
    metadata["do_sample"] = "false"

    try:
        num_questions = len(questions)
        num_batches = (len(pending_flat) + batch_size - 1) // batch_size
        for batch_number, start in enumerate(
            range(0, len(pending_flat), batch_size), start=1
        ):
            batch_flat = pending_flat[start : start + batch_size]
            row_indices = batch_flat // num_questions
            question_indices = batch_flat % num_questions
            prompts = [
                config.prompt.format(
                    example=sentences[row_index],
                    question=questions[question_index],
                )
                for row_index, question_index in zip(
                    row_indices, question_indices
                )
            ]
            decoded = generate_one_token(
                prompts, model, tokenizer, args.max_input_length
            )
            if len(decoded) != len(batch_flat):
                raise RuntimeError(
                    f"Model returned {len(decoded)} tokens for "
                    f"{len(batch_flat)} prompts"
                )
            for flat_index, token in zip(batch_flat, decoded):
                row_index = flat_index // num_questions
                question_index = flat_index % num_questions
                tokens[row_index, question_index] = token
                values[row_index, question_index] = score_decoded_token(token)

            if (
                batch_number % args.save_every == 0
                or batch_number == num_batches
            ):
                save_raw_cache(raw_path, values, tokens, metadata)
            print(
                f"{config.name}: batch {batch_number}/{num_batches} "
                f"({np.count_nonzero(values >= 0)}/{values.size} complete)",
                flush=True,
            )
    finally:
        del model
        del tokenizer
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    return raw_path


def _load_complete_raw(
    path: Path,
    expected_shape: tuple[int, int],
) -> tuple[np.ndarray, dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing raw cache: {path}")
    with np.load(path, allow_pickle=False) as cached:
        values = cached["values"].astype(np.int8, copy=False)
        metadata = json.loads(str(cached["metadata"].item()))
    if values.shape != expected_shape:
        raise ValueError(
            f"{path} has shape {values.shape}; expected {expected_shape}"
        )
    if not np.isin(values, (0, 1)).all():
        pending = int(np.count_nonzero(values < 0))
        raise ValueError(f"{path} is incomplete ({pending} answers pending)")
    return values, metadata


def prepare_output_metadata(
    output_dir: Path,
    questions_file: Path,
    questions: pd.DataFrame,
    stimulus_file: Path,
    stimuli: pd.DataFrame,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_questions = output_dir / "questions.csv"
    if questions_file.resolve() != output_questions.resolve():
        shutil.copyfile(questions_file, output_questions)

    stimulus_columns = ["row_index", "sentence"]
    output_stimuli = pd.DataFrame(
        {
            "row_index": np.arange(len(stimuli)),
            "sentence": stimuli["sentence"].astype(str),
        }
    )
    for optional_column in ("item_id", "stimsetid", "sent_identifier"):
        if optional_column in stimuli:
            output_stimuli[optional_column] = stimuli[optional_column]
            stimulus_columns.append(optional_column)
    stimuli_path = output_dir / "stimuli.csv"
    output_stimuli[stimulus_columns].to_csv(stimuli_path, index=False)
    return output_questions, stimuli_path


def aggregate_outputs(
    output_dir: Path,
    questions: pd.DataFrame,
    stimuli: pd.DataFrame,
    questions_path: Path,
    stimuli_path: Path,
) -> np.ndarray:
    expected_shape = (len(stimuli), len(questions))
    matrices = []
    raw_metadata: dict[str, dict[str, str]] = {}
    for config_name in ENSEMBLE_CONFIG_NAMES:
        raw_path = output_dir / "raw_answers" / f"{config_name}.npz"
        values, metadata = _load_complete_raw(raw_path, expected_shape)
        expected_metadata = _raw_cache_metadata(
            QA_CONFIGS[config_name],
            questions["question"].astype(str).tolist(),
            stimuli["sentence"].astype(str).tolist(),
            revision=metadata.get("requested_revision", ""),
        )
        for key, expected_value in expected_metadata.items():
            if metadata.get(key) != expected_value:
                raise ValueError(
                    f"Raw cache {raw_path} metadata mismatch for {key!r}: "
                    f"{metadata.get(key)!r} != {expected_value!r}"
                )
        matrices.append(values.astype(np.float32))
        raw_metadata[config_name] = metadata

    features = np.mean(np.stack(matrices, axis=0), axis=0, dtype=np.float32)
    if features.shape != expected_shape:
        raise AssertionError("Unexpected ensemble feature shape")
    allowed_values = np.array([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    if not np.isclose(features[..., None], allowed_values).any(axis=-1).all():
        raise AssertionError("Three-model binary ensemble has unexpected values")

    pickle_path = output_dir / "qa_emb.pkl"
    _atomic_write_bytes(
        pickle_path,
        pickle.dumps(features, protocol=pickle.HIGHEST_PROTOCOL),
    )
    csv_path = output_dir / "qa_emb.csv"
    feature_frame = pd.DataFrame(
        features,
        columns=questions["feature_name"].astype(str).tolist(),
    )
    feature_frame.to_csv(csv_path, index=False)

    manifest = {
        "baseline": "QA-Emb-29 (frozen transfer)",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "description": (
            "Fixed questions selected on Benara et al.'s original dataset; "
            "no feature selection was performed on the current study's responses."
        ),
        "shape": list(features.shape),
        "dtype": str(features.dtype),
        "row_order": "stimuli.csv row_index",
        "column_order": "questions.csv feature_index",
        "aggregation": {
            "operation": "unscaled arithmetic mean",
            "config_order": list(ENSEMBLE_CONFIG_NAMES),
            "per_config_values": [0, 1],
            "final_possible_values": allowed_values.tolist(),
        },
        "answer_scoring": (
            "Generate one token; 1 iff decoded_token.lower() contains 'yes', "
            "else 0."
        ),
        "upstream": {
            "repository": UPSTREAM_REPOSITORY,
            "commit": UPSTREAM_COMMIT,
            "implementation_files": [
                "neuro1/features/qa_embedder.py",
                "neuro1/features/feature_utils.py",
                "neuro1/features/questions/v3_boostexamples.json",
            ],
        },
        "configs": {
            name: {
                "model_id": QA_CONFIGS[name].model_id,
                "prompt": QA_CONFIGS[name].prompt,
                "prompt_sha256": sha256_strings([QA_CONFIGS[name].prompt]),
                "raw_cache": f"raw_answers/{name}.npz",
                "raw_metadata": raw_metadata[name],
            }
            for name in ENSEMBLE_CONFIG_NAMES
        },
        "checksums": {
            "questions_csv_sha256": sha256_file(questions_path),
            "stimuli_csv_sha256": sha256_file(stimuli_path),
            "qa_emb_pickle_sha256": sha256_file(pickle_path),
            "qa_emb_csv_sha256": sha256_file(csv_path),
            "matrix_bytes_sha256": sha256_bytes(features.tobytes(order="C")),
        },
        "downstream_note": (
            "No normalization or surprisal concatenation was applied. Fit any "
            "scaler and encoding model within downstream training folds."
        ),
    }
    manifest_path = output_dir / "manifest.json"
    _atomic_write_bytes(
        manifest_path,
        (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode(
            "utf-8"
        ),
    )
    print(f"Wrote {pickle_path} with shape {features.shape}")
    return features


def main() -> None:
    args = parse_args()
    config_names = resolve_config_names(args.configs)
    questions = load_questions(args.questions_file)
    stimuli = load_stimuli(
        args.stimulus_file,
        args.participants_dir,
        check_alignment=not args.skip_alignment_check,
    )
    if len(stimuli) != 200:
        raise ValueError(
            f"Expected the paper's baseline stimulus set to contain 200 rows, "
            f"found {len(stimuli)}"
        )
    question_strings = questions["question"].astype(str).tolist()
    sentences = stimuli["sentence"].astype(str).tolist()
    total_answers = len(question_strings) * len(sentences)
    print(
        f"Validated {len(sentences)} aligned stimuli and "
        f"{len(question_strings)} frozen questions ({total_answers} answers/config)."
    )
    print("Configurations:", ", ".join(config_names))
    if args.dry_run:
        for name in config_names:
            config = QA_CONFIGS[name]
            batch_size = args.batch_size or config.default_batch_size
            print(
                f"  {name}: {config.model_id}, batch_size={batch_size}, "
                f"prompts={total_answers}"
            )
        return

    questions_path, stimuli_path = prepare_output_metadata(
        args.output_dir,
        args.questions_file,
        questions,
        args.stimulus_file,
        stimuli,
    )

    if not args.aggregate_only:
        for name in config_names:
            run_config(
                QA_CONFIGS[name],
                question_strings,
                sentences,
                args.output_dir,
                args,
            )

    try:
        aggregate_outputs(
            args.output_dir,
            questions,
            stimuli,
            questions_path,
            stimuli_path,
        )
    except FileNotFoundError:
        # Named subsets are useful for separate GPU jobs. They should leave a
        # valid raw cache even while another ensemble member is still pending.
        if args.aggregate_only or set(config_names) == set(ENSEMBLE_CONFIG_NAMES):
            raise
        print(
            "Saved the requested raw configuration(s). The three-member "
            "ensemble is not complete yet; run the remaining configurations "
            "and then pass --aggregate-only."
        )


if __name__ == "__main__":
    main()
