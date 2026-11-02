#!/usr/bin/env python3
"""Fit cutoff-aware sparse Natural Stories encoding models.

For every outer leave-one-story-out fold, candidate LM features are screened
with univariate F tests and selected with L1 regression using only bins after
the first 100 seconds of each training story. Standardized surprisal enters
the selection design, as in the sentence-level analysis, and is forced into every
final L2 refit. Ridge regularization is selected with inner held-out-story
validation, and final performance is evaluated after the same 100-second
cutoff in the untouched outer story. Word count is deliberately excluded.

This preserves the original two-stage procedure: F-screening precedes the
inner LassoCV, and selected features are fixed during inner Ridge tuning.
Thus inner tuning is not an end-to-end nested feature-selection pipeline;
the outer held-out story remains excluded from selection and tuning.
LM features retain their native scales. Surprisal scaling uses all outer
training bins (including the first 100 seconds); feature-selection responses
are standardized after trimming. Use separate output directories for actual/control fits and
for different feature extractors. Existing results are resumed only when
input contents, settings, this source file, and software versions match.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
import pickle
import platform
import time
import warnings
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import pearsonr
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LassoCV, Ridge
from sklearn.preprocessing import StandardScaler


STORIES = ("1", "2", "3", "4", "5", "6", "7", "9", "10")
STORY_NAMES = {
    "1": "boar",
    "2": "aqua",
    "3": "matchstickseller",
    "4": "kingofbirds",
    "5": "elvis",
    "6": "mrsticky",
    "7": "highschool",
    "9": "tulips",
    "10": "tree",
}
RESPONSE_FILES = {
    "Lang": "d_shift_3",
    "IFGorb": "d_shift_3_IFGorb",
    "IFG": "d_shift_3_IFG",
    "MFG": "d_shift_3_MFG",
    "AntTemp": "d_shift_3_AntTemp",
    "PostTemp": "d_shift_3_PostTemp",
}
FEATURE_MODELS = {
    "jumprelu": ("sae", 0, 16384),
    "sae_full": ("sae", 0, 32768),
    "sae_0_128": ("sae", 0, 128),
    "sae_128_512": ("sae", 128, 512),
    "sae_512_2048": ("sae", 512, 2048),
    "sae_2048_8192": ("sae", 2048, 8192),
    "sae_8192_32768": ("sae", 8192, 32768),
    "sae_finer_union": ("sae", 128, 32768),
    "residual": ("residual", 0, 2304),
}
MODEL_CHOICES = ("surprisal",) + tuple(FEATURE_MODELS)
DEFAULT_MODELS = ("surprisal", "sae_full", "residual")
LASSO_ALPHAS = np.logspace(-2, 0, 10)
RIDGE_ALPHAS = np.asarray([10.0**power for power in range(-2, 6)])
SUPPORT_COLUMNS = (
    "roi",
    "response_circular_shift_trs",
    "model",
    "held_out_story_id",
    "feature_source",
    "feature_local_index",
    "feature_global_index",
    "lasso_coefficient",
    "ridge_coefficient",
)
PREDICTION_COLUMNS = (
    "roi",
    "response_circular_shift_trs",
    "model",
    "held_out_story_id",
    "held_out_story",
    "bin",
    "observed_z",
    "predicted_z",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=here / "data")
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=here / "features",
    )
    parser.add_argument(
        "--surprisal-dir",
        type=Path,
        default=here / "features",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=here / "runs" / "encoding",
    )
    parser.add_argument("--roi", choices=tuple(RESPONSE_FILES), required=True)
    parser.add_argument("--models", nargs="+", choices=MODEL_CHOICES, default=list(DEFAULT_MODELS))
    parser.add_argument(
        "--response-shifts", nargs="+", type=int, default=[0],
        help="Use 0 for actual responses or 26 52 78 104 for the original circular-shift control.",
    )
    parser.add_argument("--held-out-stories", nargs="+", choices=STORIES, default=list(STORIES))
    parser.add_argument("--test-trim-start-bins", type=int, default=50)
    parser.add_argument("--k-best", type=int, default=8000)
    parser.add_argument("--lasso-jobs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=19)
    args = parser.parse_args(argv)
    if args.test_trim_start_bins < 0:
        parser.error("--test-trim-start-bins must be non-negative")
    if args.k_best < 1:
        parser.error("--k-best must be positive")
    if any(len(values) != len(set(values)) for values in (
        args.models, args.response_shifts, args.held_out_stories
    )):
        parser.error("Models, response shifts, and held-out stories must not contain duplicates")
    return args


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_inputs(args: argparse.Namespace):
    response_path = args.data_dir / "response" / RESPONSE_FILES[args.roi]
    response_named = load_pickle(response_path)
    response = {
        story: np.asarray(response_named[STORY_NAMES[story]], dtype=np.float32)
        for story in STORIES
    }
    surprisal = {}
    sae = {}
    residual = {}
    needed_sources = {FEATURE_MODELS[model][0] for model in args.models if model != "surprisal"}
    for story in STORIES:
        predictor_path = args.surprisal_dir / f"binned_predictors_story-{story}.csv"
        predictor_frame = pd.read_csv(predictor_path)
        surprisal[story] = predictor_frame["surprisal"].to_numpy(dtype=np.float32)[:, None]
        if "sae" in needed_sources:
            sae[story] = sp.load_npz(args.feature_dir / f"binned_sae_story-{story}.npz").tocsr()
        if "residual" in needed_sources:
            residual[story] = np.load(args.feature_dir / f"binned_residual_story-{story}.npy").astype(
                np.float32
            )
        expected = len(response[story])
        observed = {"surprisal": len(surprisal[story])}
        if "sae" in needed_sources:
            observed["sae"] = sae[story].shape[0]
        if "residual" in needed_sources:
            observed["residual"] = residual[story].shape[0]
        if any(length != expected for length in observed.values()):
            raise ValueError(f"Story {story}: response/predictor mismatch {expected} vs {observed}")
        if expected - args.test_trim_start_bins < 2:
            raise ValueError(f"Story {story}: fewer than two response bins remain after trimming")
        for model in args.models:
            if model == "surprisal":
                continue
            source, _, stop = FEATURE_MODELS[model]
            width = sae[story].shape[1] if source == "sae" else residual[story].shape[1]
            if stop > width:
                raise ValueError(f"Story {story}: {model} needs {stop} features, found {width}")
    return response_path, response, surprisal, sae, residual


def trimmed_story_arrays(
    values: Mapping[str, np.ndarray],
    stories: Sequence[str],
    trim: int,
) -> tuple[np.ndarray, np.ndarray]:
    arrays = []
    groups = []
    for group, story in enumerate(stories):
        arrays.append(np.asarray(values[story])[trim:])
        groups.append(np.full(len(values[story]) - trim, group, dtype=int))
    return np.concatenate(arrays), np.concatenate(groups)


def feature_story_slice(
    story: str,
    source: str,
    start: int,
    stop: int,
    sae: Mapping[str, sp.csr_matrix],
    residual: Mapping[str, np.ndarray],
) -> np.ndarray:
    if source == "sae":
        return sae[story][:, start:stop].toarray().astype(np.float32, copy=False)
    if source == "residual":
        return residual[story][:, start:stop].astype(np.float32, copy=False)
    raise ValueError(f"Unknown source: {source}")


def select_features(
    model_name: str,
    training_stories: Sequence[str],
    shifted_response: Mapping[str, np.ndarray],
    standardized_surprisal: Mapping[str, np.ndarray],
    sae: Mapping[str, sp.csr_matrix],
    residual: Mapping[str, np.ndarray],
    trim: int,
    k_best: int,
    lasso_jobs: int,
    seed: int,
) -> dict:
    source, start, stop = FEATURE_MODELS[model_name]
    feature_arrays = {
        story: feature_story_slice(story, source, start, stop, sae, residual)
        for story in training_stories
    }
    x_trimmed, groups = trimmed_story_arrays(feature_arrays, training_stories, trim)
    surprisal_trimmed, _ = trimmed_story_arrays(
        standardized_surprisal, training_stories, trim
    )
    n_lm_features = x_trimmed.shape[1]
    # Match the paper's pipeline: surprisal participates in F-test/L1
    # selection, but it is forced into the subsequent Ridge fit regardless of
    # whether Lasso retains it. Only LM features from this support are returned.
    x_selection = np.concatenate((x_trimmed, surprisal_trimmed), axis=1)
    y_trimmed, _ = trimmed_story_arrays(shifted_response, training_stories, trim)
    y_scaler = StandardScaler()
    y_trimmed_z = y_scaler.fit_transform(y_trimmed[:, None]).ravel()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        f_scores, _ = f_regression(x_selection, y_trimmed_z)
    f_scores = np.nan_to_num(f_scores, nan=-np.inf, neginf=-np.inf, posinf=np.inf)
    finite_or_positive = np.isfinite(f_scores) | np.isposinf(f_scores)
    available = np.flatnonzero(finite_or_positive)
    if available.size == 0:
        return {
            "source": source,
            "range_start": start,
            "range_stop": stop,
            "screened_local_indices": np.asarray([], dtype=int),
            "surprisal_screened": False,
            "selected_local_indices": np.asarray([], dtype=int),
            "selected_global_indices": np.asarray([], dtype=int),
            "lasso_coefficients": np.asarray([], dtype=float),
            "surprisal_lasso_coefficient": 0.0,
            "lasso_alpha": np.nan,
        }
    if x_selection.shape[1] > k_best and available.size > k_best:
        available_scores = f_scores[available]
        top_positions = np.argpartition(available_scores, -k_best)[-k_best:]
        screened = np.sort(available[top_positions])
    else:
        screened = available
    x_screened = x_selection[:, screened]
    cv_splits = []
    for group in np.unique(groups):
        validation = np.flatnonzero(groups == group)
        training = np.flatnonzero(groups != group)
        cv_splits.append((training, validation))
    lasso = LassoCV(
        alphas=LASSO_ALPHAS,
        cv=cv_splits,
        tol=1e-3,
        selection="random",
        max_iter=5000,
        random_state=seed,
        fit_intercept=True,
        n_jobs=lasso_jobs,
    )
    with warnings.catch_warnings(record=True) as lasso_warnings:
        warnings.simplefilter("always")
        lasso.fit(x_screened, y_trimmed_z)
    support = np.flatnonzero(lasso.coef_ != 0)
    selected_original = screened[support]
    lm_support = selected_original < n_lm_features
    selected_local = selected_original[lm_support]
    baseline_positions = np.flatnonzero(selected_original == n_lm_features)
    baseline_lasso_coefficient = (
        float(lasso.coef_[support[baseline_positions[0]]])
        if baseline_positions.size
        else 0.0
    )
    return {
        "source": source,
        "range_start": start,
        "range_stop": stop,
        "screened_local_indices": screened[screened < n_lm_features],
        "surprisal_screened": bool(np.any(screened == n_lm_features)),
        "selected_local_indices": selected_local,
        "selected_global_indices": start + selected_local,
        "lasso_coefficients": lasso.coef_[support][lm_support],
        "surprisal_lasso_coefficient": baseline_lasso_coefficient,
        "lasso_alpha": float(lasso.alpha_),
        "lasso_n_iter": int(lasso.n_iter_),
        "lasso_dual_gap": float(lasso.dual_gap_),
        "lasso_warning_count": len(lasso_warnings),
        "lasso_warnings": dict(Counter(
            f"{item.category.__name__}: {item.message}" for item in lasso_warnings
        )),
    }


def make_design(
    story: str,
    surprisal: Mapping[str, np.ndarray],
    selection: dict | None,
    sae: Mapping[str, sp.csr_matrix],
    residual: Mapping[str, np.ndarray],
) -> np.ndarray:
    baseline = surprisal[story]
    if selection is None or len(selection["selected_local_indices"]) == 0:
        return baseline.copy()
    indices = selection["selected_local_indices"]
    start = int(selection["range_start"])
    stop = int(selection["range_stop"])
    if selection["source"] == "sae":
        features = sae[story][:, start:stop][:, indices].toarray().astype(np.float32)
    else:
        features = residual[story][:, start:stop][:, indices].astype(np.float32)
    return np.concatenate((baseline, features), axis=1)


def select_ridge_alpha(
    designs: Mapping[str, np.ndarray],
    response: Mapping[str, np.ndarray],
    training_stories: Sequence[str],
    trim: int,
) -> tuple[float, dict[float, float]]:
    alpha_to_mse = {}
    for alpha in RIDGE_ALPHAS:
        fold_mse = []
        for validation_story in training_stories:
            inner_training = [story for story in training_stories if story != validation_story]
            x_train = np.concatenate([designs[story] for story in inner_training])
            y_train = np.concatenate([response[story] for story in inner_training])
            x_validation = designs[validation_story]
            y_validation = response[validation_story]
            y_scaler = StandardScaler()
            y_train_z = y_scaler.fit_transform(y_train[:, None]).ravel()
            y_validation_z = y_scaler.transform(y_validation[:, None]).ravel()
            ridge = Ridge(alpha=float(alpha), fit_intercept=True)
            ridge.fit(x_train, y_train_z)
            prediction = ridge.predict(x_validation)
            fold_mse.append(float(np.mean((y_validation_z[trim:] - prediction[trim:]) ** 2)))
        alpha_to_mse[float(alpha)] = float(np.mean(fold_mse))
    best_alpha = min(alpha_to_mse, key=lambda alpha: (alpha_to_mse[alpha], alpha))
    return float(best_alpha), alpha_to_mse


def fit_outer_fold(
    model_name: str,
    held_out: str,
    shifted_response: Mapping[str, np.ndarray],
    surprisal: Mapping[str, np.ndarray],
    sae: Mapping[str, sp.csr_matrix],
    residual: Mapping[str, np.ndarray],
    args: argparse.Namespace,
) -> tuple[dict, list[dict], list[dict]]:
    training_stories = [story for story in STORIES if story != held_out]
    surprisal_scaler = StandardScaler()
    surprisal_scaler.fit(np.concatenate([surprisal[story] for story in training_stories]))
    standardized_surprisal = {
        story: surprisal_scaler.transform(surprisal[story]).astype(np.float32)
        for story in STORIES
    }
    if model_name == "surprisal":
        selection = None
    else:
        selection = select_features(
            model_name=model_name,
            training_stories=training_stories,
            shifted_response=shifted_response,
            standardized_surprisal=standardized_surprisal,
            sae=sae,
            residual=residual,
            trim=args.test_trim_start_bins,
            k_best=args.k_best,
            lasso_jobs=args.lasso_jobs,
            seed=args.seed,
        )
    designs = {
        story: make_design(story, standardized_surprisal, selection, sae, residual)
        for story in STORIES
    }
    ridge_alpha, alpha_to_mse = select_ridge_alpha(
        designs=designs,
        response=shifted_response,
        training_stories=training_stories,
        trim=args.test_trim_start_bins,
    )
    x_train = np.concatenate([designs[story] for story in training_stories])
    y_train = np.concatenate([shifted_response[story] for story in training_stories])
    x_test = designs[held_out]
    y_test = shifted_response[held_out]
    y_scaler = StandardScaler()
    y_train_z = y_scaler.fit_transform(y_train[:, None]).ravel()
    y_test_z = y_scaler.transform(y_test[:, None]).ravel()
    ridge = Ridge(alpha=ridge_alpha, fit_intercept=True)
    ridge.fit(x_train, y_train_z)
    prediction = ridge.predict(x_test)
    scored = slice(args.test_trim_start_bins, None)
    correlation = float(pearsonr(y_test_z[scored], prediction[scored]).statistic)
    result = {
        "roi": args.roi,
        "response_circular_shift_trs": args.current_shift,
        "model": model_name,
        "held_out_story_id": held_out,
        "held_out_story": STORY_NAMES[held_out],
        "r": correlation,
        "fisher_z": float(np.arctanh(np.clip(correlation, -0.999999, 0.999999))),
        "ridge_alpha": ridge_alpha,
        "ridge_inner_mean_mse_json": json.dumps(alpha_to_mse, sort_keys=True),
        "lasso_alpha": np.nan if selection is None else selection["lasso_alpha"],
        "lasso_n_iter": None if selection is None else selection.get("lasso_n_iter"),
        "lasso_dual_gap": None if selection is None else selection.get("lasso_dual_gap"),
        "lasso_warning_count": 0 if selection is None else selection.get("lasso_warning_count", 0),
        "lasso_warnings_json": json.dumps(
            {} if selection is None else selection.get("lasso_warnings", {}), sort_keys=True
        ),
        "surprisal_lasso_coefficient": (
            np.nan if selection is None else selection["surprisal_lasso_coefficient"]
        ),
        "surprisal_was_f_test_screened": (
            True if selection is None else selection["surprisal_screened"]
        ),
        "n_features_screened": 0 if selection is None else len(selection["screened_local_indices"]),
        "n_lm_features_selected": 0 if selection is None else len(selection["selected_local_indices"]),
        "n_final_predictors": int(x_train.shape[1]),
        "surprisal_coefficient": float(ridge.coef_[0]),
        "n_train_full": int(len(y_train)),
        "n_test_full": int(len(y_test)),
        "n_test_scored": int(len(y_test) - args.test_trim_start_bins),
        "feature_selection_training_bins_trimmed": True,
        "final_ridge_training_bins_trimmed": False,
        "test_trim_start_bins": args.test_trim_start_bins,
    }
    support_rows = []
    if selection is not None:
        for position, (local_index, global_index, lasso_coefficient) in enumerate(
            zip(
                selection["selected_local_indices"],
                selection["selected_global_indices"],
                selection["lasso_coefficients"],
            ),
            start=1,
        ):
            support_rows.append(
                {
                    "roi": args.roi,
                    "response_circular_shift_trs": args.current_shift,
                    "model": model_name,
                    "held_out_story_id": held_out,
                    "feature_source": selection["source"],
                    "feature_local_index": int(local_index),
                    "feature_global_index": int(global_index),
                    "lasso_coefficient": float(lasso_coefficient),
                    "ridge_coefficient": float(ridge.coef_[position]),
                }
            )
    prediction_rows = [
        {
            "roi": args.roi,
            "response_circular_shift_trs": args.current_shift,
            "model": model_name,
            "held_out_story_id": held_out,
            "held_out_story": STORY_NAMES[held_out],
            "bin": bin_index,
            "observed_z": float(y_test_z[bin_index]),
            "predicted_z": float(prediction[bin_index]),
        }
        for bin_index in range(args.test_trim_start_bins, len(y_test))
    ]
    return result, support_rows, prediction_rows


def checkpoint(
    results: list[dict],
    support: list[dict],
    predictions: list[dict],
    result_path: Path,
    support_path: Path,
    prediction_path: Path,
) -> None:
    for frame, path in (
        (pd.DataFrame(results), result_path),
        (pd.DataFrame(support, columns=SUPPORT_COLUMNS), support_path),
        (pd.DataFrame(predictions, columns=PREDICTION_COLUMNS), prediction_path),
    ):
        temporary = path.with_suffix(path.suffix + ".tmp")
        frame.to_csv(temporary, index=False)
        temporary.replace(path)


def load_records(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        return pd.read_csv(path).to_dict("records")
    except pd.errors.EmptyDataError:
        return []


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_paths(args: argparse.Namespace) -> dict[str, Path]:
    """All nine stories are needed even when only one outer fold is requested."""
    paths = {"response": args.data_dir / "response" / RESPONSE_FILES[args.roi]}
    needed_sources = {FEATURE_MODELS[model][0] for model in args.models if model != "surprisal"}
    for story in STORIES:
        paths[f"surprisal/story-{story}"] = args.surprisal_dir / f"binned_predictors_story-{story}.csv"
        if "sae" in needed_sources:
            paths[f"sae/story-{story}"] = args.feature_dir / f"binned_sae_story-{story}.npz"
        if "residual" in needed_sources:
            paths[f"residual/story-{story}"] = args.feature_dir / f"binned_residual_story-{story}.npy"
    return paths


def build_fingerprint(args: argparse.Namespace) -> dict:
    """Content-based provenance; relocating identical inputs does not invalidate it."""
    paths = input_paths(args)
    payload = {
        "schema_version": 1,
        "procedure": "original-outer-loso-fscreen-lasso-ridge-v1",
        "configuration": {
            "roi": args.roi,
            "models": sorted(args.models),
            "response_shifts": sorted(args.response_shifts),
            "held_out_stories": sorted(args.held_out_stories),
            "test_trim_start_bins": args.test_trim_start_bins,
            "k_best": args.k_best,
            "lasso_jobs": args.lasso_jobs,
            "seed": args.seed,
            "stories": list(STORIES),
            "feature_models": FEATURE_MODELS,
            "lasso_alphas": LASSO_ALPHAS.tolist(),
            "ridge_alphas": RIDGE_ALPHAS.tolist(),
        },
        "inputs": {
            key: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for key, path in paths.items()
        },
        "source_sha256": sha256_file(Path(__file__)),
        "software_versions": {
            "python": platform.python_version(),
            **{package: version(package) for package in ("numpy", "pandas", "scipy", "scikit-learn")},
        },
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "fingerprint": fingerprint,
        "payload": payload,
        "input_paths": {key: str(path.resolve()) for key, path in paths.items()},
        "checkpoint_hashes": {},
    }


def write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def prepare_resume(
    provenance_path: Path,
    expected: dict,
    result_paths: Sequence[Path],
    manifest_path: Path,
) -> dict:
    """Refuse legacy, incompatible, altered, or incomplete checkpoints."""
    if provenance_path.exists():
        saved = json.loads(provenance_path.read_text())
        if saved.get("fingerprint") != expected["fingerprint"]:
            raise ValueError(
                "Refusing incompatible cached results: inputs, configuration, source, or "
                "software versions differ. Use a new --output-dir; existing results were not changed."
            )
        observed = {path.name: sha256_file(path) for path in result_paths if path.exists()}
        if saved.get("checkpoint_hashes") != observed:
            raise ValueError(
                "Checkpoint files are missing, modified, or only partly written. "
                "Use a new --output-dir; existing results were not changed."
            )
        return saved
    if manifest_path.exists() or any(path.exists() for path in result_paths):
        raise ValueError(
            "Refusing unverified legacy outputs without a resume fingerprint. "
            "Use a new --output-dir; existing results were not changed."
        )
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(provenance_path, expected)
    return expected


def main() -> None:
    args = parse_args()
    started = time.time()
    expected_provenance = build_fingerprint(args)
    response_path, response, surprisal, sae, residual = load_inputs(args)
    result_path = args.output_dir / f"held_out_story_results_{args.roi}.csv"
    support_path = args.output_dir / f"selected_features_{args.roi}.csv"
    prediction_path = args.output_dir / f"held_out_predictions_{args.roi}.csv"
    manifest_path = args.output_dir / f"run_manifest_{args.roi}.json"
    provenance_path = args.output_dir / f"resume_fingerprint_{args.roi}.json"
    result_paths = (result_path, support_path, prediction_path)
    provenance = prepare_resume(provenance_path, expected_provenance, result_paths, manifest_path)
    existing_results = load_records(result_path)
    existing_support = load_records(support_path)
    existing_predictions = load_records(prediction_path)
    completed = {
        (int(row["response_circular_shift_trs"]), str(row["model"]), str(row["held_out_story_id"]))
        for row in existing_results
    }

    for shift in args.response_shifts:
        args.current_shift = int(shift)
        shifted_response = {
            story: np.roll(response[story], shift) if shift else response[story].copy()
            for story in STORIES
        }
        for model_name in args.models:
            for held_out in args.held_out_stories:
                key = (int(shift), model_name, held_out)
                if key in completed:
                    print(f"[skip] roi={args.roi} shift={shift} model={model_name} story={held_out}")
                    continue
                print(
                    f"[fit] roi={args.roi} shift={shift} model={model_name} "
                    f"held_out={held_out}",
                    flush=True,
                )
                result, support_rows, prediction_rows = fit_outer_fold(
                    model_name=model_name,
                    held_out=held_out,
                    shifted_response=shifted_response,
                    surprisal=surprisal,
                    sae=sae,
                    residual=residual,
                    args=args,
                )
                existing_results.append(result)
                existing_support.extend(support_rows)
                existing_predictions.extend(prediction_rows)
                completed.add(key)
                checkpoint(
                    existing_results,
                    existing_support,
                    existing_predictions,
                    result_path,
                    support_path,
                    prediction_path,
                )
                provenance["checkpoint_hashes"] = {
                    path.name: sha256_file(path) for path in result_paths
                }
                write_json_atomic(provenance_path, provenance)
                print(
                    f"[done] roi={args.roi} model={model_name} story={held_out} "
                    f"r={result['r']:.4f} selected={result['n_lm_features_selected']} "
                    f"lasso_alpha={result['lasso_alpha']} ridge_alpha={result['ridge_alpha']}",
                    flush=True,
                )

    manifest = {
        "analysis": "Natural Stories cutoff-aware F-test, L1 selection, L2 refit",
        "resume_fingerprint": provenance["fingerprint"],
        "provenance_file": provenance_path.name,
        "roi": args.roi,
        "models_requested": args.models,
        "response_shifts_requested": args.response_shifts,
        "held_out_stories_requested": args.held_out_stories,
        "response_path": str(response_path.resolve()),
        "feature_dir": str(args.feature_dir.resolve()),
        "surprisal_dir": str(args.surprisal_dir.resolve()),
        "word_count_included": False,
        "surprisal_always_in_final_ridge": True,
        "surprisal_standardization": "fit on all bins of each outer training fold",
        "feature_selection": {
            "f_test_k_best": args.k_best,
            "lasso_alphas": LASSO_ALPHAS.tolist(),
            "lasso_cv": f"leave one training story out; first {args.test_trim_start_bins} bins omitted from all stories",
            "screening_scope": "once per outer training fold, before inner LassoCV",
            "feature_standardization": False,
        },
        "ridge": {
            "alphas": RIDGE_ALPHAS.tolist(),
            "alpha_cv": f"leave one training story out; score after first {args.test_trim_start_bins} bins",
            "selection_scope": "feature support and surprisal scaling fixed from outer training fold",
            "final_training": "all bins from all outer training stories",
        },
        "test_trim_start_bins": args.test_trim_start_bins,
        "test_trim_start_seconds": 2 * args.test_trim_start_bins,
        "elapsed_seconds_this_invocation": time.time() - started,
    }
    write_json_atomic(manifest_path, manifest)


if __name__ == "__main__":
    main()
