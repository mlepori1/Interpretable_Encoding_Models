"""Lightweight tests; no released brain data, feature extraction, or jobs required.

Optional exact parity check against the original script:
NATURAL_STORIES_REFERENCE_FIT=/path/to/fit_sae_naturalstories.py pytest -q ...
"""

import importlib.util
import os
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fit = load_module(REPO_ROOT / "replications/natural_stories/fit.py", "natural_stories_fit")


@pytest.fixture
def synthetic():
    rng = np.random.default_rng(341)
    response, surprisal, sae, residual = {}, {}, {}, {}
    for story in fit.STORIES:
        features = rng.normal(size=(64, 128)).astype(np.float32)
        baseline = rng.normal(size=(64, 1)).astype(np.float32)
        response[story] = (
            0.8 * features[:, 0] - 0.4 * features[:, 1]
            + 0.5 * baseline[:, 0] + 0.3 * rng.normal(size=64)
        ).astype(np.float32)
        surprisal[story] = baseline
        sae[story] = sp.csr_matrix(features)
        residual[story] = features[:, :6].copy()
    args = SimpleNamespace(
        roi="AntTemp", test_trim_start_bins=50, k_best=8,
        lasso_jobs=1, seed=19, current_shift=0,
    )
    return response, surprisal, sae, residual, args


def test_portable_defaults():
    args = fit.parse_args(["--roi", "AntTemp"])
    package = REPO_ROOT / "replications/natural_stories"
    assert args.data_dir == package / "data"
    assert args.feature_dir == args.surprisal_dir == package / "features"
    assert args.output_dir == package / "runs/encoding"
    assert args.models == ["surprisal", "sae_full", "residual"]
    assert args.held_out_stories == list(fit.STORIES)
    with pytest.raises(SystemExit):
        fit.parse_args(["--roi", "AntTemp", "--models", "surprisal", "surprisal"])


def test_trim_and_force_surprisal(synthetic):
    response, surprisal, sae, residual, args = synthetic
    result, support, predictions = fit.fit_outer_fold(
        "sae_0_128", "1", response, surprisal, sae, residual, args
    )
    assert result["n_train_full"] == 8 * 64
    assert result["n_test_scored"] == 14
    assert result["n_final_predictors"] == len(support) + 1
    assert result["feature_selection_training_bins_trimmed"] is True
    assert result["final_ridge_training_bins_trimmed"] is False
    assert [row["bin"] for row in predictions] == list(range(50, 64))
    assert result["lasso_warning_count"] >= 0
    assert result["lasso_n_iter"] > 0
    observed = np.asarray([row["observed_z"] for row in predictions])
    predicted = np.asarray([row["predicted_z"] for row in predictions])
    assert result["r"] == pytest.approx(np.corrcoef(observed, predicted)[0, 1], abs=1e-7)


def test_outer_target_does_not_affect_selected_model(synthetic):
    response, surprisal, sae, residual, args = synthetic
    original = fit.fit_outer_fold("sae_0_128", "1", response, surprisal, sae, residual, args)
    changed_response = dict(response)
    changed_response["1"] = response["1"][::-1].copy() * 100 + 10
    changed = fit.fit_outer_fold(
        "sae_0_128", "1", changed_response, surprisal, sae, residual, args
    )
    for key in ("ridge_alpha", "lasso_alpha", "surprisal_coefficient", "n_lm_features_selected"):
        assert original[0][key] == changed[0][key]
    assert original[1] == changed[1]
    np.testing.assert_array_equal(
        [row["predicted_z"] for row in original[2]],
        [row["predicted_z"] for row in changed[2]],
    )


def test_fingerprint_content_config_and_relocation(tmp_path, monkeypatch):
    original = tmp_path / "input.txt"
    original.write_bytes(b"input-v1")
    paths = {"response": original}
    monkeypatch.setattr(fit, "input_paths", lambda args: paths)
    args = fit.parse_args(["--roi", "AntTemp", "--lasso-jobs", "1"])
    first = fit.build_fingerprint(args)
    moved = tmp_path / "moved.txt"
    moved.write_bytes(original.read_bytes())
    paths["response"] = moved
    assert fit.build_fingerprint(args)["fingerprint"] == first["fingerprint"]
    moved.write_bytes(b"input-v2")
    assert fit.build_fingerprint(args)["fingerprint"] != first["fingerprint"]
    paths["response"] = original
    args.seed += 1
    assert fit.build_fingerprint(args)["fingerprint"] != first["fingerprint"]


def test_resume_refuses_legacy_incompatible_and_damaged_files(tmp_path):
    result = tmp_path / "results.csv"
    provenance = tmp_path / "resume.json"
    manifest = tmp_path / "manifest.json"
    expected = {"fingerprint": "abc", "checkpoint_hashes": {}}
    result.write_text("old results\n")
    with pytest.raises(ValueError, match="legacy"):
        fit.prepare_resume(provenance, expected, (result,), manifest)
    assert result.read_text() == "old results\n"
    # A separate fresh output directory is the supported recovery path.
    fresh = tmp_path / "fresh"
    provenance, result, manifest = fresh / "resume.json", fresh / "results.csv", fresh / "manifest.json"
    saved = fit.prepare_resume(provenance, expected, (result,), manifest)
    assert fit.prepare_resume(provenance, expected, (result,), manifest) == saved
    with pytest.raises(ValueError, match="incompatible"):
        fit.prepare_resume(provenance, {"fingerprint": "different"}, (result,), manifest)
    result.write_text("complete\n")
    saved["checkpoint_hashes"] = {result.name: fit.sha256_file(result)}
    fit.write_json_atomic(provenance, saved)
    assert fit.prepare_resume(provenance, expected, (result,), manifest) == saved
    result.write_text("modified\n")
    with pytest.raises(ValueError, match="partly written"):
        fit.prepare_resume(provenance, expected, (result,), manifest)
    assert result.read_text() == "modified\n"


def test_main_all_control_shifts_and_safe_resume(tmp_path, monkeypatch, synthetic):
    response, surprisal, sae, residual, _ = synthetic
    data_dir, feature_dir = tmp_path / "data", tmp_path / "features"
    (data_dir / "response").mkdir(parents=True)
    feature_dir.mkdir()
    response_path = data_dir / "response/d_shift_3_AntTemp"
    with response_path.open("wb") as handle:
        pickle.dump({fit.STORY_NAMES[story]: value for story, value in response.items()}, handle)
    for story in fit.STORIES:
        pd.DataFrame({"surprisal": surprisal[story].ravel(), "word_count": 999}).to_csv(
            feature_dir / f"binned_predictors_story-{story}.csv", index=False
        )
    args = fit.parse_args([
        "--roi", "AntTemp", "--models", "surprisal", "--held-out-stories", "1",
        "--data-dir", str(data_dir), "--feature-dir", str(feature_dir),
        "--surprisal-dir", str(feature_dir), "--output-dir", str(tmp_path / "outputs"),
        "--response-shifts", "0", "26", "52", "78", "104", "--lasso-jobs", "1",
    ])
    monkeypatch.setattr(fit, "parse_args", lambda: args)
    assert len(fit.input_paths(args)) == 10  # response + nine surprisal CSVs, no LM cache
    fit.main()
    result_path = args.output_dir / "held_out_story_results_AntTemp.csv"
    frame = pd.read_csv(result_path)
    assert frame.response_circular_shift_trs.tolist() == [0, 26, 52, 78, 104]
    assert (frame.n_final_predictors == 1).all()
    assert (frame.n_lm_features_selected == 0).all()
    predictions = pd.read_csv(args.output_dir / "held_out_predictions_AntTemp.csv")
    shifted = {story: np.roll(response[story], 26) for story in fit.STORIES}
    training = np.concatenate([shifted[story] for story in fit.STORIES if story != "1"])
    expected_observed = ((shifted["1"] - training.mean()) / training.std())[50:]
    np.testing.assert_allclose(
        predictions.loc[predictions.response_circular_shift_trs == 26, "observed_z"],
        expected_observed, atol=3e-7,
    )
    before = result_path.read_bytes()
    monkeypatch.setattr(fit, "fit_outer_fold", lambda **kwargs: pytest.fail("Completed fits reran"))
    fit.main()
    assert result_path.read_bytes() == before
    args.k_best += 1
    with pytest.raises(ValueError, match="incompatible"):
        fit.main()
    assert result_path.read_bytes() == before


@pytest.mark.parametrize("model", ["surprisal", "sae_0_128", "residual"])
def test_exact_numerical_parity_with_optional_reference(synthetic, model):
    reference_path = os.environ.get("NATURAL_STORIES_REFERENCE_FIT")
    if not reference_path:
        pytest.skip("Set NATURAL_STORIES_REFERENCE_FIT for an original-script parity check")
    reference = load_module(reference_path, "natural_stories_original_fit")
    response, surprisal, sae, residual, args = synthetic
    old = reference.fit_outer_fold(model, "1", response, surprisal, sae, residual, args)
    new = fit.fit_outer_fold(model, "1", response, surprisal, sae, residual, args)
    for key, value in old[0].items():
        if isinstance(value, float) and np.isnan(value):
            assert np.isnan(new[0][key])
        else:
            assert new[0][key] == value, key
    assert old[1] == new[1]
    assert old[2] == new[2]
