"""Offline tests: no model downloads, GPU allocation, or historical run paths."""

import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replications.natural_stories import extract


def test_context_includes_target_without_future_words():
    words = [f"word{i}" for i in range(105)]
    text, start, first = extract.target_context(words, 103)
    assert first == 4
    assert len(text.split()) == 100
    assert text[start:] == "word103"
    assert "word104" not in text
    assert extract.target_context(words, 0) == ("word0", 0, 0)


def test_offsets_exclude_specials_and_select_all_target_subtokens():
    # BOS/padding have zero offsets; tokens may include a preceding space.
    offsets = np.array([[[0, 0], [0, 0], [0, 1], [1, 4], [4, 6]],
                        [[0, 0], [0, 1], [1, 2], [0, 0], [0, 0]]])
    masks = extract.target_masks_from_offsets(offsets, ["a word", "hi"], [2, 0])
    assert np.flatnonzero(masks[0]).tolist() == [3, 4]
    assert np.flatnonzero(masks[1]).tolist() == [1, 2]
    with pytest.raises(ValueError, match="No token"):
        extract.target_masks_from_offsets(np.zeros((1, 2, 2)), ["word"], [0])


def test_word_end_boundaries_and_invalid_grid():
    assert extract.word_bins([.5, 2., 2.01, 4., 6.], 3).tolist() == [0, 0, 1, 1, 2]
    for ends in ([0], [-1], [np.nan], [6.01]):
        with pytest.raises(ValueError):
            extract.word_bins(ends, 3)


def test_equal_word_weighting_and_nonempty_bin_imputation():
    words = np.array([[2., 0.], [6., 2.], [12., 10.]], dtype=np.float32)
    sae, residual, counts = extract.bin_word_features(sp.csr_matrix(words), words, [1, 2, 3], 4)
    # Nonempty bins [4,1] and [12,10]; empty bins get their unweighted mean.
    expected = np.array([[4., 1.], [12., 10.], [8., 5.5], [8., 5.5]])
    np.testing.assert_allclose(sae.toarray(), expected)
    np.testing.assert_allclose(residual, expected)
    assert counts.tolist() == [2, 1, 0, 0]


def test_used_surprisal_is_word_sum_not_subtoken_mean():
    words = pd.DataFrame({"end": [1., 2., 3.], "surprisal_nats": [6., 2., 10.],
                          "mean_target_token_surprisal_nats": [2., 2., 5.]})
    result = extract.bin_surprisal(words, 3)
    assert result.columns.tolist() == ["surprisal"]  # no word-count predictor
    np.testing.assert_array_equal(result.surprisal, [4., 10., 7.])


def cache_fixture(tmp_path):
    data = tmp_path / "features.csv"
    data.write_text("surprisal\n2.0\n")
    manifest = tmp_path / "manifest.json"
    config, inputs = {"representation": "surprisal"}, {"input_sha": "abc"}
    saved = {"config": config, "inputs": inputs,
             "model_artifacts": {"weights": {"sha256": "def"}},
             "resolved_revisions": {"model": extract.MODEL_REVISION},
             "outputs": {"bins": {"sha256": extract.sha256(data)}}}
    manifest.write_text(json.dumps(saved))
    return data, manifest, config, inputs, saved


def test_cache_accepts_matching_hashes_and_configuration(tmp_path):
    data, manifest, config, inputs, saved = cache_fixture(tmp_path)
    assert extract.checked_cache(manifest, {"bins": data}, config, inputs) == saved
    assert extract.checked_cache(tmp_path / "new.json", {"bins": tmp_path / "new.csv"}, config, inputs) is None


@pytest.mark.parametrize("change", ["output", "input", "config", "missing_output", "missing_manifest", "provenance"])
def test_cache_rejects_changed_or_incomplete_files(tmp_path, change):
    data, manifest, config, inputs, saved = cache_fixture(tmp_path)
    if change == "output":
        data.write_text("surprisal\n3.0\n")
    elif change == "input":
        inputs = {"input_sha": "changed"}
    elif change == "config":
        config = {"representation": "jumprelu"}
    elif change == "missing_output":
        data.unlink()
    elif change == "missing_manifest":
        manifest.unlink()
    else:
        saved.pop("model_artifacts")
        manifest.write_text(json.dumps(saved))
    with pytest.raises(ValueError):
        extract.checked_cache(manifest, {"bins": data}, config, inputs)


def test_sae_downloads_are_revision_pinned_and_hashed(tmp_path, monkeypatch):
    spec = extract.SAES["matryoshka"]
    weight = tmp_path / "sae_weights.safetensors"
    weight.write_bytes(b"offline-test-artifact")
    calls = []
    hub = types.ModuleType("huggingface_hub")

    def download(**kwargs):
        calls.append(kwargs)
        return str(weight)

    hub.hf_hub_download = download
    loaders = types.ModuleType("sae_lens.loading.pretrained_sae_loaders")
    loaders.hf_hub_download = lambda **kwargs: pytest.fail("Unpinned loader was called")
    directory = types.ModuleType("sae_lens.loading.pretrained_saes_directory")
    directory.get_pretrained_saes_directory = lambda: {
        spec["release"]: types.SimpleNamespace(repo_id=spec["repo"], saes_map={spec["id"]: "folder"})}
    loading = types.ModuleType("sae_lens.loading")
    loading.pretrained_sae_loaders = loaders
    lens = types.ModuleType("sae_lens")

    class FakeSAE:
        @staticmethod
        def from_pretrained_with_cfg_and_sparsity(**kwargs):
            assert kwargs["dtype"] == "float32"
            loaders.hf_hub_download(repo_id=spec["repo"], filename="weights", subfolder="folder")
            return "sae", {"config": "test"}, None

    lens.SAE = FakeSAE
    for name, module in [("huggingface_hub", hub), ("sae_lens", lens), ("sae_lens.loading", loading),
                         ("sae_lens.loading.pretrained_sae_loaders", loaders),
                         ("sae_lens.loading.pretrained_saes_directory", directory)]:
        monkeypatch.setitem(sys.modules, name, module)
    sae, cfg, artifacts = extract.pinned_sae_loader(spec, "cpu", True)
    assert sae == "sae"
    assert calls[0]["revision"] == spec["revision"]
    assert calls[0]["local_files_only"] is True
    assert artifacts["folder/weights"]["sha256"] == extract.sha256(weight)


def test_help_imports_no_model_libraries():
    root = Path(__file__).resolve().parents[1]
    code = ("import sys; from replications.natural_stories import extract; "
            "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules; "
            "assert 'sae_lens' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], cwd=root, check=True)
    output = subprocess.run([sys.executable, "-m", "replications.natural_stories.extract", "--help"],
                            cwd=root, check=True, text=True, capture_output=True)
    assert "--data-dir" in output.stdout
    assert "--validation-reference-dir" in output.stdout


def test_optional_reference_directory_is_explicit_and_complete(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing optional sentence-reference"):
        extract.sentence_reference_paths(tmp_path)
    for filename in ("stimset.csv", "sae.pkl", "residual.pkl"):
        (tmp_path / filename).write_bytes(b"test reference")
    assert extract.sentence_reference_paths(tmp_path) == {
        "stimuli": tmp_path / "stimset.csv", "sae": tmp_path / "sae.pkl",
        "residual": tmp_path / "residual.pkl"}


def test_missing_reference_files_fail_before_model_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(extract, "load_models", lambda *args: pytest.fail("Must check references first"))
    with pytest.raises(FileNotFoundError, match="Missing optional sentence-reference"):
        extract.main(["--data-dir", str(tmp_path), "--output-dir", str(tmp_path / "output"),
                      "--representation", "matryoshka", "--validation-reference-dir", str(tmp_path / "references")])
    assert not (tmp_path / "output").exists()


def test_sentence_reference_option_requires_sae_representation(tmp_path):
    with pytest.raises(SystemExit):
        extract.main(["--data-dir", str(tmp_path), "--output-dir", str(tmp_path),
                      "--representation", "surprisal", "--validation-reference-dir", str(tmp_path)])


def test_model_loader_uses_local_snapshot_and_preserves_left_padding(tmp_path, monkeypatch):
    snapshot = tmp_path / extract.MODEL_REVISION
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    calls = {}
    tokenizer = types.SimpleNamespace(is_fast=True, add_bos_token=True,
                                      padding_side="right", pad_token_id=0)

    class Model:
        def eval(self):
            return self

        def to(self, *args, **kwargs):
            return self

    model = Model()
    model.tokenizer = tokenizer
    hub = types.ModuleType("huggingface_hub")

    def snapshot_download(repo, **kwargs):
        calls["snapshot"] = (repo, kwargs)
        return str(snapshot)

    hub.snapshot_download = snapshot_download
    transformers = types.ModuleType("transformers")

    def load_tokenizer(path, **kwargs):
        calls["tokenizer"] = (path, kwargs)
        return tokenizer

    def load_model(path, **kwargs):
        calls["model"] = (path, kwargs)
        return model

    transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=load_tokenizer)
    transformers.AutoModelForCausalLM = types.SimpleNamespace(from_pretrained=load_model)
    torch = types.ModuleType("torch")
    torch.bfloat16 = "bfloat16"
    lens = types.ModuleType("sae_lens")

    def load_hooked(name, **kwargs):
        calls["hooked"] = kwargs
        tokenizer.padding_side = kwargs["default_padding_side"]
        assert kwargs["hf_model"] is model
        return model

    lens.HookedSAETransformer = types.SimpleNamespace(from_pretrained_no_processing=load_hooked)
    for name, module in [("huggingface_hub", hub), ("transformers", transformers), ("torch", torch), ("sae_lens", lens)]:
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(extract, "pinned_sae_loader", lambda *args: (Model(), {}, {"weights": {"sha256": "abc"}}))
    config = {"model_revision": extract.MODEL_REVISION, "representation": "matryoshka", "sae": extract.SAES["matryoshka"]}
    loaded_tokenizer, _, _, provenance = extract.load_models(config, "cpu", True)
    assert loaded_tokenizer.padding_side == "left"
    assert calls["snapshot"][1]["revision"] == extract.MODEL_REVISION
    assert calls["model"][0] == str(snapshot)
    assert calls["model"][1]["local_files_only"] is True
    assert calls["hooked"]["revision"] == extract.MODEL_REVISION
    assert calls["hooked"]["local_files_only"] is True
    assert provenance["resolved_revisions"]["tokenizer"] == extract.MODEL_REVISION


def test_cli_rejects_unpinned_revisions_before_loading_models(tmp_path):
    with pytest.raises(SystemExit):
        extract.main(["--data-dir", str(tmp_path), "--output-dir", str(tmp_path),
                      "--representation", "surprisal", "--model-revision", "main"])
