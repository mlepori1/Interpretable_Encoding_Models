import json
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import featurize_qa_emb as qa


def test_frozen_question_vocabulary():
    questions = qa.load_questions(qa.DEFAULT_QUESTIONS_FILE)
    assert len(questions) == 29
    assert (
        qa.sha256_strings(questions["question"].tolist())
        == qa.FROZEN_QUESTIONS_SHA256
    )
    assert questions.iloc[0]["question"].startswith(
        "Is the sentence expressing skepticism"
    )
    assert questions.iloc[-2]["question"] != questions.iloc[-1]["question"]
    assert questions["source_candidate_index"].nunique() == 29


def test_prompts_and_released_answer_scoring():
    assert qa.MISTRAL_ZERO_SHOT_PROMPT.startswith("<s>[INST]'Input text:")
    assert "and i just kept on laughing" in qa.LLAMA_FIVE_SHOT_PROMPT
    assert qa.sha256_strings([qa.MISTRAL_ZERO_SHOT_PROMPT]) == (
        "b7afe2064aceb70c1f78e143864079ef9772ef5ab91fcbaa56e652dcb4815984"
    )
    assert qa.sha256_strings([qa.LLAMA_ZERO_SHOT_PROMPT]) == (
        "ad3ba8044118ceab744388f887d53560f33a89d6a6f64a21feee9cf45fce1bbd"
    )
    assert qa.sha256_strings([qa.LLAMA_FIVE_SHOT_PROMPT]) == (
        "7c315276b24699ec9a4adab196e627a23cc7f2eaf705d88d8437fd51d961dbd3"
    )
    assert qa.score_decoded_token("Yes") == 1
    assert qa.score_decoded_token(" yes") == 1
    assert qa.score_decoded_token("No") == 0
    # This intentionally preserves the authors' substring-based conversion.
    assert qa.score_decoded_token("Yesterday") == 1


def test_participant_alignment_check(tmp_path):
    participants_dir = tmp_path / "participants"
    sentences = ["one", "two"]
    for participant_index in range(1, 9):
        participant_dir = participants_dir / f"p{participant_index}"
        participant_dir.mkdir(parents=True)
        pd.DataFrame({"sentence": sentences}).to_csv(
            participant_dir / "stimset.csv", index=False
        )

    stimuli = qa.load_stimuli(
        participants_dir / "p1" / "stimset.csv",
        participants_dir,
        check_alignment=True,
    )
    assert stimuli["sentence"].tolist() == sentences

    pd.DataFrame({"sentence": ["one", "changed"]}).to_csv(
        participants_dir / "p8" / "stimset.csv", index=False
    )
    with pytest.raises(ValueError, match="do not match"):
        qa.load_stimuli(
            participants_dir / "p1" / "stimset.csv",
            participants_dir,
            check_alignment=True,
        )


def test_raw_cache_round_trip_and_metadata_guard(tmp_path):
    config = qa.QA_CONFIGS["mistral_zero_shot"]
    metadata = qa._raw_cache_metadata(
        config, ["question?"], ["sentence"], revision="main"
    )
    path = tmp_path / "raw.npz"
    values, tokens, loaded_metadata = qa.load_or_initialize_raw_cache(
        path, (1, 1), metadata
    )
    assert values.item() == -1
    values[0, 0] = 1
    tokens[0, 0] = "Yes"
    qa.save_raw_cache(path, values, tokens, loaded_metadata)

    values2, tokens2, _ = qa.load_or_initialize_raw_cache(
        path, (1, 1), metadata
    )
    assert values2.item() == 1
    assert tokens2.item() == "Yes"

    bad_metadata = dict(metadata, requested_revision="different")
    with pytest.raises(ValueError, match="metadata mismatch"):
        qa.load_or_initialize_raw_cache(path, (1, 1), bad_metadata)


def test_three_way_aggregation_outputs_unscaled_matrix(tmp_path):
    questions = qa.load_questions(qa.DEFAULT_QUESTIONS_FILE)
    stimuli = pd.DataFrame(
        {
            "sentence": [f"sentence {i}" for i in range(200)],
            "item_id": np.arange(200),
        }
    )
    questions_path, stimuli_path = qa.prepare_output_metadata(
        tmp_path,
        qa.DEFAULT_QUESTIONS_FILE,
        questions,
        tmp_path / "input.csv",
        stimuli,
    )
    shape = (200, 29)
    raw_values = {
        "mistral_zero_shot": np.zeros(shape, dtype=np.int8),
        "llama_zero_shot": np.ones(shape, dtype=np.int8),
        "llama_five_shot": np.ones(shape, dtype=np.int8),
    }
    for config_name, values in raw_values.items():
        config = qa.QA_CONFIGS[config_name]
        metadata = qa._raw_cache_metadata(
            config,
            questions["question"].tolist(),
            stimuli["sentence"].tolist(),
            revision="main",
        )
        qa.save_raw_cache(
            tmp_path / "raw_answers" / f"{config_name}.npz",
            values,
            np.full(shape, "Yes", dtype="<U128"),
            metadata,
        )

    features = qa.aggregate_outputs(
        tmp_path, questions, stimuli, questions_path, stimuli_path
    )
    assert features.shape == shape
    assert features.dtype == np.float32
    np.testing.assert_allclose(features, np.float32(2.0 / 3.0))

    with (tmp_path / "qa_emb.pkl").open("rb") as handle:
        pickled = pickle.load(handle)
    np.testing.assert_array_equal(features, pickled)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["baseline"] == "QA-Emb-29 (frozen transfer)"
    assert manifest["shape"] == [200, 29]
    assert manifest["upstream"]["commit"] == qa.UPSTREAM_COMMIT


def test_resolve_config_names():
    assert qa.resolve_config_names(["all"]) == qa.ENSEMBLE_CONFIG_NAMES
    assert qa.resolve_config_names(
        ["llama_five_shot", "mistral_zero_shot"]
    ) == ("mistral_zero_shot", "llama_five_shot")
    with pytest.raises(ValueError, match="cannot be combined"):
        qa.resolve_config_names(["all", "mistral_zero_shot"])


def test_rebuild_frozen_features_from_released_answers(tmp_path):
    """Exercise the CLI on the released caches without loading either LM."""
    shutil.copytree(qa.DEFAULT_OUTPUT_DIR / "raw_answers", tmp_path / "raw_answers")
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "src" / "featurize_qa_emb.py"),
         "--aggregate-only", "--output-dir", str(tmp_path)],
        check=True,
    )
    with (tmp_path / "qa_emb.pkl").open("rb") as handle:
        rebuilt = pickle.load(handle)
    with (qa.DEFAULT_OUTPUT_DIR / "qa_emb.pkl").open("rb") as handle:
        released = pickle.load(handle)
    np.testing.assert_array_equal(rebuilt, released)
    released_manifest = json.loads((qa.DEFAULT_OUTPUT_DIR / "manifest.json").read_text())
    assert qa.sha256_bytes(rebuilt.tobytes(order="C")) == (
        released_manifest["checksums"]["matrix_bytes_sha256"]
    )
    for config_name in qa.ENSEMBLE_CONFIG_NAMES:
        with np.load(tmp_path / "raw_answers" / f"{config_name}.npz") as cache:
            rescored = np.array(
                [qa.score_decoded_token(token) for token in cache["tokens"].flat],
                dtype=np.int8,
            ).reshape(cache["values"].shape)
            np.testing.assert_array_equal(rescored, cache["values"])
    for filename, key in (
        ("qa_emb.csv", "qa_emb_csv_sha256"),
        ("questions.csv", "questions_csv_sha256"),
        ("stimuli.csv", "stimuli_csv_sha256"),
    ):
        assert qa.sha256_file(tmp_path / filename) == released_manifest["checksums"][key]


@pytest.mark.parametrize("key", ["sentences_sha256", "questions_sha256", "prompt_sha256", "model_id"])
def test_aggregation_rejects_mismatched_cache_metadata(tmp_path, key):
    shutil.copytree(qa.DEFAULT_OUTPUT_DIR / "raw_answers", tmp_path / "raw_answers")
    raw_path = tmp_path / "raw_answers" / "mistral_zero_shot.npz"
    with np.load(raw_path) as cache:
        values, tokens = cache["values"], cache["tokens"]
        metadata = json.loads(str(cache["metadata"].item()))
    metadata[key] = "mismatched"
    qa.save_raw_cache(raw_path, values, tokens, metadata)
    questions = qa.load_questions(qa.DEFAULT_QUESTIONS_FILE)
    stimuli = qa.load_stimuli(qa.DEFAULT_STIMULUS_FILE, qa.DEFAULT_PARTICIPANTS_DIR)
    with pytest.raises(ValueError, match=key):
        qa.aggregate_outputs(tmp_path, questions, stimuli, qa.DEFAULT_QUESTIONS_FILE,
                             qa.DEFAULT_STIMULUS_FILE)
    assert not (tmp_path / "qa_emb.pkl").exists()


def test_dry_run_does_not_create_output(tmp_path):
    output_dir = tmp_path / "not_created"
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "src" / "featurize_qa_emb.py"),
         "--dry-run", "--output-dir", str(output_dir)],
        check=True,
    )
    assert not output_dir.exists()
