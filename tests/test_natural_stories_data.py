"""Input provenance checks without network or model dependencies."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


MODULE = Path(__file__).resolve().parents[1] / "replications/natural_stories/prepare_data.py"
spec = importlib.util.spec_from_file_location("natural_stories_prepare_data", MODULE)
data_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_module)


def fixture_manifest(tmp_path, payload=b"released input\n", relative="response/d_shift_3"):
    source = tmp_path / "source"
    (source / "response").mkdir(parents=True)
    (source / "response/d_shift_3").write_bytes(payload)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "source_repository": "https://github.com/example/source",
        "source_commit": "a" * 40,
        "source_subdirectory": "data",
        "files": [{"path": relative, "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}],
    }))
    return source, manifest


def test_copy_validate_and_skip(tmp_path):
    source, manifest = fixture_manifest(tmp_path)
    target = tmp_path / "target"
    first = data_module.prepare_data(target, source_dir=source, manifest_path=manifest)
    assert first[0].read_bytes() == b"released input\n"
    before = first[0].stat().st_mtime_ns
    assert data_module.prepare_data(target, manifest_path=manifest, verify_only=True) == first
    assert first[0].stat().st_mtime_ns == before


def test_reject_wrong_existing_input(tmp_path):
    source, manifest = fixture_manifest(tmp_path)
    target = tmp_path / "target"
    (target / "response").mkdir(parents=True)
    file = target / "response/d_shift_3"
    file.write_bytes(b"not the release")
    with pytest.raises(ValueError, match="mismatch"):
        data_module.prepare_data(target, source_dir=source, manifest_path=manifest)
    assert file.read_bytes() == b"not the release"


def test_reject_bad_download_before_writing(tmp_path):
    source, manifest = fixture_manifest(tmp_path)
    (source / "response/d_shift_3").write_bytes(b"bad")
    target = tmp_path / "target"
    with pytest.raises(ValueError, match="mismatch"):
        data_module.prepare_data(target, source_dir=source, manifest_path=manifest)
    assert not target.exists()


def test_reject_path_traversal(tmp_path):
    source, manifest = fixture_manifest(tmp_path, relative="../outside")
    with pytest.raises(ValueError, match="Invalid relative path"):
        data_module.prepare_data(tmp_path / "target", source_dir=source, manifest_path=manifest)


def test_missing_verify_input(tmp_path):
    _, manifest = fixture_manifest(tmp_path)
    with pytest.raises(FileNotFoundError):
        data_module.prepare_data(tmp_path / "missing", manifest_path=manifest, verify_only=True)


def test_pinned_manifest_contains_only_required_inputs():
    manifest = json.loads(data_module.MANIFEST.read_text())
    files = manifest["files"]
    assert len(files) == 15
    assert len({row["path"] for row in files}) == 15
    assert sum(row["bytes"] for row in files) == 247002
    assert {Path(row["path"]).name for row in files if row["path"].startswith("transcribed/")} == {
        f"{story}.csv" for story in (1, 2, 3, 4, 5, 6, 7, 9, 10)
    }
