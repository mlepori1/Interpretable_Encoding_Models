"""Fetch only the released Natural Stories transcripts and shift-3 responses."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tempfile
from urllib.request import urlopen


HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "data_manifest.json"


def validate_bytes(payload: bytes, entry: dict) -> None:
    if len(payload) != entry["bytes"]:
        raise ValueError(f"Size mismatch for {entry['path']}")
    if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {entry['path']}")


def prepare_data(
    data_dir: Path,
    *,
    manifest_path: Path = MANIFEST,
    source_dir: Path | None = None,
    verify_only: bool = False,
) -> list[Path]:
    """Validate all inputs before creating any missing files; never overwrite."""
    manifest = json.loads(manifest_path.read_text())
    commit = manifest["source_commit"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Data source must be an immutable 40-character Git commit")
    repository = manifest["source_repository"]
    if not repository.startswith("https://github.com/"):
        raise ValueError("Expected a GitHub source repository")
    raw_root = repository.replace("https://github.com/", "https://raw.githubusercontent.com/", 1)
    raw_root += f"/{commit}/{manifest['source_subdirectory']}"
    pending = []
    paths = []
    for entry in manifest["files"]:
        relative = PurePosixPath(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid relative path: {relative}")
        destination = data_dir / relative
        paths.append(destination)
        if destination.exists():
            validate_bytes(destination.read_bytes(), entry)
            continue
        if verify_only:
            raise FileNotFoundError(destination)
        if source_dir is not None:
            payload = (source_dir / relative).read_bytes()
        else:
            with urlopen(f"{raw_root}/{relative}", timeout=60) as response:
                payload = response.read()
        validate_bytes(payload, entry)
        pending.append((destination, payload))

    for destination, payload in pending:
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Atomic creation without overwriting another process's existing file.
        with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
            staged = Path(temporary) / "download"
            staged.write_bytes(payload)
            try:
                destination.hardlink_to(staged)
            except FileExistsError:
                if destination.read_bytes() != payload:
                    raise ValueError(f"Conflicting input appeared at {destination}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE / "data")
    parser.add_argument(
        "--source-dir", type=Path,
        help="Optional existing NaturalStories directory; no other checkout is required.",
    )
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    paths = prepare_data(args.data_dir, source_dir=args.source_dir, verify_only=args.verify_only)
    print(f"Verified {len(paths)} released input files in {args.data_dir.resolve()}")


if __name__ == "__main__":
    main()
