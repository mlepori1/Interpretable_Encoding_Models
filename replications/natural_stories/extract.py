#!/usr/bin/env python3
"""Extract pinned Gemma features; no encoding models or word-count diagnostics.

Run with ``python -m replications.natural_stories.extract --help``. Input data
must contain transcribed/{story}.csv and response/d_shift_3 (a trusted pickle).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import pickle
import platform
import re
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import scipy.sparse as sp

STORY_NAMES = {"1": "boar", "2": "aqua", "3": "matchstickseller", "4": "kingofbirds",
               "5": "elvis", "6": "mrsticky", "7": "highschool", "9": "tulips", "10": "tree"}
MODEL_ID = "google/gemma-2-2b"
MODEL_REVISION = "c5ebcd40d208330abc697524c919956e692655cf"
HOOK_NAME = "blocks.12.hook_resid_post"
# SAE revisions are the sole locally cached snapshots found during the audit.
# Unlike the LM revision, these were NOT recorded in the historical manifests.
SAES = {
    "matryoshka": {
        "release": "gemma-2-2b-res-matryoshka-dc", "id": HOOK_NAME,
        "repo": "chanind/gemma-2-2b-batch-topk-matryoshka-saes-w-32k-l0-40",
        "revision": "4c28e790b0213430d52dd8e5a09f681a53f5b79d",
    },
    "jumprelu": {
        "release": "gemma-scope-2b-pt-res-canonical", "id": "layer_12/width_16k/canonical",
        "repo": "google/gemma-scope-2b-pt-res",
        "revision": "fd571b47c1c64851e9b1989792367b9babb4af63",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment() -> dict:
    result = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "scipy", "torch", "transformers", "tokenizers",
                 "sae-lens", "transformer-lens", "huggingface-hub", "safetensors"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def read_transcript(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if not {"text", "start", "end"}.issubset(frame.columns) or frame["text"].isna().any():
        raise ValueError(f"Invalid transcript columns or missing words: {path}")
    times = frame[["start", "end"]].to_numpy(dtype=float)
    if not len(frame) or not np.isfinite(times).all() or np.any(times[:, 0] > times[:, 1]):
        raise ValueError(f"Invalid transcript timestamps: {path}")
    if np.any(np.diff(times[:, 1]) < 0):
        raise ValueError(f"Word end timestamps are not sorted: {path}")
    return frame.reset_index(drop=True)


def target_context(words: list[str], index: int, context_words: int = 100):
    """Return text, target character start, and first word index; target included."""
    start = max(0, index - context_words + 1)
    chunk = words[start:index + 1]
    prefix = " ".join(chunk[:-1])
    return " ".join(chunk), len(prefix) + bool(prefix), start


def target_masks_from_offsets(offsets, texts, target_starts):
    """Select tokens overlapping target text; (0,0) special/pad offsets excluded."""
    masks = []
    for row, (text, start) in enumerate(zip(texts, target_starts)):
        mask = (offsets[row, :, 1] > start) & (offsets[row, :, 0] < len(text))
        if not bool(mask.any()):
            raise ValueError(f"No token overlaps target word in {text!r}")
        masks.append(mask)
    return masks


def word_bins(ends, number_of_bins: int) -> np.ndarray:
    ends = np.asarray(ends, dtype=float)
    if number_of_bins < 1 or not np.isfinite(ends).all() or np.any(ends <= 0):
        raise ValueError("Word ends must be finite and positive on a nonempty bin grid")
    if np.any(ends > 2 * number_of_bins):
        raise ValueError("Transcript extends beyond response grid")
    return np.searchsorted(np.arange(number_of_bins) * 2, ends, side="left") - 1


def bin_word_features(word_sae, word_residual, ends, number_of_bins):
    bins = word_bins(ends, number_of_bins)
    counts = np.bincount(bins, minlength=number_of_bins)
    nonempty = counts > 0
    if not nonempty.any():
        raise ValueError("No words in response grid")
    aggregator = sp.csr_matrix((1.0 / counts[bins], (bins, np.arange(len(bins)))),
                               shape=(number_of_bins, len(bins)), dtype=np.float32)
    binned_sae = (aggregator @ word_sae).tocsr()
    binned_residual = np.asarray(aggregator @ word_residual, dtype=np.float32)
    if not nonempty.all():
        mean_sae = sp.csr_matrix(binned_sae[nonempty].mean(axis=0))
        binned_sae = sp.vstack([binned_sae.getrow(i) if nonempty[i] else mean_sae
                               for i in range(number_of_bins)], format="csr")
        binned_residual[~nonempty] = binned_residual[nonempty].mean(axis=0)
    return binned_sae, binned_residual, counts


def bin_surprisal(frame: pd.DataFrame, number_of_bins: int) -> pd.DataFrame:
    bins = word_bins(frame["end"], number_of_bins)
    values = frame["surprisal_nats"].to_numpy(dtype=float)
    result = np.full(number_of_bins, np.nan)
    for index in range(number_of_bins):
        selected = values[bins == index]
        if selected.size:
            result[index] = selected.mean()
    if not np.isfinite(result).any():
        raise ValueError("No finite surprisal values")
    result[np.isnan(result)] = np.nanmean(result)
    return pd.DataFrame({"surprisal": result})


def token_nlls_for_targets(model, inputs, masks):
    """Sum is performed later, after causally scoring every target subtoken."""
    import torch
    import torch.nn.functional as F

    hidden = model.base_model(**inputs, use_cache=False, return_dict=True).last_hidden_state
    predictor_states, target_ids, lengths = [], [], []
    for row, mask in enumerate(masks):
        positions = torch.where(mask.to(hidden.device))[0]
        if bool((positions <= 0).any()) or bool((inputs["attention_mask"][row, positions - 1] == 0).any()):
            raise ValueError("Target token has no non-padding causal predecessor")
        predictor_states.append(hidden[row, positions - 1, :])
        target_ids.append(inputs["input_ids"][row, positions])
        lengths.append(len(positions))
    logits = model.lm_head(torch.cat(predictor_states))
    softcap = getattr(model.config, "final_logit_softcapping", None)
    if softcap is not None:
        logits = torch.tanh(logits / softcap) * softcap
    losses = F.cross_entropy(logits.float(), torch.cat(target_ids), reduction="none")
    return [part.detach().cpu().numpy() for part in losses.split(lengths)]


def output_paths(output_dir: Path, story: str, representation: str) -> dict:
    folder = output_dir / "features"
    if representation == "surprisal":
        return {"words": folder / f"gemma2_2b_surprisal_story-{story}.csv",
                "bins": folder / f"binned_predictors_story-{story}.csv"}
    return {"word_sae": folder / f"word_sae_story-{story}.npz",
            "word_residual": folder / f"word_residual_story-{story}.npy",
            "word_metadata": folder / f"word_metadata_story-{story}.csv",
            "binned_sae": folder / f"binned_sae_story-{story}.npz",
            "binned_residual": folder / f"binned_residual_story-{story}.npy"}


def checked_cache(manifest_path: Path, paths: dict, config: dict, inputs: dict):
    """Only reuse a complete, configuration-matched, hash-verified cache."""
    if not manifest_path.exists() and not any(p.exists() for p in paths.values()):
        return None
    if not manifest_path.is_file():
        raise ValueError("Unmanifested existing features; use a new output directory or --overwrite")
    saved = json.loads(manifest_path.read_text())
    if saved.get("config") != config or saved.get("inputs") != inputs:
        raise ValueError("Cached feature configuration/input mismatch; use new output directory or --overwrite")
    if not saved.get("model_artifacts") or not saved.get("resolved_revisions"):
        raise ValueError("Cached features lack model artifact provenance")
    for key, path in paths.items():
        if not path.is_file() or sha256(path) != saved.get("outputs", {}).get(key, {}).get("sha256"):
            raise ValueError(f"Cached feature missing or hash mismatch: {path}")
    return saved


def pinned_sae_loader(spec: dict, device: str, local_files_only: bool):
    """Keep SAELens conversion, but force its downloads to one immutable revision.

    SAELens 6.29's SAE loader has no revision argument. Scope the download
    override to that loader call; no global cache refs or source files change.
    """
    from huggingface_hub import hf_hub_download
    from sae_lens import SAE
    from sae_lens.loading import pretrained_sae_loaders as loaders
    from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory

    registry = get_pretrained_saes_directory()[spec["release"]]
    if registry.repo_id != spec["repo"] or spec["id"] not in registry.saes_map:
        raise ValueError("SAELens release mapping differs from the pinned specification")
    artifacts = {}

    def download(repo_id, filename, **kwargs):
        if repo_id != spec["repo"]:
            raise ValueError(f"Unexpected SAE repository: {repo_id}")
        kwargs.pop("revision", None)
        kwargs["local_files_only"] = local_files_only
        path = Path(hf_hub_download(repo_id=repo_id, filename=filename,
                                   revision=spec["revision"], **kwargs))
        relative = "/".join(filter(None, [kwargs.get("subfolder"), filename]))
        artifacts[relative] = {"sha256": sha256(path), "bytes": path.stat().st_size}
        return str(path)

    with patch.object(loaders, "hf_hub_download", download):
        sae, cfg, _ = SAE.from_pretrained_with_cfg_and_sparsity(
            release=spec["release"], sae_id=spec["id"], device=device, dtype="float32")
    return sae, cfg, artifacts


def load_models(config: dict, device: str, local_files_only: bool):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    snapshot = Path(snapshot_download(MODEL_ID, revision=config["model_revision"],
        allow_patterns=["*.json", "*.model", "*.safetensors"], local_files_only=local_files_only))
    if snapshot.name != config["model_revision"]:
        raise ValueError("Resolved LM snapshot does not match requested revision")
    artifacts = {"lm/" + str(p.relative_to(snapshot)): {"sha256": sha256(p), "bytes": p.stat().st_size}
                 for p in sorted(snapshot.rglob("*")) if p.is_file()}
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), use_fast=True, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.is_fast or not tokenizer.add_bos_token:
        raise ValueError("Expected a fast Gemma tokenizer with BOS enabled")
    model = AutoModelForCausalLM.from_pretrained(str(snapshot), dtype=torch.bfloat16,
                                               local_files_only=True).eval()
    sae, sae_cfg = None, None
    revisions = {"model": snapshot.name, "tokenizer": snapshot.name}
    if config["representation"] != "surprisal":
        from sae_lens import HookedSAETransformer
        model = HookedSAETransformer.from_pretrained_no_processing("gemma-2-2b", hf_model=model,
            tokenizer=tokenizer, device=device, dtype=torch.bfloat16,
            default_padding_side="left", revision=config["model_revision"],
            local_files_only=local_files_only)
        tokenizer = model.tokenizer
        tokenizer.padding_side = "left"
        sae, sae_cfg, sae_artifacts = pinned_sae_loader(config["sae"], device, local_files_only)
        sae = sae.to(device=device, dtype=torch.bfloat16)
        sae.use_error_term = True
        artifacts.update({"sae/" + key: value for key, value in sae_artifacts.items()})
        revisions["sae"] = config["sae"]["revision"]
    model = model.eval().to(device)
    return tokenizer, model, sae, {"model_artifacts": artifacts, "resolved_revisions": revisions,
        "sae_config": sae_cfg, "hardware": torch.cuda.get_device_name(0) if device.startswith("cuda") else "cpu"}


def residual_cache(model, input_ids, attention_mask):
    _, cache = model.run_with_cache(input_ids, attention_mask=attention_mask,
        names_filter=lambda name: name == HOOK_NAME, return_type=None)
    return cache[HOOK_NAME]


def sentence_reference_paths(root: Path) -> dict[str, Path]:
    """Resolve explicit, optional trusted reference files before loading models."""
    paths = {"stimuli": root / "stimset.csv", "sae": root / "sae.pkl",
             "residual": root / "residual.pkl"}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing optional sentence-reference files: " + ", ".join(missing))
    return paths


def validate_sentence_features(model, sae, paths: dict[str, Path], number: int = 5):
    """Optional historical whole-sentence check; not a word-offset validation."""
    import torch
    stimuli = pd.read_csv(paths["stimuli"])
    if "sentence" not in stimuli or len(stimuli) < number or stimuli["sentence"].head(number).isna().any():
        raise ValueError(f"Sentence reference requires at least {number} nonmissing sentences")
    expected = {}
    for key in ("sae", "residual"):
        with paths[key].open("rb") as handle:
            expected[key] = np.asarray(pickle.load(handle), dtype=np.float32)[:number]
        if expected[key].ndim != 2 or len(expected[key]) != number or not np.isfinite(expected[key]).all():
            raise ValueError(f"Invalid {key} reference feature matrix")
    observed = {"sae": [], "residual": []}
    with torch.inference_mode():
        for sentence in stimuli["sentence"].astype(str).head(number):
            tokens = model.to_tokens(sentence, prepend_bos=True)
            residual = residual_cache(model, tokens, torch.ones_like(tokens))[0, 1:, :]
            observed["residual"].append(residual.mean(0).float().cpu().numpy())
            observed["sae"].append(sae.encode(residual).mean(0).float().cpu().numpy())
    result = {"n_sentences": number, "input_hashes": {key: sha256(path) for key, path in paths.items()}}
    for key in observed:
        corr = float(np.corrcoef(np.asarray(observed[key]).ravel(), expected[key].ravel())[0, 1])
        result[key + "_flat_correlation"] = corr
        if not np.isfinite(corr) or corr < .999:
            raise ValueError(f"Sentence-feature validation failed: {result}")
    return result


def extract_story(frame, n_bins, paths, config, tokenizer, model, sae, device):
    import torch
    words = frame["text"].astype(str).tolist()
    rows, sparse_parts, residual_parts = [], [], []
    with torch.inference_mode():
        for offset in range(0, len(words), config["batch_size"]):
            indices = list(range(offset, min(offset + config["batch_size"], len(words))))
            contexts = [target_context(words, i, config["context_words_including_target"]) for i in indices]
            texts, starts, context_starts = zip(*contexts)
            encoded = tokenizer(list(texts), add_special_tokens=True, padding=True,
                                return_offsets_mapping=True, return_tensors="pt")
            masks = target_masks_from_offsets(encoded.pop("offset_mapping"), texts, starts)
            inputs = {key: value.to(device) for key, value in encoded.items()}
            if sae is None:
                losses = token_nlls_for_targets(model, inputs, masks)
                counts = [len(loss) for loss in losses]
            else:
                residual = residual_cache(model, inputs["input_ids"], inputs["attention_mask"])
                targets = [residual[row, mask.to(device), :] for row, mask in enumerate(masks)]
                counts = [len(target) for target in targets]
                activations = sae.encode(torch.cat(targets))
                sparse_parts.append(sp.csr_matrix(torch.stack([a.mean(0) for a in activations.split(counts)]).float().cpu().numpy()))
                residual_parts.append(torch.stack([target.mean(0) for target in targets]).float().cpu().numpy())
            for row, (index, start, count) in enumerate(zip(indices, context_starts, counts)):
                record = {"word_index": index, "text": words[index], "start": float(frame.iloc[index]["start"]),
                    "end": float(frame.iloc[index]["end"]), "context_start_word_index": start,
                    "context_word_count": index - start + 1, "target_token_count": count}
                if sae is None:
                    record.update(surprisal_nats=float(np.sum(losses[row])),
                                  mean_target_token_surprisal_nats=float(np.mean(losses[row])))
                rows.append(record)
    metadata = pd.DataFrame(rows)
    bins = word_bins(metadata["end"], n_bins)
    counts = np.bincount(bins, minlength=n_bins)
    if sae is None:
        if not np.isfinite(metadata["surprisal_nats"]).all():
            raise ValueError("Nonfinite extracted surprisal")
        metadata.to_csv(paths["words"], index=False)
        bin_surprisal(metadata, n_bins).to_csv(paths["bins"], index=False)
    else:
        word_sae = sp.vstack(sparse_parts, format="csr", dtype=np.float32)
        word_residual = np.concatenate(residual_parts).astype(np.float32)
        if not np.isfinite(word_sae.data).all() or not np.isfinite(word_residual).all():
            raise ValueError("Nonfinite extracted model features")
        binned_sae, binned_residual, counts = bin_word_features(word_sae, word_residual, metadata["end"], n_bins)
        metadata["response_bin"] = bins
        metadata.to_csv(paths["word_metadata"], index=False)
        sp.save_npz(paths["word_sae"], word_sae)
        sp.save_npz(paths["binned_sae"], binned_sae)
        np.save(paths["word_residual"], word_residual.astype(np.float16))
        np.save(paths["binned_residual"], binned_residual.astype(np.float16))
    return {"word_count": len(words), "response_bin_count": n_bins, "empty_bins": np.where(counts == 0)[0].tolist()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--representation", choices=["surprisal", *SAES], required=True)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--sae-revision", help="Optional immutable commit override for the selected SAE")
    parser.add_argument("--stories", nargs="+", choices=list(STORY_NAMES), default=list(STORY_NAMES))
    parser.add_argument("--context-words", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validation-reference-dir", type=Path,
                        help="Optional directory with stimset.csv, residual.pkl, and sae.pkl for a five-sentence check")
    args = parser.parse_args(argv)
    if args.context_words < 2 or args.batch_size < 1:
        parser.error("Context must be >=2 words and batch size >=1")
    spec = dict(SAES[args.representation]) if args.representation in SAES else None
    if args.sae_revision and spec:
        spec["revision"] = args.sae_revision
    if args.sae_revision and spec is None:
        parser.error("--sae-revision requires an SAE representation")
    if args.validation_reference_dir is not None and spec is None:
        parser.error("--validation-reference-dir requires an SAE representation")
    for revision in [args.model_revision] + ([spec["revision"]] if spec else []):
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            parser.error("Model and SAE revisions must be immutable 40-character commit hashes")
    references = sentence_reference_paths(args.validation_reference_dir) if args.validation_reference_dir is not None else None
    config = {"schema_version": 1, "representation": args.representation, "model_id": MODEL_ID,
        "model_revision": args.model_revision, "sae": spec, "layer": 12, "hook_name": HOOK_NAME,
        "context_words_including_target": args.context_words, "batch_size": args.batch_size,
        "dtype": "bfloat16", "device": args.device, "environment": environment(),
        "extractor_sha256": sha256(Path(__file__)), "surprisal": "sum subtoken NLL; mean words per bin",
        "features": "mean target-subtoken activations, then mean words per bin",
        "binning": "word end timestamp; exact boundary goes to preceding bin",
        "empty_bins": "within-story mean of nonempty binned features",
        "sentence_reference_hashes": {key: sha256(path) for key, path in references.items()} if references else None}
    response_path = args.data_dir / "response/d_shift_3"
    with response_path.open("rb") as handle:
        responses = pickle.load(handle)
    response_hash = sha256(response_path)
    model_objects = None
    all_manifests = {}
    for story in args.stories:
        transcript_path = args.data_dir / "transcribed" / f"{story}.csv"
        frame = read_transcript(transcript_path)
        n_bins = len(responses[STORY_NAMES[story]])
        word_bins(frame["end"], n_bins)
        inputs = {"transcript_sha256": sha256(transcript_path), "response_sha256": response_hash,
                  "story": story, "n_bins": n_bins}
        paths = output_paths(args.output_dir, story, args.representation)
        manifest_path = args.output_dir / "features" / f"extraction_{args.representation}_story-{story}.json"
        saved = None if args.overwrite else checked_cache(manifest_path, paths, config, inputs)
        if saved is None:
            if model_objects is None:
                model_objects = load_models(config, args.device, args.local_files_only)
                tokenizer, model, sae, provenance = model_objects
                if references:
                    provenance["sentence_feature_validation"] = validate_sentence_features(model, sae, references)
            tokenizer, model, sae, provenance = model_objects
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            stats = extract_story(frame, n_bins, paths, config, tokenizer, model, sae, args.device)
            saved = {"config": config, "inputs": inputs, **provenance, **stats,
                "outputs": {key: {"filename": path.name, "sha256": sha256(path)} for key, path in paths.items()}}
            manifest_path.write_text(json.dumps(saved, indent=2, sort_keys=True, default=str) + "\n")
        elif references and "sentence_feature_validation" not in saved:
            raise ValueError("Cache lacks requested sentence-feature validation; rerun with --overwrite")
        all_manifests[story] = saved
        print(f"Story {story}: {saved['word_count']} words, {saved['response_bin_count']} bins; verified {manifest_path}", flush=True)
    summary = args.output_dir / f"feature_extraction_manifest_{args.representation}.json"
    summary.write_text(json.dumps(all_manifests, indent=2, sort_keys=True, default=str) + "\n")


if __name__ == "__main__":
    main()
