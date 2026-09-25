"""
Fits a PCA basis on Gemma-2-2b residual stream activations collected from FineWeb documents.
The fitted PCA object is pickled for later use as a featurizer (e.g. projecting sentence
embeddings onto the top-k principal components).
"""

import argparse
import os
import pickle as pkl

import numpy as np
import torch
from datasets import load_dataset
from sae_lens import HookedSAETransformer
from sklearn.decomposition import PCA
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        default="gemma-2-2b",
        help="Model to use as backbone for activation collection",
    )
    parser.add_argument(
        "--layer",
        default=12,
        type=int,
        help="Layer to extract residual stream activations from",
    )
    parser.add_argument(
        "--n_documents",
        default=10000,
        type=int,
        help="Number of FineWeb documents to sample",
    )
    parser.add_argument(
        "--tokens_per_doc",
        default=128,
        type=int,
        help="Max number of tokens (excluding BOS) to take per document",
    )
    parser.add_argument(
        "--n_components",
        default=128,
        type=int,
        help="Number of principal components to keep",
    )
    parser.add_argument(
        "--fineweb_subset",
        default="sample-10BT",
        help="FineWeb config/subset name to stream documents from",
    )
    parser.add_argument(
        "-e",
        "--embedding_method",
        default="mean",
        help="Embedding method used when the stimulus activations were featurized (used to locate them for --featurize_activations)",
    )
    parser.add_argument(
        "--fit_pca",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether to collect FineWeb activations and fit a new PCA",
    )
    parser.add_argument(
        "--featurize_activations",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether to project the stimulus activations for this layer onto the fitted PCA basis. "
        "If --fit_pca is also set, uses the PCA just fit; otherwise loads it from disk.",
    )
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    return args


def get_hookpoint(layer):
    return f"blocks.{layer}.hook_resid_post"


def get_fineweb_documents(n_documents, subset, seed):
    """Stream and sample text documents from FineWeb."""
    ds = load_dataset("HuggingFaceFW/fineweb", name=subset, split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    documents = []
    for example in ds:
        text = example["text"].strip()
        if not text:
            continue
        documents.append(text)
        if len(documents) == n_documents:
            break
    return documents


def collect_activations(model, documents, layer, tokens_per_doc):
    """Collect per-token residual stream activations for a set of documents.

    Returns:
        activations: (n_tokens_total, d_model) array, BOS tokens omitted since
            they are OOD relative to the rest of the sequence.
    """
    hookpoint = get_hookpoint(layer)
    all_activations = []

    for doc in tqdm(documents):
        tokens = model.to_tokens(doc, prepend_bos=True)
        tokens = tokens[:, : tokens_per_doc + 1]  # +1 to account for BOS
        if tokens.shape[1] < 2:
            continue

        _, cache = model.run_with_cache(tokens, names_filter=hookpoint)
        acts = cache[hookpoint][0, 1:]  # Omit BOS token
        all_activations.append(acts.cpu().to(torch.float32).numpy())

    return np.concatenate(all_activations, axis=0)


if __name__ == "__main__":
    args = parse_arguments()

    root_dir = os.path.join("..", "features", args.model, "pca")
    os.makedirs(root_dir, exist_ok=True)
    pca_file = os.path.join(root_dir, f"{args.layer}.pkl")

    pca = None

    if args.fit_pca:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_grad_enabled(False)
        model = HookedSAETransformer.from_pretrained_no_processing(
            args.model,
            device=device,
            dtype=torch.bfloat16,
        )

        print(f"Sampling {args.n_documents} documents from FineWeb ({args.fineweb_subset})")
        documents = get_fineweb_documents(args.n_documents, args.fineweb_subset, args.seed)
        print(f"Collected {len(documents)} documents")

        print(f"Collecting layer {args.layer} activations ({args.tokens_per_doc} tokens/doc)")
        activations = collect_activations(model, documents, args.layer, args.tokens_per_doc)
        print(f"Activation matrix shape: {activations.shape}")

        print(f"Fitting PCA with {args.n_components} components")
        pca = PCA(n_components=args.n_components, random_state=args.seed)
        pca.fit(activations)
        print(f"Explained variance ratio (top {args.n_components}): {pca.explained_variance_ratio_.sum():.4f}")

        pkl.dump(pca, open(pca_file, "wb"))
        print(f"Saved PCA object to {pca_file}")

    if args.featurize_activations:
        if pca is None:
            print(f"Loading PCA from {pca_file}")
            pca = pkl.load(open(pca_file, "rb"))

        stimulus_activations_file = os.path.join(
            "..", "features", args.model, "hidden_states", args.embedding_method, f"{args.layer}.pkl"
        )
        print(f"Loading stimulus activations from {stimulus_activations_file}")
        stimulus_activations = pkl.load(open(stimulus_activations_file, "rb"))

        reduced_activations = pca.transform(np.asarray(stimulus_activations, dtype=np.float32))

        reduced_file = os.path.join(root_dir, f"{args.layer}_reduced_activations.pkl")
        pkl.dump(reduced_activations, open(reduced_file, "wb"))
        print(f"Saved reduced activations {reduced_activations.shape} to {reduced_file}")
