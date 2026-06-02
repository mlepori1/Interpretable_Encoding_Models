"""
This file featurizes the stimuli found in stimset using an LM.
Optionally generate feature vectors using a pretrained SAE.
"""

import argparse
import os
import pickle as pkl
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from sae_lens import HookedSAETransformer, SAE

def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        default="gemma-2-2b",
        help="Model to use as backbone for analysis",
    )

    parser.add_argument(
        "--sae_release",
        default="gemma-scope-2b-pt-res-canonical",
        type=str,
        help="SAE Release",
    )

    parser.add_argument(
        "--sae_id",
        default="layer_{layer}/width_16k/canonical",
        type=str,
        help="SAE id template",
    )

    parser.add_argument("-e", "--embedding_method", default="mean")

    parser.add_argument(
        "--use_sae",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--layer",
        default=None,
        type=int,
        help="If specified, only featurize this layer",
    )

    parser.add_argument(
        "--scale_by_decoder",
        default=False,
        action="store_true",
        help="Whether to scale SAE features by decoder norm. Makes features more comparable.",
    )

    parser.add_argument(
        "--compute_logprobs",
        default=False,
        action="store_true",
        help="Whether to compute logprobs for each sentence",
    )
    parser.add_argument(
        "--three_t",
        default=False,
        action="store_true",
        help="Whether to featurize 3T data",
    )
    args = parser.parse_args()
    return args


def get_sentences(three_t=False):
    # Get stimset sentences

    if three_t:
        sentences = pd.read_csv(
            os.path.join("..", "3T", "drive_suppress_data.csv")
        )["sentence"]
    else:
        sentences = pd.read_csv(
            os.path.join("..", "data", "processed_csvs_anon", "p1", "stimset.csv")
        )["sentence"]
    return sentences




def get_sae_hookpoint(layer):
    return f"blocks.{layer}.hook_resid_post.hook_sae_acts_post"


def get_hookpoint(layer):
    return f"blocks.{layer}.hook_resid_post"


def compute_mean_logprob(model, sentence):
    """Compute the mean log probability of a sentence.
    
    Args:
        model: HookedSAETransformer model
        sentence: String sentence to compute logprob for
        
    Returns:
        mean_logprob: Mean log probability of the sentence (scalar float)
    """
    # Tokenize the sentence
    tokens = model.to_tokens(sentence, prepend_bos=True)
    
    # Run model forward to get logits
    logits = model(tokens, return_type="logits")
    
    # Compute log probabilities using log_softmax
    logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
    
    # Extract logprobs for the actual tokens (shift by 1 for next token prediction)
    # Shape: (batch, seq_len, vocab_size) -> (batch, seq_len-1)
    token_logprobs = logprobs[0, :-1].gather(
        dim=-1, 
        index=tokens[0, 1:].unsqueeze(-1)
    ).squeeze(-1)
    
    # Compute mean logprob (excluding BOS token)
    mean_logprob = token_logprobs.mean().item()
    
    return mean_logprob


# Get forward and backward passes of SAE features
def get_activations(model, sentences, layers, embedding_method, use_sae, scale_by_decoder, decoder_norms):
    """Extract the featurized representations of all sentences.

    Args:
        model: HookedSAEModel
        sentences: A list of sentences to featurize
        layers: Layers to extract embeddings
        embedding_method: "mean" or "last_tok", how to aggregate embeddings for a sentence
        use_sae: Whether to use an SAE or standard residual stream embeddings
        scale_by_decoder: Whether to scale SAE features by decoder norm
        decoder_norms: Dict of decoder norms per layer (only needed if scale_by_decoder is True)

    Returns:
        activations: Features for each sentence
    """

    activations = []

    for sentence in tqdm(sentences):
        layer2act = {}
        # Run Model
        tokens = model.to_tokens(sentence, prepend_bos=True)
        _, cache = model.run_with_cache_with_saes(tokens)

        # Iterate through layers, collecting sae representations and aggregating them
        for layer in layers:
            if use_sae:
                vect = cache[get_sae_hookpoint(layer)]
            else:
                vect = cache[get_hookpoint(layer)]
            if embedding_method == "last_tok":
                vect = vect[0, -1]
            elif embedding_method == "mean":
                vect = torch.mean(
                    vect[0, 1:], dim=0
                )  # Omit BOS token because its OOD for SAE
            else:
                raise ValueError(f"unsupported embedding aggregation: {embedding_method}")
            vect = vect.cpu().to(torch.float16)
            if use_sae and scale_by_decoder:
                vect = vect * decoder_norms[layer]
            layer2act[layer] = vect
        activations.append(layer2act)

    return activations


if __name__ == "__main__":
    # Parse Args
    args = parse_arguments()
    three_t = args.three_t
    sae_str = os.path.join("sae", args.sae_release) if args.use_sae else "hidden_states"
    if args.scale_by_decoder:
        sae_str += "_scaled"

    if args.three_t:
        root_dir = os.path.join(
            "..",
            "features",
            args.model,
            "3T",
            sae_str,
            args.embedding_method,
        )
    else:
        root_dir = os.path.join(
            "..",
            "features",
            args.model,
            sae_str,
            args.embedding_method,
        ) 

    os.makedirs(root_dir, exist_ok=True)

    # Set up model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)
    model = HookedSAETransformer.from_pretrained_no_processing(
        args.model,
        device=device,
        dtype=torch.bfloat16,
    )

    # Set up data
    sentences = get_sentences(three_t=three_t)

    if args.layer is not None:
        layers = [args.layer]
    else:
        layers = list(range(model.cfg.n_layers))

    decoder_norms = {}

    if args.use_sae:
        # Add all SAEs to the transformer
        saes = {}
        for layer in layers:
            sae, cfg_dict, sparsity = SAE.from_pretrained(
                release=args.sae_release,
                sae_id=args.sae_id.format(layer=str(layer)),
                device="cpu",
            )
            sae.to(device=device, dtype=torch.bfloat16)  # Convert SAE to bfloat16
            sae.use_error_term = True

            saes[layer] = sae
            decoder_norms[layer] = torch.norm(sae.W_dec, dim=1).to(torch.float16).cpu()

            if args.scale_by_decoder:
                if len(decoder_norms[layer]) == torch.sum(decoder_norms[layer]):
                    print(f"Exiting: Decoder norms for layer {layer} are all ones, scaling will have no effect.")
                    sys.exit(1)

            model.add_sae(sae)


    if args.compute_logprobs:
        # Compute logprobs
        logprobs = []
        for sentence in sentences:
            logprob = compute_mean_logprob(model, sentence)
            logprobs.append(logprob)
        logprobs = np.array(logprobs)
        pkl.dump(logprobs, open(os.path.join(root_dir, "logprobs.pkl"), "wb"))

    # Compute activations to use as features for regressions
    print("Computing Activations")
    activations_per_layer = get_activations(
        model, sentences, layers, args.embedding_method, args.use_sae, args.scale_by_decoder, decoder_norms
    )

    for layer in layers:
        print(f"Processing Layer {layer}")
        feature_file = os.path.join(root_dir, str(layer) + ".pkl")

        # Reformat Activations to feed into classifier
        activations = []
        activations = [act[layer] for act in activations_per_layer]
        activations = np.stack(activations, axis=0)

        pkl.dump(activations, open(feature_file, "wb"))
