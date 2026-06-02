"""
This file performs regressions to predict voxel responses from either residual streams
or SAE latents. The analysis proceeds by first running a Lasso Regression for feature selection,
then refitting these features using Ridge regression.

Qualitative Analysis only makes sense for SAE features, so this script does not support residual stream regressions.

For per-participant experiments, run example:
python run_qualitative_experiment.py --sae_release gemma-2-2b-res-matryoshka-dc -n 10 --per_participant
--use_logprobs
"""

import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import regression_utils


def list_of_ints(arg):
    return [int(a) for a in arg.split(",")]


def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        default="gemma-2-2b",
        help="Model to use as backbone for analysis",
    )

    parser.add_argument(
        "-n",
        default=20,
        type=int,
        help="Top N neuroids to report features for",
    )

    parser.add_argument(
        "-d",
        "--datasets",
        nargs="+",
        default=[
            "abstract",
            "concrete",
            "hard_to_process",
            "easy_to_process",
            "ghost"
        ],
    )

    parser.add_argument("-e", "--embedding_method", default="mean")

    parser.add_argument(
        "--sae_release",
        default="gemma-2-2b-res-matryoshka-dc",
        # default="gemma-scope-2b-pt-res-canonical",
        type=str,
        help="SAE Release",
    )

    parser.add_argument(
        "--scale_by_decoder",
        default=False,
        action="store_true",
        help="Whether to scale SAE features by decoder norm. Makes features more comparable.",
    )

    parser.add_argument(
        "-l",
        "--layer",
        metavar="N",
        type=int,
        default=12,
        help="Layer to analyze",
    )

    parser.add_argument(
        "-p",
        "--participants",
        type=str,
        nargs="+",
        default= [
            "p1", "p2", "p3", "p4",
            "p5", "p6", "p7", "p8",
        ]
    )

    parser.add_argument(
        "--standardize_features",
        default=False,
        action="store_true",
        help="Whether to standardize features before regression. Not recommended for SAE features.",
    )

    parser.add_argument(
        "--standardize_betas",
        default=False,
        action="store_true",
        help="Whether to standardize betas before regression.",
    )

    parser.add_argument(
        "--use_logprobs",
        default=False,
        action="store_true",
        help="Whether to include logprobs as features in addition to hidden states.",
    )

    parser.add_argument(
        "--per_participant",
        default=False,
        action="store_true",
        help="Whether to select top N voxels per participant (instead of globally pooled).",
    )

    parser.add_argument(
        "--all_voxels",
        default=False,
        action="store_true",
        help="Process all voxels for specified datasets (ignores -n).",
    )

    parser.add_argument(
        "--save_meta",
        default=False,
        action="store_true",
        help="Merge meta columns from {dataset}_meta.csv into the output CSV.",
    )

    parser.add_argument(
        "--k_best",
        default=8000,
        type=int,
        help="Number of features to filter using F-Test"
    )

    parser.add_argument(
        "--results_suffix",
        default="",
        type=str,
        help="Suffix for results filename to read, e.g. 'langfroi12345' reads results_langfroi12345.csv",
    )

    args = parser.parse_args()
    return args



def create_root_dir(args):
    root_dir = "../results"

    langfroi_analysis = any(["langfroi" in ds for ds in args.datasets])
    dataset_type_str = "fROI" if langfroi_analysis else "categories"
    root_dir = os.path.join(root_dir, dataset_type_str)

    if args.use_logprobs:
        root_dir = os.path.join(root_dir, "full_features", "regressions")
    else:
        root_dir = os.path.join(root_dir, "content_only", "regressions")

    root_dir = os.path.join(root_dir, "sae", args.sae_release)
    root_dir = os.path.join(root_dir, args.embedding_method, str(args.layer))

    # Mirror expt 1 folder logic for standardized vs raw features/betas
    if args.standardize_features and args.standardize_betas:
        root_dir = os.path.join(root_dir, "standardized_features_and_betas", "regressions")
    elif args.standardize_features:
        root_dir = os.path.join(root_dir, "standardized", "regressions")
    elif args.standardize_betas:
        root_dir = os.path.join(root_dir, "standardized_betas", "regressions")
    else:
        root_dir = os.path.join(root_dir, "raw", "regressions")

    return root_dir


def filter_voxels(args, root_dir):
    """Filters voxels to the top N best predicted per dataset (and optionally per participant).
    If args.all_voxels is True, returns all voxels for the specified datasets.
    """
    dfs = []
    results_fname = f"results_{args.results_suffix}.csv" if args.results_suffix else "results.csv"
    for participant in args.participants:
        df = pd.read_csv(os.path.join(root_dir, participant, results_fname))
        df["participant"] = [participant] * len(df)
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)

    if args.all_voxels:
        # Return all voxels for the specified datasets
        data = data[data["dataset"].isin(args.datasets)]
        if args.per_participant:
            for participant in args.participants:
                for dataset in args.datasets:
                    sub = data[(data["participant"] == participant) & (data["dataset"] == dataset)]
                    if len(sub) > 0:
                        print(f"{participant} - {dataset}: all {len(sub)} voxels")
        else:
            for dataset in args.datasets:
                sub = data[data["dataset"] == dataset]
                if len(sub) > 0:
                    print(f"{dataset}: all {len(sub)} voxels")
        return data

    if args.per_participant:
        # Group by both participant and dataset
        grouped_data = data.groupby(["participant", "dataset"])

        best_predicted = []
        for (participant, dataset), group_data in grouped_data:
            if dataset in args.datasets:
                total_voxels = len(group_data)
                group_data = group_data.sort_values(by="NC Normalized R", ascending=False)
                top_n = group_data.iloc[:args.n]
                pct = (args.n / total_voxels) * 100
                print(f"{participant} - {dataset}: top {args.n} of {total_voxels} voxels ({pct:.1f}%)")
                print(f"  NC Normalized R range: {top_n['NC Normalized R'].min():.3f} - {top_n['NC Normalized R'].max():.3f}")
                best_predicted.append(top_n)
        best_predicted = pd.concat(best_predicted, ignore_index=True)
    else:
        # Original behavior: group by dataset only (pooled across participants)
        grouped_data = data.groupby("dataset")

        best_predicted = []
        for dataset, dataset_data in grouped_data:
            if dataset in args.datasets:
                dataset_data = dataset_data.sort_values(by="NC Normalized R", ascending=False)
                best_predicted_voxels = dataset_data.iloc[:args.n]
                print(f"{dataset}: top {args.n} of {len(dataset_data)} voxels")
                print(f"  NC Normalized R range: {best_predicted_voxels['NC Normalized R'].min():.3f} - {best_predicted_voxels['NC Normalized R'].max():.3f}")
                best_predicted.append(best_predicted_voxels)
        best_predicted = pd.concat(best_predicted, ignore_index=True)
    return best_predicted


def select_features(args, voxel_row, store_selective=False):
    """This function performs feature selection for a voxel 
    and reports the selected feature
    """

    # Load Up Activations to feed into classifier
    activations = regression_utils.get_activations(
        args.model, args.layer, args.embedding_method, True, args.sae_release, args.scale_by_decoder
    )
    activations = np.array(activations, dtype=np.float32)

    logprobs = regression_utils.get_logprobs(args.model)
    logprobs = np.array(logprobs, dtype=np.float32).reshape(-1, 1)

    # Always standardize logprobs
    scaler = StandardScaler()
    logprobs = scaler.fit_transform(logprobs)
    
    dataset = voxel_row["dataset"]
    participant = voxel_row["participant"]
    neuroid = voxel_row["neuroid"]
    nc_normalized_correlation = voxel_row["NC Normalized R"]
    correlation = voxel_row["R"]
    fischer_correlation = voxel_row["R Fischer"]
    nc_normalized_fischer_correlation = voxel_row["NC Normalized R Fischer"]

    betas, sentences, neuroids, metadata = regression_utils.set_up_datasets(participant, dataset, n=args.n)
    betas = betas[str(neuroid)].to_numpy()

    # Optionally include logprobs as features
    if args.use_logprobs:
        activations = np.concatenate([activations, logprobs], axis=1)
        logprob_index = activations.shape[1] - 1  # Index of logprob feature
    
    if args.standardize_features:
        scaler = StandardScaler()
        activations = scaler.fit_transform(activations)

    if args.standardize_betas:
        scaler = StandardScaler()
        betas = scaler.fit_transform(betas.reshape(-1, 1)).flatten() # When we are in this function, we just have one voxel's betas, so shape of betas is (200,)

    # Compute support features
    support_features = regression_utils.compute_support_features(activations, betas, omit_features=[], model="lasso", k_best=args.k_best)

    # Handle case where Lasso selects no features
    if support_features is None:
        support_features = np.zeros(activations.shape[1], dtype=bool)

    if args.use_logprobs:
        # Ensure logprob feature is always included for ridge regression
        support_features[logprob_index] = True

    feature_indices = np.where(support_features)[0]

    support_acts = activations[:, support_features]

    # First, search for the best Alpha hyperparameter
    alpha = regression_utils.select_alpha(support_acts, betas)

    # Fit Model to get coefficients
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(support_acts, betas)

    # Indicate whether each feature is used positively or negatively by
    # Ridge Regression
    sign = np.where(model.coef_ >= 0, 1, -1).astype(int) 
    feature_indices = sign * feature_indices
    if store_selective:
        selectivity = voxel_row["language selectivity"]
        return {
            "neuroid": neuroid,
            "participant": participant,
            "dataset": dataset,
            "selective": selectivity,
            "feature_indices": feature_indices,
            "NC Normalized R": nc_normalized_correlation,
            "R": correlation,
            "R Fischer": fischer_correlation,
            "NC Normalized R Fischer": nc_normalized_fischer_correlation
        }
    else:
        return {
            "neuroid": neuroid,
            "participant": participant,
            "dataset": dataset,
            "feature_indices": feature_indices,
            "NC Normalized R": nc_normalized_correlation,
            "R": correlation,
            "R Fischer": fischer_correlation,
            "NC Normalized R Fischer": nc_normalized_fischer_correlation
        }

if __name__ == "__main__":
    # Parse Args
    args = parse_arguments()
    root_dir = create_root_dir(args)

    # Determine output file suffix
    suffix = "_per_participant" if args.per_participant else ""
    if len(args.participants) == 1:
        suffix += f"_{args.participants[0]}"

    # Determine output file label
    if args.all_voxels:
        label = "_".join(args.datasets)
    else:
        label = f"n{args.n}"

    best_voxels = filter_voxels(args, root_dir)

    results = {
        "neuroid": [],
        "participant": [],
        "dataset": [],
        "NC Normalized R": [],
        "R": [],
        "R Fischer": [],
        "NC Normalized R Fischer": [],
        "feature_indices": [],
    }
    for row_idx, row in tqdm(best_voxels.iterrows(), total=len(best_voxels), desc="Feature selection"):
        neuroid_results = select_features(args, row)
        results["neuroid"].append(neuroid_results["neuroid"])
        results["participant"].append(neuroid_results["participant"])
        results["dataset"].append(neuroid_results["dataset"])
        results["NC Normalized R"].append(neuroid_results["NC Normalized R"])
        results["R"].append(neuroid_results["R"])
        results["R Fischer"].append(neuroid_results["R Fischer"])
        results["NC Normalized R Fischer"].append(neuroid_results["NC Normalized R Fischer"])
        results["feature_indices"].append(neuroid_results["feature_indices"])

    results_df = pd.DataFrame.from_dict(results)

    # Optionally merge meta columns
    if args.save_meta:
        meta_dfs = []
        for (participant, dataset), group in results_df.groupby(["participant", "dataset"]):
            meta_path = os.path.join("..", "data", "processed_csvs_anon", participant, f"{dataset}_meta.csv")
            if os.path.exists(meta_path):
                meta = pd.read_csv(meta_path)
                merged = group.merge(meta, left_on="neuroid", right_on="neuroid_id", how="left")
                meta_dfs.append(merged)
            else:
                print(f"Warning: meta file not found: {meta_path}")
                meta_dfs.append(group)
        results_df = pd.concat(meta_dfs, ignore_index=True)

    source_suffix = f"_{args.results_suffix}" if args.results_suffix else ""

    output_dir = os.path.join(root_dir, "qualitative")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"Qualitative_Analysis_{args.k_best}_{label}{suffix}{source_suffix}.csv")
    results_df.to_csv(output_file, index=False)
    print(f"\nSaved results to: {output_file}")



