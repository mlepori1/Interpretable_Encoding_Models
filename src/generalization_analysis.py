"""
This file performs a regressions to understand whether different voxels respond to the 
same feature set. This analysis proceeds by first running a Lasso regression for feature selection
for a base voxel, then refitting these features using Ridge regression to get the coefficient directions.

Then, for each generalization voxel, we use the feature set selected from the base voxel to refit a 
Ridge regression (where coefficients are constraint to be the same direction as the), and evaluate performance.
"""

import argparse
import os
import pickle as pkl
from collections import defaultdict
import copy

from matplotlib.pylab import normal
import numpy as np
import pandas as pd
from tqdm import tqdm
import json

from sklearn.model_selection import cross_val_score, LeaveOneOut, KFold
from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV, ElasticNetCV
from sklearn.metrics import make_scorer
from scipy.stats import pearsonr

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
        help="Number of neuroids to include per dataset",
    )

    parser.add_argument(
        "-d",
        "--datasets",
        nargs="+",
        default=[
            "abstract",
            "concrete",
            "easy_to_process",
            "hard_to_process",
        ],
    )

    parser.add_argument("-e", "--embedding_method", default="mean")

    parser.add_argument(
        "--use_sae",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--sae_release",
        default="gemma-scope-2b-pt-res-canonical",
        type=str,
        help="SAE Release",
    )

    parser.add_argument(
        "-l",
        "--layers",
        metavar="N",
        type=int,
        nargs="+",
        default=[12],
        help="Layer to analyze",
    )

    parser.add_argument(
        "-p",
        "--participants",
        type=str,
        nargs="+",
        default=[
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
        "--scale_by_decoder",
        default=False,
        action="store_true",
        help="Whether to scale SAE features by decoder norm. Makes features more comparable.",
    )

    parser.add_argument(
        "--use_logprobs",
        default=False,
        action="store_true",
        help="Whether to include logprobs as features in addition to hidden states.",
    )

    parser.add_argument(
        "--alpha_per_gen_fold",
        default=False,
        action="store_true",
        help="Whether to compute alpha for ridge for each generalization fold.",
    )


    parser.add_argument(
        "--results_fname",
        default="results.csv",
        help="Filename for results csv",
    )

    parser.add_argument(
        "--skip_self",
        default="False",
        type=str,
        help="Whether to skip the base voxel when running regressions."
    )

    parser.add_argument(
        "--filter_csv_path",
        default="",
        type=str,
        help="Points to a csv of participants and neuroids to downselect, if you want"
    )

    parser.add_argument(
        "--k_best",
        default=8000,
        type=int,
        help="How many features to subselect using f_test"
    )

    parser.add_argument(
        "--base_participant",
        default=None,
        type=str,
        help="Restrict base voxels to come from one particular participant to parallelize"
    )

    parser.add_argument(
        "--gen_participant",
        default=None,
        type=str,
        help="Restrict gen voxels to come from one particular participant to parallelize"
    )

    args = parser.parse_args()
    return args


def generalization_analysis(args, data, root_dir):
    """This function performs the generalization analysis across voxels.

    First, for each base voxel, we perform feature selection using Lasso regression.
    Then, we fit a Ridge regression model using the selected features to get coefficient directions.
    Finally, for each generalization voxel, we use the selected features to fit a Ridge regression
    model (with coefficients constrained to the directions from the base voxel), and evaluate performance.
    """

    for layer in args.layers:
        print(f"Processing Layer {layer}")

        if args.standardize_features and args.standardize_betas:
            subdir = "standardized_features_and_betas"
        elif args.standardize_features:
            subdir = "standardized"
        elif args.standardize_betas:
            subdir = "standardized_betas"
        else:
            subdir = "raw"
        outfolder = os.path.join(root_dir, str(layer), subdir, "generalization")
        os.makedirs(outfolder, exist_ok=True)

        results = {
            "base_participant": [],
            "base_dataset": [],
            "base_neuroid": [],
            "base_lang_t_stat": [],
            "gen_participant": [],
            "gen_dataset": [],
            "gen_neuroid": [],
            "gen_lang_t_stat": [],
            "R": [],
            "NC Normalized R": [],
            "R Fischer": [],
            "NC Normalized R Fischer": [],
            "NC Normalized Voxel Correlation": [],
            "NC Normalized PC1 Correlation": [],
            "NC Normalized PC2 Correlation": [],
        }

        activations = regression_utils.get_activations(
                args.model, layer, args.embedding_method, args.use_sae, args.sae_release, args.scale_by_decoder
            )
        logprobs = regression_utils.get_logprobs(args.model)
        logprobs = np.array(logprobs, dtype=np.float32).reshape(-1, 1)

        # optionally only include base participant
        if args.base_participant is not None:
            base_data = data[data["participants"] == args.base_participant]
        else:
            base_data = data

        if args.gen_participant is not None:
            gen_data = data[data["participants"] == args.gen_participant]
        else:
            gen_data = data

        for base_idx, base_row in tqdm(base_data.iterrows(), total=len(base_data)):

            base_betas = base_row["betas"].copy()
            if args.standardize_betas:
                base_betas = StandardScaler().fit_transform(base_betas.reshape(-1, 1)).flatten()

            if args.standardize_features:
                scaler = StandardScaler()
                acts = scaler.fit_transform(activations)
            else:
                acts = activations

            # Optionally include logprobs as features
            if args.use_logprobs:
                scaler = StandardScaler()
                logprobs_scaled = scaler.fit_transform(logprobs)
                X_full = np.concatenate([acts, logprobs_scaled], axis=1)
                logprob_index = X_full.shape[1] - 1  # Index of logprob feature
            else:
                X_full = acts

            support_features = regression_utils.compute_support_features(
                X_full, base_betas, omit_features=[], model="lasso", k_best=args.k_best
            )

            if support_features is None:
                if not args.use_logprobs:
                    continue
                support_features = np.zeros(X_full.shape[1], dtype=bool)
                support_features[logprob_index] = True  # Ensure logprob feature is included
                selected_features = X_full[:, support_features]
            else:
                if args.use_logprobs: # Ensure logprob is always included
                    support_features[logprob_index] = True
                selected_features = X_full[:, support_features]
        
            alpha = regression_utils.select_alpha(selected_features, base_betas)

            model = Ridge(alpha=alpha)
            model.fit(selected_features, base_betas)
            sign = np.where(model.coef_ < 0, -1.0, 1.0)

            for gen_idx, gen_row in gen_data.iterrows():
                # Skip if same voxel
                if args.skip_self != "False":
                    if (base_row["participants"] == gen_row["participants"]) and \
                    (base_row["datasets"] == gen_row["datasets"]) and \
                    (base_row["neuroids"] == gen_row["neuroids"]):
                        continue

                gen_betas = gen_row["betas"]

                # 5 Fold CV for Alpha Estimation and Testing
                kfold = KFold(n_splits=5, shuffle=True, random_state=19)

                predictions = []
                ground_truths = []

                for train_indices, test_indices in kfold.split(activations):
                    feats_train, feats_test = activations[train_indices], activations[test_indices]
                    logprobs_train, logprobs_test = logprobs[train_indices], logprobs[test_indices]

                    # Always standardize logprobs
                    scaler = StandardScaler()
                    logprobs_train = scaler.fit_transform(logprobs_train)
                    logprobs_test = scaler.transform(logprobs_test)

                    betas_train, betas_test = gen_betas[train_indices], gen_betas[test_indices]
                    if args.standardize_betas:
                        scaler = StandardScaler()
                        betas_train = scaler.fit_transform(betas_train.reshape(-1, 1)).flatten()
                        betas_test = scaler.transform(betas_test.reshape(-1, 1)).flatten()

                    if args.standardize_features:
                        scaler = StandardScaler()
                        feats_train = scaler.fit_transform(feats_train)
                        feats_test = scaler.transform(feats_test)

                    if args.use_logprobs:
                        X_full_train = np.concatenate([feats_train, logprobs_train], axis=1)
                        X_full_test = np.concatenate([feats_test, logprobs_test], axis=1)
                    else:
                        X_full_train = feats_train
                        X_full_test = feats_test

                    # Ensures that enforcing a positive coefficient on
                    # generalization ridge regression means that coefficients
                    # go in the same direction as base voxel
                    selected_features_train = X_full_train[:, support_features] * sign
                    selected_features_test = X_full_test[:, support_features]  * sign

                    # # First, use the train set to search for the best Alpha hyperparameter (if specified)
                    if args.alpha_per_gen_fold:
                        alpha = regression_utils.select_alpha(selected_features_train, betas_train, positive=True)

                    # Use best alpha to predict held-out voxels
                    model = Ridge(alpha=alpha, positive=True)
                    model.fit(selected_features_train, betas_train)
                    pred_betas = model.predict(selected_features_test)
                    predictions.append(pred_betas.reshape(-1))
                    ground_truths.append(betas_test.reshape(-1))

                # Compute Pearson R per fold (concordant with regression_analysis.py)
                rs = [] # R's per fold (everything here is per voxel)
                for prediction, ground_truth in zip(predictions, ground_truths):
                    try:
                        r, _ = pearsonr(ground_truth, prediction)
                        if np.isnan(r):  # Handle constant inputs gracefully
                            r = 0
                    except ValueError:  # If empty Support Set (should not happen because we always have log prob)
                        r = np.nan # We can then search for NaN predictions downstream
                    rs.append(r)

                mean_r = np.mean(rs)
                normalized_r = mean_r / gen_row["noise_ceilings"]
                z_rs = np.arctanh(rs)
                mean_r_after_fischer = np.tanh(np.mean(z_rs))
                normalized_mean_r_from_fischer = mean_r_after_fischer / gen_row["noise_ceilings"]

                # BASELINE: Compute Voxel Correlation
                try:
                    voxel_r, _ = pearsonr(base_row["betas"], gen_row["betas"])
                except ValueError:
                    voxel_r = np.nan
                normalized_voxel_r = voxel_r / gen_row["noise_ceilings"]

                # BASELINE: Compute Normalized PC1 + PC2 Correlation
                normalized_pc1_r = gen_row["pc1_corrs"] / gen_row["noise_ceilings"]
                normalized_pc2_r = gen_row["pc2_corrs"] / gen_row["noise_ceilings"]

                results["base_participant"].append(base_row["participants"])
                results["base_dataset"].append(base_row["datasets"])
                results["base_neuroid"].append(base_row["neuroids"])
                results["base_lang_t_stat"].append(base_row["language_t_stats"])
                results["gen_participant"].append(gen_row["participants"])
                results["gen_dataset"].append(gen_row["datasets"])
                results["gen_neuroid"].append(gen_row["neuroids"])
                results["gen_lang_t_stat"].append(gen_row["language_t_stats"])
                results["R"].append(mean_r)
                results["NC Normalized R"].append(normalized_r)
                results["R Fischer"].append(mean_r_after_fischer)
                results["NC Normalized R Fischer"].append(normalized_mean_r_from_fischer)
                results["NC Normalized Voxel Correlation"].append(normalized_voxel_r)
                results["NC Normalized PC1 Correlation"].append(normalized_pc1_r)
                results["NC Normalized PC2 Correlation"].append(normalized_pc2_r)

        results = pd.DataFrame.from_dict(results)

        results_fname = args.results_fname
        if args.gen_participant:
            results_fname = f"gen_{args.gen_participant}_{results_fname}"
        if args.base_participant:
            results_fname = f"base_{args.base_participant}_{results_fname}"

        results.to_csv(os.path.join(outfolder, results_fname), index=False)


def collect_data(args):
    """Collect relevant data across participants and datasets for analysis."""
    data = {
        "betas": [],
        "noise_ceilings": [],
        "language_t_stats": [],
        "pc1_corrs": [],
        "pc2_corrs": [],
        "participants": [],
        "neuroids": [],
        "datasets": [],
    }

    if args.filter_csv_path:
        filter_set = pd.read_csv(args.filter_csv_path)
    else:
        filter_set = None

    for participant in args.participants:
        for dataset in args.datasets:
            betas, _, neuroids, metadata = regression_utils.set_up_datasets(participant, dataset, args.n)

            for neuroid in neuroids:

                # Filter out neuroids that are not in the filter set (if one is specified)
                if filter_set is not None:
                    if len(filter_set[(filter_set["neuroid"] == int(neuroid)) & (filter_set["Participant"] == participant)]) != 1:
                        continue

                neuroid_betas = betas[neuroid].to_numpy()
                neuroid_metadata = metadata[metadata["neuroid_id"] == int(neuroid)]

                noise_ceiling = neuroid_metadata["nc"].iloc[0]
                noise_ceiling = np.sqrt(noise_ceiling/100)
            
                language_t_stat = neuroid_metadata["tstat_langloc_SN"].iloc[0]
                pc1_corr = neuroid_metadata["corr_SentPC1"].iloc[0]
                pc2_corr = neuroid_metadata["corr_SentPC2"].iloc[0]
                data["betas"].append(neuroid_betas)
                data["noise_ceilings"].append(noise_ceiling)
                data["language_t_stats"].append(language_t_stat)
                data["pc1_corrs"].append(pc1_corr)
                data["pc2_corrs"].append(pc2_corr)
                data["participants"].append(participant)
                data["neuroids"].append(neuroid)
                data["datasets"].append(dataset)
    data = pd.DataFrame.from_dict(data)
    return data

if __name__ == "__main__":
    # Parse Args
    args = parse_arguments()

    args.skip_self = args.skip_self.strip()
    if args.skip_self not in ["True", "False"]:
        raise ValueError("skip_self must in one of [True, False]")
    
    sae_str = f"sae/{args.sae_release}" if args.use_sae else "hidden_states"

    if args.standardize_features:
        sae_str += "_standardized"

    features_str = "full_features" if args.use_logprobs else "content_only"

    langfroi_analysis = any(["langfroi" in ds for ds in args.datasets])
    dataset_type_str = "fROI" if langfroi_analysis else "categories"

    root_dir = os.path.join(
        "..", "results", dataset_type_str, features_str, f"generalization_{args.k_best}", sae_str, args.embedding_method
    )
    os.makedirs(root_dir, exist_ok=True)

    print(args.datasets)
    data = collect_data(args)
    generalization_analysis(args, data, root_dir)



