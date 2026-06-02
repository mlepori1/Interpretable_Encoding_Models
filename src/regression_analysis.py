"""
This file performs a regressions to predict voxel reponses from either residual streams
or SAE latents. The analysis proceeds by first running a Lasso Regression for feature selection,
then refitting these features using Ridge regression.
"""

import argparse
import os

import numpy as np
import pandas as pd

from tqdm import tqdm

from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
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
        "--additional_voxels",
        default=0,
        type=int,
        help="Number of additional voxels to include",
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
            "ghost",
            "additional",
        ],
    )

    parser.add_argument("-e", "--embedding_method", default="mean")

    parser.add_argument(
        "--use_topic_model",
        default=False,
        action="store_true",
        help="Whether to use topic model features instead of model activations. Will error if --use_sae is set.",
    )

    parser.add_argument(
        "--logprobs_only",
        default=False,
        action="store_true",
        help="Whether to use logprobs as the only feature.",
    )

    parser.add_argument(
        "--use_sae",
        default=False,
        action="store_true",
        help="Whether to use SAE features instead of model activations. Will error if --use_topic_model is set.",
    )

    parser.add_argument(
        "--sae_release",
        default="gemma-scope-2b-pt-res-canonical",
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
        help="Whether to standardize features before regression (no CV leakage). Not recommended for SAE features.",
    )

    parser.add_argument(
        "--standardize_betas",
        default=False,
        action="store_true",
        help="Whether to standardize betas before regression (no CV leakage).",
    )

    parser.add_argument(
        "--min_feature_index",
        default=0,
        type=int,
        help="Minimum index of features to use for regression",
    )

    parser.add_argument(
        "--max_feature_index",
        default=None,
        type=int,
        help="Max index of features to use for regression",
    )

    parser.add_argument(
        "--use_logprobs",
        default=False,
        action="store_true",
        help="Whether to include logprobs as features in addition to hidden states.",
    )

    parser.add_argument(
        "--participant_shuffle_control",
        default=False,
        action="store_true",
        help="Whether to shuffle activations + logprobs per participant before regression. Only use this as a control.",
    )

    parser.add_argument(
        "--voxel_shuffle_control",
        default=False,
        action="store_true",
        help="Whether to shuffle activations + logprobs per voxel before regression. Only use this as a control.",
    )

    parser.add_argument(
        "--output_suffix",
        default="",
        type=str,
        help="Suffix for results filename, e.g. 'langfroi12345' saves to results_langfroi12345.csv",
    )

    args = parser.parse_args()
    return args


def process_participant(args, participant, root_dir):
    """This function iterates through datasets, computing
    voxel predictivity for each voxel/dataset.
    """

    for layer in args.layers:
        print(f"Processing Layer {layer}")

        ### Set Up Output ###
        results = {
            "neuroid": [],
            "dataset": [],
            "R": [],
            "NC Normalized R": [],
            "R Fischer": [],
            "NC Normalized R Fischer": [],
            "Noise Ceiling": [],
            "Feature Intersection": [],
            "Feature Union": [],
            "Feature Mean": [],
            "Feature Std": [],
            "Alpha Mean": [],
            "Alpha Median": [],
            "Language T-Stat": [],
            "PC1 Correlation": [],
            "PC2 Correlation": [],
            "Parcel Name": [],
            "R Fischer SEM": [],
            "NC Normalized R Fischer SEM": [],
        }
        if args.standardize_features and args.standardize_betas:
            outfolder = os.path.join(
                root_dir, str(layer), "standardized_features_and_betas", "regressions", participant
            )
        elif args.standardize_features:
            outfolder = os.path.join(
                root_dir, str(layer), "standardized", "regressions", participant
            )
        elif args.standardize_betas:
            outfolder = os.path.join(
                root_dir, str(layer), "standardized_betas", "regressions", participant
            )
        else:
            outfolder = os.path.join(
                root_dir, str(layer), "raw", "regressions", participant
            )
        os.makedirs(outfolder, exist_ok=True)

        if args.participant_shuffle_control and args.voxel_shuffle_control:
            raise ValueError("Choose only one of participant_shuffle_control or voxel_shuffle_control.")

        # Load in logprobs
        logprobs = regression_utils.get_logprobs(args.model)
        logprobs = np.array(logprobs, dtype=np.float32).reshape(-1, 1)

        # Load Up Activations to feed into classifier
        if not args.logprobs_only:
            activations = regression_utils.get_activations(
                args.model,
                layer,
                args.embedding_method,
                args.use_sae,
                args.sae_release,
                args.scale_by_decoder,
                args.use_topic_model,
            )
            activations = np.array(activations, dtype=np.float32)
            activations = activations[:, args.min_feature_index:args.max_feature_index]
        else:
            activations = np.empty(
                (len(logprobs), 1)
            )  # Placeholder if not using any activations

        if args.participant_shuffle_control:
            print("Shuffling activations and logprobs for control analysis")
            perm = np.random.permutation(len(logprobs))
            activations = activations[perm]
            logprobs = logprobs[perm]

        for dataset in args.datasets:

            if dataset == "additional":
                n_voxels = args.additional_voxels
            else:
                n_voxels = args.n

            betas, sentences, neuroids, metadata = regression_utils.set_up_datasets(
                participant, dataset, n_voxels
            )

            # Keep the activations such that when we shuffle per voxel, we do not shuffle "on top" of already shuffled activations
            # (a bit cleaner, but similar)
            orig_activations = activations.copy()
            orig_logprobs = logprobs.copy()

            # Iterate through neuroids to do feature selection
            print(f"Processing Participant {participant} Dataset: {dataset}")
            for neuroid_idx, neuroid in enumerate(tqdm(neuroids)):
                activations_all = orig_activations
                logprobs_all = orig_logprobs

                # 5 Fold CV for Feature Selection, Alpha Estimation, and Testing
                kfold = KFold(n_splits=5, shuffle=True, random_state=19)

                predictions = []
                ground_truths = []
                support_feature_sets = []
                alphas = []

                if args.voxel_shuffle_control:
                    print("Shuffling activations and logprobs (per voxel) for control analysis")
                    perm = np.random.permutation(len(orig_logprobs)) # a shuffle per voxel
                    activations_all = orig_activations[perm]
                    logprobs_all = orig_logprobs[perm]

                for train_indices, test_indices in kfold.split(activations):
                    acts_train, acts_test = (
                        activations_all[train_indices],
                        activations_all[test_indices],
                    )
                    logprobs_train, logprobs_test = (
                        logprobs_all[train_indices],
                        logprobs_all[test_indices],
                    )
                    betas_train, betas_test = (
                        betas[neuroid][train_indices].to_numpy(),
                        betas[neuroid][test_indices].to_numpy(),
                    )

                    # Always standardize logprobs
                    scaler = StandardScaler()
                    logprobs_train = scaler.fit_transform(logprobs_train) # always fit on train to avoid CV leakage
                    logprobs_test = scaler.transform(logprobs_test)

                    if args.standardize_features:
                        scaler = StandardScaler()
                        acts_train = scaler.fit_transform(acts_train)
                        acts_test = scaler.transform(acts_test)

                    if args.standardize_betas:
                        scaler = StandardScaler()
                        betas_train = scaler.fit_transform(betas_train.reshape(-1, 1)).reshape(-1)
                        betas_test = scaler.transform(betas_test.reshape(-1, 1)).reshape(-1)

                    # Optionally include logprobs as features
                    if args.use_logprobs:
                        if args.logprobs_only:
                            # If using logprobs only, set acts_train and acts_test to logprobs
                            acts_train = logprobs_train
                            acts_test = logprobs_test
                        else:
                            # Otherwise, concatenate logprobs to activations
                            acts_train = np.concatenate(
                                [acts_train, logprobs_train], axis=1
                            )
                            acts_test = np.concatenate(
                                [acts_test, logprobs_test], axis=1
                            )
                        logprob_index = (
                            acts_train.shape[1] - 1
                        )  # Index of logprob feature

                    # Compute support features for this split
                    if args.logprobs_only:
                        # If using logprobs_only, skip Lasso feature selection
                        support_features = np.ones(acts_train.shape[1], dtype=bool)
                    else:
                        support_features = regression_utils.compute_support_features(
                            acts_train, betas_train, omit_features=[], model="lasso"
                        )

                        if support_features is None:
                            # If not using logprobs, skip this fold, else continue with only logprobs
                            if not args.use_logprobs:
                                predictions.append(
                                    np.full(len(betas_test.reshape(-1)), np.nan)
                                )
                                ground_truths.append(betas_test.reshape(-1))
                                support_features = np.zeros(
                                    acts_train.shape[1], dtype=bool
                                )
                                support_feature_sets.append(support_features)
                                continue
                            else:
                                # Log prob feature will be forced below
                                support_features = np.zeros(
                                    acts_train.shape[1], dtype=bool
                                )

                        if args.use_logprobs:
                            # Ensure logprob feature is always included for ridge regression
                            support_features[logprob_index] = True

                    support_feature_sets.append(support_features)

                    acts_train = acts_train[:, support_features]
                    acts_test = acts_test[:, support_features]

                    # First, use the train set to search for the best Alpha hyperparameter
                    alpha = regression_utils.select_alpha(acts_train, betas_train)
                    alphas.append(alpha)

                    # Use best alpha to predict held-out voxels
                    model = Ridge(alpha=alpha, fit_intercept=True)
                    model.fit(acts_train, betas_train)
                    pred_betas = model.predict(acts_test)
                    predictions.append(pred_betas.reshape(-1))
                    ground_truths.append(betas_test.reshape(-1))

                # Compute relevant quantities for the support features
                support_feature_sets = np.asarray(support_feature_sets, dtype=bool)
                intersection = np.sum(np.all(support_feature_sets, axis=0))
                union = np.sum(np.any(support_feature_sets, axis=0))
                mean = np.mean(np.sum(support_feature_sets, axis=1))
                std = np.std(np.sum(support_feature_sets, axis=1))

                # Retrieve relevant metadata
                neuroid_metadata = metadata[metadata["neuroid_id"] == int(neuroid)]

                noise_ceiling = neuroid_metadata["nc"].iloc[0]
                noise_ceiling = np.sqrt(noise_ceiling / 100)

                language_t_stat = neuroid_metadata["tstat_langloc_SN"].iloc[0]
                pc1_corr = neuroid_metadata["corr_SentPC1"].iloc[0]
                pc2_corr = neuroid_metadata["corr_SentPC2"].iloc[0]
                parcel = neuroid_metadata["parc_name_glasser"].iloc[0]

                # Compute Pearson R
                rs = [] # R's per fold (everything here is per voxel)
                for prediction, ground_truth in zip(predictions, ground_truths):
                    try:
                        r, _ = pearsonr(ground_truth, prediction)
                        if np.isnan(r):  # Handle constant inputs gracefully
                            r = 0
                    except ValueError:  # If empty Support Set (should not happen because we always have log prob)
                        r = np.nan # We can then search for NaN predictions downstream
                        print(f"Warning: NaN r value for neuroid {neuroid} dataset {dataset} fold with predictions {prediction} and ground truth {ground_truth}")
                    rs.append(r)

                normalized_r = np.mean(rs) / noise_ceiling

                # Do another version where we first Fischer Z transform the r values, average, then inverse transform
                z_rs = np.arctanh(rs)
                mean_z_r = np.mean(z_rs)
                # invert back
                mean_r_after_fischer = np.tanh(mean_z_r)
                # also compute noise ceiling normalized version of this
                normalized_mean_r_from_fischer = mean_r_after_fischer / noise_ceiling

                results["neuroid"].append(neuroid)
                results["dataset"].append(dataset)
                results["R"].append(np.mean(rs))
                results["NC Normalized R"].append(normalized_r)
                results["R Fischer"].append(mean_r_after_fischer)
                results["NC Normalized R Fischer"].append(normalized_mean_r_from_fischer)
                results["Noise Ceiling"].append(noise_ceiling)
                results["Feature Intersection"].append(intersection)
                results["Feature Union"].append(union)
                results["Feature Mean"].append(mean)
                results["Feature Std"].append(std)
                results["Alpha Mean"].append(np.mean(alphas) if alphas else np.nan)
                results["Alpha Median"].append(np.median(alphas) if alphas else np.nan)
                results["Language T-Stat"].append(language_t_stat)
                results["PC1 Correlation"].append(pc1_corr)
                results["PC2 Correlation"].append(pc2_corr)
                results["Parcel Name"].append(parcel)
                # SEM of Fischer-transformed R across folds
                sem_z = np.std(z_rs, ddof=1) / np.sqrt(len(z_rs))
                r_fischer_sem = np.tanh(mean_z_r + sem_z) - mean_r_after_fischer
                results["R Fischer SEM"].append(r_fischer_sem)
                results["NC Normalized R Fischer SEM"].append(r_fischer_sem / noise_ceiling)

        results = pd.DataFrame.from_dict(results)
        # Add participant column for ease of analysis later, even though it's redundant with the folder structure
        results["Participant"] = participant

        if args.output_suffix:
            results_fname = f"results_{args.output_suffix}.csv"
        else:
            results_fname = "results.csv"
        results.to_csv(os.path.join(outfolder, results_fname))


if __name__ == "__main__":
    # Set Random Seed (for shuffle control)
    np.random.seed(19)

    # Parse Args
    args = parse_arguments()

    if sum([args.use_sae, args.use_topic_model, args.logprobs_only]) > 1:
        raise ValueError(
            "Only one of --use_sae, --use_topic_model, or --logprobs_only can be set."
        )

    if args.logprobs_only:
        args.use_logprobs = True

    if args.use_sae:
        featurizer_str = f"sae/{args.sae_release}"
        if args.min_feature_index != 0 or args.max_feature_index is not None:
            featurizer_str = f"{featurizer_str}_{args.min_feature_index}_{args.max_feature_index}"
    elif args.use_topic_model:
        featurizer_str = "topic_model"
    elif args.logprobs_only:
        featurizer_str = "logprobs_only"
    else:
        featurizer_str = "hidden_states"

    control_str = "control" if args.participant_shuffle_control else "voxel_control" if args.voxel_shuffle_control else ""
    incl_logprob_str = "full_features" if args.use_logprobs else "content_only"

    langfroi_analysis = any(["langfroi" in ds for ds in args.datasets])
    dataset_type_str = "fROI" if langfroi_analysis else "categories"

    root_dir = os.path.join(
        "..",
        "results",
        dataset_type_str,
        incl_logprob_str,
        "regressions",
        featurizer_str,
        control_str,
        args.embedding_method,
    )
    os.makedirs(root_dir, exist_ok=True)

    for participant in args.participants:
        print(f"Processing Participant {participant}")
        process_participant(args, participant, root_dir)
