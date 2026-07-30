"""
This file performs a regressions to predict voxel reponses from SAE latents. 
Importantly, this file does NOT include any feature selection.

Instead, the user specifies a specific feature set to use for regression.
These features are direclty fed into a cross-validated ridge regression.

The user must also specify a direction for each feature in the set (+/-). 
After fitting the regression model for each fold, the number of features whose signs
match the direction specified are counted.
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
        default=10000,
        type=int,
        help="Number of neuroids to include per dataset",
    )

    parser.add_argument(
        "-d",
        "--datasets",
        nargs="+",
        default=[
            "ghost_additional",
            "abstract",
            "concrete",
        ],
    )

    parser.add_argument(
        "-f",
        "--features",
        nargs="+",
        type=int,
        default=[
            40, 94, 79, 44, 49, 64, 199, 16, 669, 884,
        ],
    )

    parser.add_argument(
        "-s",
        "--directions",
        nargs="+",
        type=int,
        default=[
            1,
            1,
            -1,
            1,
            1,
            1,
            1,
            -1,
            -1,
            1,
        ],
    )

    parser.add_argument("-e", "--embedding_method", default="mean")

    parser.add_argument(
        "--sae_release",
        default="gemma-2-2b-res-matryoshka-dc",
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
        "--standardize_betas",
        default=False,
        action="store_true",
        help="Whether to standardize betas before regression (no CV leakage).",
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
            "Num Features": [],
            "Average Num Features Agree Direction": [],
            "Alpha Mean": [],
            "Alpha Median": [],
            "Language T-Stat": [],
            "PC1 Correlation": [],
            "PC2 Correlation": [],
            "Parcel Name": [],
            "R Fischer SEM": [],
            "NC Normalized R Fischer SEM": [],
        }

        if args.standardize_betas:
            outfolder = os.path.join(
                root_dir, str(layer), "standardized_betas", "regressions", participant
            )
        else:
            outfolder = os.path.join(
                root_dir, str(layer), "raw", "regressions", participant
            )
        os.makedirs(outfolder, exist_ok=True)

        # Load Up Activations to feed into classifier
        activations = regression_utils.get_activations(
            args.model,
            layer,
            args.embedding_method,
            True,
            args.sae_release,
        )
        activations = np.array(activations, dtype=np.float32)
        activations = activations[:, args.features] # subset to only the features of interest


        for dataset in args.datasets:

            betas, sentences, neuroids, metadata = regression_utils.set_up_datasets(
                participant, dataset, args.n
            )

            # Keep the activations such that when we shuffle per voxel, we do not shuffle "on top" of already shuffled activations
            # (a bit cleaner, but similar)
            orig_activations = activations.copy()

            # Iterate through neuroids to do feature selection
            print(f"Processing Participant {participant} Dataset: {dataset}")
            for neuroid_idx, neuroid in enumerate(tqdm(neuroids)):
                activations_all = orig_activations

                # 5 Fold CV for Feature Selection, Alpha Estimation, and Testing
                kfold = KFold(n_splits=5, shuffle=True, random_state=19)

                predictions = []
                ground_truths = []
                alphas = []
                num_features_agree_directions = []

                for train_indices, test_indices in kfold.split(activations):
                    acts_train, acts_test = (
                        activations_all[train_indices],
                        activations_all[test_indices],
                    )

                    betas_train, betas_test = (
                        betas[neuroid][train_indices].to_numpy(),
                        betas[neuroid][test_indices].to_numpy(),
                    )

                    if args.standardize_betas:
                        scaler = StandardScaler()
                        betas_train = scaler.fit_transform(betas_train.reshape(-1, 1)).reshape(-1)
                        betas_test = scaler.transform(betas_test.reshape(-1, 1)).reshape(-1)

                    # First, use the train set to search for the best Alpha hyperparameter
                    alpha = regression_utils.select_alpha(acts_train, betas_train)
                    alphas.append(alpha)

                    # Use best alpha to predict held-out voxels
                    model = Ridge(alpha=alpha, fit_intercept=True)
                    model.fit(acts_train, betas_train)

                    # Save whether the coefficients agree with the direction specified
                    ridge_coefs = model.coef_
                    num_features_agree_direction = np.sum(ridge_coefs * np.array(args.directions) > 0)
                    num_features_agree_directions.append(num_features_agree_direction)

                    pred_betas = model.predict(acts_test)
                    predictions.append(pred_betas.reshape(-1))
                    ground_truths.append(betas_test.reshape(-1))

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
                results["Num Features"].append(len(args.features))
                results["Average Num Features Agree Direction"].append(np.mean(num_features_agree_directions))
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


    featurizer_str = f"sae/{args.sae_release}"
    langfroi_analysis = any(["langfroi" in ds for ds in args.datasets])
    dataset_type_str = "fROI" if langfroi_analysis else "categories"

    root_dir = os.path.join(
        "..",
        "results_anon",
        dataset_type_str,
        "spotlight",
        featurizer_str,
        args.embedding_method,
    )
    os.makedirs(root_dir, exist_ok=True)

    for participant in args.participants:
        print(f"Processing Participant {participant}")
        process_participant(args, participant, root_dir)
