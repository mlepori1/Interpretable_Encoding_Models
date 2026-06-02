"""
Analyze feature sharing across participants for qualitative analysis results.

For each dataset category, this script reports how many participants share
the same features (with the same sign/direction).

For per-participant visualizatiions, run e.g.
python analyze_feature_sharing.py --csv ../results/full_features/Exp1/sae/gemma-2-2b-res-matryoshka-dc/mean/12/raw/Experiment_1/Qualitative_Analysis_n5_per_participant.csv
--min_participants 2 --exclude_logprob --plot --save_plots  ../results/feature_sharing_plots
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


def parse_feature_indices(s, exclude_logprob=False, logprob_idx=32768):
    """Parse string representation of feature indices array.

    Handles numpy array string format like "[ 123 -456  789]"
    Returns list of signed integers.
    """
    s = s.strip("[]")
    features = [int(x) for x in s.split() if x]
    if exclude_logprob:
        features = [f for f in features if abs(f) != logprob_idx]
    return features


def analyze_feature_sharing(csv_path, min_participants=2, exclude_logprob=False):
    """Analyze which features are shared across participants for each dataset.

    Args:
        csv_path: Path to Qualitative_Analysis CSV file
        min_participants: Minimum number of participants to consider a feature "shared"
        exclude_logprob: Whether to exclude the logprob feature (index 32768)

    Returns:
        Dictionary with results per dataset
    """
    df = pd.read_csv(csv_path)

    results = {}
    for dataset in sorted(df["dataset"].unique()):
        dataset_df = df[df["dataset"] == dataset]
        participants = dataset_df["participant"].unique()
        n_participants = len(participants)

        # Count signed features per participant
        # Key: signed feature index, Value: set of participants who have it
        feature_counts = defaultdict(set)

        for _, row in dataset_df.iterrows():
            features = parse_feature_indices(row["feature_indices"], exclude_logprob=exclude_logprob)
            for feat in features:
                feature_counts[feat].add(row["participant"])

        # Find shared features (appear in >= min_participants)
        shared = {f: p for f, p in feature_counts.items()
                  if len(p) >= min_participants}

        # Sort by number of participants sharing (descending)
        shared_sorted = sorted(shared.items(),
                               key=lambda x: len(x[1]), reverse=True)

        results[dataset] = {
            "n_participants": n_participants,
            "shared_features": shared_sorted,
            "total_unique_features": len(feature_counts),
            "feature_counts": feature_counts  # Keep for plotting
        }

    return results


def print_results(results, top_n=20, min_participants=2):
    """Print feature sharing results in a readable format."""
    for dataset, data in results.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset}")
        print(f"{'='*60}")
        print(f"Total participants: {data['n_participants']}")
        print(f"Total unique signed features: {data['total_unique_features']}")

        shared = data["shared_features"]
        if shared:
            # Count features by number of participants sharing
            sharing_counts = defaultdict(int)
            for feat, participants in shared:
                sharing_counts[len(participants)] += 1

            print(f"\nSummary of shared features:")
            for n_shared in sorted(sharing_counts.keys(), reverse=True):
                print(f"  {sharing_counts[n_shared]} features shared by {n_shared}/{data['n_participants']} participants")

            print(f"\nTop {min(top_n, len(shared))} shared features:")
            for feat, participants in shared[:top_n]:
                sign = "+" if feat > 0 else "-"
                print(f"  Feature {abs(feat):>6} ({sign}): "
                      f"{len(participants)}/{data['n_participants']} participants "
                      f"({', '.join(sorted(participants))})")
        else:
            print(f"\nNo features shared by {min_participants}+ participants")


def plot_sharing_summary(results, save_path=None, min_shared=2):
    """Plot bar chart showing number of features shared by N participants per dataset.

    Args:
        results: Dictionary with results per dataset
        save_path: Path to save the plot (optional)
        min_shared: Minimum number of participants to include in plot (default: 2)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    datasets = list(results.keys())
    n_participants = results[datasets[0]]["n_participants"]
    x = np.arange(min_shared, n_participants + 1)  # min_shared to max participants
    width = 0.8 / len(datasets)

    colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))

    for i, dataset in enumerate(datasets):
        sharing_counts = defaultdict(int)
        for feat, participants in results[dataset]["shared_features"]:
            sharing_counts[len(participants)] += 1

        counts = [sharing_counts.get(n, 0) for n in x]
        offset = (i - len(datasets)/2 + 0.5) * width
        ax.bar(x + offset, counts, width, label=dataset, color=colors[i])

    ax.set_xlabel("Number of participants sharing feature")
    ax.set_ylabel("Number of features")
    ax.set_title(f"Feature Sharing Across Participants by Dataset (>= {min_shared}/{n_participants})")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}/{n_participants}" for n in x])
    ax.legend(title="Dataset")
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved summary plot to: {save_path}")
    plt.show()


def plot_top_features_heatmap(results, top_n=15, save_path=None):
    """Plot heatmap of top shared features across datasets."""
    # Collect top features from each dataset
    all_top_features = set()
    for dataset, data in results.items():
        for feat, participants in data["shared_features"][:top_n]:
            all_top_features.add(feat)

    if not all_top_features:
        print("No shared features to plot")
        return

    # Sort features by total sharing across all datasets
    feature_totals = {}
    for feat in all_top_features:
        total = 0
        for dataset, data in results.items():
            for f, p in data["shared_features"]:
                if f == feat:
                    total += len(p)
                    break
        feature_totals[feat] = total

    sorted_features = sorted(all_top_features, key=lambda f: feature_totals[f], reverse=True)[:top_n]

    # Build heatmap matrix
    datasets = list(results.keys())
    n_participants = results[datasets[0]]["n_participants"]
    matrix = np.zeros((len(sorted_features), len(datasets)))

    for j, dataset in enumerate(datasets):
        feature_dict = {f: len(p) for f, p in results[dataset]["shared_features"]}
        for i, feat in enumerate(sorted_features):
            matrix[i, j] = feature_dict.get(feat, 0)

    # Create labels
    feature_labels = []
    for feat in sorted_features:
        sign = "+" if feat > 0 else "-"
        feature_labels.append(f"{abs(feat)} ({sign})")

    fig, ax = plt.subplots(figsize=(8, max(6, len(sorted_features) * 0.4)))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=n_participants)

    ax.set_xticks(np.arange(len(datasets)))
    ax.set_yticks(np.arange(len(sorted_features)))
    ax.set_xticklabels(datasets)
    ax.set_yticklabels(feature_labels)

    # Add text annotations
    for i in range(len(sorted_features)):
        for j in range(len(datasets)):
            val = int(matrix[i, j])
            if val > 0:
                color = 'white' if val > n_participants/2 else 'black'
                ax.text(j, i, f"{val}", ha='center', va='center', color=color, fontsize=9)

    ax.set_xlabel("Dataset")
    ax.set_ylabel("Feature (signed)")
    ax.set_title(f"Top {len(sorted_features)} Shared Features Across Datasets\n(values = # participants)")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f"Participants (max={n_participants})")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved heatmap to: {save_path}")
    plt.show()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Analyze feature sharing across participants"
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to Qualitative_Analysis CSV file"
    )
    parser.add_argument(
        "--min_participants",
        type=int,
        default=2,
        help="Minimum participants to consider a feature 'shared' (default: 2)"
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=20,
        help="Number of top shared features to display per dataset (default: 20)"
    )
    parser.add_argument(
        "--exclude_logprob",
        action="store_true",
        default=False,
        help="Exclude the logprob feature (index 32768) from analysis"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help="Generate plots"
    )
    parser.add_argument(
        "--save_plots",
        type=str,
        default=None,
        help="Directory to save plots (if not specified, plots are shown but not saved)"
    )
    return parser.parse_args()


def extract_n_from_filename(csv_path):
    """Extract the n value from CSV filename (e.g., 'Qualitative_Analysis_n10_per_participant.csv' -> 'n10')."""
    import re
    basename = os.path.basename(csv_path)
    match = re.search(r'_(n\d+)', basename)
    return match.group(1) if match else None


if __name__ == "__main__":
    import os
    args = parse_arguments()

    print(f"Loading: {args.csv}")
    print(f"Minimum participants for shared feature: {args.min_participants}")
    if args.exclude_logprob:
        print("Excluding logprob feature (32768)")

    # Extract n value from CSV filename for output naming
    n_suffix = extract_n_from_filename(args.csv)
    if n_suffix:
        print(f"Detected {n_suffix} from input filename")

    results = analyze_feature_sharing(args.csv, args.min_participants, args.exclude_logprob)
    print_results(results, args.top_n, args.min_participants)

    if args.plot:
        if args.save_plots:
            os.makedirs(args.save_plots, exist_ok=True)

        # Generate 3 versions of the summary bar plot with different min_shared thresholds
        for min_shared in [2, 3, 4]:
            if args.save_plots:
                suffix = f"_{n_suffix}" if n_suffix else ""
                summary_path = os.path.join(args.save_plots, f"feature_sharing_summary_min{min_shared}{suffix}.png")
            else:
                summary_path = None
            plot_sharing_summary(results, save_path=summary_path, min_shared=min_shared)

        # Generate heatmap
        if args.save_plots:
            suffix = f"_{n_suffix}" if n_suffix else ""
            heatmap_path = os.path.join(args.save_plots, f"feature_sharing_heatmap{suffix}.png")
        else:
            heatmap_path = None
        plot_top_features_heatmap(results, top_n=args.top_n, save_path=heatmap_path)
