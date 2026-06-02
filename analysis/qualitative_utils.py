"""
Qualitative / feature-sharing analysis utilities.
Used by: feature_sharing.ipynb, category_regressions_analysis.ipynb (qualitative section)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, Counter


# ---------------------------------------------------------------------------
# Feature index parsing
# ---------------------------------------------------------------------------

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


def parse_int_list(s):
    """Parse a bracket-enclosed whitespace-separated list of integers (unsigned)."""
    return [int(x) for x in s.strip("[]").split() if x]


def parse_voxel_list(voxels):
    """Parse list of 'participant_neuroid' strings into (participant, neuroid) tuples."""
    parsed = []
    for voxel in voxels:
        parts = voxel.split("_")
        parsed.append((parts[0], int(parts[1])))
    return parsed


# ---------------------------------------------------------------------------
# Feature overlap / sharing analysis
# ---------------------------------------------------------------------------

def analyze_raw_feature_sharing(csv_path, datasets=["abstract", "concrete"], voxels=None, logprob_idx=32768):
    """Analyze which features are shared across participants for each dataset.

    Args:
        csv_path: Path to Qualitative_Analysis CSV file
        datasets: List of raw dataset names to analyze
        voxels: A list of (participant, neuroid) pairs to restrict analysis to
        logprob_idx: Feature index corresponding to log-probability token (excluded)

    Returns:
        Dictionary keyed by readable dataset name with shared feature information
    """
    df = pd.read_csv(csv_path)

    dataset_map = {
        "hard_to_process": "Hard to Process",
        "abstract": "Abstract",
        "concrete": "Concrete",
        "ghost": "Ghost",
    }

    results = {}
    for dataset in datasets:
        dataset_df = df[df["dataset"] == dataset].copy()
        dataset_df["dataset"] = dataset_map.get(dataset, dataset)

        feature_counts = defaultdict(int)

        if voxels:
            for voxel in voxels:
                row = dataset_df[(dataset_df["participant"] == voxel[0]) & (dataset_df["neuroid"] == voxel[1])]
                row = row.iloc[0]
                features = parse_feature_indices(row["feature_indices"], exclude_logprob=True, logprob_idx=logprob_idx)
                for feat in features:
                    feature_counts[feat] += 1
        else:
            for _, row in dataset_df.iterrows():
                features = parse_feature_indices(row["feature_indices"], exclude_logprob=True, logprob_idx=logprob_idx)
                for feat in features:
                    feature_counts[feat] += 1

        features_sorted = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)

        results[dataset_map.get(dataset, dataset)] = {
            "shared_features": features_sorted,
            "total_unique_features": len(feature_counts),
            "feature_counts": feature_counts,
        }

    return results


# ---------------------------------------------------------------------------
# Heatmap visualization
# ---------------------------------------------------------------------------

def plot_top_features_heatmap(
    results,
    max_voxels,
    fig_width=8,
    top_n=15,
    format="portrait",
):
    """Plot heatmap of top shared features across datasets.

    Args:
        results: Dict from analyze_raw_feature_sharing
        max_voxels: Maximum voxel count for colorbar scaling
        fig_width: Figure width in inches
        top_n: Number of top features to display
        format: "portrait" (features on y-axis) or "landscape" (features on x-axis)

    Returns:
        (fig, ax) tuple
    """
    all_top_features = set()
    for dataset, data in results.items():
        for feat, count in data["shared_features"][:top_n]:
            all_top_features.add(feat)

    if not all_top_features:
        print("No shared features to plot")
        return

    feature_totals = {}
    for feat in all_top_features:
        total = 0
        for dataset, data in results.items():
            for f, count in data["shared_features"]:
                if f == feat:
                    total += count
                    break
        feature_totals[feat] = total

    sorted_features = sorted(all_top_features, key=lambda f: feature_totals[f], reverse=True)[:top_n]

    datasets = list(results.keys())
    matrix = np.zeros((len(sorted_features), len(datasets)))

    for j, dataset in enumerate(datasets):
        feature_dict = {f: count for f, count in results[dataset]["shared_features"]}
        for i, feat in enumerate(sorted_features):
            matrix[i, j] = feature_dict.get(feat, 0)

    feature_labels = []
    for feat in sorted_features:
        sign = "+" if feat > 0 else "-"
        feature_labels.append(f"{abs(feat)} ({sign})")

    if format == "portrait":
        cell_size = 0.65
        fig_h = max(4, len(sorted_features) * cell_size)
        fig_w = max(3, len(datasets) * cell_size) + 1.5  # extra for colorbar
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=400)
        im = ax.imshow(matrix, cmap="YlOrRd", aspect="equal", vmin=0, vmax=max_voxels)

        ax.set_xticks(np.arange(len(datasets)))
        ax.set_yticks(np.arange(len(sorted_features)))
        ax.set_xticklabels(datasets, fontsize=13)
        ax.set_yticklabels(feature_labels, fontsize=14)

        for i in range(len(sorted_features)):
            for j in range(len(datasets)):
                val = int(matrix[i, j])
                if val > 0:
                    color = "white" if val > max_voxels / 2 else "black"
                    ax.text(j, i, f"{val}", ha="center", va="center", color=color, fontsize=12, fontweight="bold")

        ax.set_xlabel("Dataset", fontsize=14)
        ax.set_ylabel("Feature (signed)", fontsize=14)
        ax.set_title(f"Top {len(sorted_features)} Shared Features Across Voxels\n(values = # Voxels)", fontsize=14)
    else:
        cell_size = 0.8
        fig_w = max(4, len(sorted_features) * cell_size)
        fig_h = max(3, len(datasets) * cell_size) + 1.5
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=400)
        im = ax.imshow(matrix.T, cmap="YlOrRd", aspect="equal", vmin=0, vmax=max_voxels)

        ax.set_yticks(np.arange(len(datasets)))
        ax.set_xticks(np.arange(len(sorted_features)))
        ax.set_yticklabels(datasets, fontsize=13)
        ax.set_xticklabels(feature_labels, fontsize=11)

        for i in range(len(datasets)):
            for j in range(len(sorted_features)):
                val = int(matrix.T[i, j])
                if val > 0:
                    color = "white" if val > max_voxels / 2 else "black"
                    ax.text(j, i, f"{val}", ha="center", va="center", color=color, fontsize=12, fontweight="bold")

        ax.set_ylabel("Dataset", fontsize=14)
        ax.set_xlabel("Feature (signed)", fontsize=14)
        ax.set_title(f"Top {len(sorted_features)} Shared Features Across Voxels (values = # Voxels)", fontsize=14)

    cbar_kwargs = {"ax": ax}
    if format != "portrait":
        cbar_kwargs["shrink"] = 0.4
    cbar = plt.colorbar(im, **cbar_kwargs)
    cbar.set_label(f"Voxels (max={max_voxels})", fontsize=13)
    cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.show()

    return fig, ax
