"""
Qualitative / feature-sharing analysis utilities.
Used by: feature_sharing.ipynb, category_regressions_analysis.ipynb (qualitative section)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from scipy.stats import pearsonr, spearmanr


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

DATASET_LABELS = {
    "hard_to_process": "Hard to Process",
    "abstract": "Abstract",
    "concrete": "Concrete",
    "ghost": "Ghost",
}


def analyze_raw_feature_sharing(csv_path, datasets=["abstract", "concrete"], voxels=None, logprob_idx=32768, split_idx=None):
    """Analyze which features are shared across participants for each dataset.

    Args:
        csv_path: Path to Qualitative_Analysis CSV file
        datasets: List of raw dataset names to analyze
        voxels: A list of (participant, neuroid) pairs to restrict analysis to
        logprob_idx: Feature index corresponding to log-probability token (excluded)
        split_idx: Split index to restrict analysis to (0 or 1) if the data was split into two halves.
    Returns:
        Dictionary keyed by readable dataset name with shared feature information
    """
    df = pd.read_csv(csv_path)

    dataset_map = DATASET_LABELS

    results = {}
    for dataset in datasets:
        dataset_df = df[df["dataset"] == dataset].copy()
        dataset_df["dataset"] = dataset_map.get(dataset, dataset)

        feature_counts = defaultdict(int)

        if voxels:
            for voxel in voxels:
                row = dataset_df[(dataset_df["participant"] == voxel[0]) & (dataset_df["neuroid"] == voxel[1])]
                row = row.iloc[0]
                if split_idx is not None and row["split_idx"] != split_idx:
                    continue
                features = parse_feature_indices(row["feature_indices"], exclude_logprob=True, logprob_idx=logprob_idx)
                for feat in features:
                    feature_counts[feat] += 1
        else:
            for _, row in dataset_df.iterrows():
                if split_idx is not None and row["split_idx"] != split_idx:
                    continue
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

def get_combined_top_features(results, top_n=15):
    """Return the top_n features shown in plot_top_features_heatmap: the union of each
    dataset's top_n most-shared features, ranked by total voxel count summed across
    datasets, and truncated back down to top_n.

    Args:
        results: Dict from analyze_raw_feature_sharing, keyed by readable dataset name
        top_n: Number of top features to keep per dataset (and in the combined ranking)

    Returns:
        List of (feature, total_count) tuples, sorted by total_count descending.
    """
    all_top_features = set()
    for dataset, data in results.items():
        for feat, count in data["shared_features"][:top_n]:
            all_top_features.add(feat)

    if not all_top_features:
        return []

    counts_by_dataset = [dict(data["shared_features"]) for data in results.values()]
    feature_totals = {
        feat: sum(counts.get(feat, 0) for counts in counts_by_dataset)
        for feat in all_top_features
    }

    sorted_features = sorted(all_top_features, key=lambda f: feature_totals[f], reverse=True)[:top_n]
    return [(f, feature_totals[f]) for f in sorted_features]


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
    combined_top = get_combined_top_features(results, top_n=top_n)

    if not combined_top:
        print("No shared features to plot")
        return

    sorted_features = [feat for feat, _ in combined_top]

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


# ---------------------------------------------------------------------------
# Split-half IoU analysis
# ---------------------------------------------------------------------------

def compute_iou(features_a, features_b):
    """Compute Intersection over Union between two collections of feature indices.

    Returns NaN if both collections are empty (undefined overlap).
    """
    set_a, set_b = set(features_a), set(features_b)
    union = set_a | set_b
    if not union:
        return np.nan
    return len(set_a & set_b) / len(union)


def analyze_split_iou(csv_path, datasets=None, logprob_idx=32768, signed=False):
    """Compute per-voxel Intersection over Union (IoU) of selected features between
    the two halves of a split qualitative analysis run (see --split in qualitative_analysis.py).

    Args:
        csv_path: Path to a Qualitative_Analysis CSV produced with --split (must have a split_idx column).
        datasets: Raw dataset names to include (default: all datasets present in the file).
        logprob_idx: Feature index corresponding to log-probability token (excluded).
        signed: If True, a feature must match in both identity and sign (direction of effect)
            to count as an overlap. If False (default), only feature identity is compared.

    Returns:
        DataFrame with one row per voxel: participant, neuroid, dataset, iou, n_features_0, n_features_1.
    """
    df = pd.read_csv(csv_path)
    if "split_idx" not in df.columns:
        raise ValueError(f"{csv_path} has no 'split_idx' column - was it generated with --split?")

    if datasets is None:
        datasets = sorted(df["dataset"].unique())

    rows = []
    for dataset in datasets:
        dataset_df = df[df["dataset"] == dataset]
        for (participant, neuroid), group in dataset_df.groupby(["participant", "neuroid"]):
            split_rows = {row["split_idx"]: row for _, row in group.iterrows()}
            if 0 not in split_rows or 1 not in split_rows:
                continue  # Both halves must be present to compare

            def get_features(row):
                feats = parse_feature_indices(row["feature_indices"], exclude_logprob=True, logprob_idx=logprob_idx)
                return feats if signed else [abs(f) for f in feats]

            features_0 = get_features(split_rows[0])
            features_1 = get_features(split_rows[1])

            rows.append({
                "participant": participant,
                "neuroid": neuroid,
                "dataset": dataset,
                "iou": compute_iou(features_0, features_1),
                "n_features_0": len(features_0),
                "n_features_1": len(features_1),
            })

    return pd.DataFrame(rows)


def plot_iou_distribution(iou_df, fig_width=5, fig_height=4, bins=10):
    """Plot per-voxel split-half IoU as a histogram, one panel per dataset.

    Args:
        iou_df: DataFrame from analyze_split_iou
        fig_width: Width in inches of each per-dataset panel
        fig_height: Figure height in inches
        bins: Number of histogram bins

    Returns:
        (fig, axes) tuple
    """
    datasets = sorted(iou_df["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(fig_width * len(datasets), fig_height), dpi=400, sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        vals = iou_df.loc[iou_df["dataset"] == dataset, "iou"].dropna()
        ax.hist(vals, bins=bins, range=(0, 1), color="#4C72B0", edgecolor="white")
        ax.axvline(vals.mean(), color="black", linestyle="--", linewidth=1.5, label=f"mean = {vals.mean():.2f}")
        ax.set_title(DATASET_LABELS.get(dataset, dataset), fontsize=13)
        ax.set_xlabel("IoU between splits", fontsize=12)
        ax.legend(fontsize=10)

    axes[0].set_ylabel("# Voxels", fontsize=12)
    plt.tight_layout()
    plt.show()

    return fig, axes


# ---------------------------------------------------------------------------
# Top-N shared-feature comparison (split-half vs. main analysis)
# ---------------------------------------------------------------------------

def compare_combined_top_features(results_a, results_b, top_n=15):
    """Compare the combined top-N features shown in plot_top_features_heatmap (the union
    of per-dataset top-N features, ranked by total count across datasets) between two
    feature-sharing analyses.

    Computes a single IoU over that combined feature set - i.e. whether the same features
    would appear in the heatmap for both analyses - plus, for each dataset present in both
    analyses, the Pearson/Spearman correlation of per-feature voxel-sharing counts
    (restricted to the union of the two combined feature sets).

    Args:
        results_a, results_b: Dicts from analyze_raw_feature_sharing (e.g. split halves or
            the full/main analysis), keyed by readable dataset name
        top_n: Number of top features (matches the top_n passed to plot_top_features_heatmap)

    Returns:
        DataFrame with one row per dataset present in both inputs. The 'iou' column is
        constant across rows - it is a property of the combined feature set, not of a
        single dataset.
    """
    top_a = {feat for feat, _ in get_combined_top_features(results_a, top_n)}
    top_b = {feat for feat, _ in get_combined_top_features(results_b, top_n)}
    iou = compute_iou(top_a, top_b)
    union_top = sorted(top_a | top_b)

    rows = []
    common_datasets = [d for d in results_a if d in results_b]
    for dataset in common_datasets:
        counts_a = dict(results_a[dataset]["shared_features"])
        counts_b = dict(results_b[dataset]["shared_features"])

        vec_a = np.array([counts_a.get(f, 0) for f in union_top])
        vec_b = np.array([counts_b.get(f, 0) for f in union_top])

        if len(union_top) >= 2 and vec_a.std() > 0 and vec_b.std() > 0:
            pearson_r, pearson_p = pearsonr(vec_a, vec_b)
            spearman_r, spearman_p = spearmanr(vec_a, vec_b)
        else:
            pearson_r = pearson_p = spearman_r = spearman_p = np.nan

        rows.append({
            "dataset": dataset,
            "iou": iou,
            "n_features": len(union_top),
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
        })

    return pd.DataFrame(rows)


def compare_splits_and_main(main_results, split0_results, split1_results, top_n=15):
    """Compare the combined top-N heatmap features (see compare_combined_top_features)
    for split0 vs split1, split0 vs main, and split1 vs main, in one combined DataFrame.

    Returns:
        DataFrame with a 'comparison' column indicating which pair was compared.
    """
    comparisons = [
        ("split0_vs_split1", split0_results, split1_results),
        ("split0_vs_main", split0_results, main_results),
        ("split1_vs_main", split1_results, main_results),
    ]

    dfs = []
    for label, results_a, results_b in comparisons:
        df = compare_combined_top_features(results_a, results_b, top_n=top_n)
        df["comparison"] = label
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)
