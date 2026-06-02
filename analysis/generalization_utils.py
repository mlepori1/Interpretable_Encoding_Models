import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, leaves_list
from scipy.spatial.distance import squareform

import seaborn as sns
from matplotlib.patches import Rectangle


def create_similarity_matrix(df):
    # Creates a similarity matrix from generalization results
    unique_voxels = pd.Index(df["base"].unique())

    matrix_df = df.pivot_table(
        index="base",
        columns="target",
        values="NC Normalized R Fischer",
        aggfunc="first",
    )

    matrix_df = matrix_df.reindex(index=unique_voxels, columns=unique_voxels)

    similarity_matrix = matrix_df.to_numpy()

    return similarity_matrix, unique_voxels.to_numpy()


def run_cluster_generalization_analysis(df):
    """Reorders voxels for plotting after passing
    through a hierarchical clustering algorithm.
    """
    # Load up results
    df["base"] = df["base_participant"] + "_" + df["base_neuroid"].astype(str)
    df["target"] = df["gen_participant"] + "_" + df["gen_neuroid"].astype(str)

    similarity_matrix, voxels = create_similarity_matrix(df)

    # Run off-the-shelf clustering algorithm to order voxels
    distances = 1 - np.minimum(similarity_matrix, similarity_matrix.T)
    distances = np.clip(distances, a_min=0, a_max=1)
    np.fill_diagonal(distances, 0)

    # Convert to condensed distance format
    condensed = squareform(distances)

    Z = linkage(condensed, method="average")
    Z_opt = optimal_leaf_ordering(Z, condensed)
    order = leaves_list(Z_opt)

    # Reorder similarity matrix
    S_reordered = similarity_matrix[order][:, order]
    voxels_reordered = voxels[order]

    return S_reordered, voxels_reordered


# Plot reordered similarity matrix
def plot_similarity_matrix(
    similarity_matrix,
    rectangle_edges=None,
    title=None,
    dpi=500,
):
    """Plots a heatmap indicating generalization, optionally outlining
    a set of rectangles to highlight them to a reader.
    """
    fig = plt.figure(figsize=(8, 8), dpi=dpi)
    ax = sns.heatmap(
        similarity_matrix,
        cmap="viridis",
        square=True,
        vmin=0,
        vmax=1,
        cbar_kws={"shrink": 0.6},
    )

    plt.title(title, fontsize=18)
    plt.tight_layout()

    # Rasterize the heatmap “tiles” only
    ax.collections[0].set_rasterized(True)

    if rectangle_edges:
        for edges in rectangle_edges:
            # ---- Define rectangle ----
            r0, r1 = edges[0], edges[1]
            c0, c1 = edges[0], edges[1]

            # ---- Add rectangle ----
            rect = Rectangle(
                (c0, r0),  # (x, y) bottom-left corner
                c1 - c0,  # width
                r1 - r0,  # height
                fill=False,
                edgecolor="red",
                linewidth=1.5,
            )

            ax.add_patch(rect)

    plt.ylabel("Source Voxel", fontsize=16)
    plt.xlabel("Target Voxel", fontsize=16)

    plt.show()

    return fig


def create_nested_heatmap(
    generalization_df,
    meta_basename="langfroi12345",
    sort_order="participant_first",
):
    """Nested heatmap of generalization results, organized by participant × fROI.

    Uses numpy + imshow instead of pivot_table + sns.heatmap for performance
    (~5M cells renders in seconds instead of minutes).

    Args:
        generalization_df: long-format generalization results.
        meta_basename: basename for the per-participant meta CSVs.
        sort_order: "participant_first" (default; voxels grouped by participant,
            then fROI within each participant) or "fROI_first" (voxels grouped
            by fROI, then participant within each fROI). Thick separator lines
            and large axis labels mark the primary dimension; thin separators
            and small secondary-axis labels mark the secondary dimension.
    """
    if sort_order not in ("participant_first", "fROI_first"):
        raise ValueError(f"sort_order must be 'participant_first' or 'fROI_first', got {sort_order!r}")

    froi_order = [
        "lang_LH_IFGorb",
        "lang_LH_IFG",
        "lang_LH_MFG",
        "lang_LH_AntTemp",
        "lang_LH_PostTemp",
    ]

    # Build neuroid -> fROI lookup
    def _load_neuroid_froi_map(participant):
        meta_path = f"../data/processed_csvs_anon/{participant}/{meta_basename}_meta.csv"
        meta = pd.read_csv(meta_path, usecols=["neuroid_id", "top10_tstat_langloc_SN_parc_lang_froi"])
        return dict(zip(meta["neuroid_id"], meta["top10_tstat_langloc_SN_parc_lang_froi"]))

    PARTICIPANT_ORDER = {
        "p1": "P1", "p2": "P2", "p3": "P3",
        "p4": "P4", "p5": "P5", "p6": "P6",
        "p7": "P7", "p8": "P8",
    }

    flat_lookup = {}
    all_participants = sorted(
        set(generalization_df["base_participant"].unique()) |
        set(generalization_df["gen_participant"].unique()),
        key=lambda p: list(PARTICIPANT_ORDER.keys()).index(p),
    )
    for participant in all_participants:
        for neuroid, froi in _load_neuroid_froi_map(participant).items():
            flat_lookup[f"{participant}_{neuroid}"] = froi

    # Sort key depends on sort_order; tertiary is always neuroid for stability
    def _sort_key(k):
        p = k.rsplit("_", 1)[0]
        nid = int(k.rsplit("_", 1)[1])
        f = flat_lookup.get(k, froi_order[-1])
        p_idx = all_participants.index(p)
        f_idx = froi_order.index(f)
        if sort_order == "participant_first":
            return (p_idx, f_idx, nid)
        return (f_idx, p_idx, nid)

    base_keys = (
        generalization_df["base_participant"] + "_" + generalization_df["base_neuroid"].astype(str)
    )
    unique_voxel_keys = sorted(set(base_keys), key=_sort_key)

    n = len(unique_voxel_keys)
    key_to_idx = {k: i for i, k in enumerate(unique_voxel_keys)}

    # Build matrix directly via numpy
    matrix = np.full((n, n), np.nan)
    gen_keys = (
        generalization_df["gen_participant"] + "_" + generalization_df["gen_neuroid"].astype(str)
    )
    base_idx = base_keys.map(key_to_idx).values
    gen_idx = gen_keys.map(key_to_idx).values
    vals = generalization_df["NC Normalized R Fischer"].values
    matrix[base_idx, gen_idx] = vals

    # Per-voxel primary/secondary dimension sequences for break + label computation
    voxel_participants = [k.rsplit("_", 1)[0] for k in unique_voxel_keys]
    voxel_frois = [flat_lookup.get(k, "") for k in unique_voxel_keys]

    froi_short = {
        "lang_LH_IFGorb": "IFGorb", "lang_LH_IFG": "IFG",
        "lang_LH_MFG": "MFG", "lang_LH_AntTemp": "AntTemp",
        "lang_LH_PostTemp": "PostTemp",
    }

    if sort_order == "participant_first":
        primary_seq, secondary_seq = voxel_participants, voxel_frois
        primary_label_map = PARTICIPANT_ORDER
        secondary_label_map = froi_short
        axis_label = "Participant → fROI"
    else:  # fROI_first
        primary_seq, secondary_seq = voxel_frois, voxel_participants
        primary_label_map = froi_short
        secondary_label_map = PARTICIPANT_ORDER
        axis_label = "fROI → Participant"

    # Thick breaks: primary-dimension changes. Thin breaks: any change.
    primary_breaks = [i for i in range(1, n) if primary_seq[i] != primary_seq[i - 1]]
    secondary_breaks = [
        i for i in range(1, n)
        if primary_seq[i] != primary_seq[i - 1] or secondary_seq[i] != secondary_seq[i - 1]
    ]

    def _midpoints_and_labels(breaks, seq, label_map):
        bounds = [0] + breaks + [n]
        labels, positions = [], []
        for i in range(len(bounds) - 1):
            start, end = bounds[i], bounds[i + 1]
            positions.append((start + end) / 2)
            labels.append(label_map.get(seq[start], seq[start]))
        return labels, positions

    primary_labels, primary_positions = _midpoints_and_labels(primary_breaks, primary_seq, primary_label_map)
    secondary_labels, secondary_positions = _midpoints_and_labels(secondary_breaks, secondary_seq, secondary_label_map)

    # Plot with imshow — match similarity matrix colormap
    fig, ax = plt.subplots(figsize=(8, 8), dpi=400)
    im = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1, aspect="equal",
                   interpolation="none", rasterized=True)
    cbar = fig.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label("NC Normalized R Fischer", fontsize=12)

    # Separator lines: thin at all breaks, thick at primary breaks (overdraws thin)
    for pos in secondary_breaks:
        ax.axhline(pos - 0.5, color="white", linewidth=0.5)
        ax.axvline(pos - 0.5, color="white", linewidth=0.5)
    for pos in primary_breaks:
        ax.axhline(pos - 0.5, color="white", linewidth=1.5)
        ax.axvline(pos - 0.5, color="white", linewidth=1.5)

    # Primary labels on main axis (left/bottom, larger)
    ax.set_yticks(primary_positions)
    ax.set_yticklabels(primary_labels, fontsize=8)
    ax.set_xticks(primary_positions)
    ax.set_xticklabels(primary_labels, fontsize=8, rotation=45, ha="right")

    # Secondary labels on secondary axis (right/top, smaller)
    ax_right = ax.secondary_yaxis("right")
    ax_right.set_yticks(secondary_positions)
    ax_right.set_yticklabels(secondary_labels, fontsize=5)
    ax_top = ax.secondary_xaxis("top")
    ax_top.set_xticks(secondary_positions)
    ax_top.set_xticklabels(secondary_labels, fontsize=5, rotation=90)

    ax.set_xlabel(f"Generalization Target: {axis_label}", fontsize=12)
    ax.set_ylabel(f"Base Voxel: {axis_label}", fontsize=12)

    plt.tight_layout()
    return matrix, ax
