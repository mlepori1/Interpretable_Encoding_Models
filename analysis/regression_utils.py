"""
Regression analysis utilities.
Used by: category_regressions_analysis.ipynb, langfroi_analysis.ipynb
"""
import re
import glob as _glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, entropy
from collections import Counter, defaultdict
from pathlib import Path
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from helpers import (
    PARTICIPANTS, PALETTE, DATASET_MAP, DATASET_ORDER,
    FROI_MAP, FROI_ORDER,
    p_to_stars, bonferroni, build_path, load_data,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FROI_PALETTE = {
    "IFGorb":   "#FDAE6B",
    "IFG":      "#E6550D",
    "MFG":      "#C44E8B",
    "AntTemp":  "#8856A7",
    "PostTemp": "#4A1486",
}

DATASET_PALETTE = {
    "Hard to Process": "firebrick",
    "Easy to Process": "steelblue",
    "Abstract": "hotpink",
    "Concrete": "limegreen",
    "Ghost": "purple",
}

FEATURIZERS = ["Shuffled Control", "Surprisal", "JumpReLU", "Matryoshka", "Residual"]
YCOLS = ["R Fischer", "NC Normalized R Fischer"]

LOGPROB_MARKER = 32768
MATRYOSHKA_LEVELS = [128, 512, 2048, 8192, 32768]


# ---------------------------------------------------------------------------
# fROI Voxel Summary
# ---------------------------------------------------------------------------

def print_voxel_summary(df, label):
    """Print voxel counts per participant × fROI, plus summary statistics."""
    feat0 = "Surprisal"
    sub = df[df["Featurizer"] == feat0]

    counts = sub.groupby(["Participant", "fROI"], observed=True).size().unstack("fROI", fill_value=0)
    counts = counts.reindex(columns=FROI_ORDER, fill_value=0)
    counts["Total"] = counts.sum(axis=1)
    print(f"=== {label} ===")
    print(counts.to_string())

    totals = counts["Total"]
    print(f"\nVoxels per participant: mean={totals.mean():.1f}, median={totals.median():.1f}, "
          f"min={totals.min()}, max={totals.max()}, std={totals.std():.1f}")
    print(f"Total voxels (all participants): {totals.sum()}")

    print(f"\nVoxels per fROI (summed across participants):")
    for froi in FROI_ORDER:
        vals = counts[froi]
        print(f"  {froi:>10s}: total={vals.sum():>5d}, mean={vals.mean():.1f}, "
              f"min={vals.min()}, max={vals.max()}")
    print()


# ---------------------------------------------------------------------------
# Unified bar plot (category and fROI)
# ---------------------------------------------------------------------------

def predictivity_bar_plot(
    grouping="Dataset",
    ycol="NC Normalized R Fischer",
    gap_after="Surprisal",
    gap_frac=0.5,
    figsize=(13, 4.2),
    dpi=300,
    err="sem",
    add_significance=False,
    significance_annotations=None,
    ylim=None,
    layer=12,
    add_control=True,
    meta_basename="langfroi12345",
    filter_set=None,
    results_suffix="_langfroi12345_20260311"
):
    """Grouped bar plot of predictivity by featurizer and dataset or fROI.

    Args:
        grouping: "Dataset" or "fROI"
        ycol: Column to plot
        gap_after: Featurizer name after which to add a visual gap
        gap_frac: Gap size as fraction of one bar width
        figsize: Figure size tuple
        dpi: Figure DPI
        err: "sem" or "sd"
        add_significance: Whether to annotate bars with significance stars
        significance_annotations: Dict (group_label, featurizer) -> star string
        ylim: Optional (ymin, ymax) tuple
        layer: Layer index for build_path
        add_control: Whether to include Shuffled Control bars
        meta_basename: Metadata file basename for fROI merging (passed to load_data)
        filter_set: A list of either datasets or fROIs to include

    Returns:
        (fig, ax)
    """
    featurizers = ["Surprisal", "JumpReLU", "Matryoshka", "Residual"]
    if add_control:
        featurizers = ["Shuffled Control"] + featurizers

    if grouping == "Dataset":
        group_order = DATASET_ORDER
        if filter_set:
            group_order = [item for item in group_order if item in filter_set]
        xlabel = "Dataset"
        group_col = "Dataset"
        dataset = "categories"
    elif grouping == "fROI":
        group_order = FROI_ORDER
        if filter_set:
            group_order = [item for item in group_order if item in filter_set]
        xlabel = "Language fROI"
        group_col = "fROI"
        dataset = "fROI"
    else:
        raise ValueError(f"grouping must be 'Dataset' or 'fROI', got '{grouping}'")

    dfs = []
    for featurizer in featurizers:
        for participant_id in PARTICIPANTS:
            feat_str = {"Shuffled Control": "Residual", "Surprisal": "Log Probabilities"}.get(featurizer, featurizer)
            control = featurizer == "Shuffled Control"
            tmp = load_data(feat_str, participant_id, dataset=dataset, layer=layer,
                            control=control, meta_basename=meta_basename, results_suffix=results_suffix)
            tmp["Featurizer"] = featurizer
            tmp["Participant"] = participant_id
            dfs.append(tmp)
    df = pd.concat(dfs, ignore_index=True)

    if grouping == "Dataset":
        df["Dataset"] = df["dataset"].map(DATASET_MAP)
        df["Dataset"] = pd.Categorical(df["Dataset"], categories=DATASET_ORDER, ordered=True)
        if filter_set:
            df = df[df["Dataset"].isin(filter_set)]
    else:
        df["fROI"] = df["top10_tstat_langloc_SN_parc_lang_int"].map(FROI_MAP)
        df["fROI"] = pd.Categorical(df["fROI"], categories=FROI_ORDER, ordered=True)
        if filter_set:
            df = df[df["fROI"].isin(filter_set)]

    df["Featurizer"] = pd.Categorical(df["Featurizer"], categories=featurizers, ordered=True)

    df_participant_level = (
        df.groupby(["Participant", group_col, "Featurizer"], as_index=False, observed=True)[ycol].mean()
    )

    grouped = df_participant_level.groupby([group_col, "Featurizer"], observed=True)[ycol]
    mean = grouped.mean()
    if err == "sem":
        errvals = grouped.sem()
    elif err == "sd":
        errvals = grouped.std()
    else:
        raise ValueError("err must be 'sem' or 'sd'")

    mean_tbl = mean.unstack("Featurizer").reindex(index=group_order, columns=featurizers)
    err_tbl = errvals.unstack("Featurizer").reindex(index=group_order, columns=featurizers)

    n_groups = len(group_order)
    n_hue = len(featurizers)
    x = np.arange(n_groups)
    bar_w = 0.15
    gap = gap_frac * bar_w
    offsets = (np.arange(n_hue) - (n_hue - 1) / 2.0) * bar_w
    gap_idx = featurizers.index(gap_after)
    offsets = offsets + (np.arange(n_hue) > gap_idx) * gap

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    colors = [PALETTE[f] for f in featurizers]

    for j, feat in enumerate(featurizers):
        ax.bar(
            x + offsets[j],
            mean_tbl[feat].to_numpy(),
            width=bar_w,
            yerr=err_tbl[feat].to_numpy(),
            capsize=0,
            linewidth=1,
            label=feat,
            color=colors[j],
            edgecolor="black",
            zorder=3,
        )
        for gi, gname in enumerate(group_order):
            sub = df_participant_level[
                (df_participant_level[group_col] == gname) &
                (df_participant_level["Featurizer"] == feat)
            ]
            jitter = np.random.uniform(-bar_w * 0.3, bar_w * 0.3, len(sub))
            ax.scatter(
                np.full(len(sub), x[gi] + offsets[j]) + jitter,
                sub[ycol].to_numpy(),
                s=16, alpha=0.7, color=colors[j], edgecolor="black", linewidth=1, zorder=4,
            )

    ax.axhline(0, color="black", linestyle="--", linewidth=1.0, zorder=2)
    if "NC Normalized" in ycol:
        ax.set_ylabel("Normalized Predictivity ($r$)", fontsize=14)
    else:
        ax.set_ylabel("Predictivity ($r$)", fontsize=14)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(group_order, rotation=45, ha="center", fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.yaxis.grid(True, linestyle="-", linewidth=0.5, alpha=0.3, zorder=0)
    ax.xaxis.grid(False)
    sns.despine(ax=ax)
    ax.legend(title="Featurizer", frameon=False, loc="center left",
               bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    if add_significance and significance_annotations is not None:
        if ylim is not None:
            y_span = ylim[1] - ylim[0]
        else:
            y0, y1 = ax.get_ylim()
            y_span = y1 - y0
        star_pad = 0.02 * y_span

        for gi, gname in enumerate(group_order):
            for feat in featurizers:
                if feat in ("Surprisal", "Shuffled Control"):
                    continue
                label = significance_annotations.get((gname, feat), "")
                if not label:
                    continue
                j = featurizers.index(feat)
                y_bar = mean_tbl.loc[gname, feat]
                y_err = err_tbl.loc[gname, feat]
                if pd.isna(y_bar):
                    continue
                y_pos = y_bar + (0 if pd.isna(y_err) else y_err) + star_pad
                ax.text(x[gi] + offsets[j], y_pos, label,
                        ha="center", va="bottom", fontsize=9, fontweight="bold", zorder=5)

    compare_feats = ["JumpReLU", "Matryoshka", "Residual"]
    if all(f in mean_tbl.columns for f in ["Surprisal"] + compare_feats):
        if ylim is not None:
            y_span = ylim[1] - ylim[0]
        else:
            y0, y1 = ax.get_ylim()
            y_span = y1 - y0
        bracket_pad = 0.04 * y_span
        text_pad = 0.01 * y_span

        for gi, gname in enumerate(group_order):
            lp = mean_tbl.loc[gname, "Surprisal"]
            avg_other = mean_tbl.loc[gname, compare_feats].mean()
            delta = avg_other - lp

            # Bracket spans only the 3 SAE featurizers (delta = their avg - log probs)
            sae_feats_in_plot = [f for f in compare_feats if f in featurizers]
            x_left = x[gi] + offsets[featurizers.index(sae_feats_in_plot[0])]
            x_right = x[gi] + offsets[featurizers.index(sae_feats_in_plot[-1])]
            cluster_x = (x_left + x_right) / 2

            all_feats = sae_feats_in_plot + ["Surprisal"]
            cluster_height = max(
                mean_tbl.loc[gname, all_feats] + err_tbl.loc[gname, all_feats].fillna(0)
            )
            bracket_y = cluster_height + bracket_pad

            # Horizontal line spanning the 3 SAE featurizers
            ax.plot([x_left, x_right], [bracket_y, bracket_y],
                    color="black", linewidth=1.2, zorder=6, clip_on=False)


            # Delta text above bracket
            ax.text(cluster_x, bracket_y + text_pad, f"Δ={delta:.3f}",
                    ha="center", va="bottom", fontsize=11, fontweight="bold", zorder=6)

    if ylim is not None:
        ax.set_ylim(*ylim)

    fig.tight_layout()
    return fig, ax


def strip_plot_by_froi(df, ycol="NC Normalized R Fischer", figsize=(16, 5), dpi=300, title=None):
    """Strip plot showing voxel-level dots and participant means, faceted by fROI.

    Returns:
        (fig, axes)
    """
    featurizers = FEATURIZERS

    fig, axes = plt.subplots(1, len(FROI_ORDER), figsize=figsize, dpi=dpi, sharey=True)
    for i, froi in enumerate(FROI_ORDER):
        ax = axes[i]
        sub = df[df["fROI"] == froi]

        sns.stripplot(data=sub, x="Featurizer", y=ycol, order=featurizers,
                      hue="Featurizer", palette=PALETTE, hue_order=featurizers,
                      alpha=0.06, size=2, jitter=0.3, dodge=False, legend=False, ax=ax, zorder=1)

        pmeans = sub.groupby(["Participant", "Featurizer"], as_index=False, observed=True)[ycol].mean()
        sns.stripplot(data=pmeans, x="Featurizer", y=ycol, order=featurizers,
                      hue="Featurizer", palette=PALETTE, hue_order=featurizers,
                      alpha=0.8, size=5, jitter=0.15, dodge=False,
                      edgecolor="black", linewidth=0.5, legend=False, ax=ax, zorder=2)

        ax.set_title(froi, fontsize=10)
        ax.set_xlabel("")
        labels = []
        for f in featurizers:
            if f == "Surprisal":
                labels.append("Surprisal")
            elif f == "Topic Model":
                labels.append("Topic\nModel")
            elif f == "Shuffled Control":
                labels.append("Shuffled\nCtrl")
            else:
                labels.append(f)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
        if i > 0:
            ax.set_ylabel("")
        ax.axhline(0, color="black", linestyle="--", linewidth=0.5, alpha=0.5)
        sns.despine(ax=ax)

    if title:
        fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Restricted-set analysis
# ---------------------------------------------------------------------------


RESTRICTED_LABELS_MAP = {
    "_0_128": "0–128",
    "_128_512": "128–512",
    "_512_2048": "512-2048",
    "_2048_8192": "2048-8192",
    "_8192_None": "8192+",
    "_128_None": "128+",
    "Surprisal": "Surprisal",
}

RESTRICTED_FEATURES_PALETTE = {
    "_0_128": "mediumseagreen",
    "_128_512": "darkturquoise",
    "_512_2048": "plum",
    "_2048_8192": "lightcoral",
    "_8192_None": "goldenrod",
    "_128_None": "darkorange",
    "Surprisal": "dimgray",
}

def restricted_set_analysis(
    grouping="Dataset",
    ycol="NC Normalized R Fischer",
    feature_sets=["Surprisal", "_0_128", "_128_None"],
    figsize=(5, 4.2),
    dpi=300,
    err="sem",
    ylim=None,
    layer=12,
    meta_basename="langfroi12345",
    results_suffix="_langfroi12345_20260311",
    categories=None,
):
    """Bar plot comparing predictivity for restricted Matryoshka feature index ranges.

    Args:
        grouping: "Dataset" or "fROI" — controls which results directory is loaded
        ycol: Column to plot
        feature_sets: List of feature index suffixes (and "Surprisal") to compare
        figsize: Figure size tuple
        dpi: Figure DPI
        err: "sem" or "sd"
        ylim: Optional (ymin, ymax) tuple
        layer: Layer index for build_path
        meta_basename: Metadata file basename for fROI merging (passed to load_data)
        results_suffix: Fallback to try and find results.
        categories: Only consider this set of categories (if grouping == "Dataset")

    Returns:
        (fig, ax)
    """
    if grouping not in ("Dataset", "fROI"):
        raise ValueError(f"grouping must be 'Dataset' or 'fROI', got '{grouping}'")

    dataset = "categories" if grouping == "Dataset" else "fROI"
    xlabel = "All Datasets" if grouping == "Dataset" else "All fROIs"
    dfs = []

    for feature_set in feature_sets:
        for participant_id in PARTICIPANTS:
            if feature_set == "Surprisal":
                tmp = load_data("Log Probabilities", participant_id, dataset=dataset,
                                layer=layer, meta_basename=meta_basename, results_suffix=results_suffix)
            else:
                tmp = load_data("Matryoshka", participant_id, dataset=dataset,
                                layer=layer, feature_idx=feature_set, meta_basename=meta_basename, results_suffix=results_suffix)
            tmp["Featurizer"] = feature_set
            tmp["Participant"] = participant_id
            dfs.append(tmp)

    df = pd.concat(dfs, ignore_index=True)
    if grouping == "fROI":
        df["fROI"] = df["top10_tstat_langloc_SN_parc_lang_int"].map(FROI_MAP)
    if grouping == "Dataset":
        if categories is not None:
            print(f"Only considering the following categories: {categories}")
            df = df[df["dataset"].isin(categories)]

    df["Featurizer"] = pd.Categorical(df["Featurizer"], categories=feature_sets, ordered=True)

    df_participant_level = (
        df.groupby(["Participant", "Featurizer"], as_index=False)[ycol].mean()
    )

    grouped = df_participant_level.groupby("Featurizer")[ycol]
    mean = grouped.mean().reindex(feature_sets)
    if err == "sem":
        errvals = grouped.sem().reindex(feature_sets)
    elif err == "sd":
        errvals = grouped.std().reindex(feature_sets)
    else:
        raise ValueError("err must be 'sem' or 'sd'")

    n_hue = len(feature_sets)
    x = np.array([0.0])
    total_width = 0.8
    bar_w = total_width / n_hue
    offsets = (np.arange(n_hue) - (n_hue - 1) / 2) * bar_w

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    colors = [RESTRICTED_FEATURES_PALETTE[f] for f in feature_sets]

    for j, feat in enumerate(feature_sets):
        xpos = x + offsets[j]
        ax.bar(xpos, [mean.loc[feat]], width=bar_w, yerr=[errvals.loc[feat]], capsize=0,
               linewidth=1, label=RESTRICTED_LABELS_MAP[feat], color=colors[j], edgecolor="black", zorder=3)

        sub = df_participant_level[df_participant_level["Featurizer"] == feat]
        jitter = np.random.uniform(-bar_w * 0.2, bar_w * 0.2, len(sub))
        ax.scatter(np.full(len(sub), xpos[0]) + jitter, sub[ycol].to_numpy(),
                   s=16, alpha=0.7, color=colors[j], edgecolor="black", linewidth=1, zorder=4)

    ax.axhline(0, color="black", linestyle="--", linewidth=1.0, zorder=2)
    ax.set_title(f"Predictivity by Restricted Feature Set ({grouping})", fontsize=14)
    ax.set_ylabel("Normalized Predictivity ($r$)", fontsize=14)
    ax.set_xlabel("", fontsize=14)
    ax.set_xticks([0])
    ax.set_xticklabels([xlabel], rotation=0, ha="center", fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.yaxis.grid(True, linestyle="-", linewidth=0.5, alpha=0.3, zorder=0)
    ax.xaxis.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(title="Feature Set", frameon=False, loc="center left",
               bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    if ylim is not None:
        ax.set_ylim(*ylim)

    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Feature basis / support set size
# ---------------------------------------------------------------------------

def feature_basis_plot(
    grouping="Dataset",
    layer=12,
    meta_basename="langfroi12345",
    results_suffix="_langfroi12345_20260311"
):
    """Bar plot of average support set size (Feature Mean) by dataset or fROI and featurizer.

    Args:
        grouping: "Dataset" or "fROI"
        layer: Layer index for build_path
        meta_basename: Metadata file basename for fROI merging (passed to load_data)

    Returns:
        g.fig from catplot
    """
    if grouping not in ("Dataset", "fROI"):
        raise ValueError(f"grouping must be 'Dataset' or 'fROI', got '{grouping}'")

    featurizers = ["JumpReLU", "Matryoshka", "Residual"]
    dataset = "categories" if grouping == "Dataset" else "fROI"

    dfs = []
    for participant_id in PARTICIPANTS:
        for featurizer in featurizers:
            tmp = load_data(featurizer, participant_id, dataset=dataset, layer=layer,
                            meta_basename=meta_basename, results_suffix=results_suffix)
            tmp["Featurizer"] = featurizer
            tmp["Participant"] = participant_id
            dfs.append(tmp)

    df = pd.concat(dfs, ignore_index=True)

    if grouping == "Dataset":
        group_order = ["Abstract", "Concrete"]
        df["Dataset"] = df["dataset"].map(DATASET_MAP)

        include_in_avg = ["Hard to Process", "Easy to Process", "Abstract", "Concrete"]
        matryoshka_df = df[df["Featurizer"] == "Matryoshka"]
        matryoshka_df = matryoshka_df[matryoshka_df["Dataset"].isin(include_in_avg)]
        overall_avg = matryoshka_df["Feature Mean"].mean()
        print(f"\nOverall voxel-level average Matryoshka Feature Mean: {overall_avg:.3f}")

        df["Dataset"] = pd.Categorical(df["Dataset"], categories=group_order, ordered=True)
        group_col = "Dataset"
    else:
        group_order = FROI_ORDER
        df["fROI"] = df["top10_tstat_langloc_SN_parc_lang_int"].map(FROI_MAP)
        df["fROI"] = pd.Categorical(df["fROI"], categories=group_order, ordered=True)
        group_col = "fROI"

    df["Featurizer"] = pd.Categorical(df["Featurizer"], categories=featurizers, ordered=True)

    df_participant_level = (
        df.groupby(["Participant", group_col, "Featurizer"], as_index=False)["Feature Mean"].mean()
    )

    g = sns.catplot(
        data=df_participant_level, x=group_col, y="Feature Mean", hue="Featurizer",
        order=group_order, palette=PALETTE, kind="bar", errorbar="se",
        edgecolor="black", linewidth=1, height=4.2, aspect=5.5 / 4.2,
    )
    g.fig.set_dpi(400)
    plt.ylabel("Average Support Set Size")
    plt.title(f"Feature Basis Analysis: Support Set Size by {grouping} and Featurizer")
    plt.show()
    return g.fig


# ---------------------------------------------------------------------------
# Unified granularity bar plot (category and fROI)
# ---------------------------------------------------------------------------

def _load_granularity_data(qual_dir, qual_glob):
    """Load qualitative CSVs for fROI granularity analysis, parse feature indices, map fROI.

    Args:
        qual_dir: Directory containing qualitative CSV files
        qual_glob: Glob pattern for the CSV files

    Returns:
        DataFrame with parsed feature_abs and fROI columns
    """
    files = sorted(_glob.glob(str(Path(qual_dir) / qual_glob)))
    dfs = []
    for f in files:
        dfs.append(pd.read_csv(f))
    df = pd.concat(dfs, ignore_index=True)

    df["feature_abs"] = (
        df["feature_indices"]
        .apply(lambda s: [abs(int(x)) for x in re.findall(r"-?\d+", str(s))])
        .apply(lambda lst: [x for x in lst if x != LOGPROB_MARKER])
    )
    df["fROI"] = df["top10_tstat_langloc_SN_parc_lang_int"].map(FROI_MAP)
    df["fROI"] = pd.Categorical(df["fROI"], categories=FROI_ORDER, ordered=True)
    return df


def granularity_bar_plot(
    grouping="fROI",
    levels=None,
    figsize=(8.5, 4.2),
    dpi=400,
    layer=12,
    qual_dir=None,
    qual_glob=None,
):
    """Bar plot of average feature count per Matryoshka granularity bin, grouped by fROI or Dataset.

    Args:
        grouping: "fROI" or "Dataset"
        levels: Matryoshka bin boundaries (defaults to MATRYOSHKA_LEVELS)
        figsize: Figure size tuple
        dpi: Figure DPI
        layer: Layer index for build_path (used for Dataset path; ignored for fROI)
        qual_dir: Directory of per-participant qualitative CSVs (required for grouping="fROI")
        qual_glob: Glob pattern for qualitative CSVs (required for grouping="fROI")

    Returns:
        g.fig from catplot
    """
    if levels is None:
        levels = MATRYOSHKA_LEVELS

    if grouping == "fROI":
        if qual_dir is None or qual_glob is None:
            raise ValueError("qual_dir and qual_glob must be provided when grouping='fROI'")
        group_col = "fROI"
        group_order = FROI_ORDER
        palette = FROI_PALETTE
        title = "Feature Granularity by fROI"
        df = _load_granularity_data(qual_dir, qual_glob)
    elif grouping == "Dataset":
        group_col = "Dataset"
        group_order = DATASET_ORDER
        palette = DATASET_PALETTE
        title = "Feature Granularity by Dataset"
        qual_path = f"{build_path('Matryoshka', dataset='categories', layer=layer)}/qualitative/Qualitative_Analysis_8000_n20_per_participant.csv"
        df = pd.read_csv(qual_path)
        df["feature_abs"] = (
            df["feature_indices"]
            .apply(lambda s: [abs(int(x)) for x in re.findall(r"-?\d+", str(s))])
            .apply(lambda lst: [x for x in lst if x != LOGPROB_MARKER])
        )
        df["Dataset"] = df["dataset"].map(DATASET_MAP)
        df["Dataset"] = pd.Categorical(df["Dataset"], categories=DATASET_ORDER, ordered=True)
    else:
        raise ValueError(f"grouping must be 'fROI' or 'Dataset', got '{grouping}'")

    rows = []
    for _, row in df.iterrows():
        counts = np.histogram(row["feature_abs"], bins=[0] + levels)[0]
        for level, count in zip(levels, counts):
            rows.append({
                "participant": row["participant"],
                group_col: row[group_col],
                "Feature Bin": level,
                "Feature Count": int(count),
            })
    df_exp = pd.DataFrame(rows)
    df_exp[group_col] = pd.Categorical(df_exp[group_col], categories=group_order, ordered=True)

    df_part = df_exp.groupby(
        ["participant", group_col, "Feature Bin"], as_index=False, observed=True
    )["Feature Count"].mean()

    g = sns.catplot(
        data=df_part, x="Feature Bin", y="Feature Count", hue=group_col,
        hue_order=group_order, palette=palette, kind="bar", errorbar="se",
        edgecolor="black", linewidth=1, height=figsize[1], aspect=figsize[0] / figsize[1],
    )
    g._legend.set_title(group_col)
    g._legend.set_frame_on(True)
    frame = g._legend.get_frame()
    frame.set_edgecolor("black")
    frame.set_linewidth(1)
    frame.set_facecolor("white")
    g.fig.set_dpi(dpi)
    for ax in g.axes.flat:
        ax.grid(False)
        sns.despine(ax=ax)
        ax.tick_params(axis='both', labelsize=14)
    plt.ylabel("Average Number of Features", fontsize=14)
    plt.xlabel("Feature Bin", fontsize=14)
    plt.title(title, fontsize=14)
    plt.show()
    return g.fig


# ---------------------------------------------------------------------------
# Feature index histograms
# ---------------------------------------------------------------------------

def feature_index_plot(
    grouping="fROI",
    collapsed=False,
    levels=None,
    figsize=None,
    dpi=300,
    log_x=False,
    qual_dir=None,
    qual_glob=None,
    categories=None,
):
    """Histogram of Matryoshka feature usage by SAE feature index.

    Works for both fROI and Dataset groupings, and in both faceted (one panel
    per group) and collapsed (all groups combined) modes.

    Args:
        grouping: "fROI" or "Dataset"
        collapsed: If True, produce a single histogram collapsed across all groups.
            If False (default), produce one subplot per group.
        levels: Matryoshka bin boundaries used for axvline markers
            (defaults to MATRYOSHKA_LEVELS)
        figsize: Figure size tuple. Defaults to (16, 8) for faceted, (14, 3) for collapsed.
        dpi: Figure DPI
        log_x: Whether to use log2 x-axis scale
        qual_dir: Directory of per-participant qualitative CSVs (required for grouping="fROI")
        qual_glob: Glob pattern for qualitative CSVs (required for grouping="fROI")
        categories: List of raw dataset names to include (Dataset only; None = all)

    Returns:
        fig
    """
    if levels is None:
        levels = MATRYOSHKA_LEVELS

    # ------------------------------------------------------------------
    # Load and prepare data
    # ------------------------------------------------------------------
    if grouping == "fROI":
        if qual_dir is None or qual_glob is None:
            raise ValueError("qual_dir and qual_glob must be provided when grouping='fROI'")
        df = _load_granularity_data(qual_dir, qual_glob)
        group_col = "fROI"
        group_order = FROI_ORDER
        palette = FROI_PALETTE
    elif grouping == "Dataset":
        path = f"{build_path('Matryoshka', dataset='categories', layer=12)}/qualitative/Qualitative_Analysis_8000_n20_per_participant.csv"
        df = pd.read_csv(path)
        df["feature_abs"] = (
            df["feature_indices"]
            .apply(lambda s: [abs(int(x)) for x in re.findall(r"-?\d+", str(s))])
            .apply(lambda lst: [x for x in lst if x != LOGPROB_MARKER])
        )
        df["Dataset"] = df["dataset"].map(DATASET_MAP)
        df["Dataset"] = pd.Categorical(df["Dataset"], categories=DATASET_ORDER, ordered=True)
        if categories is not None:
            df = df[df["dataset"].isin(categories)]
        group_col = "Dataset"
        group_order = [d for d in DATASET_ORDER if d in df["Dataset"].values]
        palette = DATASET_PALETTE
    else:
        raise ValueError(f"grouping must be 'fROI' or 'Dataset', got '{grouping}'")

    n_participants = df["participant"].nunique()

    # ------------------------------------------------------------------
    # Shared helper: draw bars/vlines on a single axes
    # ------------------------------------------------------------------
    def _draw(ax, indices, counts, color):
        if log_x:
            plot_indices = [i if i > 0 else 0.5 for i in indices]
            ax.vlines(plot_indices, 0, counts, colors=color, linewidth=0.8)
            ax.set_xscale("log", base=2)
            ax.set_xlim(left=0.3)
        else:
            ax.bar(indices, counts, width=40, color=color, edgecolor="none")
        for level in levels:
            ax.axvline(x=level, color="red", linestyle="--", alpha=0.8, linewidth=1)
        ax.set_ylim(bottom=0)
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ------------------------------------------------------------------
    # Collapsed: single histogram across all groups
    # ------------------------------------------------------------------
    if collapsed:
        if figsize is None:
            figsize = (14, 3)
        counter = Counter()
        for feat_list in df["feature_abs"]:
            counter.update(feat_list)
        indices = sorted(counter.keys())
        counts = [counter[i] for i in indices]

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        _draw(ax, indices, counts, "slategray")
        for i, level in enumerate(levels):
            ax.text(level, ax.get_ylim()[1], str(level),
                    ha="right" if i == 0 else "center", va="bottom", fontsize=10, color="black")
        ax.set_xlabel("SAE Feature Index", fontsize=14)
        ax.set_ylabel("Count", fontsize=14)
        ax.tick_params(axis='both', labelsize=14)
        ax.set_title(
            f"Feature Usage by SAE Feature Index ({n_participants} participants, all {grouping}s)",
            fontsize=14
        )

    # ------------------------------------------------------------------
    # Faceted: one subplot per group
    # ------------------------------------------------------------------
    else:
        if figsize is None:
            figsize = (16, 8)
        feat_counts = {}
        for group in group_order:
            counter = Counter()
            for feat_list in df[df[group_col] == group]["feature_abs"]:
                counter.update(feat_list)
            feat_counts[group] = counter

        fig, axes = plt.subplots(len(group_order), 1, figsize=figsize, sharex=True, dpi=dpi)
        if len(group_order) == 1:
            axes = [axes]

        for ax, group in zip(axes, group_order):
            counter = feat_counts[group]
            indices = sorted(counter.keys())
            counts = [counter[i] for i in indices]
            _draw(ax, indices, counts, palette[group])
            ax.set_ylabel("Count", fontsize=14)
            ax.tick_params(axis='both', labelsize=14)
            ax.text(0.01, 0.85, group, transform=ax.transAxes, fontsize=12,
                    fontweight="bold", color=palette[group], va="top")

        for i, level in enumerate(levels):
            axes[0].text(level, axes[0].get_ylim()[1], str(level),
                         ha="right" if i == 0 else "center", va="bottom", fontsize=10, color="black")
        axes[-1].set_xlabel("SAE Feature Index", fontsize=14)
        fig.suptitle(
            f"Feature Usage by SAE Feature Index ({n_participants} participants)",
            y=1.02, fontsize=14,
        )

    plt.tight_layout()
    plt.show()
    return fig

# ---------------------------------------
#  Feature Specificity Analysis
# ---------------------------------------

def feature_specificity_analysis(
    qual_dir,
    qual_glob,
    norm_per_participant=False,
    grouping="participant",
    figsize=(6.5, 2.5),
    dpi=400,
):
    """Scatter plot of feature diagnosticity vs. Prevalence.

    Args:
        qual_dir: Directory of per-participant qualitative CSVs.
        qual_glob: Glob pattern for qualitative CSVs.
        norm_per_participant: If True, normalize per-group counts by group voxel totals
            before entropy. (Name preserved for backward compatibility; applies to
            whichever grouping is chosen.)
        grouping: "participant" (entropy across the 8 participants, ceiling = log2(8) = 3 bits)
            or "fROI" (entropy across the 5 language fROIs, ceiling = log2(5) ≈ 2.32 bits).
        figsize: Figure size tuple.
        dpi: Figure DPI.

    Returns:
        (fig, plot_df) tuple.
    """

    df = _load_granularity_data(qual_dir, qual_glob)

    if grouping == "participant":
        group_col = "participant"
        group_keys = list(PARTICIPANTS)
        ylabel = "Entropy across Participants"
        title = "Feature Prevalence vs. Participant Entropy"
    elif grouping == "fROI":
        group_col = "fROI"
        group_keys = list(FROI_ORDER)
        ylabel = "Entropy across fROIs"
        title = "Feature Prevalence vs. fROI Entropy"
    else:
        raise ValueError(f"grouping must be 'participant' or 'fROI', got {grouping!r}")

    group_voxel_totals = {g: int((df[group_col] == g).sum()) for g in group_keys}

    feature2properties = {}
    for _, row in df.iterrows():
        features = row["feature_abs"]
        g = row[group_col]

        for feature in features:
            if feature not in feature2properties:
                feature2properties[feature] = {"all": 0}
            if g not in feature2properties[feature]:
                feature2properties[feature][g] = 0

            feature2properties[feature][g] += 1
            feature2properties[feature]["all"] += 1

    rows = []

    for feature, props in feature2properties.items():

        if norm_per_participant:
            dist = [props.get(g, 0) / max(group_voxel_totals[g], 1) for g in group_keys]
        else:
            dist = [props.get(g, 0) for g in group_keys]

        rows.append({
            "Feature": feature,
            "Prevalence": props["all"],
            "Entropy": entropy(dist, base=2),
            "Type": "Feature",
        })

    # Baseline: maximally prevalent feature, distributed like group voxel totals
    if norm_per_participant:
        rows.append({
            "Feature": "Max prevalence baseline",
            "Prevalence": sum(group_voxel_totals.values()),
            "Entropy": entropy([1.0 for _ in group_keys], base=2),
            "Type": "Max Prevalence Baseline",
        })
    else:
        rows.append({
            "Feature": "Max prevalence baseline",
            "Prevalence": sum(group_voxel_totals.values()),
            "Entropy": entropy([group_voxel_totals[g] for g in group_keys], base=2),
            "Type": "Max Prevalence Baseline",
        })
    # Baseline: maximally group-specific features
    for g in group_keys:
        rows.append({
            "Feature": f"Max specificity baseline: {g}",
            "Prevalence": group_voxel_totals[g],
            "Entropy": 0.0,
            "Type": "Max Specificity Baselines",
        })

    plot_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    for label, subdf in plot_df.groupby("Type"):
        ax.scatter(
            subdf["Prevalence"],
            subdf["Entropy"],
            label=label,
            alpha=0.7,
            s=30 if label == "Feature" else 80,
            edgecolors="black",
            linewidths=.5
        )

    ax.set_xlabel("Prevalence")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)

    fig.tight_layout()

    # after plot_df is created
    feature_df = plot_df[plot_df["Type"] == "Feature"].copy()

    top_right = feature_df[
        (feature_df["Prevalence"] >= feature_df["Prevalence"].quantile(0.96)) &
        (feature_df["Entropy"] >= feature_df["Entropy"].quantile(0.90))
    ]

    labeled = top_right

    for _, row in labeled.iterrows():
        ax.annotate(
            str(row["Feature"]),
            xy=(row["Prevalence"], row["Entropy"]),
            xytext=(-6, -12),
            textcoords="offset points",
            fontsize=8,
            alpha=0.9,
        )

  # choose zoom limits for bottom-left region
    x1, x2 = 25, 400
    y1, y2 = -0.02, 1.558

    axins = inset_axes(
        ax,
        width="25%",
        height="40%",
        bbox_to_anchor=(-.4, -.3, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=1.75,
    )

    # replot only features in inset
    feature_df = plot_df[plot_df["Type"] == "Feature"]

    axins.scatter(
        feature_df["Prevalence"],
        feature_df["Entropy"],
        alpha=0.7,
        s=20,
        edgecolors="black",
        linewidths=.5

    )

    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)

    # annotate dots inside inset
    inset_df = feature_df[
        feature_df["Prevalence"].between(x1, x2) &
        feature_df["Entropy"].between(y1, y2)
    ].copy()

    # only keep high-prevalence within inset
    inset_df = inset_df[
        inset_df["Prevalence"] >= inset_df["Prevalence"].quantile(0.8)
    ]

    for _, row in inset_df.iterrows():
        axins.annotate(
            str(row["Feature"]),
            xy=(row["Prevalence"], row["Entropy"]),
            xytext=(3, 2),
            textcoords="offset points",
            fontsize=5,
            alpha=0.9,
        )

    axins.tick_params(axis="both", labelsize=7)

    # optional: draw box on main plot showing zoomed region
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.4", lw=0.8)


    return fig, plot_df
