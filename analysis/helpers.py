import numpy as np
import pandas as pd
from scipy import stats

PARTICIPANTS = [
    "p1",
    "p2",
    "p3",
    "p4",
    "p5",
    "p6",
    "p7",
    "p8",
]

# Shared featurizer color palette
PALETTE = {
    "Shuffled Control": "lightgray",
    "Surprisal": "dimgray",
    "JumpReLU": "dodgerblue",
    "Matryoshka": "skyblue",
    "Residual": "firebrick",
}

# Shared dataset label mapping and ordering
DATASET_MAP = {
    "hard_to_process": "Hard to Process",
    "easy_to_process": "Easy to Process",
    "abstract": "Abstract",
    "concrete": "Concrete",
    "ghost": "Ghost",
}
DATASET_ORDER = ["Hard to Process", "Easy to Process", "Abstract", "Concrete", "Ghost"]

FROI_MAP = {1: "IFGorb", 2: "IFG", 3: "MFG", 4: "AntTemp", 5: "PostTemp"}
FROI_ORDER = ["IFGorb", "IFG", "MFG", "AntTemp", "PostTemp"]


def p_to_stars(p):
    if pd.isna(p):
        return ""
    if p < 1e-3: return "***"
    if p < 1e-2: return "**"
    if p < 5e-2: return "*"
    if p < 1e-1: return "."
    return ""

def bonferroni(pvals):
    """
    Implement bonferroni correction.
    Returns adjusted p-values in the same order as input.
    """
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    return np.minimum(pvals * m, 1.0)

def build_path(feature_str, dataset="categories", analysis="regression", standardized=False, layer=12, control=False, feature_idx=""):
    """Helper function to build paths to results from different experimental conditions

    Args:
        feature_str: {Matryoshka, JumpReLU, Residual, Log Probabilties}
        dataset: Dataset to analyze (categories or fROI)
        analysis: Regression or generalization
        standardized: Whether analyses had standardized features, or just standardized betas
        layer: Which layer were the features drawn from
        control: Whether you're querying results from the shuffled control setting
        feature_idx: Whether or not the feature set was restricted to certain indices.

    Returns:
        Path string to the results directory (not including per-participant subfolder or results.csv)
    """
    if feature_str == "Matryoshka":
        sae_or_hidden = "sae"
        sae_release = "gemma-2-2b-res-matryoshka-dc"
    elif feature_str == "JumpReLU":
        sae_or_hidden = "sae"
        sae_release = "gemma-scope-2b-pt-res-canonical"
    elif feature_str == "Residual":
        sae_or_hidden = "hidden_states"
        sae_release = ""
    elif feature_str == "Log Probabilities":
        sae_or_hidden = "logprobs_only"
        layer = 12
        sae_release = ""
    else:
        raise ValueError(f"Unknown feature_str: '{feature_str}'. Expected one of: Matryoshka, JumpReLU, Residual, Log Probabilities")

    if analysis == "generalization":
        analysis_top_level = "generalization_8000"
        analysis_lower_level = "generalization"
    else:
        analysis_top_level = "regressions"
        analysis_lower_level = "regressions"

    if standardized:
        standardized_str = "standardized"
    else:
        standardized_str = "standardized_betas"

    if control:
        control_str = "voxel_control/mean"
    else:
        control_str = "mean"

    return f"../results/{dataset}/full_features/{analysis_top_level}/{sae_or_hidden}/{sae_release}{feature_idx}/{control_str}/{layer}/{standardized_str}/{analysis_lower_level}"


def load_data(
    feature_str,
    participant_id,
    dataset="categories",
    analysis="regression",
    standardized=False,
    layer=12,
    control=False,
    feature_idx="",
    meta_basename="langfroi12345",
    results_suffix="",
):
    """Load a results CSV for one participant, merging fROI metadata if needed.

    Args:
        feature_str: Featurizer name passed to build_path
        participant_id: Participant identifier string
        dataset: "categories" or "fROI" — passed to build_path; if "fROI", the
            participant metadata CSV is merged in to add top10_tstat_langloc_SN_parc_lang_int
        analysis: Passed to build_path
        standardized: Passed to build_path
        layer: Passed to build_path
        control: Passed to build_path
        feature_idx: Passed to build_path
        meta_basename: Basename of the metadata file (without _meta.csv suffix),
            used only when dataset="fROI"; also used to locate the betas CSV for
            deduplication ({meta_basename}_betas.csv)
        results_suffix: Suffix of results file. Try normal results.csv first. If not found, try appending.

    Returns:
        DataFrame with results, plus fROI metadata columns if dataset="fROI".
        For fROI data, voxels with duplicate beta patterns are removed, keeping
        only the first occurrence of each unique beta vector per participant.
    """
    try:
        path = f"{build_path(feature_str, dataset=dataset, analysis=analysis, standardized=standardized, layer=layer, control=control, feature_idx=feature_idx)}/{participant_id}/results.csv"
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Attempting results suffix: results{results_suffix}.csv ...")
        path = f"{build_path(feature_str, dataset=dataset, analysis=analysis, standardized=standardized, layer=layer, control=control, feature_idx=feature_idx)}/{participant_id}/results{results_suffix}.csv"
        df = pd.read_csv(path)

    if dataset == "fROI":
        meta_path = f"../data/processed_csvs_anon/{participant_id}/{meta_basename}_meta.csv"
        meta = pd.read_csv(meta_path)
        df = df.merge(
            meta[["neuroid_id", "top10_tstat_langloc_SN_parc_lang_int"]],
            left_on="neuroid", right_on="neuroid_id", how="left",
        )
        # Deduplicate voxels with identical beta patterns
        betas_path = f"../data/processed_csvs_anon/{participant_id}/{meta_basename}_betas.csv"
        betas = pd.read_csv(betas_path, index_col=0)
        seen = {}
        unique_nids = set()
        for col in betas.columns:
            key = tuple(betas[col].values)
            if key not in seen:
                seen[key] = col
                unique_nids.add(int(col))
        df = df[df["neuroid"].isin(unique_nids)]

    return df


def compute_significance_from_logprob(
    ycol="NC Normalized R Fischer",
    grouping="Dataset",
    correction=True,
    layer=12,
    meta_basename="langfroi12345",
    results_suffix="_langfroi12345_20260311",
):
    # ----------------------------
    # 1) Load and assemble data
    # ----------------------------
    dfs = []
    featurizers = ["Surprisal", "JumpReLU", "Matryoshka", "Residual"]
    _feat_to_load = {"Surprisal": "Log Probabilities"}
    dataset = "categories" if grouping == "Dataset" else "fROI"

    for featurizer in featurizers:
        for participant_id in PARTICIPANTS:
            load_name = _feat_to_load.get(featurizer, featurizer)
            tmp = load_data(load_name, participant_id, dataset=dataset, layer=layer,
                            meta_basename=meta_basename, results_suffix=results_suffix)
            tmp["Featurizer"] = featurizer
            tmp["Participant"] = participant_id
            dfs.append(tmp)

    df = pd.concat(dfs, ignore_index=True)

    if grouping == "Dataset":
        df["Dataset"] = df["dataset"].map(DATASET_MAP)
        df[grouping] = pd.Categorical(df[grouping], categories=DATASET_ORDER, ordered=True)
    else:
        df["fROI"] = df["top10_tstat_langloc_SN_parc_lang_int"].map(FROI_MAP)
        df[grouping] = pd.Categorical(df[grouping], categories=FROI_ORDER, ordered=True)

    df["Featurizer"] = pd.Categorical(df["Featurizer"], categories=featurizers, ordered=True)

    # ----------------------------
    # 2) Aggregate mean + error
    # ----------------------------
    # First aggregate by participant, then compute mean and error across participants.
    df_participant_level = (
        df.groupby(["Participant", grouping, "Featurizer"], as_index=False)[ycol]
        .mean()
    )

    wide = df_participant_level.pivot_table(
        index=["Participant", grouping],
        columns="Featurizer",
        values=ycol,
        aggfunc="mean",

    )
    groups = list(df_participant_level[grouping].dropna().unique())
    results: dict[tuple[str, str], dict] = {}

    for g in groups:
        p_raw = []
        n_list = []

        for feat in ["Matryoshka", "JumpReLU", "Residual"]:
            cols = ["Surprisal", feat]
            sub = wide.xs(g, level=grouping)[cols].dropna()
            n = len(sub)
            n_list.append(n)
            _, p = stats.ttest_rel(sub[feat], sub["Surprisal"])
            p_raw.append(p)

        p_adj = np.array(p_raw, dtype=float)
        if correction:
            p_adj = bonferroni(p_adj)

        for feat, pr, pa, n in zip(["Matryoshka", "JumpReLU", "Residual"], p_raw, p_adj, n_list):
            results[(g, feat)] = {"p_raw": pr, "p_adj": pa, "n": n}

    return results
