#!/usr/bin/env python3
"""Render the four Natural Stories manuscript panels from saved encoding fits.

Inputs use MODEL/held_out_story_results_ROI.csv and, for sae_full,
MODEL/selected_features_ROI.csv. No model fitting or feature extraction occurs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FROIS = ("IFGorb", "IFG", "MFG", "AntTemp", "PostTemp")
STORIES = (1, 2, 3, 4, 5, 6, 7, 9, 10)
SHIFTS = (26, 52, 78, 104)
KEY_MODELS = ("surprisal", "jumprelu", "sae_full", "residual")
BIN_EDGES = (0, 128, 512, 2048, 8192, 32768)
BIN_LABELS = tuple(f"{a}-{b}" for a, b in zip(BIN_EDGES[:-1], BIN_EDGES[1:]))
BUCKET_MODELS = tuple(f"sae_{a}_{b}" for a, b in zip(BIN_EDGES[:-1], BIN_EDGES[1:]))
MODEL_COLORS = dict(control="lightgray", surprisal="dimgray", jumprelu="dodgerblue",
                    sae_full="skyblue", residual="firebrick")
ROI_COLORS = dict(zip(FROIS, ("#FDAE6B", "#E6550D", "#C44E8B", "#8856A7", "#4A1486")))
BIN_COLORS = ("mediumseagreen", "darkturquoise", "plum", "lightcoral", "goldenrod")
STEMS = {
    "a": "key_model_predictivity",
    "b": "fig5b_matryoshka_selected_feature_counts_by_roi",
    "c": "fig5c_matryoshka_bucket_predictivity",
    "d": "full_sae_most_consistently_selected_features",
}


def _require_columns(frame: pd.DataFrame, columns, name: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")


def _check_grid(frame: pd.DataFrame, keys, expected, name: str) -> None:
    expected = set(expected)
    if frame.duplicated(keys).any():
        raise ValueError(f"{name}: duplicate fit keys {keys}")
    observed = set(frame[keys].itertuples(index=False, name=None))
    if observed != expected:
        raise ValueError(f"{name}: incomplete or unexpected fit grid; "
                         f"missing={sorted(expected - observed)}, "
                         f"extra={sorted(observed - expected)}")


def load_inputs(actual_dir: Path, control_dir: Path):
    """Read saved tables; extra legacy models may exist but are not plotted."""
    result_paths = sorted(actual_dir.glob("*/held_out_story_results_*.csv"))
    control_paths = sorted((control_dir / "residual").glob("held_out_story_results_*.csv"))
    support_paths = sorted((actual_dir / "sae_full").glob("selected_features_*.csv"))
    for name, paths in (("actual fits", result_paths), ("residual controls", control_paths),
                        ("full-SAE selected features", support_paths)):
        if not paths:
            raise ValueError(f"No {name} found in the supplied directories")
    return tuple(pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
                 for paths in (result_paths, control_paths, support_paths))


def summarize(actual: pd.DataFrame, control: pd.DataFrame, selected: pd.DataFrame,
              feature_count_targets: str = "frois") -> dict[str, pd.DataFrame]:
    """Compute panel tables with explicit observation units and denominators."""
    if feature_count_targets not in {"frois", "all"}:
        raise ValueError("feature_count_targets must be 'frois' or 'all'")
    _require_columns(actual, ["roi", "model", "held_out_story_id", "r"], "actual")
    _require_columns(control, ["roi", "model", "held_out_story_id", "r",
                               "response_circular_shift_trs"], "control")
    _require_columns(selected, ["roi", "held_out_story_id", "feature_global_index",
                                "ridge_coefficient"], "selected")
    actual, control, selected = actual.copy(), control.copy(), selected.copy()
    for frame in (actual, control, selected):
        frame["held_out_story_id"] = pd.to_numeric(frame.held_out_story_id).astype(int)
    if "response_circular_shift_trs" in actual and actual.response_circular_shift_trs.ne(0).any():
        raise ValueError("Actual results contain shifted-response fits")
    if "response_circular_shift_trs" in selected and selected.response_circular_shift_trs.ne(0).any():
        raise ValueError("Selected-feature input contains shifted-response fits")
    if "model" in selected and selected.model.ne("sae_full").any():
        raise ValueError("Selected-feature input must contain only full-SAE supports")
    needed_models = KEY_MODELS + BUCKET_MODELS
    fit_keys = ["roi", "model", "held_out_story_id"]
    five = actual.loc[actual.roi.isin(FROIS) & actual.model.isin(needed_models)].copy()
    _check_grid(five, fit_keys, ((r, m, s) for r in FROIS for m in needed_models
                               for s in STORIES), "five-fROI actual results")
    if not np.isfinite(five.r).all():
        raise ValueError("Actual correlations must be finite")
    null = control.loc[control.roi.isin(FROIS) & control.model.eq("residual")].copy()
    _check_grid(null, fit_keys + ["response_circular_shift_trs"],
                ((r, "residual", s, shift) for r in FROIS for s in STORIES for shift in SHIFTS),
                "residual controls")
    if not np.isfinite(null.r).all():
        raise ValueError("Control correlations must be finite")

    a_actual = five.loc[five.model.isin(KEY_MODELS), fit_keys + ["r"]].copy()
    a_actual["observation_unit"] = "held_out_story"
    a_null = null.groupby(["roi", "response_circular_shift_trs"], as_index=False).r.mean()
    a_null["model"] = "control"
    a_null["observation_unit"] = "shift_mean_across_nine_stories"
    a_points = pd.concat([a_actual, a_null], ignore_index=True)
    a_summary = a_points.groupby(["roi", "model", "observation_unit"], as_index=False).agg(
        mean=("r", "mean"), sem=("r", "sem"), n_observations=("r", "size"))
    means = a_actual.groupby(["roi", "model"]).r.mean().unstack().reindex(FROIS)
    delta = pd.DataFrame({"roi": FROIS, "surprisal_mean_r": means.surprisal.to_numpy(),
                          "three_model_mean_r": means[["jumprelu", "sae_full", "residual"]].mean(axis=1).to_numpy()})
    delta["delta_r"] = delta.three_model_mean_r - delta.surprisal_mean_r
    delta["percent_increase"] = 100 * delta.delta_r / delta.surprisal_mean_r

    targets = FROIS + (("Lang",) if feature_count_targets == "all" else ())
    d_fits = actual.loc[actual.roi.isin(targets) & actual.model.eq("sae_full")]
    _check_grid(d_fits, fit_keys, ((r, "sae_full", s) for r in targets for s in STORIES),
                "feature-count targets")
    support = selected.loc[selected.roi.isin(targets)].copy()
    if support.duplicated(["roi", "held_out_story_id", "feature_global_index"]).any():
        raise ValueError("Duplicate selected-feature entries")
    if not support.held_out_story_id.isin(STORIES).all():
        raise ValueError("Unexpected story in selected-feature input")
    if "n_lm_features_selected" in d_fits:
        recorded = d_fits.set_index(["roi", "held_out_story_id"])["n_lm_features_selected"]
        observed = support.groupby(["roi", "held_out_story_id"]).size().reindex(recorded.index, fill_value=0)
        if not recorded.eq(observed).all():
            raise ValueError("Selected-feature table does not match recorded support sizes")
    features = pd.to_numeric(support.feature_global_index)
    if not (np.isfinite(features).all() and features.between(0, 32767).all()
            and features.eq(np.floor(features)).all()):
        raise ValueError("Full-SAE feature indices must be integers in [0, 32768)")
    if not np.isfinite(support.ridge_coefficient).all() or support.ridge_coefficient.eq(0).any():
        raise ValueError("Selected-feature directions require finite, nonzero ridge coefficients")
    support["feature_global_index"] = features.astype(int)
    support["feature_bin"] = pd.cut(support.feature_global_index, BIN_EDGES,
                                     labels=BIN_LABELS, right=False)
    counts = support.loc[support.roi.isin(FROIS)].groupby(
        ["roi", "held_out_story_id", "feature_bin"], observed=True).size()
    complete = pd.MultiIndex.from_product([FROIS, STORIES, BIN_LABELS],
                                          names=["roi", "held_out_story_id", "feature_bin"])
    b_points = counts.reindex(complete, fill_value=0).rename("feature_count").reset_index()
    b_summary = b_points.groupby(["roi", "feature_bin"], as_index=False).agg(
        mean=("feature_count", "mean"), sem=("feature_count", "sem"), n_fits=("feature_count", "size"))

    c_points = five.loc[five.model.isin(("surprisal",) + BUCKET_MODELS)].groupby(
        ["model", "held_out_story_id"], as_index=False).agg(r=("r", "mean"), n_frois=("roi", "nunique"))
    c_summary = c_points.groupby("model", as_index=False).agg(
        mean=("r", "mean"), sem=("r", "sem"), n_stories=("r", "size"))
    support["direction"] = np.where(support.ridge_coefficient.gt(0), "positive", "negative")
    d_all = support.groupby(["feature_global_index", "direction"]).size().rename("fits_selected").reset_index()
    d_all = d_all.sort_values(["fits_selected", "feature_global_index", "direction"],
                              ascending=[False, True, True]).reset_index(drop=True)
    d_all["total_fits"] = len(d_fits)
    d_all["selection_percent"] = 100 * d_all.fits_selected / len(d_fits)
    d_all["targets"] = feature_count_targets
    d_by_roi = support.groupby(["roi", "feature_global_index", "direction"]).size().rename("folds_selected").reset_index()
    d_by_roi["total_folds"] = len(STORIES)
    return {"panel_a_points": a_points, "panel_a_summary": a_summary, "panel_a_deltas": delta,
            "panel_b_counts_by_fit": b_points, "panel_b_summary": b_summary,
            "panel_c_points": c_points, "panel_c_summary": c_summary,
            "panel_d_all_signed_features": d_all, "panel_d_top15": d_all.head(15).copy(),
            "panel_d_counts_by_roi": d_by_roi}


def _style(ax, ylabel):
    ax.set_ylabel(ylabel, fontsize=16)
    ax.tick_params(labelsize=14, colors="black")
    ax.spines[["top", "right"]].set_visible(False)


def _save(fig, out, panel):
    for extension in ("pdf", "png", "svg"):
        fig.savefig(out / f"{STEMS[panel]}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def render(tables: dict[str, pd.DataFrame], out: Path) -> None:
    """Save only the four final panels, their source tables, and plot metadata."""
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "ps.fonttype": 42, "svg.fonttype": "none"})

    fig, ax = plt.subplots(figsize=(9.2, 4.9))
    displayed = ("control",) + KEY_MODELS
    labels = ("Circular-shift control", "Surprisal", "JumpReLU", "Matryoshka", "Residual")
    offsets, width, rng = np.linspace(-.30, .30, len(displayed)), .135, np.random.default_rng(19)
    for j, (model, label) in enumerate(zip(displayed, labels)):
        rows = tables["panel_a_summary"].loc[lambda d: d.model.eq(model)].set_index("roi").reindex(FROIS)
        x = np.arange(len(FROIS)) + offsets[j]
        ax.bar(x, rows["mean"], yerr=rows["sem"], width=width, color=MODEL_COLORS[model],
               edgecolor="black", linewidth=1, capsize=0, label=label, zorder=2)
        for i, roi in enumerate(FROIS):
            values = tables["panel_a_points"].loc[lambda d: d.model.eq(model) & d.roi.eq(roi), "r"].to_numpy()
            ax.scatter(x[i] + rng.uniform(-width * .22, width * .22, len(values)), values,
                       s=17, color=MODEL_COLORS[model], edgecolor="black", linewidth=.45, alpha=.70, zorder=3)
    for i, roi in enumerate(FROIS):
        row = tables["panel_a_deltas"].set_index("roi").loc[roi]
        y = tables["panel_a_points"].loc[lambda d: d.roi.eq(roi), "r"].max() + .045
        left, right = i + offsets[2], i + offsets[-1]
        ax.plot([left, left, right, right], [y-.01, y, y, y-.01], color="black", lw=1, clip_on=False)
        ax.text((left+right)/2, y+.007, f"Δ={row.delta_r:.3f} (+{row.percent_increase:.0f}%)",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    _style(ax, "Held-out-story predictivity (r)")
    ax.yaxis.label.set_fontsize(14)
    ax.set_xticks(range(len(FROIS)), FROIS)
    ax.set_xlabel("Language fROI", fontsize=15)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.5, -.19), fontsize=11)
    fig.tight_layout(rect=(0, .08, 1, 1))
    _save(fig, out, "a")

    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    x, width = np.arange(len(BIN_LABELS)), .15
    for j, roi in enumerate(FROIS):
        rows = tables["panel_b_summary"].loc[lambda d: d.roi.eq(roi)].set_index("feature_bin").reindex(BIN_LABELS)
        ax.bar(x+(j-2)*width, rows["mean"], yerr=rows["sem"], width=width,
               color=ROI_COLORS[roi], edgecolor="black", linewidth=1, label=roi, capsize=0)
    ax.set_xticks(x, BIN_LABELS)
    _style(ax, "Average number of selected features")
    ax.yaxis.label.set_fontsize(14)
    ax.tick_params(axis="x", labelsize=12)
    ax.set_xlabel("Matryoshka feature bin", fontsize=15)
    ax.legend(title="fROI", frameon=True, facecolor="white", edgecolor="black", fontsize=11)
    fig.tight_layout()
    _save(fig, out, "b")

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ordered = ("surprisal",) + BUCKET_MODELS
    width = .84 / len(ordered)
    offsets = (np.arange(len(ordered)) - (len(ordered)-1)/2) * width
    c_summary = tables["panel_c_summary"].set_index("model")
    ax.axhline(0, color="black", linestyle="--", lw=1)
    ax.axhline(c_summary.loc["surprisal", "mean"], color="black", linestyle="--", lw=1.2)
    for j, (model, color, label) in enumerate(zip(ordered, ("dimgray",)+BIN_COLORS, ("Surprisal",)+BIN_LABELS)):
        row = c_summary.loc[model]
        ax.bar(offsets[j], row["mean"], yerr=row["sem"], width=width, color=color,
               edgecolor="black", linewidth=1, label=label, capsize=0, zorder=3)
        values = tables["panel_c_points"].loc[lambda d: d.model.eq(model)].sort_values("held_out_story_id").r.to_numpy()
        jitter = np.random.default_rng(1000+j).uniform(-.15*width, .15*width, len(values))
        ax.scatter(offsets[j]+jitter, values, s=25, facecolor=color, edgecolor="black",
                   linewidth=.5, alpha=.8, zorder=4)
    _style(ax, "Held-out-story predictivity (r)")
    ax.set_xticks([0], ["All 5 Language fROIs"], fontsize=16)
    ax.set_xlim(-.60, .60)
    ax.legend(title="Feature Bin Used in Regression", frameon=False, loc="center left",
              bbox_to_anchor=(1.02, .5), fontsize=11, title_fontsize=11)
    fig.tight_layout()
    _save(fig, out, "c")

    top = tables["panel_d_top15"]
    if top.empty:
        raise ValueError("No selected SAE features available for panel D")
    total = int(top.total_fits.iloc[0])
    fig, ax = plt.subplots(figsize=(12.4, 2.25))
    values = top.fits_selected.to_numpy()[None, :]
    im = ax.imshow(values, cmap="YlOrRd", vmin=0, vmax=total, aspect="equal")
    for j, value in enumerate(values.ravel()):
        ax.text(j, 0, str(int(value)), ha="center", va="center", fontsize=12, fontweight="bold",
                color="white" if value >= .7*total else "black")
    labels = [f"{r.feature_global_index} ({'+' if r.direction == 'positive' else '−'})"
              for r in top.itertuples()]
    ax.set_xticks(range(len(top)), labels, fontsize=11)
    ax.set_yticks([0], ["Natural Stories"], fontsize=12)
    ax.set_xlabel("Feature (signed)", fontsize=14)
    target_label = "five fROIs + network average" if top.targets.iloc[0] == "all" else "five fROIs"
    ax.set_title(f"Most consistently selected full-SAE features ({target_label})", fontsize=13, pad=9)
    ax.tick_params(length=0)
    cbar = fig.colorbar(im, ax=ax, fraction=.025, pad=.02)
    cbar.set_label(f"Fits (max = {total})", fontsize=12)
    fig.tight_layout()
    _save(fig, out, "d")
    metadata = {"font": "DejaVu Sans", "pdf_fonttype": 42,
                "feature_count_targets": str(top.targets.iloc[0]), "feature_count_total_fits": total,
                "panel_a_actual_unit": "held-out story (9)",
                "panel_a_control_unit": "residual-model shift mean across nine stories (4)",
                "panel_c_unit": "held-out story, averaging five individual fROI scores first (9)"}
    (out / "plot_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual-dir", type=Path, required=True)
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-count-targets", choices=("frois", "all"), default="frois",
                        help="Panel D: five fROIs (45 fits; default), or add the network average (54; legacy)")
    args = parser.parse_args()
    tables = summarize(*load_inputs(args.actual_dir, args.control_dir), args.feature_count_targets)
    render(tables, args.output_dir)
    print(f"Saved four panels and source tables to {args.output_dir}")


if __name__ == "__main__":
    main()
