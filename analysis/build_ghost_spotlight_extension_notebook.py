"""Build the standalone Ghost spotlight extension analysis notebook.

The generated notebook contains all analysis code and prose. This builder is
only a convenient way to maintain the notebook JSON and is not imported or
required when the notebook is executed.
"""

import os
from pathlib import Path

import nbformat as nbf


NOTEBOOK_PATH = Path(
    os.environ.get(
        "GHOST_NOTEBOOK_OUTPUT",
        Path(__file__).with_name("ghost_spotlight_extension_2026-09-22.ipynb"),
    )
)


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    md(
        r"""
# Extending the people-centered Ghost analysis

This notebook reproduces the full extension of the Ghost analysis described in
the reviewer response. It:

1. Loads the held-out `ghost_additional` voxel results.
2. Applies the prediction-quality and feature-direction criteria.
3. Reports how the retained voxels are distributed across participants.
4. Quantifies overlap with an independent functional LTPJ parcel before and
   after feature-direction alignment.
5. Audits the strict 10/10-aligned subset.
6. Tests robustness to alternative raw and noise-ceiling-normalized prediction
   thresholds and to alternative direction thresholds.

The original 10 Matryoshka SAE features and their directions are held fixed
throughout. No features are selected using the additional Ghost voxels.
"""
    ),
    code(
        r"""
import os
from pathlib import Path

from IPython import get_ipython

ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic("matplotlib", "inline")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
pd.set_option("display.max_columns", 100)
pd.set_option("display.precision", 3)

# The notebook works when launched from either the repository root or analysis/.
CWD = Path.cwd().resolve()
if (CWD / "data").exists() and (CWD / "analysis").exists():
    REPO_ROOT = CWD
elif CWD.name == "analysis" and (CWD.parent / "data").exists():
    REPO_ROOT = CWD.parent
else:
    raise RuntimeError(
        "Run this notebook from the repository root or analysis/ directory."
    )

PUBLIC_REPO = REPO_ROOT
RESULTS_DIR = (
    PUBLIC_REPO
    / "results_anon/categories/spotlight/sae/"
      "gemma-2-2b-res-matryoshka-dc/mean/12/raw/regressions"
)
META_DIR = REPO_ROOT / "data/processed_csvs_anon"
SURFACE_ASSET_DIR = Path(os.environ.get(
    "GHOST_SURFACE_ASSET_DIR", REPO_ROOT / "analysis/ghost_surface_assets"
)).expanduser()
SAXE_LTPJ_PATH = (
    SURFACE_ASSET_DIR / "lh.Saxe_Dufour2013_LTPJ_fsaverage7_nearest.gii"
)

PARTICIPANTS = [f"p{i}" for i in range(1, 9)]
PEOPLE_FEATURES = [40, 94, 79, 44, 49, 64, 199, 16, 669, 884]
PEOPLE_DIRECTIONS = [1, 1, -1, 1, 1, 1, 1, -1, -1, 1]
PGI_TPOJ = ["PGi", "TPOJ1", "TPOJ2", "TPOJ3"]

PREDICTION_COLUMN = "NC Normalized R Fischer"
PREDICTION_THRESHOLD = 0.50
RAW_PREDICTION_THRESHOLD = 0.25
DIRECTION_THRESHOLD = 7.0
MAX_LOW_CONTRIBUTION = 3  # participants with <=3 aligned voxels are omitted from anatomy

print(f"Spotlight results: {RESULTS_DIR.relative_to(REPO_ROOT)}")
"""
    ),
    md(
        r"""
## Question and held-out voxel pool

We performed a substantial extension of the Ghost analysis. Specifically, we
asked: can the people-centered signature that we discovered from the original
32-voxel cluster, using the feature-identification method in Study 2, be used
to find additional voxels with similar tuning?

We first fixed the original people-centered signature: 10 Matryoshka SAE
features and their directions. We then extracted all voxels across the eight
participants that satisfied the original Ghost criterion—NCSNR \(>0.4\) and
correlations with both main Sentence PCs within \((-0.14,0.14)\). This yielded
1,490 additional voxels that did not overlap with the original Ghost voxels.
"""
    ),
    code(
        r"""
frames = []
for participant in PARTICIPANTS:
    path = RESULTS_DIR / participant / "results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing spotlight results: {path}")
    frame = pd.read_csv(path)
    frame["Participant"] = participant
    frames.append(frame)

all_results = pd.concat(frames, ignore_index=True)
ghost = all_results[all_results["dataset"].eq("ghost_additional")].copy()
comparison = all_results[
    all_results["dataset"].isin(["abstract", "concrete"])
].copy()

assert len(ghost) == 1_490
assert len(comparison) == 320
assert ghost["neuroid"].nunique() < len(ghost)  # IDs repeat across participants
assert not ghost.duplicated(["Participant", "neuroid"]).any()
np.testing.assert_allclose(
    ghost["NC Normalized R Fischer"],
    ghost["R Fischer"] / ghost["Noise Ceiling"],
)

# Load the independent Saxe/Dufour functional LTPJ surface parcel without
# requiring nibabel in the notebook kernel.
import base64
import xml.etree.ElementTree as ET
import zlib


def read_gifti_float_array(path):
    arrays = ET.parse(path).getroot().findall("DataArray")
    if len(arrays) != 1:
        raise ValueError("Expected exactly one GIFTI DataArray")
    data_array = arrays[0]
    if data_array.get("DataType") not in ("NIFTI_TYPE_FLOAT32", "16"):
        raise ValueError("Expected a float32 GIFTI DataArray")
    if data_array.get("Encoding") != "GZipBase64Binary":
        raise ValueError("Expected GZipBase64Binary GIFTI encoding")
    endian = data_array.get("Endian")
    if endian not in ("LittleEndian", "BigEndian"):
        raise ValueError("Expected LittleEndian or BigEndian GIFTI byte order")
    if data_array.get("Dimensionality") != "1":
        raise ValueError("Expected a one-dimensional GIFTI surface array")
    data = data_array.find("Data")
    if data is None or data.text is None:
        raise ValueError("Missing GIFTI Data payload")
    compressed = base64.b64decode("".join(data.text.split()), validate=True)
    byte_order = "<" if endian == "LittleEndian" else ">"
    values = np.frombuffer(zlib.decompress(compressed), dtype=byte_order + "f4")
    if len(values) != int(data_array.attrib["Dim0"]):
        raise ValueError("GIFTI payload length does not match Dim0")
    return values


def validate_binary_surface(values, expected_size=163_842):
    values = np.asarray(values)
    if values.shape != (expected_size,):
        raise ValueError("Unexpected LTPJ surface shape")
    if not np.isfinite(values).all() or not np.isin(values, [0, 1]).all():
        raise ValueError("LTPJ surface values must be finite and binary (0 or 1)")
    return values.astype(bool)


def validated_vertices(values, expected_size=163_842):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("Vertex IDs must be one-dimensional")
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValueError("Vertex IDs must be finite integers")
    if np.any(values < 0) or np.any(values >= expected_size):
        raise ValueError("Vertex IDs are outside the fsaverage7 surface range")
    return values.astype(np.int64)


if not SAXE_LTPJ_PATH.exists():
    raise FileNotFoundError(
        "See analysis/saxe_ltpj_surface_transform/README.md for the canonical "
        "neuromaps transform and expected figure assets."
    )
saxe_ltpj = validate_binary_surface(read_gifti_float_array(SAXE_LTPJ_PATH))
assert saxe_ltpj.shape == (163_842,)
assert int(saxe_ltpj.sum()) == 6_587
ghost["neuroid"] = validated_vertices(ghost["neuroid"])
comparison["neuroid"] = validated_vertices(comparison["neuroid"])
ghost["Inside Saxe functional LTPJ"] = saxe_ltpj[
    ghost["neuroid"].to_numpy()
]

ghost["Well predicted"] = ghost[PREDICTION_COLUMN] >= PREDICTION_THRESHOLD
ghost["Feature direction-aligned"] = (
    ghost["Well predicted"]
    & (
        ghost["Average Num Features Agree Direction"]
        >= DIRECTION_THRESHOLD
    )
)

well_predicted = ghost[ghost["Well predicted"]].copy()
aligned = ghost[ghost["Feature direction-aligned"]].copy()

print(f"Additional Ghost candidates: {len(ghost):,}")
print(
    f"Well predicted (NC-normalized R Fisher >= {PREDICTION_THRESHOLD}): "
    f"{len(well_predicted):,}"
)
print(
    f"Feature direction-aligned (>= {DIRECTION_THRESHOLD:g}/10): "
    f"{len(aligned):,}"
)
"""
    ),
    md(
        r"""
## Prediction and feature-direction criteria

Of the 1,490 additional voxels, 275 were well predicted by the fixed
10-feature model, using a noise-ceiling-normalized cross-validated threshold
of \(r_{\mathrm{NC}}\geq0.50\). Here,
\(r_{\mathrm{NC}}=r_{\mathrm{Fisher}}/r_{\mathrm{ceiling}}\), so the model
captures at least half of each voxel's estimated predictable correlation.

We next asked whether the features predicted these voxels in the same direction
as in the original cluster. We required at least 7 of the 10 coefficient
directions to agree, on average across the cross-validation folds. This left
181 **feature direction-aligned** voxels. For comparison, none of the 320
Abstract or Concrete voxels reached 7/10 directional agreement. This suggests
that the alignment is not simply a consequence of being well predicted by the
10 features and is not shared with these comparison voxel populations.
"""
    ),
    code(
        r"""
comparison_max_direction = comparison[
    "Average Num Features Agree Direction"
].max()
comparison_at_primary_gate = comparison[
    comparison["Average Num Features Agree Direction"] >= DIRECTION_THRESHOLD
]

funnel = pd.DataFrame(
    {
        "Analysis stage": [
            "Additional Ghost candidates",
            (
                "Well predicted "
                f"(NC-normalized R Fisher >= {PREDICTION_THRESHOLD})"
            ),
            f"Feature direction-aligned (>= {DIRECTION_THRESHOLD:g}/10)",
        ],
        "Voxels": [len(ghost), len(well_predicted), len(aligned)],
    }
)
funnel["Percent of initial pool"] = 100 * funnel["Voxels"] / len(ghost)
display(funnel)

print(
    "Maximum direction agreement in Abstract/Concrete comparison voxels: "
    f"{comparison_max_direction:.1f}/10"
)
print(
    f"Comparison voxels reaching {DIRECTION_THRESHOLD:g}/10: "
    f"{len(comparison_at_primary_gate)}"
)
"""
    ),
    md(
        r"""
## Distribution across participants

The additional analysis does not suggest that the feature-direction alignment
is present equally strongly in every participant. Rather, it is clearly
present in a subset: p1, p2, p3, p5, and p8 each contribute multiple
additional feature direction-aligned voxels, whereas p4, p6, and p7 contribute
three, zero, and zero voxels, respectively.
"""
    ),
    code(
        r"""
participant_rows = []
for participant in PARTICIPANTS:
    participant_ghost = ghost[ghost["Participant"].eq(participant)]
    participant_well = well_predicted[
        well_predicted["Participant"].eq(participant)
    ]
    participant_aligned = aligned[aligned["Participant"].eq(participant)]
    participant_rows.append(
        {
            "Participant": participant,
            "Additional candidates": len(participant_ghost),
            "Well-predicted voxels": len(participant_well),
            "Well-predicted share (%)": (
                100 * len(participant_well) / len(well_predicted)
            ),
            "Feature direction-aligned voxels": len(participant_aligned),
            "Aligned-set contribution (%)": (
                100 * len(participant_aligned) / len(aligned)
            ),
        }
    )

participant_table = pd.DataFrame(participant_rows)
display(
    participant_table.style.format(
        {
            "Well-predicted share (%)": "{:.1f}",
            "Aligned-set contribution (%)": "{:.1f}",
        }
    )
)

aligned_counts = participant_table.set_index("Participant")[
    "Feature direction-aligned voxels"
]
ANATOMY_PARTICIPANTS = aligned_counts[
    aligned_counts > MAX_LOW_CONTRIBUTION
].index.tolist()
LOW_CONTRIBUTORS = aligned_counts[
    aligned_counts <= MAX_LOW_CONTRIBUTION
].index.tolist()

low_contribution_n = int(aligned_counts.loc[LOW_CONTRIBUTORS].sum())
print(f"Participants retained for anatomy: {ANATOMY_PARTICIPANTS}")
print(f"Minimal contributors omitted from anatomy: {LOW_CONTRIBUTORS}")
print(
    f"Minimal contributors account for {low_contribution_n}/{len(aligned)} "
    f"voxels ({100 * low_contribution_n / len(aligned):.1f}%)."
)
"""
    ),
    md(
        r"""
p4, p6, and p7 together contribute only 3 of the 181 feature
direction-aligned voxels, or 1.7%. We therefore exclude these three
participants from the following anatomical summary, while reporting their
results in the participant-level analysis above.
"""
    ),
    md(
        r"""
## Anatomical composition

Our primary anatomical reference is the independent Saxe/Dufour functional
LTPJ parcel. Anatomy played no role in selecting the Ghost candidates,
screening them for prediction quality, or measuring feature-direction
agreement.

The three minimal contributors account for only three aligned voxels, leaving
**178 aligned voxels** in the five primary contributing participants. Of
these, 143 (80.3%) fall inside the functional LTPJ parcel. In contrast, only
20 of the 86 well-predicted but non-aligned voxels (23.3%) and 372 of all
1,284 additional Ghost candidates (29.0%) from the same participants fall
inside the parcel.

The anatomical Glasser PGi/TPOJ1/2/3 analysis is retained as a supplement.
"""
    ),
    code(
        r"""
core_ghost = ghost[
    ghost["Participant"].isin(ANATOMY_PARTICIPANTS)
].copy()
core_well = well_predicted[
    well_predicted["Participant"].isin(ANATOMY_PARTICIPANTS)
].copy()
core_aligned = aligned[
    aligned["Participant"].isin(ANATOMY_PARTICIPANTS)
].copy()
core_well_nonaligned = core_well[
    ~core_well["Feature direction-aligned"]
].copy()

candidate_ltpj = int(core_ghost["Inside Saxe functional LTPJ"].sum())
well_nonaligned_ltpj = int(
    core_well_nonaligned["Inside Saxe functional LTPJ"].sum()
)
aligned_ltpj = int(core_aligned["Inside Saxe functional LTPJ"].sum())

def count_and_percent(n, denominator):
    return f"{n:,} ({100 * n / denominator:.1f}%)"


anatomy_table = pd.DataFrame(
    [
        {
            "Anatomical location": "Inside functional LTPJ",
            "All additional Ghost candidates": count_and_percent(
                candidate_ltpj, len(core_ghost)
            ),
            "Well-predicted but non-aligned": count_and_percent(
                well_nonaligned_ltpj, len(core_well_nonaligned)
            ),
            "Well-predicted and aligned": count_and_percent(
                aligned_ltpj, len(core_aligned)
            ),
        },
        {
            "Anatomical location": "Outside functional LTPJ",
            "All additional Ghost candidates": count_and_percent(
                len(core_ghost) - candidate_ltpj, len(core_ghost)
            ),
            "Well-predicted but non-aligned": count_and_percent(
                len(core_well_nonaligned) - well_nonaligned_ltpj,
                len(core_well_nonaligned),
            ),
            "Well-predicted and aligned": count_and_percent(
                len(core_aligned) - aligned_ltpj, len(core_aligned)
            ),
        },
        {
            "Anatomical location": "Total",
            "All additional Ghost candidates": count_and_percent(
                len(core_ghost), len(core_ghost)
            ),
            "Well-predicted but non-aligned": count_and_percent(
                len(core_well_nonaligned), len(core_well_nonaligned)
            ),
            "Well-predicted and aligned": count_and_percent(
                len(core_aligned), len(core_aligned)
            ),
        },
    ]
)
display(anatomy_table.style.hide(axis="index"))

assert len(core_ghost) == 1_284
assert len(core_well) == 264
assert len(core_aligned) == 178
assert len(core_well_nonaligned) == 86
assert candidate_ltpj == 372
assert well_nonaligned_ltpj == 20
assert aligned_ltpj == 143
"""
    ),
    md(
        r"""
Thus, the feature-direction alignment step substantially changes the
anatomical composition of the set. The aligned voxels are 3.5 times as likely
to fall inside functional LTPJ as the well-predicted but non-aligned voxels
(80.3% versus 23.3%). They are also substantially more concentrated there
than the complete eligible Ghost pool (29.0%). This provides an anatomical
validation of the feature-defined population using a parcel that was never
used during voxel selection.

For completeness, including all eight participants gives 79.0%, rather than
80.3%, of the feature direction-aligned voxels falling inside
the functional LTPJ parcel. We treat the parcel as a functional anatomical
reference, not as evidence that these voxels were independently localized as
theory-of-mind cortex in our participants.
"""
    ),
    md(
        r"""
<!-- ghost-ltpj-reviewer-tables-2026-11-09 -->
## Functional LTPJ: threshold sensitivity and participant counts

The prediction and agreement thresholds are descriptive heuristics, not
significance tests. NC-normalized (r=0.5) means half the estimated ceiling
**correlation**, not half the explainable variance. Direction agreement is
averaged across the five cross-validation folds.

The first table keeps the original five-participant cohort fixed when changing
either threshold; only the second row includes all eight participants. Both
comparison groups are well predicted: aligned voxels meet the agreement
threshold, and non-aligned voxels fall below it. Changing that threshold also
changes the comparison group. Each fraction is the number inside functional
LTPJ divided by the total in that voxel set. Counts pool participant–vertex
observations, not unique vertices or participant-wise percentages.

The second table reports all eight participants at the original thresholds,
including participants with no aligned voxels. An em dash means the denominator
is zero, not 0% overlap. The pooled result need not hold equally within every
participant.
"""
    ),
    code(
        r"""
# ghost-ltpj-reviewer-tables-2026-11-09
def reviewer_ltpj_percent(inside, total):
    return 100 * inside / total if total else np.nan


def reviewer_ltpj_fraction(inside, total):
    if not total:
        return "—"
    return f"{inside:,}/{total:,} ({reviewer_ltpj_percent(inside, total):.1f}%)"


def reviewer_ltpj_counts(frame, prediction_threshold, agreement_threshold):
    predicted = frame["NC Normalized R Fischer"].ge(prediction_threshold)
    agreement = frame["Average Num Features Agree Direction"].ge(agreement_threshold)
    aligned_mask = predicted & agreement
    nonaligned_mask = predicted & ~agreement
    aligned_frame = frame.loc[aligned_mask]
    nonaligned_frame = frame.loc[nonaligned_mask]
    aligned_inside = int(aligned_frame["Inside Saxe functional LTPJ"].sum())
    nonaligned_inside = int(nonaligned_frame["Inside Saxe functional LTPJ"].sum())
    assert int(predicted.sum()) == len(aligned_frame) + len(nonaligned_frame)
    return {
        "candidate_count": len(frame),
        "well_predicted_count": int(predicted.sum()),
        "aligned_count": len(aligned_frame),
        "aligned_ltpj_count": aligned_inside,
        "aligned_ltpj_percent": reviewer_ltpj_percent(aligned_inside, len(aligned_frame)),
        "well_predicted_nonaligned_count": len(nonaligned_frame),
        "well_predicted_nonaligned_ltpj_count": nonaligned_inside,
        "well_predicted_nonaligned_ltpj_percent": reviewer_ltpj_percent(
            nonaligned_inside, len(nonaligned_frame)
        ),
    }


def build_reviewer_ltpj_tables(frame, baseline_participants, participants):
    # Freeze this cohort once; do not reselect contributors at other thresholds.
    baseline_participants = tuple(baseline_participants)
    participants = tuple(participants)
    specifications = [
        ("Original", 0.5, 7, baseline_participants),
        ("No participant exclusion", 0.5, 7, participants),
        ("Lower predictivity", 0.4, 7, baseline_participants),
        ("Higher predictivity", 0.6, 7, baseline_participants),
        ("Lower agreement", 0.5, 6, baseline_participants),
        ("Higher agreement", 0.5, 8, baseline_participants),
    ]
    sensitivity_rows = []
    for analysis, prediction_threshold, agreement_threshold, cohort in specifications:
        subset = frame.loc[frame["Participant"].isin(cohort)]
        sensitivity_rows.append({
            "analysis": analysis,
            "nc_normalized_r_threshold": prediction_threshold,
            "mean_direction_agreement_threshold": agreement_threshold,
            "participants": ";".join(cohort),
            "participant_count": len(cohort),
            **reviewer_ltpj_counts(subset, prediction_threshold, agreement_threshold),
        })
    participant_rows = []
    for participant in participants:
        subset = frame.loc[frame["Participant"].eq(participant)]
        participant_rows.append({
            "participant": participant.upper(),
            "row_type": "participant",
            "nc_normalized_r_threshold": 0.5,
            "mean_direction_agreement_threshold": 7,
            **reviewer_ltpj_counts(subset, 0.5, 7),
        })
    participant_rows.append({
        "participant": "All eight (pooled)",
        "row_type": "pooled_total",
        "nc_normalized_r_threshold": 0.5,
        "mean_direction_agreement_threshold": 7,
        **reviewer_ltpj_counts(frame.loc[frame["Participant"].isin(participants)], 0.5, 7),
    })
    return pd.DataFrame(sensitivity_rows), pd.DataFrame(participant_rows)


ltpj_threshold_sensitivity, ltpj_participant_counts = build_reviewer_ltpj_tables(
    ghost, ANATOMY_PARTICIPANTS, PARTICIPANTS
)

sensitivity_display = pd.DataFrame({
    "Analysis": ltpj_threshold_sensitivity["analysis"],
    "NC-normalized r ≥": ltpj_threshold_sensitivity["nc_normalized_r_threshold"],
    "Mean agreement ≥ (of 10)": ltpj_threshold_sensitivity["mean_direction_agreement_threshold"],
    "Participant cohort": ltpj_threshold_sensitivity["participants"].str.upper().str.replace(";", ", "),
    "Aligned in LTPJ, n/N (%)": [
        reviewer_ltpj_fraction(row.aligned_ltpj_count, row.aligned_count)
        for row in ltpj_threshold_sensitivity.itertuples()
    ],
    "Well-predicted non-aligned in LTPJ, n/N (%)": [
        reviewer_ltpj_fraction(row.well_predicted_nonaligned_ltpj_count, row.well_predicted_nonaligned_count)
        for row in ltpj_threshold_sensitivity.itertuples()
    ],
})
display(sensitivity_display.style.hide(axis="index").format({
    "NC-normalized r ≥": "{:.1f}", "Mean agreement ≥ (of 10)": "{:.0f}",
}))

participant_display = pd.DataFrame({
    "Participant": ltpj_participant_counts["participant"],
    "Candidates": ltpj_participant_counts["candidate_count"],
    "Well predicted": ltpj_participant_counts["well_predicted_count"],
    "Aligned": ltpj_participant_counts["aligned_count"],
    "Aligned in LTPJ, n/N (%)": [
        reviewer_ltpj_fraction(row.aligned_ltpj_count, row.aligned_count)
        for row in ltpj_participant_counts.itertuples()
    ],
    "Well-predicted non-aligned in LTPJ, n/N (%)": [
        reviewer_ltpj_fraction(row.well_predicted_nonaligned_ltpj_count, row.well_predicted_nonaligned_count)
        for row in ltpj_participant_counts.itertuples()
    ],
})
display(participant_display.style.hide(axis="index"))

reviewer_table_dir = REPO_ROOT / "analysis/figures"
reviewer_table_dir.mkdir(parents=True, exist_ok=True)
ltpj_threshold_sensitivity.to_csv(
    reviewer_table_dir / "ghost_ltpj_threshold_sensitivity.csv", index=False, na_rep="NA"
)
ltpj_participant_counts.to_csv(
    reviewer_table_dir / "ghost_ltpj_participant_counts.csv", index=False, na_rep="NA"
)
print("Saved raw counts and percentages: ghost_ltpj_threshold_sensitivity.csv and ghost_ltpj_participant_counts.csv")
"""
    ),
    code(
        r"""
# Composition figure used to keep all denominators explicit.
participant_colors = dict(
    zip(PARTICIPANTS, plt.cm.tab10(np.linspace(0, 0.8, len(PARTICIPANTS))))
)
COLOR_POOL = "#C8CDD3"
COLOR_PREDICTED = "#4C97C2"
COLOR_ALIGNED = "#7651A3"
COLOR_OTHER = "#D0D3D6"

fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
ax_a, ax_b, ax_c, ax_d = axes.ravel()

# A: funnel
stage_counts = [len(ghost), len(well_predicted), len(aligned)]
stage_labels = [
    "Additional Ghost candidates",
    f"Well predicted: NC-normalized r >= {PREDICTION_THRESHOLD}",
    f"Feature direction-aligned: >= {DIRECTION_THRESHOLD:g}/10",
]
stage_percent = 100 * np.array(stage_counts) / len(ghost)
ax_a.barh(
    np.arange(3),
    stage_percent,
    color=[COLOR_POOL, COLOR_PREDICTED, COLOR_ALIGNED],
)
ax_a.set_yticks(np.arange(3), stage_labels)
ax_a.invert_yaxis()
ax_a.set_xlabel("Percentage of initial candidate pool")
for y, n, pct in zip(range(3), stage_counts, stage_percent):
    ax_a.text(pct + 1, y, f"{n:,} ({pct:.1f}%)", va="center")
ax_a.set_xlim(0, 110)
ax_a.set_title("A. Analysis funnel", loc="left", fontweight="bold")

# B: participant composition of the two retained sets
for y, (label, frame) in enumerate(
    [("Well predicted", well_predicted), ("Feature direction-aligned", aligned)]
):
    left = 0
    for participant in PARTICIPANTS:
        n = frame["Participant"].eq(participant).sum()
        pct = 100 * n / len(frame)
        ax_b.barh(
            y, pct, left=left, color=participant_colors[participant], height=0.55
        )
        if pct >= 4:
            ax_b.text(
                left + pct / 2,
                y,
                f"{participant}\n{pct:.1f}%",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
            )
        left += pct
ax_b.set_yticks([0, 1], ["Well predicted\n(n=275)", "Direction-aligned\n(n=181)"])
ax_b.invert_yaxis()
ax_b.set_xlim(0, 100)
ax_b.set_xlabel("Percentage composition")
ax_b.set_title("B. Participant composition", loc="left", fontweight="bold")

# C: joint prediction and direction space, including comparison voxels
nonaligned = ghost.drop(aligned.index)
ax_c.scatter(
    nonaligned[PREDICTION_COLUMN],
    nonaligned["Average Num Features Agree Direction"],
    s=12,
    alpha=0.20,
    color="#777777",
    linewidths=0,
    label="Additional Ghost",
)
for dataset, color, label in [
    ("abstract", "#D870AD", "Abstract comparison"),
    ("concrete", "#63A65F", "Concrete comparison"),
]:
    frame = comparison[comparison["dataset"].eq(dataset)]
    ax_c.scatter(
        frame[PREDICTION_COLUMN],
        frame["Average Num Features Agree Direction"],
        s=17,
        alpha=0.45,
        color=color,
        linewidths=0,
        label=label,
    )
ax_c.scatter(
    aligned[PREDICTION_COLUMN],
    aligned["Average Num Features Agree Direction"],
    s=18,
    alpha=0.75,
    color=COLOR_ALIGNED,
    linewidths=0,
    label="Feature direction-aligned",
)
ax_c.axvline(PREDICTION_THRESHOLD, color="black", ls="--", lw=1)
ax_c.axhline(DIRECTION_THRESHOLD, color="black", ls="--", lw=1)
ax_c.set_xlabel("Noise-ceiling-normalized cross-validated R Fisher")
ax_c.set_ylabel("Average number of feature directions aligned (of 10)")
ax_c.legend(frameon=False, fontsize=8)
ax_c.set_title("C. Prediction and feature direction", loc="left", fontweight="bold")

# D: anatomical composition for the five contributing participants
for y, (label, frame) in enumerate(
    [
        ("Final feature direction-aligned", core_aligned),
        ("Well-predicted but non-aligned", core_well_nonaligned),
    ]
):
    inside_n = int(frame["Inside Saxe functional LTPJ"].sum())
    inside_pct = 100 * inside_n / len(frame)
    outside_n = len(frame) - inside_n
    outside_pct = 100 - inside_pct
    ax_d.barh(y, inside_pct, color=COLOR_ALIGNED, height=0.55)
    ax_d.barh(
        y,
        outside_pct,
        left=inside_pct,
        color=COLOR_OTHER,
        height=0.55,
    )
    ax_d.text(
        inside_pct / 2,
        y,
        f"Functional LTPJ\n{inside_n} ({inside_pct:.1f}%)",
        ha="center",
        va="center",
        fontsize=9,
        color="white",
    )
    ax_d.text(
        inside_pct + outside_pct / 2,
        y,
        f"Outside\n{outside_n} ({outside_pct:.1f}%)",
        ha="center",
        va="center",
        fontsize=9,
    )
ax_d.set_yticks(
    [0, 1],
    [
        f"Final direction-aligned\n(n={len(core_aligned)})",
        f"Well-predicted non-aligned\n(n={len(core_well_nonaligned)})",
    ],
)
ax_d.invert_yaxis()
ax_d.set_xlim(0, 100)
ax_d.set_xlabel("Percentage composition")
ax_d.set_title("D. Anatomical composition", loc="left", fontweight="bold")

fig.suptitle(
    "Held-out search for the people-centered feature directions",
    fontsize=15,
)
fig.tight_layout()
plt.show()
"""
    ),
    md(
        r"""
## Strict 10/10 feature-direction agreement

As a strict descriptive check, we identify voxels for which all 10 feature
directions agree in every cross-validation fold. This is intentionally stricter
than the primary 7/10 criterion and is not used to establish cross-participant
recurrence.
"""
    ),
    code(
        r"""
ten_of_ten = ghost[
    np.isclose(ghost["Average Num Features Agree Direction"], 10.0)
].copy()
ten_of_ten["Passes primary prediction threshold"] = (
    ten_of_ten[PREDICTION_COLUMN] >= PREDICTION_THRESHOLD
)
ten_of_ten = ten_of_ten.sort_values(PREDICTION_COLUMN, ascending=False)

ten_table = ten_of_ten[
    [
        "Participant",
        "neuroid",
        "R Fischer",
        "NC Normalized R Fischer",
        "R Fischer SEM",
        "Parcel Name",
        "Inside Saxe functional LTPJ",
        "Passes primary prediction threshold",
    ]
].reset_index(drop=True)
display(ten_table)

print(f"10/10-aligned voxels: {len(ten_of_ten)}")
print("Participants:", ten_of_ten["Participant"].value_counts().to_dict())
print("Parcels:", ten_of_ten["Parcel Name"].value_counts().to_dict())

assert len(ten_of_ten) == 9
assert set(ten_of_ten["Participant"]) == {"p2"}
assert set(ten_of_ten["Parcel Name"]) == {"PGi"}
assert ten_of_ten["Inside Saxe functional LTPJ"].all()
assert ten_of_ten["Passes primary prediction threshold"].all()
"""
    ),
    code(
        r"""
# Response similarity among the nine strict voxels.
p2_ids = ten_of_ten["neuroid"].astype(int).tolist()
p2_betas = pd.read_csv(
    META_DIR / "p2" / "ghost_additional_betas.csv",
    index_col=0,
)
p2_columns = [str(neuroid) for neuroid in p2_ids]
strict_corr = p2_betas[p2_columns].corr()
upper = strict_corr.values[np.triu_indices(len(strict_corr), 1)]

print(
    "Pairwise response correlation among the nine 10/10 voxels: "
    f"median={np.median(upper):.3f}, "
    f"mean={np.mean(upper):.3f}, "
    f"range=({np.min(upper):.3f}, {np.max(upper):.3f})"
)
display(strict_corr.style.background_gradient(cmap="Purples", vmin=0, vmax=1))
"""
    ),
    md(
        r"""
All nine 10/10 voxels are well predicted, come from p2, and fall inside both
PGi and the independent functional LTPJ parcel. Their response profiles are
also strongly correlated. Perfect agreement
therefore identifies a particularly coherent p2–PGi group, but it does not
establish cross-participant recurrence. The 7/10 criterion is more appropriate
for the main question of whether the broader feature alignment occurs across
people.
"""
    ),
    md(
        r"""
## Supplement: robustness across prediction scales and thresholds

The primary analysis uses \(r_{\mathrm{NC}}\geq0.50\). To show that the
result does not depend on this exact cutoff, we repeat it at one lower
(\(0.40\)) and one higher (\(0.60\)) noise-ceiling-normalized threshold. We
also move the former raw-\(r\) analysis to this supplement and report the
original \(r\geq0.25\) rule with one lower (\(0.20\)) and one higher
(\(0.30\)) threshold.

The first table gives the number and percentage of candidate voxels that are
well predicted, and the number and percentage of those well-predicted voxels
that are feature direction-aligned. The second table gives functional LTPJ
overlap within the five primary contributing participants. Percentages in
that table are within the corresponding well-predicted or feature
direction-aligned set.
"""
    ),
    code(
        r"""
supplemental_rules = [
    {
        "Prediction scale": "NC-normalized R Fisher",
        "Column": "NC Normalized R Fischer",
        "Threshold": 0.40,
        "Role": "Lower",
    },
    {
        "Prediction scale": "NC-normalized R Fisher",
        "Column": "NC Normalized R Fischer",
        "Threshold": 0.50,
        "Role": "Primary",
    },
    {
        "Prediction scale": "NC-normalized R Fisher",
        "Column": "NC Normalized R Fischer",
        "Threshold": 0.60,
        "Role": "Higher",
    },
    {
        "Prediction scale": "Raw R Fisher",
        "Column": "R Fischer",
        "Threshold": 0.20,
        "Role": "Lower",
    },
    {
        "Prediction scale": "Raw R Fisher",
        "Column": "R Fischer",
        "Threshold": 0.25,
        "Role": "Former specification",
    },
    {
        "Prediction scale": "Raw R Fisher",
        "Column": "R Fischer",
        "Threshold": 0.30,
        "Role": "Higher",
    },
]

threshold_rows = []
ltpj_composition_rows = []
supplemental_sets = {}

for rule in supplemental_rules:
    scale = rule["Prediction scale"]
    column = rule["Column"]
    threshold = rule["Threshold"]
    role = rule["Role"]
    rule_key = (scale, threshold)

    threshold_well = ghost[ghost[column] >= threshold].copy()
    threshold_aligned = threshold_well[
        threshold_well["Average Num Features Agree Direction"]
        >= DIRECTION_THRESHOLD
    ].copy()
    supplemental_sets[rule_key] = (threshold_well, threshold_aligned)
    participant_counts = threshold_aligned["Participant"].value_counts()

    threshold_rows.append(
        {
            "Prediction scale": scale,
            "Threshold": threshold,
            "Role": role,
            "Well-predicted voxels": len(threshold_well),
            "Well-predicted candidates (%)": (
                100 * len(threshold_well) / len(ghost)
            ),
            "Feature direction-aligned voxels": len(threshold_aligned),
            "Aligned among well-predicted (%)": (
                100 * len(threshold_aligned) / len(threshold_well)
            ),
            "Participants with >3 aligned voxels": int(
                (participant_counts > MAX_LOW_CONTRIBUTION).sum()
            ),
        }
    )

    for set_name, frame in [
        ("Well predicted", threshold_well),
        ("Feature direction-aligned", threshold_aligned),
    ]:
        core_frame = frame[
            frame["Participant"].isin(ANATOMY_PARTICIPANTS)
        ]
        inside_n = int(core_frame["Inside Saxe functional LTPJ"].sum())
        ltpj_composition_rows.append({
            "Prediction scale": scale,
            "Threshold": threshold,
            "Role": role,
            "Voxel set": set_name,
            "Total voxels": len(core_frame),
            "Inside functional LTPJ n": inside_n,
            "Inside functional LTPJ (%)": (
                100 * inside_n / len(core_frame) if len(core_frame) else np.nan
            ),
        })

threshold_robustness = pd.DataFrame(threshold_rows)
display(
    threshold_robustness.style.format(
        {
            "Threshold": "{:.2f}",
            "Well-predicted candidates (%)": "{:.1f}",
            "Aligned among well-predicted (%)": "{:.1f}",
        }
    )
)

ltpj_robustness = pd.DataFrame(ltpj_composition_rows)
display(
    ltpj_robustness.style.format(
        {
            "Threshold": "{:.2f}",
            "Inside functional LTPJ (%)": "{:.1f}",
        }
    )
)
"""
    ),
    md(
        r"""
The result is stable on both prediction scales. Across the normalized
thresholds \(0.40\), \(0.50\), and \(0.60\), 254, 181, and 104 voxels are
feature direction-aligned across all participants; the functional LTPJ parcel
contains 74.0%, 80.3%, and 89.3% of the aligned voxels in the five primary
contributing participants. Across
raw thresholds \(0.20\), \(0.25\), and \(0.30\), 290, 245, and 191 voxels are
aligned across all participants, and the corresponding functional LTPJ
percentages are 72.0%, 75.9%, and 77.7%. The same five participants contribute
more than three aligned voxels under every specification.
"""
    ),
    md(
        r"""
## Robustness to the feature-direction threshold

Next, we hold the primary prediction threshold at
\(r_{\mathrm{NC}}\geq0.50\) and vary the required average directional
agreement. The comparison-voxel column shows how many Abstract or Concrete
voxels pass the same prediction and direction gates.
"""
    ),
    code(
        r"""
direction_thresholds = [6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
direction_robustness_rows = []

comparison_well = comparison[
    comparison[PREDICTION_COLUMN] >= PREDICTION_THRESHOLD
]
for threshold in direction_thresholds:
    threshold_aligned = well_predicted[
        well_predicted["Average Num Features Agree Direction"] >= threshold
    ]
    threshold_comparison = comparison_well[
        comparison_well["Average Num Features Agree Direction"] >= threshold
    ]
    participant_counts = threshold_aligned["Participant"].value_counts()
    threshold_core = threshold_aligned[
        threshold_aligned["Participant"].isin(ANATOMY_PARTICIPANTS)
    ]
    direction_robustness_rows.append(
        {
            "Direction threshold": threshold,
            "Feature direction-aligned voxels": len(threshold_aligned),
            "Abstract/Concrete comparison voxels": len(threshold_comparison),
            "Participants with >3 aligned voxels": int(
                (participant_counts > MAX_LOW_CONTRIBUTION).sum()
            ),
            "Functional LTPJ among primary participants (%)": (
                100
                * threshold_core["Inside Saxe functional LTPJ"].mean()
                if len(threshold_core)
                else np.nan
            ),
            **{
                participant: int(participant_counts.get(participant, 0))
                for participant in PARTICIPANTS
            },
        }
    )

direction_robustness = pd.DataFrame(direction_robustness_rows)
display(
    direction_robustness.style.format(
        {
            "Direction threshold": "{:.1f}",
            "Functional LTPJ among primary participants (%)": "{:.1f}",
        }
    )
)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(
    direction_robustness["Direction threshold"],
    direction_robustness["Feature direction-aligned voxels"],
    marker="o",
    color=COLOR_ALIGNED,
    label="Additional Ghost",
)
axes[0].plot(
    direction_robustness["Direction threshold"],
    direction_robustness["Abstract/Concrete comparison voxels"],
    marker="o",
    color="#666666",
    label="Abstract/Concrete",
)
axes[0].axvline(DIRECTION_THRESHOLD, color="black", ls="--", lw=1)
axes[0].set_xlabel("Required average feature directions aligned")
axes[0].set_ylabel("Number of voxels")
axes[0].legend(frameon=False)

for participant in PARTICIPANTS:
    axes[1].plot(
        direction_robustness["Direction threshold"],
        direction_robustness[participant],
        marker="o",
        label=participant,
        color=participant_colors[participant],
    )
axes[1].axvline(DIRECTION_THRESHOLD, color="black", ls="--", lw=1)
axes[1].set_xlabel("Required average feature directions aligned")
axes[1].set_ylabel("Feature direction-aligned voxels")
axes[1].legend(frameon=False, ncol=2, fontsize=8)
fig.tight_layout()
plt.show()
"""
    ),
    md(
        r"""
The 7/10 criterion is a simple supermajority rule and is also the first
threshold at which no Abstract or Concrete comparison voxel remains. The same
five principal participants contribute more than three voxels through 7.5/10.
At 8/10, p8 falls to three voxels; at 9/10 and above, the result becomes
dominated by p2. At 10/10, all nine surviving voxels come from p2–PGi.
"""
    ),
    md(
        r"""
## Draft response to the reviewer

We performed a substantial extension of the Ghost analysis. Specifically, we
asked: can the people-centered signature that we discovered from the original
32-voxel cluster, using the feature-identification method in Study 2, be used
to find additional voxels with similar tuning?

We took the original people-centered signature—10 Matryoshka SAE features and
their directions—held it fixed, and searched for it in an independent pool of
additional Ghost voxels. We extracted all voxels across the eight participants
that satisfied the original Ghost criterion (NCSNR \(>0.4\) and correlations
with both main Sentence PCs within \((-0.14,0.14)\)). This yielded 1,490
additional voxels that did not overlap with the original Ghost voxels.

Of these, 275 were well predicted by the fixed 10-feature model, using a
noise-ceiling-normalized cross-validated threshold of
\(r_{\mathrm{NC}}\geq0.50\). This requires the model to capture at least half
of each voxel's estimated predictable correlation. We next asked whether the
features predicted these voxels in the same direction as in the original
cluster. We required at least 7 of the 10 coefficient directions to agree, on
average across the cross-validation folds. This left 181 **feature
direction-aligned** voxels. For comparison, none of the 320 Abstract or
Concrete voxels reached 7/10 directional agreement. This suggests that the
alignment is not simply a consequence of being well predicted by the 10
features and is not shared with these comparison voxel populations.

The additional analysis does not suggest that this feature alignment is
present equally strongly in every participant. Rather, it is clearly present
in a subset. p1, p2, p3, p5, and p8 contribute 58, 74, 9, 12, and 25
additional feature direction-aligned voxels, respectively. p4, p6, and p7
contribute only 3, 0, and 0 voxels. Together, these three participants account
for only 3 of the 181 aligned voxels (1.7%), so we report them in the
participant table but exclude them from the following anatomical summary.

After omitting these three minimal contributors, 178 of the original 181
feature direction-aligned voxels remained for the anatomical summary. Of
these, 138 (77.5%) were in PGi or TPOJ1/2/3, whereas 40 (22.5%) were located
elsewhere. For comparison, within the same five participants, the broader set
that passed the prediction threshold before feature-direction alignment
contained 264 voxels: 148 (56.1%) in PGi or TPOJ1/2/3 and 116 (43.9%) in
other parcels. Thus, the 264 voxels are a pre-alignment comparison set, not an
additional selection stage. Put another way, 138 of the 148 well-predicted
PGi/TPOJ voxels retained the feature directions, compared with 40 of the 116
well-predicted voxels elsewhere.

Thus, the additional analysis recovers a substantially larger
feature-defined population across multiple participants and shows that it is
particularly concentrated in left PGi and TPOJ areas. Because all analyzed
voxels were in the left hemisphere, we describe this as a left
temporoparietal concentration. We did not collect a theory-of-mind functional
localizer and therefore do not identify these voxels as belonging to the ToM
network.

The result is robust to both the normalized threshold and the prediction
scale. Using lower and higher normalized thresholds
(\(r_{\mathrm{NC}}\geq0.40\) and \(0.60\)) yields 254 and 104 feature
direction-aligned voxels, of which 72.0% and 83.5%, respectively, are in PGi
or TPOJ1/2/3. Using raw-\(r\) thresholds of \(0.20\), \(0.25\), and \(0.30\)
yields 290, 245, and 191 aligned voxels, with corresponding four-parcel
percentages of 69.9%, 73.4%, and 75.0%. The same five participants contribute
more than three aligned voxels under every specification.
"""
    ),
    md(
        r"""
## Summary

- We searched 1,490 non-overlapping additional Ghost voxels using a fixed
  10-feature people-centered basis.
- 275 voxels passed the primary \(r_{\mathrm{NC}}\geq0.50\) prediction screen.
- 181 also had at least 7/10 feature directions aligned.
- Five participants contributed more than three aligned voxels; three
  participants together contributed only three.
- Within the five contributing participants, the independent functional LTPJ
  parcel contained 80.3% of feature direction-aligned voxels, compared with
  23.3% of well-predicted but non-aligned voxels and 29.0% of all additional
  Ghost candidates.
- The main participant and anatomical conclusions are stable across a broad
  range of raw, noise-ceiling-normalized, and direction thresholds; the
  supplement reports functional LTPJ overlap for each specification and
  retains the Glasser PGi/TPOJ analysis as a secondary anatomical reference.
- Perfect 10/10 agreement is a strict p2 phenomenon inside functional LTPJ and
  is therefore a useful descriptive subset, not the primary cross-participant
  criterion.
"""
    ),
]

# The reviewer-response prose is maintained separately from this analysis
# artifact.
cells = [
    cell
    for cell in cells
    if not (
        cell.cell_type == "markdown"
        and cell.source.lstrip().startswith("## Draft response to the reviewer")
    )
]


for cell in cells:
    if "ghost-ltpj-reviewer-tables-2026-11-09" in cell.source.splitlines()[0]:
        cell.metadata["tags"] = ["ghost-ltpj-reviewer-tables-2026-11-09"]


notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10",
        },
    },
)
nbf.write(notebook, NOTEBOOK_PATH)
print(f"Wrote {NOTEBOOK_PATH.name}")
