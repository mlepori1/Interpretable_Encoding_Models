#!/usr/bin/env python3
"""Add the exact fsaverage7 vertex-overlap audit to the dated Ghost notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat


NOTEBOOK = Path(__file__).with_name(
    "ghost_spotlight_extension_2026-09-22.ipynb"
)
CELL_TAG = "ghost-id-overlap-2026-09-22"


def tagged(cell):
    return CELL_TAG in cell.get("metadata", {}).get("tags", [])


def main() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)

    # Correct a stale sentence in the logging copy. The executed counts show
    # p4=3, p6=0, and p7=0, for three minimal-contributor voxels in total.
    stale = (
        "whereas p4, p6, and p7 contribute\n"
        "three, zero, and one voxel, respectively."
    )
    corrected = (
        "whereas p4 contributes three voxels and p6 and p7 contribute none."
    )
    for cell in notebook.cells:
        if cell.cell_type == "markdown" and stale in cell.source:
            cell.source = cell.source.replace(stale, corrected)

    # Make this updater safe to rerun while we iterate on the working copy.
    notebook.cells = [cell for cell in notebook.cells if not tagged(cell)]

    markdown = nbformat.v4.new_markdown_cell(
        """## Exact fsaverage7 vertex overlap across participants

Because all participants are represented on the same left-hemisphere
fsaverage7 mesh, `neuroid` is directly comparable as a surface-vertex index.
We therefore ask how often the **exact same vertex ID** appears in more than
one participant at three stages: the full additional Ghost candidate pool,
the well-predicted set, and the feature direction-aligned set.

This is deliberately an exact-index audit; it does not treat nearby vertices
as overlapping and does not apply spatial smoothing. Candidate eligibility is
participant-specific, so low exact overlap can reflect both anatomical or
functional variability and differences in which vertices entered each
participant's candidate pool.""",
        metadata={"tags": [CELL_TAG]},
    )

    code = nbformat.v4.new_code_cell(
        """overlap_sets = {
    "All additional Ghost candidates": ghost,
    "Well predicted": well_predicted,
    "Feature direction-aligned": aligned,
}


def exact_vertex_overlap(frame):
    participant_sets = {
        participant: set(
            frame.loc[frame["Participant"].eq(participant), "neuroid"].astype(int)
        )
        for participant in PARTICIPANTS
    }
    multiplicity = frame.groupby("neuroid")["Participant"].nunique()
    counts = pd.DataFrame(0, index=PARTICIPANTS, columns=PARTICIPANTS, dtype=int)
    jaccard = pd.DataFrame(
        0.0, index=PARTICIPANTS, columns=PARTICIPANTS, dtype=float
    )
    for participant_a in PARTICIPANTS:
        for participant_b in PARTICIPANTS:
            set_a = participant_sets[participant_a]
            set_b = participant_sets[participant_b]
            intersection = len(set_a & set_b)
            union = len(set_a | set_b)
            counts.loc[participant_a, participant_b] = intersection
            jaccard.loc[participant_a, participant_b] = (
                100 * intersection / union if union else np.nan
            )
    return participant_sets, multiplicity, counts, jaccard


overlap_audits = {
    name: exact_vertex_overlap(frame) for name, frame in overlap_sets.items()
}

overlap_rows = []
for name, frame in overlap_sets.items():
    _, multiplicity, _, _ = overlap_audits[name]
    shared = multiplicity[multiplicity >= 2]
    overlap_rows.append(
        {
            "Analysis stage": name,
            "Participant–vertex observations": len(frame),
            "Unique vertex IDs": len(multiplicity),
            "IDs in >=2 participants": len(shared),
            "Shared IDs among unique IDs (%)": (
                100 * len(shared) / len(multiplicity)
            ),
            "Maximum participants at one ID": int(multiplicity.max()),
        }
    )

exact_overlap_summary = pd.DataFrame(overlap_rows)
display(
    exact_overlap_summary.style.format(
        {"Shared IDs among unique IDs (%)": "{:.2f}"}
    )
)

# Report every non-zero between-participant intersection and its Jaccard rate.
pairwise_rows = []
for name, (_, _, counts, jaccard) in overlap_audits.items():
    for index_a, participant_a in enumerate(PARTICIPANTS):
        for participant_b in PARTICIPANTS[index_a + 1:]:
            overlap_n = int(counts.loc[participant_a, participant_b])
            if overlap_n:
                pairwise_rows.append(
                    {
                        "Analysis stage": name,
                        "Participant pair": f"{participant_a}–{participant_b}",
                        "Shared exact vertex IDs": overlap_n,
                        "Jaccard overlap (%)": jaccard.loc[
                            participant_a, participant_b
                        ],
                    }
                )
pairwise_exact_overlap = pd.DataFrame(pairwise_rows)
display(pairwise_exact_overlap.style.format({"Jaccard overlap (%)": "{:.2f}"}))

# Inspect the exact vertices shared in the final aligned set.
aligned_multiplicity = overlap_audits["Feature direction-aligned"][1]
shared_aligned_ids = aligned_multiplicity[
    aligned_multiplicity >= 2
].index.astype(int)
shared_aligned_details = (
    aligned.loc[
        aligned["neuroid"].astype(int).isin(shared_aligned_ids),
        [
            "neuroid",
            "Participant",
            "Parcel Name",
            "NC Normalized R Fischer",
            "Average Num Features Agree Direction",
        ],
    ]
    .sort_values(["neuroid", "Participant"])
    .reset_index(drop=True)
)
display(shared_aligned_details)

# Visualization: overall shared-ID counts plus off-diagonal pairwise overlaps.
fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))

stage_labels = ["Candidates", "Well\\npredicted", "Direction\\naligned"]
shared_counts = exact_overlap_summary["IDs in >=2 participants"].to_numpy()
unique_counts = exact_overlap_summary["Unique vertex IDs"].to_numpy()
bars = axes[0].bar(
    stage_labels,
    shared_counts,
    color=[COLOR_POOL, COLOR_PREDICTED, COLOR_ALIGNED],
)
axes[0].set_ylabel("Exact vertex IDs in >=2 participants")
axes[0].set_title("A. Cross-participant overlap", loc="left", fontweight="bold")
for bar, shared_n, unique_n in zip(bars, shared_counts, unique_counts):
    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        shared_n + 0.35,
        f"{shared_n}/{unique_n}",
        ha="center",
        va="bottom",
        fontsize=9,
    )
axes[0].spines[["top", "right"]].set_visible(False)

for ax, (name, (_, _, counts, _)), panel in zip(
    axes[1:], overlap_audits.items(), ["B", "C", "D"]
):
    off_diagonal = counts.to_numpy(dtype=float)
    diagonal_sizes = np.diag(off_diagonal).astype(int).copy()
    np.fill_diagonal(off_diagonal, np.nan)
    masked = np.ma.masked_invalid(off_diagonal)
    finite = off_diagonal[np.isfinite(off_diagonal)]
    vmax = max(1, int(finite.max()))
    image = ax.imshow(masked, cmap="Blues", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(PARTICIPANTS)), PARTICIPANTS, rotation=45)
    ax.set_yticks(range(len(PARTICIPANTS)), PARTICIPANTS)
    ax.set_title(f"{panel}. {name}", loc="left", fontweight="bold", fontsize=10)
    for row in range(len(PARTICIPANTS)):
        for column in range(len(PARTICIPANTS)):
            if row == column:
                ax.text(
                    column,
                    row,
                    f"n={diagonal_sizes[row]}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="#555555",
                )
            elif counts.iloc[row, column] > 0:
                ax.text(
                    column,
                    row,
                    str(int(counts.iloc[row, column])),
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                )
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Shared IDs")

fig.suptitle("Exact fsaverage7 vertex overlap across participants", fontsize=14)
fig.tight_layout()
plt.show()""",
        metadata={"tags": [CELL_TAG]},
    )

    conclusion = nbformat.v4.new_markdown_cell(
        """### Implication for a participant-aggregated surface figure

Exact vertex overlap is rare. The 1,490 candidate observations correspond to
1,474 unique vertex IDs, only 16 of which occur in two participants. The 181
feature direction-aligned observations correspond to 178 unique IDs: only
three IDs occur in two participants, and no ID occurs in more than two.
Those three shared aligned IDs are vertices **11418, 42139, and 120150**; each
is shared by p1 and p2 and labeled PGi in both participants.

Consequently, a pooled map colored by the number of participants at the exact
same vertex would be nearly binary and would understate cross-participant
anatomical convergence. The most faithful primary display is therefore a set
of participant-level maps in the common fsaverage7 space. A pooled map can
still be useful as a spatial-density visualization, provided that it is
described as anatomical concentration rather than exact vertexwise overlap.""",
        metadata={"tags": [CELL_TAG]},
    )

    notebook.cells.extend([markdown, code, conclusion])
    nbformat.write(notebook, NOTEBOOK)
    print(f"Updated {NOTEBOOK.name}")


if __name__ == "__main__":
    main()
