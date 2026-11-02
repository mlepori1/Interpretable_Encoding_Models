#!/usr/bin/env python3
"""Add the cleaned fsaverage7 surface analysis to the dated Ghost notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat


NOTEBOOK = Path(__file__).with_name("ghost_spotlight_extension_2026-09-22.ipynb")
CELL_TAG = "ghost-surface-maps-2026-09-22"


def tagged(cell):
    return CELL_TAG in cell.get("metadata", {}).get("tags", [])


def markdown(source):
    return nbformat.v4.new_markdown_cell(
        source, metadata={"tags": [CELL_TAG]}
    )


def code(source):
    return nbformat.v4.new_code_cell(
        source, metadata={"tags": [CELL_TAG]}
    )


def main() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    notebook.cells = [cell for cell in notebook.cells if not tagged(cell)]

    notebook.cells.extend(
        [
            markdown(
                """## Surface visualization and anatomical references

The metadata's `neuroid` value indexes a vertex on the common left-hemisphere
fsaverage7 surface (163,842 vertices per hemisphere). This allows participant
voxels to be displayed in a common coordinate system, although a pooled map
is a spatial overlay rather than a conventional group-average activation map.

The **primary anatomical reference** is the independently defined probabilistic
functional LTPJ parcel. The five left-hemisphere language parcels are also
outlined: IFGorb, IFG, MFG, anterior temporal, and posterior temporal. No
anatomical label entered voxel selection, prediction screening, or feature-
direction scoring. The anatomical Glasser PGi/TPOJ1/2/3 definition is retained
only as a supplement.

All quantitative surface maps use one visual specification: Ghost candidates
below the prediction threshold are gray, while well-predicted voxels are
colored with `plasma` according to their graded feature-direction agreement.
Every voxel marker has the same area."""
            ),
            code(
                r'''import base64
import gzip
import struct
import sys
import urllib.request
import xml.etree.ElementTree as ET
import zlib

from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D


plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

GLASSER_PATH = SURFACE_ASSET_DIR / "lh.HCP_MMP1_fsaverage.mgz"
GLASSER_LOOKUP_PATH = SURFACE_ASSET_DIR / "HCP_MMP1_fsaverage.txt"

LANGUAGE_SN8_PATH = (
    SURFACE_ASSET_DIR / "lh.parcels_language_fsaverage_frontal_SN8.gii"
)
LANGUAGE_SN220_PATH = (
    SURFACE_ASSET_DIR / "lh.parcels_language_fsaverage_REGULARIZED.gii"
)
LANGUAGE_PARCEL_NAMES = {
    1: "IFGorb",
    2: "IFG",
    3: "MFG",
    4: "AntTemp",
    5: "PostTemp",
}
LANGUAGE_OUTLINE_COLOR = "#555B61"

INFLATED_PATH = SURFACE_ASSET_DIR / "lh.inflated"
SULC_PATH = SURFACE_ASSET_DIR / "lh.sulc"
SURFACE_URLS = {
    INFLATED_PATH: (
        "https://natural-scenes-dataset.s3.amazonaws.com/"
        "nsddata/freesurfer/fsaverage/surf/lh.inflated"
    ),
    SULC_PATH: (
        "https://natural-scenes-dataset.s3.amazonaws.com/"
        "nsddata/freesurfer/fsaverage/surf/lh.sulc"
    ),
}


def ensure_surface_assets():
    SURFACE_ASSET_DIR.mkdir(parents=True, exist_ok=True)
    for path, url in SURFACE_URLS.items():
        if not path.exists():
            print(f"Downloading {path.name} from the NSD fsaverage release...")
            urllib.request.urlretrieve(url, path)


def read_fs_surface(path):
    with open(path, "rb") as handle:
        magic = int.from_bytes(handle.read(3), byteorder="big")
        if magic != 16_777_214:
            raise ValueError(f"Unexpected FreeSurfer surface magic: {magic}")
        handle.readline()
        handle.readline()
        n_vertices, n_faces = np.fromfile(handle, dtype=">i4", count=2)
        coordinates = np.fromfile(
            handle, dtype=">f4", count=int(n_vertices) * 3
        ).reshape(int(n_vertices), 3)
        faces = np.fromfile(
            handle, dtype=">i4", count=int(n_faces) * 3
        ).reshape(int(n_faces), 3)
    return coordinates.astype(float), faces.astype(int)


def read_fs_morph(path):
    with open(path, "rb") as handle:
        magic = int.from_bytes(handle.read(3), byteorder="big")
        if magic != 16_777_215:
            raise ValueError(f"Unexpected FreeSurfer morph magic: {magic}")
        n_vertices, _, values_per_vertex = np.fromfile(
            handle, dtype=">i4", count=3
        )
        values = np.fromfile(
            handle,
            dtype=">f4",
            count=int(n_vertices) * int(values_per_vertex),
        )
    return values.reshape(int(n_vertices), int(values_per_vertex)).squeeze()


def read_mgh_labels(path):
    with gzip.open(path, "rb") as handle:
        payload = handle.read()
    _, width, height, depth, frames, mgh_type, _ = struct.unpack(
        ">7i", payload[:28]
    )
    if mgh_type != 3:
        raise ValueError(f"Expected MGH float type 3, found {mgh_type}")
    n_values = width * height * depth * frames
    return np.frombuffer(
        payload, dtype=">f4", count=n_values, offset=284
    ).astype(int)


def read_surface_gifti(path):
    return read_gifti_float_array(path)


ensure_surface_assets()
surface_coordinates, surface_faces = read_fs_surface(INFLATED_PATH)
surface_sulc = read_fs_morph(SULC_PATH)
glasser_labels = read_mgh_labels(GLASSER_PATH)
language_sn8_labels = read_surface_gifti(LANGUAGE_SN8_PATH).astype(int)
language_sn220_labels = read_surface_gifti(LANGUAGE_SN220_PATH).astype(int)
saxe_ltpj_surface = validate_binary_surface(read_surface_gifti(SAXE_LTPJ_PATH))

assert surface_coordinates.shape == (163_842, 3)
assert surface_sulc.shape == (163_842,)
assert glasser_labels.shape == (163_842,)
assert saxe_ltpj_surface.shape == (163_842,)
assert int(saxe_ltpj_surface.sum()) == 6_587
np.testing.assert_array_equal(saxe_ltpj_surface, saxe_ltpj)

glasser_lookup = pd.read_csv(
    GLASSER_LOOKUP_PATH,
    sep=r"\s+",
    names=["Parcel ID", "Parcel Name"],
)
target_glasser_ids = glasser_lookup.loc[
    glasser_lookup["Parcel Name"].isin(PGI_TPOJ), "Parcel ID"
].astype(int).tolist()
glasser_tpj_surface = np.isin(glasser_labels, target_glasser_ids)

# Reconstruct the five-parcel surface definition and validate its assignments
# on the released, selected language-fROI subset (not all surface vertices).
language_parcel_labels = np.zeros(163_842, dtype=int)
for parcel_id in (1, 2, 3):
    language_parcel_labels[language_sn8_labels == parcel_id] = parcel_id
for parcel_id in (4, 5):
    language_parcel_labels[language_sn220_labels == parcel_id] = parcel_id

language_checks = []
for participant in PARTICIPANTS:
    meta = pd.read_csv(META_DIR / participant / "langfroi12345_meta.csv")
    vertices = validated_vertices(meta["neuroid_id"])
    exact = language_parcel_labels[vertices] == meta["parc_lang"].to_numpy()
    language_checks.append((participant, len(exact), int(exact.sum())))
assert all(total == exact for _, total, exact in language_checks)
print("Surface assets loaded; language-parcel assignments match the released language-fROI subset.")
print("Probabilistic functional LTPJ vertices:", int(saxe_ltpj_surface.sum()))
print("Glasser supplement labels:", target_glasser_ids)
'''
            ),
            code(
                r'''# Project the inflated fsaverage7 surface into lateral and medial views.
face_coordinates = surface_coordinates[surface_faces]
face_normal_x = np.cross(
    face_coordinates[:, 1] - face_coordinates[:, 0],
    face_coordinates[:, 2] - face_coordinates[:, 0],
)[:, 0]

view_geometry = {}
for view, faces, x_coordinates in [
    ("lateral", surface_faces[face_normal_x < 0], -surface_coordinates[:, 1]),
    ("medial", surface_faces[face_normal_x > 0], surface_coordinates[:, 1]),
]:
    vertices = np.unique(faces)
    visible = np.zeros(163_842, dtype=bool)
    visible[vertices] = True
    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    )
    unique_edges, edge_counts = np.unique(
        np.sort(edges, axis=1), axis=0, return_counts=True
    )
    rim_edges = unique_edges[edge_counts == 1]
    view_geometry[view] = {
        "faces": faces,
        "vertices": vertices,
        "visible": visible,
        "edges": unique_edges,
        "rim_edges": rim_edges,
        "x": x_coordinates,
        "y": surface_coordinates[:, 2],
    }


def segments_from_edges(edges, x_coordinates, y_coordinates):
    return np.stack(
        [
            np.column_stack(
                [x_coordinates[edges[:, 0]], y_coordinates[edges[:, 0]]]
            ),
            np.column_stack(
                [x_coordinates[edges[:, 1]], y_coordinates[edges[:, 1]]]
            ),
        ],
        axis=1,
    )


def mask_boundary_segments(mask, view):
    geometry = view_geometry[view]
    edges = geometry["edges"]
    boundary = edges[mask[edges[:, 0]] != mask[edges[:, 1]]]
    return segments_from_edges(boundary, geometry["x"], geometry["y"])


for view, geometry in view_geometry.items():
    geometry["rim_segments"] = segments_from_edges(
        geometry["rim_edges"], geometry["x"], geometry["y"]
    )
    geometry["saxe_segments"] = mask_boundary_segments(
        saxe_ltpj_surface, view
    )
    geometry["glasser_segments"] = mask_boundary_segments(
        glasser_tpj_surface, view
    )
    geometry["language_segments"] = {
        parcel_id: mask_boundary_segments(
            language_parcel_labels == parcel_id, view
        )
        for parcel_id in LANGUAGE_PARCEL_NAMES
    }

TPJ_OUTLINE_COLOR = "#111111"
GRAY_GHOST = "#9EA4AD"
agreement_cmap = plt.get_cmap("plasma")
agreement_norm = Normalize(vmin=0, vmax=10)
AGREEMENT_COLUMN = "Average Num Features Agree Direction"
MARKER_AREA = 24


def prepare_brain_axis(
    ax, view, reference="saxe", title=None, show_language=True
):
    geometry = view_geometry[view]
    vertices = geometry["vertices"]
    x_coordinates, y_coordinates = geometry["x"], geometry["y"]

    # Stronger sulcal/gyri contrast than the exploratory figures, while
    # preserving a neutral white/gray surface beneath the quantitative layer.
    lightness = np.interp(
        np.clip(surface_sulc[vertices], -1.5, 1.5),
        [-1.5, 1.5],
        [0.38, 0.995],
    )
    colors = np.column_stack(
        [lightness, lightness, lightness, np.ones_like(lightness)]
    )
    ax.scatter(
        x_coordinates[vertices],
        y_coordinates[vertices],
        color=colors,
        s=0.23,
        linewidths=0,
        rasterized=True,
        zorder=0,
    )
    ax.add_collection(
        LineCollection(
            geometry["rim_segments"],
            colors="#AEB4BA",
            linewidths=0.70,
            alpha=0.95,
            zorder=1,
        )
    )

    if show_language:
        for segments in geometry["language_segments"].values():
            ax.add_collection(
                LineCollection(
                    segments,
                    colors=LANGUAGE_OUTLINE_COLOR,
                    linewidths=0.62,
                    linestyles="solid",
                    alpha=0.78,
                    zorder=6,
                )
            )

    reference_segments = geometry[
        "saxe_segments" if reference == "saxe" else "glasser_segments"
    ]
    ax.add_collection(
        LineCollection(
            reference_segments,
            colors=TPJ_OUTLINE_COLOR,
            linewidths=2.65,
            alpha=0.98,
            zorder=7,
        )
    )

    ax.set_aspect("equal")
    ax.set_xlim(x_coordinates[vertices].min() - 3,
                x_coordinates[vertices].max() + 3)
    ax.set_ylim(y_coordinates[vertices].min() - 3,
                y_coordinates[vertices].max() + 3)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=2)


def plot_gray_and_agreement(ax, frame, view):
    geometry = view_geometry[view]
    visible = geometry["visible"]
    x_coordinates, y_coordinates = geometry["x"], geometry["y"]

    gray = frame[~frame["Well predicted"]]
    gray_vertices = gray["neuroid"].astype(int).to_numpy()
    gray_vertices = gray_vertices[visible[gray_vertices]]
    ax.scatter(
        x_coordinates[gray_vertices],
        y_coordinates[gray_vertices],
        s=MARKER_AREA,
        color=GRAY_GHOST,
        alpha=0.46,
        edgecolors="white",
        linewidths=0.18,
        rasterized=True,
        zorder=2,
    )

    predicted = frame[frame["Well predicted"]].copy()
    predicted = predicted[
        visible[predicted["neuroid"].astype(int).to_numpy()]
    ].sort_values(AGREEMENT_COLUMN)
    vertices = predicted["neuroid"].astype(int).to_numpy()
    ax.scatter(
        x_coordinates[vertices],
        y_coordinates[vertices],
        c=predicted[AGREEMENT_COLUMN],
        cmap=agreement_cmap,
        norm=agreement_norm,
        s=MARKER_AREA,
        alpha=0.74,
        edgecolors="#202124",
        linewidths=0.20,
        rasterized=True,
        zorder=3,
    )


def reference_legend(reference, show_language=True):
    label = (
        "Probabilistic functional LTPJ"
        if reference == "saxe"
        else "Glasser PGi/TPOJ1/2/3"
    )
    handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", color=GRAY_GHOST,
            label="Not well predicted", markersize=6,
        ),
        Line2D(
            [0], [0], color=TPJ_OUTLINE_COLOR, linewidth=2.65, label=label
        ),
    ]
    if show_language:
        handles.append(
            Line2D(
                [0], [0], color=LANGUAGE_OUTLINE_COLOR,
                linewidth=0.9, linestyle="solid", label="Language parcels",
            )
        )
    return handles


def add_agreement_colorbar(fig, axes, bounds=None):
    mappable = ScalarMappable(norm=agreement_norm, cmap=agreement_cmap)
    mappable.set_array([])
    if bounds is None:
        colorbar = fig.colorbar(
            mappable,
            ax=axes,
            orientation="horizontal",
            fraction=0.045,
            pad=0.035,
            aspect=52,
            ticks=[0, 2, 4, 6, 7, 8, 10],
        )
    else:
        colorbar_axis = fig.add_axes(bounds)
        colorbar = fig.colorbar(
            mappable,
            cax=colorbar_axis,
            orientation="horizontal",
            ticks=[0, 2, 4, 6, 7, 8, 10],
        )
    colorbar.set_label(
        "Mean feature directions matching the original signature (of 10)",
        fontsize=13,
        labelpad=6,
    )
    colorbar.ax.tick_params(labelsize=12, colors="black")
    colorbar.ax.axvline(7, color="white", linewidth=3.0)
    colorbar.ax.axvline(7, color="black", linewidth=1.2)
    colorbar.ax.text(
        7, 1.45, "7/10 cutoff", ha="center", va="bottom",
        fontsize=10, fontweight="bold",
    )
    return colorbar


def save_figure(fig, stem, dpi=350):
    fig.savefig(f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    print(f"Saved {Path(stem).name}.png and .pdf")
'''
            ),
            markdown(
                """## Primary group surface map

The group map retains all 1,490 additional Ghost candidates. The 1,215 voxels
below the prediction threshold are gray. The 275 well-predicted voxels are
colored by their continuous feature-direction agreement, with the reported
7/10 cutoff marked on the colorbar. The probabilistic functional LTPJ and five
language parcels are reference outlines only; parcel numbers are intentionally
omitted."""
            ),
            code(
                r'''def make_group_surface(
    reference, title, output_stem, show_language=True
):
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 6.5))
    for ax, view, panel in zip(
        axes,
        ["lateral", "medial"],
        ["A. Left lateral", "B. Left medial"],
    ):
        prepare_brain_axis(
            ax,
            view,
            reference=reference,
            title=panel,
            show_language=show_language,
        )
        plot_gray_and_agreement(ax, ghost, view)

    fig.subplots_adjust(
        left=0.01, right=0.99, top=0.78, bottom=0.18, wspace=0.04
    )
    add_agreement_colorbar(fig, axes, bounds=[0.18, 0.075, 0.64, 0.025])
    fig.legend(
        handles=reference_legend(reference, show_language=show_language),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        frameon=False,
        ncol=4,
        fontsize=10.5,
        columnspacing=1.3,
        handlelength=2.1,
    )
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.99)
    save_figure(fig, output_stem)
    plt.show()


primary_surface_stem = (
    REPO_ROOT
    / "analysis/figures/ghost_surface_agreement_plasma_"
      "functional_ltpj_primary_2026-09-22"
)
make_group_surface(
    "saxe",
    "Feature-direction agreement in additional Ghost voxels",
    primary_surface_stem,
)

primary_surface_no_language_stem = (
    REPO_ROOT
    / "analysis/figures/ghost_surface_agreement_plasma_"
      "functional_ltpj_no_language_parcels_2026-09-22"
)
make_group_surface(
    "saxe",
    "Feature-direction agreement in additional Ghost voxels",
    primary_surface_no_language_stem,
    show_language=False,
)
'''
            ),
            markdown(
                """## Primary anatomical quantification

The surface map is descriptive; the compact plots below quantify its central
contrast in the five primary contributing participants. The recommended main
panel compares all eligible candidates, well-predicted but non-aligned voxels,
and aligned voxels. A matched horizontal panel directly compares aligned and
non-aligned voxels under both the probabilistic functional LTPJ and anatomical Glasser
PGi/TPOJ definitions. A third panel shows participant-level heterogeneity."""
            ),
            code(
                r'''FIGURE_DIR = REPO_ROOT / "analysis/figures"

quantification_rows = []
for label, frame in [
    ("All Ghost\ncandidates", core_ghost),
    ("Well-predicted,\nnon-aligned", core_well_nonaligned),
    ("Feature direction-\naligned", core_aligned),
]:
    inside = int(frame["Inside Saxe functional LTPJ"].sum())
    quantification_rows.append({
        "Voxel set": label.replace("\n", " "),
        "Total": len(frame),
        "Inside probabilistic functional LTPJ": inside,
        "Inside probabilistic functional LTPJ (%)": 100 * inside / len(frame),
    })
functional_ltpj_quantification = pd.DataFrame(quantification_rows)
display(
    functional_ltpj_quantification.style.format(
        {"Inside probabilistic functional LTPJ (%)": "{:.1f}"}
    )
)
functional_ltpj_quantification.to_csv(
    FIGURE_DIR / "ghost_functional_ltpj_quantification.csv", index=False
)

# Option 1 (recommended): direct percentage bars.
labels = [
    "All Ghost\ncandidates",
    "Well-predicted,\nnon-aligned",
    "Feature direction-\naligned",
]
percentages = functional_ltpj_quantification[
    "Inside probabilistic functional LTPJ (%)"
].to_numpy()
inside_counts = functional_ltpj_quantification[
    "Inside probabilistic functional LTPJ"
].to_numpy()
totals = functional_ltpj_quantification["Total"].to_numpy()
bar_colors = ["#A9AFB7", "#7651A3", agreement_cmap(0.86)]

fig, ax = plt.subplots(figsize=(5.4, 5.4))
bars = ax.bar(
    np.arange(3), percentages, color=bar_colors, width=0.66,
    edgecolor="black", linewidth=1.1, zorder=3,
)
for bar, pct, inside, total in zip(bars, percentages, inside_counts, totals):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        pct + 2.2,
        f"{inside}/{total}\n({pct:.1f}%)",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )
ax.set_xticks(np.arange(3), labels)
ax.set_ylabel(
    "Percentage of each voxel set\nlocated inside probabilistic functional LTPJ"
)
ax.set_ylim(0, 103)
ax.set_yticks(np.arange(0, 101, 20))
ax.set_title(
    "Probabilistic functional LTPJ concentration",
    fontweight="bold",
    pad=12,
)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(colors="black")
ax.yaxis.grid(True, linewidth=0.7, alpha=0.25, zorder=0)
fig.tight_layout()
save_figure(
    fig, FIGURE_DIR / "ghost_functional_ltpj_quantification_bar_2026-09-22"
)
plt.show()

# Option 2: matched alignment contrast under the new and old TPJ references.
for frame in (core_well_nonaligned, core_aligned):
    frame["Inside Glasser PGi/TPOJ"] = glasser_tpj_surface[
        frame["neuroid"].astype(int).to_numpy()
    ]

reference_specs = [
    ("Probabilistic\nfunctional LTPJ", "Inside Saxe functional LTPJ"),
    ("Anatomical PGi/TPOJ\n(Glasser)", "Inside Glasser PGi/TPOJ"),
]
tpj_reference_rows = []
for reference_label, column in reference_specs:
    for voxel_set, frame in [
        ("Well-predicted, non-aligned", core_well_nonaligned),
        ("Feature direction-aligned", core_aligned),
    ]:
        inside = int(frame[column].sum())
        tpj_reference_rows.append({
            "TPJ reference": reference_label.replace("\n", " "),
            "Voxel set": voxel_set,
            "Inside": inside,
            "Total": len(frame),
            "Inside (%)": 100 * inside / len(frame),
        })
tpj_reference_quantification = pd.DataFrame(tpj_reference_rows)
display(tpj_reference_quantification.style.format({"Inside (%)": "{:.1f}"}))
tpj_reference_quantification.to_csv(
    FIGURE_DIR / "ghost_tpj_reference_alignment_quantification.csv",
    index=False,
)

reference_labels = [label for label, _ in reference_specs]
nonaligned_pct = np.array([
    100 * core_well_nonaligned[column].mean()
    for _, column in reference_specs
])
aligned_pct = np.array([
    100 * core_aligned[column].mean()
    for _, column in reference_specs
])

fig, ax = plt.subplots(figsize=(7.2, 4.2))
y = np.arange(len(reference_specs))
bar_height = 0.30
nonaligned_bars = ax.barh(
    y + bar_height / 1.7,
    nonaligned_pct,
    height=bar_height,
    color="#7651A3",
    edgecolor="black",
    linewidth=1.1,
    label="Well-predicted, non-aligned",
    zorder=3,
)
aligned_bars = ax.barh(
    y - bar_height / 1.7,
    aligned_pct,
    height=bar_height,
    color=agreement_cmap(0.86),
    edgecolor="black",
    linewidth=1.1,
    label="Feature direction-aligned",
    zorder=3,
)
for bars, values in [
    (nonaligned_bars, nonaligned_pct),
    (aligned_bars, aligned_pct),
]:
    for bar, pct in zip(bars, values):
        ax.text(
            pct + 1.4,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%",
            ha="left",
            va="center",
            fontsize=11.5,
            fontweight="bold",
        )
ax.set_yticks(y, reference_labels)
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xticks(np.arange(0, 101, 20))
ax.set_xlabel("Percentage of each voxel set located inside TPJ")
ax.set_title(
    "TPJ concentration by feature-direction alignment",
    fontweight="bold",
    pad=12,
)
ax.legend(
    frameon=False, ncol=2, loc="lower center",
    bbox_to_anchor=(0.5, -0.38), fontsize=10.5,
)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.tick_params(axis="y", length=0, colors="black")
ax.tick_params(axis="x", colors="black")
ax.xaxis.grid(True, linewidth=0.7, alpha=0.25, zorder=0)
fig.subplots_adjust(left=0.28, right=0.98, top=0.84, bottom=0.25)
save_figure(
    fig,
    FIGURE_DIR / "ghost_tpj_reference_alignment_comparison_2026-09-22",
)
plt.show()

# Option 2b: the primary probabilistic functional LTPJ contrast by itself.
functional_labels = [
    "Well-predicted,\nnon-aligned",
    "Feature direction-\naligned",
]
functional_pct = np.array([nonaligned_pct[0], aligned_pct[0]])
functional_inside = np.array([
    int(core_well_nonaligned["Inside Saxe functional LTPJ"].sum()),
    int(core_aligned["Inside Saxe functional LTPJ"].sum()),
])
functional_totals = np.array([
    len(core_well_nonaligned),
    len(core_aligned),
])

fig, ax = plt.subplots(figsize=(6.6, 3.5))
y = np.arange(2)
bars = ax.barh(
    y,
    functional_pct,
    height=0.56,
    color=["#7651A3", agreement_cmap(0.86)],
    edgecolor="black",
    linewidth=1.1,
    zorder=3,
)
for bar, pct, inside, total in zip(
    bars, functional_pct, functional_inside, functional_totals
):
    ax.text(
        pct + 1.4,
        bar.get_y() + bar.get_height() / 2,
        f"{inside}/{total} ({pct:.1f}%)",
        ha="left",
        va="center",
        fontsize=11.5,
        fontweight="bold",
    )
ax.set_yticks(y, functional_labels)
ax.invert_yaxis()
ax.set_xlim(0, 112)
ax.set_xticks(np.arange(0, 101, 20))
ax.set_xlabel(
    "Percentage of each voxel set located inside\n"
    "probabilistic functional LTPJ"
)
ax.set_title(
    "Concentration in probabilistic functional LTPJ",
    fontweight="bold",
    pad=12,
)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.tick_params(axis="y", length=0, colors="black")
ax.tick_params(axis="x", colors="black")
ax.xaxis.grid(True, linewidth=0.7, alpha=0.25, zorder=0)
fig.subplots_adjust(left=0.27, right=0.98, top=0.81, bottom=0.25)
save_figure(
    fig,
    FIGURE_DIR
    / "ghost_probabilistic_functional_ltpj_alignment_horizontal_2026-09-22",
)
plt.show()

# Matching vertical version of the primary two-group contrast.
fig, ax = plt.subplots(figsize=(5.0, 5.2))
x = np.arange(2)
bars = ax.bar(
    x,
    functional_pct,
    width=0.62,
    color=["#7651A3", agreement_cmap(0.86)],
    edgecolor="black",
    linewidth=1.1,
    zorder=3,
)
for bar, pct, inside, total in zip(
    bars, functional_pct, functional_inside, functional_totals
):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        pct + 2.0,
        f"{inside}/{total}\n({pct:.1f}%)",
        ha="center",
        va="bottom",
        fontsize=11.5,
        fontweight="bold",
    )
ax.set_xticks(x, functional_labels)
ax.set_ylim(0, 103)
ax.set_yticks(np.arange(0, 101, 20))
ax.set_ylabel(
    "Percentage of each voxel set located inside\n"
    "probabilistic functional LTPJ"
)
ax.set_title(
    "Concentration in probabilistic functional LTPJ",
    fontweight="bold",
    pad=12,
)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(colors="black")
ax.yaxis.grid(True, linewidth=0.7, alpha=0.25, zorder=0)
fig.tight_layout()
save_figure(
    fig,
    FIGURE_DIR
    / "ghost_probabilistic_functional_ltpj_alignment_vertical_2026-09-22",
)
plt.show()

# Option 3: participant-resolved paired percentages.
participant_quantification = []
for participant in ANATOMY_PARTICIPANTS:
    participant_non = core_well_nonaligned[
        core_well_nonaligned["Participant"].eq(participant)
    ]
    participant_aligned = core_aligned[
        core_aligned["Participant"].eq(participant)
    ]
    participant_quantification.append({
        "Participant": participant,
        "Well-predicted, non-aligned (%)": (
            100 * participant_non["Inside Saxe functional LTPJ"].mean()
        ),
        "Feature direction-aligned (%)": (
            100 * participant_aligned["Inside Saxe functional LTPJ"].mean()
        ),
        "Non-aligned n": len(participant_non),
        "Aligned n": len(participant_aligned),
    })
participant_quantification = pd.DataFrame(participant_quantification)
display(participant_quantification.style.format({
    "Well-predicted, non-aligned (%)": "{:.1f}",
    "Feature direction-aligned (%)": "{:.1f}",
}))

fig, ax = plt.subplots(figsize=(7.4, 4.5))
y = np.arange(len(participant_quantification))
for yi, row in participant_quantification.iterrows():
    ax.plot(
        [row["Well-predicted, non-aligned (%)"],
         row["Feature direction-aligned (%)"]],
        [yi, yi], color="#B8BDC4", linewidth=2.0, zorder=1,
    )
ax.scatter(
    participant_quantification["Well-predicted, non-aligned (%)"], y,
    s=72, color=agreement_cmap(0.27), label="Well-predicted, non-aligned",
    zorder=2,
)
ax.scatter(
    participant_quantification["Feature direction-aligned (%)"], y,
    s=72, color=agreement_cmap(0.86), label="Feature direction-aligned",
    zorder=3,
)
ax.set_yticks(y, participant_quantification["Participant"].str.upper())
ax.invert_yaxis()
ax.set_xlim(-3, 103)
ax.set_xlabel(
    "Percentage of each participant voxel set located inside\n"
    "probabilistic functional LTPJ"
)
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.34),
          ncol=2, fontsize=10.5)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(colors="black")
fig.tight_layout()
save_figure(
    fig,
    FIGURE_DIR / "ghost_functional_ltpj_quantification_participants_2026-09-22",
)
plt.show()
'''
            ),
            markdown(
                """## Supplement: individual-participant surface maps

Each participant is shown separately using the identical surface, outlines,
marker sizes, prediction threshold, and Plasma normalization as the primary
group map. This preserves participant identity while making the spatial
convergence directly inspectable."""
            ),
            code(
                r'''fig, axes = plt.subplots(4, 4, figsize=(14.2, 14.8))
for participant_index, participant in enumerate(PARTICIPANTS):
    row = participant_index // 2
    first_column = 2 * (participant_index % 2)
    participant_frame = ghost[ghost["Participant"].eq(participant)]
    for offset, view in enumerate(["lateral", "medial"]):
        ax = axes[row, first_column + offset]
        prepare_brain_axis(
            ax,
            view,
            reference="saxe",
            title=f"{participant.upper()}: {view.capitalize()}",
        )
        plot_gray_and_agreement(ax, participant_frame, view)

fig.subplots_adjust(
    left=0.01, right=0.99, top=0.96, bottom=0.17, hspace=0.03, wspace=0.03
)
add_agreement_colorbar(
    fig, axes, bounds=[0.18, 0.085, 0.64, 0.014]
)
fig.legend(
    handles=reference_legend("saxe"),
    loc="lower center",
    bbox_to_anchor=(0.5, 0.002),
    frameon=False,
    ncol=3,
    fontsize=10,
)
fig.suptitle(
    "Supplement: feature-direction agreement by participant",
    fontsize=16,
    fontweight="bold",
    y=0.997,
)
participant_surface_stem = (
    FIGURE_DIR
    / "ghost_surface_participants_plasma_functional_ltpj_"
      "supplement_2026-09-22"
)
save_figure(fig, participant_surface_stem)
plt.show()
'''
            ),
            markdown(
                """## Supplement: anatomical Glasser PGi/TPOJ reference

This supplement repeats the primary group display while changing only the
black reference outline to the anatomical Glasser union of PGi and
TPOJ1/2/3. The compact table provides the corresponding composition; the
probabilistic functional LTPJ parcel remains the primary anatomical
quantification."""
            ),
            code(
                r'''for frame in (core_ghost, core_well_nonaligned, core_aligned):
    frame["Inside Glasser PGi/TPOJ"] = glasser_tpj_surface[
        frame["neuroid"].astype(int).to_numpy()
    ]

glasser_rows = []
for label, frame in [
    ("All additional Ghost candidates", core_ghost),
    ("Well-predicted but non-aligned", core_well_nonaligned),
    ("Well-predicted and aligned", core_aligned),
]:
    inside = int(frame["Inside Glasser PGi/TPOJ"].sum())
    glasser_rows.append({
        "Voxel set": label,
        "Total": len(frame),
        "Inside Glasser PGi/TPOJ": inside,
        "Inside Glasser PGi/TPOJ (%)": 100 * inside / len(frame),
    })
glasser_composition = pd.DataFrame(glasser_rows)
display(
    glasser_composition.style.format(
        {"Inside Glasser PGi/TPOJ (%)": "{:.1f}"}
    )
)

glasser_surface_stem = (
    FIGURE_DIR
    / "ghost_surface_agreement_plasma_"
      "glasser_pgi_tpoj_supplement_2026-09-22"
)
make_group_surface(
    "glasser",
    "Supplement: anatomical PGi/TPOJ reference",
    glasser_surface_stem,
)
'''
            ),
            markdown(
                """## Original 20-per-participant Ghost set

The held-out search excludes the original Ghost voxels. For anatomical
context, we reconstruct the Study 2 hierarchical ordering of the original 160
Ghost voxels and identify the original 32-voxel discovery cluster using the
same fixed matrix interval (positions 70–101 inclusive). The map uses the same
probabilistic functional LTPJ and language-parcel outlines as the primary
extension map."""
            ),
            code(
                r'''if str(REPO_ROOT / "analysis") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "analysis"))
import generalization_utils

ORIGINAL_GHOST_GENERALIZATION = (
    PUBLIC_REPO
    / "results_anon/categories/full_features/generalization_8000/sae/"
      "gemma-2-2b-res-matryoshka-dc/mean/12/standardized_betas/"
      "generalization/ghost.csv"
)
original_generalization = pd.read_csv(ORIGINAL_GHOST_GENERALIZATION)
_, original_ghost_order = generalization_utils.run_cluster_generalization_analysis(
    original_generalization
)
original_cluster_keys = set(original_ghost_order[70:102])

original_frames = []
for participant in PARTICIPANTS:
    path = PUBLIC_REPO / f"data/processed_csvs_anon/{participant}/ghost_meta.csv"
    frame = pd.read_csv(path)
    frame["Participant"] = participant
    frame["neuroid"] = validated_vertices(frame["neuroid_id"])
    frame["Voxel key"] = participant + "_" + frame["neuroid"].astype(str)
    original_frames.append(frame)
original_ghost = pd.concat(original_frames, ignore_index=True)
original_ghost["Original 32-voxel cluster"] = original_ghost[
    "Voxel key"
].isin(original_cluster_keys)

assert len(original_ghost) == 160
assert original_ghost.groupby("Participant").size().eq(20).all()
assert int(original_ghost["Original 32-voxel cluster"].sum()) == 32


def plot_original_ghost(ax, view):
    geometry = view_geometry[view]
    visible = geometry["visible"]
    x_coordinates, y_coordinates = geometry["x"], geometry["y"]
    for is_cluster, color, alpha, zorder in [
        (False, GRAY_GHOST, 0.68, 2),
        (True, agreement_cmap(0.86), 0.98, 3),
    ]:
        layer = original_ghost[
            original_ghost["Original 32-voxel cluster"].eq(is_cluster)
        ]
        vertices = layer["neuroid"].astype(int).to_numpy()
        vertices = vertices[visible[vertices]]
        ax.scatter(
            x_coordinates[vertices],
            y_coordinates[vertices],
            s=MARKER_AREA,
            color=color,
            alpha=alpha,
            edgecolors="white",
            linewidths=0.22,
            rasterized=True,
            zorder=zorder,
        )


fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.7))
for ax, view, panel in zip(
    axes,
    ["lateral", "medial"],
    ["A. Left lateral", "B. Left medial"],
):
    prepare_brain_axis(ax, view, reference="saxe", title=panel)
    plot_original_ghost(ax, view)

original_legend = [
    Line2D([0], [0], marker="o", linestyle="none", color=GRAY_GHOST,
           label="Other original Ghost (n=128)", markersize=6),
    Line2D([0], [0], marker="o", linestyle="none",
           color=agreement_cmap(0.86),
           label="Original discovery cluster (n=32)", markersize=6),
]
original_legend.extend(reference_legend("saxe")[1:])
fig.legend(
    handles=original_legend,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.005),
    frameon=False,
    ncol=4,
    fontsize=10.2,
)
fig.suptitle(
    "Original Ghost voxels and discovery cluster",
    fontsize=16,
    fontweight="bold",
    y=0.99,
)
fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.18, wspace=0.04)
original_surface_stem = (
    FIGURE_DIR / "ghost_surface_original_160_functional_ltpj_2026-09-22"
)
save_figure(fig, original_surface_stem)
plt.show()
'''
            ),
            markdown(
                """### Reading the surface maps

The lateral and medial panels use identical fsaverage7 coordinates and marker
sizes. Gray points are eligible Ghost candidates that do not pass the primary
prediction screen; Plasma colors encode graded directional agreement only
among well-predicted voxels. The solid black outline is the independent
probabilistic functional LTPJ parcel in the primary maps, and thin dark-gray
outlines indicate the five language parcels. The participant-pooled map describes spatial
concentration in a common coordinate system; exact shared vertex IDs are
quantified separately above."""
            ),
        ]
    )

    nbformat.write(notebook, NOTEBOOK)
    print(f"Updated {NOTEBOOK.name}")


if __name__ == "__main__":
    main()
