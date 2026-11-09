"""Test embedded notebook safety helpers without generating/executing notebooks."""

import ast
import base64
from pathlib import Path
import xml.etree.ElementTree as ET
import zlib

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "analysis/build_ghost_spotlight_extension_notebook.py"
SURFACE_BUILDER = ROOT / "analysis/update_ghost_surface_notebook_2026_09_22.py"


def embedded_code_cells(path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "code" and node.args:
                yield ast.literal_eval(node.args[0]).strip()


@pytest.fixture(scope="module")
def helpers():
    selected = {"read_gifti_float_array", "validate_binary_surface", "validated_vertices"}
    definitions = []
    for source in embedded_code_cells(BUILDER):
        for node in ast.parse(source).body:
            if isinstance(node, ast.FunctionDef) and node.name in selected:
                definitions.append(node)
    assert {node.name for node in definitions} == selected
    namespace = {"np": np, "ET": ET, "base64": base64, "zlib": zlib}
    module = ast.Module(body=definitions, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(BUILDER), "exec"), namespace)
    return namespace


def write_gifti(path, values=(0, 1, 0), *, endian="LittleEndian", overrides=None, arrays=1):
    attributes = {
        "DataType": "NIFTI_TYPE_FLOAT32", "Encoding": "GZipBase64Binary",
        "Endian": endian, "Dimensionality": "1", "Dim0": str(len(values)),
    }
    attributes.update(overrides or {})
    dtype = ">f4" if endian == "BigEndian" else "<f4"
    payload = base64.b64encode(zlib.compress(np.asarray(values, dtype=dtype).tobytes())).decode()
    root = ET.Element("GIFTI", {"NumberOfDataArrays": str(arrays)})
    for _ in range(arrays):
        data_array = ET.SubElement(root, "DataArray", attributes)
        ET.SubElement(data_array, "Data").text = "\n" + payload + "\n"
    ET.ElementTree(root).write(path)
    return path


@pytest.mark.parametrize("endian", ["LittleEndian", "BigEndian"])
def test_gifti_float32_roundtrip(tmp_path, helpers, endian):
    path = write_gifti(tmp_path / "surface.gii", endian=endian)
    values = helpers["read_gifti_float_array"](path)
    np.testing.assert_array_equal(values, [0, 1, 0])
    np.testing.assert_array_equal(helpers["validate_binary_surface"](values, 3), [False, True, False])


@pytest.mark.parametrize("overrides, message", [
    ({"DataType": "NIFTI_TYPE_INT32"}, "float32"),
    ({"Encoding": "ASCII"}, "GZipBase64Binary"),
    ({"Endian": "UnknownEndian"}, "byte order"),
    ({"Dimensionality": "2"}, "one-dimensional"),
    ({"Dim0": "4"}, "payload length"),
])
def test_rejects_incompatible_gifti_formats(tmp_path, helpers, overrides, message):
    path = write_gifti(tmp_path / "invalid.gii", overrides=overrides)
    with pytest.raises(ValueError, match=message):
        helpers["read_gifti_float_array"](path)


@pytest.mark.parametrize("arrays", [0, 2])
def test_requires_exactly_one_gifti_array(tmp_path, helpers, arrays):
    path = write_gifti(tmp_path / "arrays.gii", arrays=arrays)
    with pytest.raises(ValueError, match="exactly one"):
        helpers["read_gifti_float_array"](path)


@pytest.mark.parametrize("value", [np.nan, np.inf, -1, 0.5, 2])
def test_binary_mask_rejects_invalid_values_before_bool_cast(helpers, value):
    with pytest.raises(ValueError, match="finite and binary"):
        helpers["validate_binary_surface"]([0, value, 1], 3)


def test_binary_mask_shape(helpers):
    with pytest.raises(ValueError, match="shape"):
        helpers["validate_binary_surface"]([[0, 1, 0]], 3)


def test_valid_vertex_boundaries(helpers):
    vertices = helpers["validated_vertices"]([0.0, 1.0, 163841.0])
    np.testing.assert_array_equal(vertices, [0, 1, 163841])
    assert np.issubdtype(vertices.dtype, np.integer)


@pytest.mark.parametrize("values, message", [
    ([np.nan], "finite integers"),
    ([np.inf], "finite integers"),
    ([1.5], "finite integers"),
    ([-1], "surface range"),
    ([163842], "surface range"),
    ([[1]], "one-dimensional"),
])
def test_invalid_vertex_ids(helpers, values, message):
    with pytest.raises(ValueError, match=message):
        helpers["validated_vertices"](values)


def test_inline_backend_selected_before_pyplot_import():
    first = next(embedded_code_cells(BUILDER))
    assert first.index('run_line_magic("matplotlib", "inline")') < first.index("import matplotlib.pyplot")


def test_public_builder_sources_compile_and_omit_private_paths():
    paths = [BUILDER, ROOT / "analysis/update_ghost_surface_notebook_2026_09_22.py",
             ROOT / "analysis/update_ghost_overlap_notebook_2026_09_22.py"]
    for path in paths:
        source = path.read_text()
        compile(source, str(path), "exec")
        assert "ANON_TO_CVN" not in source
        assert "cvn700" not in source and "cvn701" not in source
        assert "/n/holylabs/" not in source
        for cell in embedded_code_cells(path):
            compile(cell, str(path), "exec")


@pytest.fixture
def optional_glasser_loader():
    definition = next(
        node
        for source in embedded_code_cells(SURFACE_BUILDER)
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "load_optional_glasser"
    )
    namespace = {"np": np, "pd": pd}
    module = ast.Module(body=[definition], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(SURFACE_BUILDER), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("existing", [(), ("labels.mgz",), ("lookup.txt",)])
def test_missing_glasser_assets_are_optional(tmp_path, optional_glasser_loader, capsys, existing):
    for name in existing:
        (tmp_path / name).write_text("placeholder")
    optional_glasser_loader["read_mgh_labels"] = lambda path: pytest.fail("Missing asset pair was loaded")
    surface, ids = optional_glasser_loader["load_optional_glasser"](
        tmp_path / "labels.mgz", tmp_path / "lookup.txt", ["PGi", "TPOJ1"], 4
    )
    assert surface is None and ids == []
    assert "primary LTPJ analyses" in capsys.readouterr().out


def test_available_glasser_keeps_original_mask_calculation(tmp_path, optional_glasser_loader):
    labels = np.array([1, 2, 3, 2])
    labels_path, lookup_path = tmp_path / "labels.mgz", tmp_path / "lookup.txt"
    labels_path.write_text("reader stub")
    lookup_path.write_text("1 other\n2 PGi\n3 TPOJ1\n")
    optional_glasser_loader["read_mgh_labels"] = lambda path: labels
    surface, ids = optional_glasser_loader["load_optional_glasser"](
        labels_path, lookup_path, ["PGi", "TPOJ1"], 4
    )
    assert ids == [2, 3]
    np.testing.assert_array_equal(surface, np.isin(labels, ids))
    with pytest.raises(ValueError, match="shape"):
        optional_glasser_loader["load_optional_glasser"](
            labels_path, lookup_path, ["PGi", "TPOJ1"], 5
        )


def test_save_figure_creates_missing_parent_directory(tmp_path):
    definition = next(
        node
        for source in embedded_code_cells(SURFACE_BUILDER)
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "save_figure"
    )
    namespace = {"Path": Path}
    module = ast.Module(body=[definition], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(SURFACE_BUILDER), "exec"), namespace)
    saved = []

    class FigureStub:
        def savefig(self, filename, **kwargs):
            assert Path(filename).parent.is_dir()
            saved.append(Path(filename).suffix)

    stem = tmp_path / "new" / "figures" / "example"
    assert not stem.parent.exists()
    namespace["save_figure"](FigureStub(), stem)
    assert stem.parent.is_dir()
    assert saved == [".png", ".pdf"]


@pytest.mark.parametrize("glasser_available", [False, True])
def test_primary_bars_render_without_glasser(tmp_path, monkeypatch, capsys, glasser_available):
    # Run only the lightweight quantification cells on tiny synthetic data.
    # CSV/figure writes are stubbed; no notebook, data, or surface assets are changed.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nonaligned = pd.DataFrame({
        "neuroid": [0, 1], "Participant": ["p1", "p2"],
        "Inside Saxe functional LTPJ": [False, True],
    })
    aligned = pd.DataFrame({
        "neuroid": [2, 3], "Participant": ["p1", "p2"],
        "Inside Saxe functional LTPJ": [True, True],
    })
    saved = []

    def save_figure(fig, stem, **kwargs):
        saved.append(Path(stem).name)
        plt.close(fig)

    monkeypatch.setattr(pd.DataFrame, "to_csv", lambda *args, **kwargs: None)
    monkeypatch.setattr(plt, "show", lambda: None)
    namespace = {
        "pd": pd, "np": np, "plt": plt, "REPO_ROOT": tmp_path,
        "core_ghost": pd.concat([nonaligned, aligned], ignore_index=True),
        "core_well_nonaligned": nonaligned, "core_aligned": aligned,
        "ANATOMY_PARTICIPANTS": ["p1", "p2"],
        "GLASSER_AVAILABLE": glasser_available,
        "glasser_tpj_surface": np.array([False, True, True, False]) if glasser_available else None,
        "agreement_cmap": plt.get_cmap("plasma"), "save_figure": save_figure,
        "display": lambda value: None,
    }
    cells = list(embedded_code_cells(SURFACE_BUILDER))
    quantification = next(source for source in cells if source.startswith("FIGURE_DIR ="))
    exec(compile(quantification, "primary_quantification_cell", "exec"), namespace)
    np.testing.assert_array_equal(namespace["functional_pct"], [50, 100])
    assert len(namespace["tpj_reference_quantification"]) == (4 if glasser_available else 2)
    for stem in [
        "ghost_functional_ltpj_quantification_bar_2026-09-22",
        "ghost_probabilistic_functional_ltpj_alignment_horizontal_2026-09-22",
        "ghost_probabilistic_functional_ltpj_alignment_vertical_2026-09-22",
        "ghost_functional_ltpj_quantification_participants_2026-09-22",
    ]:
        assert stem in saved
    assert ("ghost_tpj_reference_alignment_comparison_2026-09-22" in saved) == glasser_available

    surfaces = []
    namespace["make_group_surface"] = lambda *args: surfaces.append(args[0])
    supplement = next(source for source in cells if source.startswith("if GLASSER_AVAILABLE:"))
    exec(compile(supplement, "glasser_supplement_cell", "exec"), namespace)
    assert surfaces == (["glasser"] if glasser_available else [])
    if not glasser_available:
        assert "Skipping the Glasser surface supplement" in capsys.readouterr().out


@pytest.fixture
def reviewer_table_helpers():
    selected = {
        "reviewer_ltpj_percent", "reviewer_ltpj_fraction", "reviewer_ltpj_counts",
        "build_reviewer_ltpj_tables",
    }
    definitions = [
        node
        for source in embedded_code_cells(BUILDER)
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name in selected
    ]
    assert {node.name for node in definitions} == selected
    namespace = {"np": np, "pd": pd}
    module = ast.Module(body=definitions, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(BUILDER), "exec"), namespace)
    return namespace


@pytest.fixture
def reviewer_candidates():
    # Repeated vertex IDs across participants intentionally count separately.
    values = {
        "p1": [(0.5, 7, True), (0.6, 8, False), (0.4, 7, True), (0.5, 6, False)],
        "p2": [(0.5, 7, True), (0.5, 5, False)],
        "p3": [(0.5, 7, False)],
        "p4": [(0.5, 7, False), (0.5, 6, True), (0.4, 7, True)],
        "p5": [(0.5, 7, True)],
        "p6": [(0.5, 5, False)],
        "p7": [(0.5, 5, True)],
        "p8": [(0.5, 7, True)],
    }
    return pd.DataFrame([
        {"Participant": participant, "neuroid": index,
         "NC Normalized R Fischer": r, "Average Num Features Agree Direction": agreement,
         "Inside Saxe functional LTPJ": inside}
        for participant, rows in values.items()
        for index, (r, agreement, inside) in enumerate(rows)
    ])


def test_reviewer_sensitivity_order_fixed_cohort_and_partition(reviewer_table_helpers, reviewer_candidates):
    baseline = ["p1", "p2", "p3", "p5", "p8"]
    participants = [f"p{i}" for i in range(1, 9)]
    sensitivity, _ = reviewer_table_helpers["build_reviewer_ltpj_tables"](
        reviewer_candidates, baseline, participants
    )
    assert sensitivity.analysis.tolist() == [
        "Original", "No participant exclusion", "Lower predictivity", "Higher predictivity",
        "Lower agreement", "Higher agreement",
    ]
    assert sensitivity.nc_normalized_r_threshold.tolist() == [0.5, 0.5, 0.4, 0.6, 0.5, 0.5]
    assert sensitivity.mean_direction_agreement_threshold.tolist() == [7, 7, 7, 7, 6, 8]
    assert sensitivity.participant_count.tolist() == [5, 8, 5, 5, 5, 5]
    assert sensitivity.candidate_count.tolist() == [9, 14, 9, 9, 9, 9]
    assert sensitivity.loc[[0, 2, 3, 4, 5], "participants"].eq(";".join(baseline)).all()
    assert sensitivity.loc[1, "participants"] == ";".join(participants)
    assert sensitivity.well_predicted_count.tolist() == [8, 12, 9, 1, 8, 8]
    assert sensitivity.aligned_count.tolist() == [6, 7, 7, 1, 7, 1]
    assert sensitivity.aligned_ltpj_count.tolist() == [4, 4, 5, 0, 4, 0]
    np.testing.assert_array_equal(
        sensitivity.well_predicted_count,
        sensitivity.aligned_count + sensitivity.well_predicted_nonaligned_count,
    )
    # Stricter agreement moves formerly aligned rows into the comparator.
    assert sensitivity.loc[5, "well_predicted_nonaligned_count"] == 7
    assert sensitivity.loc[5, "well_predicted_nonaligned_ltpj_count"] == 4


def test_reviewer_participants_include_zero_alignment_and_pooled_total(reviewer_table_helpers, reviewer_candidates):
    participants = [f"p{i}" for i in range(1, 9)]
    _, table = reviewer_table_helpers["build_reviewer_ltpj_tables"](
        reviewer_candidates, ["p1", "p2", "p3", "p5", "p8"], participants
    )
    assert table.participant.tolist() == [p.upper() for p in participants] + ["All eight (pooled)"]
    assert table.row_type.tolist() == ["participant"] * 8 + ["pooled_total"]
    pooled = table.iloc[-1]
    for column in (
        "candidate_count", "well_predicted_count", "aligned_count", "aligned_ltpj_count",
        "well_predicted_nonaligned_count", "well_predicted_nonaligned_ltpj_count",
    ):
        assert pooled[column] == table.iloc[:8][column].sum()
    assert pooled.candidate_count == len(reviewer_candidates)
    assert pooled.aligned_ltpj_percent == pytest.approx(100 * 4 / 7)
    p6 = table.loc[table.participant.eq("P6")].iloc[0]
    assert p6.aligned_count == 0 and np.isnan(p6.aligned_ltpj_percent)
    assert reviewer_table_helpers["reviewer_ltpj_fraction"](0, 0) == "—"
    assert reviewer_table_helpers["reviewer_ltpj_fraction"](0, 1) == "0/1 (0.0%)"


def test_reviewer_cell_displays_two_tables_and_saves_raw_csvs(tmp_path, reviewer_candidates):
    marker = "ghost-ltpj-reviewer-tables-2026-11-09"
    source = next(cell for cell in embedded_code_cells(BUILDER) if cell.startswith("# " + marker))
    displayed = []
    namespace = {
        "np": np, "pd": pd, "REPO_ROOT": tmp_path,
        "ghost": reviewer_candidates, "ANATOMY_PARTICIPANTS": ["p1", "p2", "p3", "p5", "p8"],
        "PARTICIPANTS": [f"p{i}" for i in range(1, 9)],
        "display": displayed.append,
    }
    exec(compile(source, "reviewer_ltpj_table_cell", "exec"), namespace)
    assert len(displayed) == 2
    directory = tmp_path / "analysis/figures"
    sensitivity = pd.read_csv(directory / "ghost_ltpj_threshold_sensitivity.csv")
    participants = pd.read_csv(directory / "ghost_ltpj_participant_counts.csv")
    assert len(sensitivity) == 6 and len(participants) == 9
    assert np.isnan(participants.loc[participants.participant.eq("P6"), "aligned_ltpj_percent"]).all()
    assert participants.loc[participants.participant.eq("P6"), "aligned_count"].item() == 0
    assert namespace["participant_display"].loc[
        namespace["participant_display"].Participant.eq("P6"), "Aligned in LTPJ, n/N (%)"
    ].item() == "—"
    assert not list(directory.glob("*.png")) and not list(directory.glob("*.pdf"))
