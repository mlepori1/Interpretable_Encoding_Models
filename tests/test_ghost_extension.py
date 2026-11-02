"""Test embedded notebook safety helpers without generating/executing notebooks."""

import ast
import base64
from pathlib import Path
import xml.etree.ElementTree as ET
import zlib

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "analysis/build_ghost_spotlight_extension_notebook.py"


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
