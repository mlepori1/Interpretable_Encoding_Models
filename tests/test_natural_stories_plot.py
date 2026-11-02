"""Check observation units, support counts, and saved-result compatibility.

Set NATURAL_STORIES_RESULTS_ROOT to the run directory to additionally compare
with existing results. That test reads saved fits; it never refits a model.
"""

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "replications/natural_stories/plot.py"
spec = importlib.util.spec_from_file_location("natural_stories_plot", SCRIPT)
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)


@pytest.fixture
def tables():
    actual, controls, selected = [], [], []
    models = plot.KEY_MODELS + plot.BUCKET_MODELS
    for i, roi in enumerate(plot.FROIS + ("Lang",)):
        for j, model in enumerate(models):
            for story in plot.STORIES:
                actual.append(dict(roi=roi, model=model, held_out_story_id=story,
                                   response_circular_shift_trs=0,
                                   r=.1 + i*.01 + j*.02 + story*.001))
        for shift in plot.SHIFTS:
            for story in plot.STORIES:
                controls.append(dict(roi=roi, model="residual", held_out_story_id=story,
                                     response_circular_shift_trs=shift,
                                     r=i*.01 + shift*.001 + story*.002))
        for story in plot.STORIES:
            for feature, coef in [(40, 1.), (79, -1.)] + ([(194, .5)] if story == 1 else []):
                selected.append(dict(roi=roi, model="sae_full", held_out_story_id=story,
                                     response_circular_shift_trs=0,
                                     feature_global_index=feature, ridge_coefficient=coef))
    return pd.DataFrame(actual), pd.DataFrame(controls), pd.DataFrame(selected)


def test_panel_a_story_and_shift_units_and_delta(tables):
    result = plot.summarize(*tables)
    summary = result["panel_a_summary"].set_index(["roi", "model"])
    actual = summary.loc[("IFG", "surprisal")]
    control = summary.loc[("IFG", "control")]
    assert actual.n_observations == 9
    assert control.n_observations == 4
    assert actual["sem"] == pytest.approx(pd.Series(plot.STORIES).std()/3*.001)
    assert control["sem"] == pytest.approx(pd.Series(plot.SHIFTS).std()/2*.001)
    delta = result["panel_a_deltas"].set_index("roi").loc["IFG"]
    assert delta.delta_r == pytest.approx(.04)
    assert delta.percent_increase == pytest.approx(100*.04/actual["mean"])


def test_panel_b_zero_bins_and_panel_c_average_frois_first(tables):
    result = plot.summarize(*tables)
    b = result["panel_b_summary"].set_index(["roi", "feature_bin"])
    assert b.loc[("IFG", "128-512"), "mean"] == pytest.approx(1/9)
    assert b.loc[("IFG", "128-512"), "sem"] == pytest.approx(1/9)
    assert b.loc[("IFG", "512-2048"), "mean"] == 0
    assert b.loc[("IFG", "512-2048"), "n_fits"] == 9
    points = result["panel_c_points"]
    assert points.n_frois.eq(5).all()
    expected = .1 + .02 + np.mean(plot.STORIES)*.001
    c = result["panel_c_summary"].set_index("model").loc["surprisal"]
    assert c["mean"] == pytest.approx(expected)
    assert c.n_stories == 9


def test_panel_d_signed_counts_and_target_choice(tables):
    all_targets = plot.summarize(*tables, feature_count_targets="all")
    five_targets = plot.summarize(*tables, feature_count_targets="frois")
    a = all_targets["panel_d_top15"]
    f = five_targets["panel_d_top15"]
    assert list(a.feature_global_index) == [40, 79, 194]
    assert list(a.direction) == ["positive", "negative", "positive"]
    assert list(a.fits_selected) == [54, 54, 6]
    assert list(f.fits_selected) == [45, 45, 5]
    assert a.total_fits.eq(54).all() and f.total_fits.eq(45).all()
    assert five_targets["panel_d_counts_by_roi"].total_folds.eq(9).all()
    pd.testing.assert_frame_equal(all_targets["panel_b_summary"], five_targets["panel_b_summary"])


def test_missing_or_duplicate_story_is_rejected(tables):
    actual, control, selected = tables
    with pytest.raises(ValueError, match="incomplete"):
        plot.summarize(actual.iloc[1:], control, selected)
    with pytest.raises(ValueError, match="duplicate"):
        plot.summarize(pd.concat([actual, actual.iloc[[0]]]), control, selected)


def test_duplicate_or_missing_feature_support_is_rejected(tables):
    actual, control, selected = tables
    with pytest.raises(ValueError, match="Duplicate selected-feature"):
        plot.summarize(actual, control, pd.concat([selected, selected.iloc[[0]]]))
    actual = actual.copy()
    actual["n_lm_features_selected"] = np.where(actual.held_out_story_id.eq(1), 3, 2)
    with pytest.raises(ValueError, match="recorded support sizes"):
        plot.summarize(actual, control, selected.iloc[1:])


def test_render_four_panels_and_provenance(tables, tmp_path):
    plot.render(plot.summarize(*tables, feature_count_targets="frois"), tmp_path)
    assert {p.stem for p in tmp_path.glob("*.pdf")} == set(plot.STEMS.values())
    assert {p.stem for p in tmp_path.glob("*.png")} == set(plot.STEMS.values())
    metadata = json.loads((tmp_path / "plot_metadata.json").read_text())
    assert metadata["feature_count_total_fits"] == 45
    assert metadata["font"] == "DejaVu Sans"
    assert (tmp_path / "panel_a_deltas.csv").is_file()


@pytest.mark.skipif(not os.environ.get("NATURAL_STORIES_RESULTS_ROOT"), reason="optional saved-data comparison")
def test_existing_saved_results(tmp_path):
    root = Path(os.environ["NATURAL_STORIES_RESULTS_ROOT"])
    actual, control, selected = plot.load_inputs(root / "encoding_actual", root / "encoding_circular_shift")
    result = plot.summarize(actual, control, selected, feature_count_targets="all")
    legacy = root / "summary"
    old_a = pd.read_csv(legacy / "predictivity_summary.csv").set_index(["roi", "model"])
    for row in result["panel_a_summary"].itertuples():
        if row.model != "control":
            assert row.mean == pytest.approx(old_a.loc[(row.roi, row.model), "actual_mean_r"], abs=1e-12)
            assert row.sem == pytest.approx(old_a.loc[(row.roi, row.model), "actual_sem_across_stories"], abs=1e-12)
    old_b = pd.read_csv(legacy / "matryoshka_selected_feature_counts_by_roi_summary.csv").set_index(["roi", "feature_bin"])
    for row in result["panel_b_summary"].itertuples():
        assert row.mean == pytest.approx(old_b.loc[(row.roi, row.feature_bin), "mean"], abs=1e-12)
        assert row.sem == pytest.approx(old_b.loc[(row.roi, row.feature_bin), "sem"], abs=1e-12)
    old_c = pd.read_csv(legacy / "matryoshka_bucket_predictivity_summary.csv").set_index("model")
    for row in result["panel_c_summary"].itertuples():
        assert row.mean == pytest.approx(old_c.loc[row.model, "mean"], abs=1e-12)
        assert row.sem == pytest.approx(old_c.loc[row.model, "sem"], abs=1e-12)
    old_d = pd.read_csv(legacy / "full_sae_signed_feature_selection_frequency.csv").head(15)
    columns = ["feature_global_index", "direction", "fits_selected"]
    pd.testing.assert_frame_equal(result["panel_d_top15"][columns].reset_index(drop=True), old_d[columns])
    five = plot.summarize(actual, control, selected, "frois")
    assert five["panel_d_top15"].iloc[0].fits_selected == 45
    assert five["panel_d_top15"].iloc[1].fits_selected == 45
    assert (194, "positive") in set(five["panel_d_top15"][["feature_global_index", "direction"]].itertuples(index=False, name=None))
    plot.render(result, tmp_path / "legacy54")
    plot.render(five, tmp_path / "five45")
