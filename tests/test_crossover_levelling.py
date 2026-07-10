import matplotlib as mpl  # pylint: disable=too-many-lines

mpl.use("Agg")

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

import airbornegeo.crossover_levelling as clv_mod
import airbornegeo.plotting as plotting_mod
from airbornegeo.crossover_levelling import (
    _crossover_network_levelling,
    _crossover_pair_levelling,
    _end_iterations,
    _line_network_crossover_errors,
    alternating_iterative_line_levelling,
    calculate_intersection_weights,
    crossover_network_levelling,
    crossover_pair_levelling,
    plot_levelling_convergence,
)
from airbornegeo.crossovers import (
    calculate_crossover_errors,
    create_intersection_table,
    interpolate_intersections,
)

CRS = "EPSG:3431"
BIAS = {"F1": 5.0, "F2": -3.0, "T1": 2.0, "T2": -1.0}


@pytest.fixture(autouse=True)
def _no_gui_plots(monkeypatch):
    monkeypatch.setattr(clv_mod.plt, "show", lambda *_a, **_k: None)
    yield
    clv_mod.plt.close("all")


def _line_points(name, eastings, northings, line_column="line", **extra_columns):
    frame = pd.DataFrame({"easting": eastings, "northing": northings, **extra_columns})
    frame[line_column] = name
    return gpd.GeoDataFrame(
        frame, geometry=gpd.points_from_xy(frame.easting, frame.northing), crs=CRS
    )


def _network_fixture(line_bias):
    """Two flight lines (type 0, horizontal) crossing two tie lines (type 1, vertical)
    at exactly-known grid coordinates, each carrying a constant per-line bias for the
    'value' column and no spatial trend. F1 crosses T1 at dist_along_line=200 and T2 at
    dist_along_line=700; F2 crosses T1 at dist=200 and T2 at dist=700 too. Because the
    field is a pure per-line constant (no spatial trend) and crossings land exactly on
    existing data points, max_dist is exactly 0 for every intersection and misties are
    EXACT: mistie(line1, line2) = line_bias[line1] - line_bias[line2], with zero
    interpolation error. This makes a degree=1 trend fit through a line's two crossings
    an EXACT fit (no residual), so hand-computed expected outputs are possible."""
    f1 = _line_points("F1", np.linspace(0, 1000, 21), [0] * 21, line_type=0)
    f2 = _line_points("F2", np.linspace(0, 1000, 21), [500] * 21, line_type=0)
    t1 = _line_points("T1", [200] * 15, np.linspace(-100, 600, 15), line_type=1)
    t2 = _line_points("T2", [700] * 15, np.linspace(-100, 600, 15), line_type=1)
    data = pd.concat([f1, f2, t1, t2], ignore_index=True)
    data["dist_along_line"] = data.groupby("line")["easting"].transform(
        lambda s: s - s.min()
    ) + data.groupby("line")["northing"].transform(lambda s: s - s.min())
    data["value"] = data["line"].map(line_bias)
    return data


def _leveling_ready(line_bias, method="groups"):
    """Build _network_fixture(line_bias), run it through the REAL crossovers.py pipeline
    (create_intersection_table -> interpolate_intersections) and return (filled, inters)
    ready to hand to crossover_pair_levelling / crossover_network_levelling. method is
    "groups" (uses line_type 0 vs 1, needed for crossover_pair_levelling and
    alternating_iterative_line_levelling) or "network" (any-line-crosses-any-line, needed
    for crossover_network_levelling)."""
    data = _network_fixture(line_bias)
    inters = create_intersection_table(data, line_column="line", method=method)
    filled, inters_valid = interpolate_intersections(
        data,
        inters,
        line_column="line",
        to_interp="value",
        interp_on="dist_along_line",
        method="linear",
    )
    return filled, inters_valid


# --------------------------------------------------------------------------- #
# _end_iterations
# --------------------------------------------------------------------------- #


def test_end_iterations_first_iteration_skips_mid_loop_checks_and_does_not_end():
    """On iteration 1, all mid-loop checks are skipped even when tolerances would otherwise be satisfied."""
    end, termination_reason = _end_iterations(
        rms_values=[1.0],
        delta_rms_values=[0.0],
        max_iterations=5,
        crossover_error_values=[1.0],
        rms_tolerance=100.0,
        rms_percent_change_tolerance=100.0,
        rms_percent_increase_tolerance=0.0,
    )
    assert end is False
    assert not termination_reason


def test_end_iterations_no_tolerances_provided_and_max_iterations_not_reached():
    """With every optional tolerance left as None and max_iterations not yet reached, the function ends nothing."""
    end, termination_reason = _end_iterations(
        rms_values=[10.0],
        delta_rms_values=[0.0],
        max_iterations=5,
    )
    assert end is False
    assert not termination_reason


def test_end_iterations_no_tolerances_provided_past_first_iteration_does_not_raise():
    """Regression test for a fixed bug: the 'RMS increasing' check used to run
    unconditionally whenever iteration > 1, dividing by rms_percent_increase_tolerance
    without a None-guard (unlike the other two tolerance checks), so calling this with
    rms_percent_increase_tolerance left at its default None and a 2+ element history
    raised TypeError. It's now guarded the same way as the other checks, so this should
    just end nothing, not raise."""
    end, termination_reason = _end_iterations(
        rms_values=[10.0, 9.0],
        delta_rms_values=[0.0, 0.0],
        max_iterations=5,
    )
    assert end is False
    assert not termination_reason


def test_end_iterations_max_iterations_one_with_single_iteration_only_fires_max_iterations():
    """With max_iterations=1 and a single-element history (iteration==1), only 'max iterations' fires even though rms_tolerance is trivially satisfied."""
    end, termination_reason = _end_iterations(
        rms_values=[1.0],
        delta_rms_values=[0.0],
        max_iterations=1,
        rms_tolerance=100.0,
    )
    assert end is True
    assert termination_reason == ["max iterations"]


def test_end_iterations_rms_increasing_above_percent_increase_tolerance():
    """RMS rising more than rms_percent_increase_tolerance percent above the minimum RMS so far triggers 'RMS increasing'."""
    end, termination_reason = _end_iterations(
        rms_values=[10.0, 13.0],
        delta_rms_values=[0.0, 0.0],
        max_iterations=5,
        rms_percent_increase_tolerance=20.0,
    )
    assert end is True
    assert termination_reason == ["RMS increasing"]


def test_end_iterations_rms_not_increasing_when_equal_to_minimum():
    """RMS equal to the minimum so far (not strictly greater) does not trigger 'RMS increasing'."""
    end, termination_reason = _end_iterations(
        rms_values=[10.0, 9.0, 8.5],
        delta_rms_values=[50.0, 4.0, 3.0],
        max_iterations=10,
        rms_percent_increase_tolerance=20.0,
    )
    assert end is False
    assert not termination_reason


def test_end_iterations_rms_percent_change_plateaued_over_two_iterations():
    """Both the current and previous delta RMS at or below rms_percent_change_tolerance triggers 'RMS percent change tolerance'."""
    end, termination_reason = _end_iterations(
        rms_values=[10.0, 9.0, 8.5],
        delta_rms_values=[50.0, 4.0, 3.0],
        max_iterations=10,
        rms_percent_increase_tolerance=20.0,
        rms_percent_change_tolerance=5.0,
    )
    assert end is True
    assert termination_reason == ["RMS percent change tolerance"]


def test_end_iterations_rms_percent_change_not_plateaued_on_second_iteration():
    """On iteration 2, previous_delta_rms defaults to infinity, so a low current delta_rms alone cannot trigger 'RMS percent change tolerance'."""
    end, termination_reason = _end_iterations(
        rms_values=[10.0, 9.0],
        delta_rms_values=[50.0, 3.0],
        max_iterations=5,
        rms_percent_increase_tolerance=20.0,
        rms_percent_change_tolerance=5.0,
    )
    assert end is False
    assert not termination_reason


def test_end_iterations_rms_below_tolerance():
    """RMS strictly below rms_tolerance triggers 'RMS tolerance'."""
    end, termination_reason = _end_iterations(
        rms_values=[10.0, 4.0],
        delta_rms_values=[0.0, 0.0],
        max_iterations=5,
        rms_percent_increase_tolerance=20.0,
        rms_tolerance=5.0,
    )
    assert end is True
    assert termination_reason == ["RMS tolerance"]


def test_end_iterations_rms_at_tolerance_does_not_trigger():
    """RMS equal to rms_tolerance (not strictly below) does not trigger 'RMS tolerance'."""
    end, termination_reason = _end_iterations(
        rms_values=[10.0, 5.0],
        delta_rms_values=[0.0, 0.0],
        max_iterations=5,
        rms_percent_increase_tolerance=20.0,
        rms_tolerance=5.0,
    )
    assert end is False
    assert not termination_reason


def test_end_iterations_crossover_mistie_increasing():
    """On iteration 3, a crossover mistie greater than the previous iteration's mistie triggers 'cross-over RMS increasing'."""
    end, termination_reason = _end_iterations(
        rms_values=[10.0, 8.0, 7.0],
        delta_rms_values=[50.0, 20.0, 10.0],
        max_iterations=10,
        crossover_error_values=[5.0, 3.0, 4.0],
        rms_percent_increase_tolerance=20.0,
    )
    assert end is True
    assert termination_reason == ["cross-over RMS increasing"]


def test_end_iterations_crossover_mistie_increase_not_checked_on_second_iteration():
    """On iteration 2, previous_mistie defaults to infinity, so a mistie increase relative to iteration 1 cannot trigger 'cross-over RMS increasing'."""
    end, termination_reason = _end_iterations(
        rms_values=[10.0, 9.0],
        delta_rms_values=[0.0, 0.0],
        max_iterations=5,
        crossover_error_values=[3.0, 10.0],
        rms_percent_increase_tolerance=20.0,
    )
    assert end is False
    assert not termination_reason


def test_end_iterations_max_iterations_reached_fires_alone():
    """Reaching max_iterations always appends 'max iterations', independent of and regardless of the other checks."""
    end, termination_reason = _end_iterations(
        rms_values=[10.0, 9.0],
        delta_rms_values=[0.0, 0.0],
        max_iterations=2,
        rms_percent_increase_tolerance=1000.0,
    )
    assert end is True
    assert termination_reason == ["max iterations"]


def test_end_iterations_max_iterations_and_rms_tolerance_both_accumulate():
    """When both max_iterations is reached and rms_tolerance is satisfied in the same call, both reasons accumulate in check order."""
    end, termination_reason = _end_iterations(
        rms_values=[10.0, 3.0],
        delta_rms_values=[0.0, 0.0],
        max_iterations=2,
        rms_percent_increase_tolerance=20.0,
        rms_tolerance=5.0,
    )
    assert end is True
    assert termination_reason == ["RMS tolerance", "max iterations"]


# --------------------------------------------------------------------------- #
# _line_network_crossover_errors
# --------------------------------------------------------------------------- #


def test_line_only_as_line1_keeps_sign_and_filters_other_lines():
    """A line appearing only as line1 should use dist_along_line1 and crossover_error_col unchanged, excluding other lines' rows."""
    inters = pd.DataFrame(
        {
            "line1": ["A", "A", "B"],
            "line2": ["B", "C", "C"],
            "dist_along_line1": [10.0, 15.0, 1.0],
            "dist_along_line2": [20.0, 25.0, 2.0],
            "crossover_error_0": [2.0, 3.0, 99.0],
        }
    )
    result = _line_network_crossover_errors(inters, "A", "crossover_error_0")
    assert list(result.columns) == ["dist_along_line", "network_mistie"]
    assert len(result) == 2
    assert result["dist_along_line"].tolist() == pytest.approx([10.0, 15.0])
    assert result["network_mistie"].tolist() == pytest.approx([2.0, 3.0])


def test_line_only_as_line2_negates_mistie():
    """A line appearing only as line2 should use dist_along_line2 and a negated crossover_error_col."""
    inters = pd.DataFrame(
        {
            "line1": ["B", "C", "B"],
            "line2": ["A", "A", "C"],
            "dist_along_line1": [20.0, 25.0, 1.0],
            "dist_along_line2": [10.0, 15.0, 2.0],
            "crossover_error_0": [2.0, 3.0, 99.0],
        }
    )
    result = _line_network_crossover_errors(inters, "A", "crossover_error_0")
    assert len(result) == 2
    assert result["dist_along_line"].tolist() == pytest.approx([10.0, 15.0])
    assert result["network_mistie"].tolist() == pytest.approx([-2.0, -3.0])


def test_line_as_both_line1_and_line2_combines_with_correct_signs():
    """A line appearing as both line1 and line2 should get concatenated rows, unflipped on the line1 side and negated on the line2 side."""
    inters = pd.DataFrame(
        {
            "line1": ["A", "C", "B"],
            "line2": ["B", "A", "C"],
            "dist_along_line1": [10.0, 5.0, 1.0],
            "dist_along_line2": [20.0, 30.0, 2.0],
            "crossover_error_0": [2.0, -4.0, 99.0],
        }
    )
    result = _line_network_crossover_errors(inters, "A", "crossover_error_0")
    expected = pd.DataFrame(
        {"dist_along_line": [10.0, 30.0], "network_mistie": [2.0, 4.0]}
    )
    pd.testing.assert_frame_equal(result, expected)


def test_intersection_weight_col_carried_through_unmodified_on_both_sides():
    """The weight column should pass through unchanged for both line1-side and line2-side rows, unlike network_mistie."""
    inters = pd.DataFrame(
        {
            "line1": ["A", "C"],
            "line2": ["B", "A"],
            "dist_along_line1": [10.0, 5.0],
            "dist_along_line2": [20.0, 30.0],
            "crossover_error_0": [2.0, -4.0],
            "weight": [0.5, 0.8],
        }
    )
    result = _line_network_crossover_errors(
        inters, "A", "crossover_error_0", intersection_weight_col="weight"
    )
    expected = pd.DataFrame(
        {
            "dist_along_line": [10.0, 30.0],
            "network_mistie": [2.0, 4.0],
            "weight": [0.5, 0.8],
        }
    )
    pd.testing.assert_frame_equal(result, expected)


def test_line_with_no_intersections_returns_empty_dataframe_with_expected_columns():
    """A line absent from both line1 and line2 should return an empty dataframe with the right columns, not raise."""
    inters = pd.DataFrame(
        {
            "line1": ["B"],
            "line2": ["C"],
            "dist_along_line1": [1.0],
            "dist_along_line2": [2.0],
            "crossover_error_0": [99.0],
        }
    )
    result = _line_network_crossover_errors(inters, "A", "crossover_error_0")
    assert result.empty
    assert list(result.columns) == ["dist_along_line", "network_mistie"]


def test_row_order_does_not_affect_result_and_index_is_reset():
    """Row order in inters shouldn't change the combined result, and the output index should be a fresh RangeIndex."""
    inters = pd.DataFrame(
        {
            "line1": ["C", "A"],
            "line2": ["A", "B"],
            "dist_along_line1": [5.0, 10.0],
            "dist_along_line2": [30.0, 20.0],
            "crossover_error_0": [-4.0, 2.0],
        }
    )
    result = _line_network_crossover_errors(inters, "A", "crossover_error_0")
    assert result.index.tolist() == list(range(len(result)))
    assert sorted(result["dist_along_line"].tolist()) == pytest.approx([10.0, 30.0])
    assert sorted(result["network_mistie"].tolist()) == pytest.approx([2.0, 4.0])


# --------------------------------------------------------------------------- #
# crossover_pair_levelling / _crossover_pair_levelling
# --------------------------------------------------------------------------- #


def test_crossover_pair_levelling_exact_correction_on_two_point_fit():
    """A degree=1 fit through exactly 2 crossings should exactly reproduce the reference lines' values at those crossings."""
    filled, inters_valid = _leveling_ready(BIAS, method="groups")
    data, inters = crossover_pair_levelling(
        filled,
        inters_valid,
        lines_to_level=["F1", "F2"],
        data_col="value",
        levelled_col="value_leveled",
        line_column="line",
        distance_column="dist_along_line",
        degree=1,
        max_iterations=1,
        plot_convergence=False,
        progressbar=False,
    )

    f1 = data[data.line == "F1"]

    def _value_at(dist):
        rows = f1[f1.dist_along_line == dist]
        assert rows.value_leveled.min() == rows.value_leveled.max()
        return rows.value_leveled.iloc[0]

    assert _value_at(0) == pytest.approx(3.2, abs=1e-6)
    assert _value_at(200) == pytest.approx(2.0, abs=1e-6)
    assert _value_at(700) == pytest.approx(-1.0, abs=1e-6)
    assert _value_at(1000) == pytest.approx(-2.8, abs=1e-6)

    for line in ["T1", "T2"]:
        subset = data[data.line == line]
        assert subset.value_leveled.to_numpy() == pytest.approx(
            subset.value.to_numpy(), abs=1e-9
        )
        assert (subset.value_leveled == BIAS[line]).all()

    error_cols = [c for c in inters.columns if c.startswith("crossover_error_")]
    final_col = f"crossover_error_{max(int(c.split('_')[-1]) for c in error_cols)}"
    assert inters[final_col].to_numpy() == pytest.approx([0.0, 0.0, 0.0, 0.0], abs=1e-9)


def test_crossover_pair_levelling_subsets_lines_to_level():
    """Levelling only F1 should leave F2's value_leveled equal to its original data_col."""
    filled, inters_valid = _leveling_ready(BIAS, method="groups")
    data, _inters = crossover_pair_levelling(
        filled,
        inters_valid,
        lines_to_level=["F1"],
        data_col="value",
        levelled_col="value_leveled",
        line_column="line",
        distance_column="dist_along_line",
        degree=1,
        max_iterations=1,
        plot_convergence=False,
        progressbar=False,
    )

    f2 = data[data.line == "F2"]
    assert f2.value_leveled.to_numpy() == pytest.approx(f2.value.to_numpy())

    f1 = data[data.line == "F1"]
    assert not np.allclose(f1.value_leveled.to_numpy(), f1.value.to_numpy())


def test__crossover_pair_levelling_degree_and_filter_type_raises_userwarning():
    """Providing both degree and filter_type should raise UserWarning naming both options."""
    filled, inters_valid = _leveling_ready(BIAS, method="groups")
    with pytest.raises(
        UserWarning, match="only provide either `filter_type` or `degree`, not both"
    ):
        _crossover_pair_levelling(
            filled,
            inters_valid,
            lines_to_level=["F1", "F2"],
            data_col="value",
            levelled_col="value_leveled",
            line_column="line",
            distance_column="dist_along_line",
            degree=1,
            filter_type="g300",
        )


def test_crossover_pair_levelling_degree_and_filter_type_public_wrapper_swallows_warning():
    """The public wrapper catches the UserWarning internally and returns after zero successful iterations, leaving data unmodified aside from the missing levelled_col."""
    filled, inters_valid = _leveling_ready(BIAS, method="groups")
    data, inters = crossover_pair_levelling(
        filled,
        inters_valid,
        lines_to_level=["F1", "F2"],
        data_col="value",
        levelled_col="value_leveled",
        line_column="line",
        distance_column="dist_along_line",
        degree=1,
        filter_type="g300",
        max_iterations=1,
        plot_convergence=False,
        progressbar=False,
    )
    assert "value_leveled" not in data.columns
    assert data.value.to_numpy() == pytest.approx(filled.value.to_numpy())
    assert list(inters.columns) == list(inters_valid.columns)


def test__crossover_pair_levelling_no_degree_no_filter_type_raises_userwarning():
    """Providing neither degree nor filter_type should raise UserWarning."""
    filled, inters_valid = _leveling_ready(BIAS, method="groups")
    with pytest.raises(
        UserWarning, match="must provide either `filter_type` or `degree`"
    ):
        _crossover_pair_levelling(
            filled,
            inters_valid,
            lines_to_level=["F1", "F2"],
            data_col="value",
            levelled_col="value_leveled",
            line_column="line",
            distance_column="dist_along_line",
        )


def test__crossover_pair_levelling_mixed_line_types_raises_valueerror():
    """Supplying line names from both intersection sides in lines_to_level should raise ValueError."""
    filled, inters_valid = _leveling_ready(BIAS, method="groups")
    with pytest.raises(ValueError, match="Supplied two types of lines to be levelled!"):
        _crossover_pair_levelling(
            filled,
            inters_valid,
            lines_to_level=["F1", "T1"],
            data_col="value",
            levelled_col="value_leveled",
            line_column="line",
            distance_column="dist_along_line",
            degree=1,
        )


def test_crossover_pair_levelling_mixed_line_types_propagates_through_public_wrapper():
    """The ValueError for mixed line types is not a UserWarning, so it propagates through the public wrapper too."""
    filled, inters_valid = _leveling_ready(BIAS, method="groups")
    with pytest.raises(ValueError, match="Supplied two types of lines to be levelled!"):
        crossover_pair_levelling(
            filled,
            inters_valid,
            lines_to_level=["F1", "T1"],
            data_col="value",
            levelled_col="value_leveled",
            line_column="line",
            distance_column="dist_along_line",
            degree=1,
            max_iterations=1,
            plot_convergence=False,
            progressbar=False,
        )


def test_crossover_pair_levelling_filter_type_runs_without_nans():
    """A GMT low-pass filter string should run and produce a levelled_col with no NaNs on the levelled lines."""
    filled, inters_valid = _leveling_ready(BIAS, method="groups")
    data, _inters = crossover_pair_levelling(
        filled,
        inters_valid,
        lines_to_level=["F1", "F2"],
        data_col="value",
        levelled_col="value_leveled",
        line_column="line",
        distance_column="dist_along_line",
        filter_type="g300",
        max_iterations=1,
        plot_convergence=False,
        progressbar=False,
    )
    levelled = data[data.line.isin(["F1", "F2"])]
    assert not levelled.value_leveled.isna().any()


def test_crossover_pair_levelling_intersection_weight_col_accepted():
    """Supplying a uniform intersection_weight_col should be accepted and match the unweighted result."""
    filled, inters_valid = _leveling_ready(BIAS, method="groups")
    inters_weighted = inters_valid.copy()
    inters_weighted["weight"] = 1.0

    unweighted, _ = crossover_pair_levelling(
        filled,
        inters_valid,
        lines_to_level=["F1", "F2"],
        data_col="value",
        levelled_col="value_leveled",
        line_column="line",
        distance_column="dist_along_line",
        degree=1,
        max_iterations=1,
        plot_convergence=False,
        progressbar=False,
    )
    weighted, _ = crossover_pair_levelling(
        filled,
        inters_weighted,
        lines_to_level=["F1", "F2"],
        data_col="value",
        levelled_col="value_leveled",
        line_column="line",
        distance_column="dist_along_line",
        intersection_weight_col="weight",
        degree=1,
        max_iterations=1,
        plot_convergence=False,
        progressbar=False,
    )
    assert weighted.value_leveled.to_numpy() == pytest.approx(
        unweighted.value_leveled.to_numpy()
    )


def test_crossover_pair_levelling_multiple_iterations_terminate_cleanly():
    """A tight rms_tolerance forcing multiple attempted iterations should still terminate and return valid data."""
    filled, inters_valid = _leveling_ready(BIAS, method="groups")
    data, inters = crossover_pair_levelling(
        filled,
        inters_valid,
        lines_to_level=["F1", "F2"],
        data_col="value",
        levelled_col="value_leveled",
        line_column="line",
        distance_column="dist_along_line",
        degree=1,
        max_iterations=3,
        rms_tolerance=1e-6,
        raise_error_if_unchanged=True,
        plot_convergence=False,
        progressbar=False,
    )
    assert len(data) == len(filled)
    assert not data.value_leveled.isna().any()
    error_cols = [c for c in inters.columns if c.startswith("crossover_error_")]
    final_col = f"crossover_error_{max(int(c.split('_')[-1]) for c in error_cols)}"
    assert inters[final_col].to_numpy() == pytest.approx([0.0, 0.0, 0.0, 0.0], abs=1e-9)


def test_crossover_pair_levelling_plot_convergence_runs_with_multiple_iterations():
    """plot_convergence=True with enough iterations to exceed 2 should trigger the convergence plot without error."""
    filled, inters_valid = _leveling_ready(BIAS, method="groups")
    data, _inters = crossover_pair_levelling(
        filled,
        inters_valid,
        lines_to_level=["F1", "F2"],
        data_col="value",
        levelled_col="value_leveled",
        line_column="line",
        distance_column="dist_along_line",
        degree=1,
        max_iterations=5,
        rms_tolerance=1e-30,
        raise_error_if_unchanged=False,
        plot_convergence=True,
        progressbar=False,
    )
    assert not data.value_leveled.isna().any()


def test_crossover_pair_levelling_trend_found_array_error_zeroes_correction(
    monkeypatch,
):
    """When airbornegeo.trend raises a ValueError whose message contains 'Found array with'
    (sklearn's empty-array error), the except (ValueError, UserWarning) handler logs and
    falls back to a zero levelling_correction instead of propagating, matching the same
    handling as a UserWarning."""
    filled, inters_valid = _leveling_ready(BIAS, method="groups")

    def fake_trend(*_args, **_kwargs):
        msg = "Found array with 0 sample(s) (shape=(0, 1)) while a minimum of 1 is required by LinearRegression."
        raise ValueError(msg)

    monkeypatch.setattr(clv_mod.airbornegeo, "trend", fake_trend)

    data, _inters = _crossover_pair_levelling(
        filled,
        inters_valid,
        lines_to_level=["F1", "F2"],
        data_col="value",
        levelled_col="value_leveled",
        line_column="line",
        distance_column="dist_along_line",
        degree=1,
        raise_error_if_unchanged=False,
    )
    for line in ["F1", "F2"]:
        subset = data[data.line == line]
        assert subset.value_leveled.to_numpy() == pytest.approx(subset.value.to_numpy())


def test_crossover_pair_levelling_trend_other_valueerror_is_reraised(monkeypatch):
    """A ValueError from airbornegeo.trend that isn't the 'Found array with' sklearn
    message should be re-raised (wrapped in a new ValueError) rather than silently
    swallowed."""
    filled, inters_valid = _leveling_ready(BIAS, method="groups")

    def fake_trend(*_args, **_kwargs):
        msg = "some unrelated failure"
        raise ValueError(msg)

    monkeypatch.setattr(clv_mod.airbornegeo, "trend", fake_trend)

    # the source re-raises via `raise ValueError from e` with no message of its own
    with pytest.raises(ValueError):  # noqa: PT011
        _crossover_pair_levelling(
            filled,
            inters_valid,
            lines_to_level=["F1", "F2"],
            data_col="value",
            levelled_col="value_leveled",
            line_column="line",
            distance_column="dist_along_line",
            degree=1,
            raise_error_if_unchanged=False,
        )


def _mismatched_pair_fixture():
    """Minimal hand-built data/inters for an F1/T1 crossing plus an unrelated, untouched
    F2/T2 crossing, bypassing the real create_intersection_table/interpolate_intersections
    pipeline. calculate_crossover_errors matches purely on (line, intersecting_line,
    easting, northing), so a NaN data_col value at the intersection row on F1 propagates
    straight into a NaN crossover_error_0 for that pair, letting us trigger the
    'matched-but-NaN' no-valid-misties path deterministically. The extra F2/T2 crossing
    (left out of lines_to_level, so passed through unchanged) keeps at least one
    non-NaN mistie in the final inters table, avoiding an all-NaN RMSE warning."""
    data = pd.DataFrame(
        {
            "line": ["F1", "F1", "T1", "T1", "F2", "F2", "T2", "T2"],
            "line_type": [0, 0, 1, 1, 0, 0, 1, 1],
            "dist_along_line": [0.0, 50.0, 0.0, 50.0, 0.0, 50.0, 0.0, 50.0],
            "value": [1.0, np.nan, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
            "easting": [10.0, 100.0, 200.0, 100.0, 20.0, 300.0, 400.0, 300.0],
            "northing": [10.0, 100.0, 200.0, 100.0, 20.0, 300.0, 400.0, 300.0],
            "is_intersection": [False, True, False, True, False, True, False, True],
            "intersecting_line": [None, "T1", None, "F1", None, "T2", None, "F2"],
        }
    )
    inters = pd.DataFrame(
        {
            "line1": ["F1", "F2"],
            "line2": ["T1", "T2"],
            "easting": [100.0, 300.0],
            "northing": [100.0, 300.0],
            "dist_along_line1": [50.0, 50.0],
            "dist_along_line2": [50.0, 50.0],
        }
    )
    return data, inters


def test_crossover_pair_levelling_filter_type_no_valid_misties_zeroes_correction(
    caplog,
):
    """When the matched mistie value for a line's only intersection is NaN (e.g. the
    underlying data value at that crossing is missing), n_valid_misties ends up 0 and the
    filter_type branch logs a warning and zero-fills the levelling correction rather than
    interpolating/filtering."""
    data, inters = _mismatched_pair_fixture()
    result, _inters = _crossover_pair_levelling(
        data,
        inters,
        lines_to_level=["F1"],
        data_col="value",
        levelled_col="value_leveled",
        line_column="line",
        distance_column="dist_along_line",
        filter_type="g300",
        raise_error_if_unchanged=False,
    )
    f1 = result[result.line == "F1"]
    assert f1.value_leveled.to_numpy() == pytest.approx(
        f1.value.to_numpy(), nan_ok=True
    )
    assert "no mistie values could be matched" in caplog.text


def test_crossover_pair_levelling_no_matching_line_type_raises_valueerror():
    """lines_to_level containing a line absent from both inters.line1 and inters.line2
    should hit the final 'else: raise ValueError' branch used when neither type could be
    determined."""
    data, inters = _mismatched_pair_fixture()
    with pytest.raises(
        ValueError, match="need to supplied either lines of type 0 or 1"
    ):
        _crossover_pair_levelling(
            data,
            inters,
            lines_to_level=["NOPE"],
            data_col="value",
            levelled_col="value_leveled",
            line_column="line",
            distance_column="dist_along_line",
            degree=1,
        )


def test_crossover_pair_levelling_wrapper_crossover_error_col_fallback(monkeypatch):
    """Regression coverage for the `except ValueError: current_crossover_error_col =
    'crossover_error_0'` fallback in the public crossover_pair_levelling wrapper: if
    `intersection_weight_col` happens to equal the sole crossover_error_* column name,
    that column gets excluded from the max()-search list, so max() raises ValueError and
    the code falls back to the literal name 'crossover_error_0' (which, in this contrived
    case, is exactly the excluded column, so the subsequent lookup still succeeds)."""
    filled, inters_valid = _leveling_ready(BIAS, method="groups")

    def fake_private(data, inters, **_kwargs):
        d = data.copy()
        d["value_leveled"] = d["value"]
        i = inters.copy()
        i["crossover_error_0"] = 0.0
        return d, i

    monkeypatch.setattr(clv_mod, "_crossover_pair_levelling", fake_private)

    data, inters = crossover_pair_levelling(
        filled,
        inters_valid,
        lines_to_level=["F1", "F2"],
        data_col="value",
        levelled_col="value_leveled",
        line_column="line",
        distance_column="dist_along_line",
        degree=1,
        intersection_weight_col="crossover_error_0",
        max_iterations=1,
        plot_convergence=False,
        progressbar=False,
    )
    assert "crossover_error_0" in inters.columns
    assert not data["value_leveled"].isna().any()


# --------------------------------------------------------------------------- #
# crossover_network_levelling / _crossover_network_levelling
# --------------------------------------------------------------------------- #


def _isolated_line_fixture(line_bias):
    """Same as _network_fixture, plus a fifth line 'ISO' far away in coordinate space
    that never crosses F1, F2, T1 or T2."""
    data = _network_fixture(line_bias)
    iso = _line_points("ISO", np.linspace(5000, 6000, 5), [5000] * 5, line_type=0)
    iso["dist_along_line"] = iso["easting"] - iso["easting"].min()
    iso["value"] = line_bias.get("ISO", 42.0)
    return pd.concat([data, iso], ignore_index=True)


class _FakeDisplayHandle:
    def __init__(self):
        self.update_calls = 0

    def update(self, *_args, **_kwargs):
        self.update_calls += 1


def _leveling_ready_with_isolated_line(line_bias, method="network"):
    data = _isolated_line_fixture(line_bias)
    inters = create_intersection_table(data, line_column="line", method=method)
    filled, inters_valid = interpolate_intersections(
        data,
        inters,
        line_column="line",
        to_interp="value",
        interp_on="dist_along_line",
        method="linear",
    )
    return filled, inters_valid


def test_crossover_network_levelling_converges_and_keeps_single_error_column():
    """Network levelling on the exact-fit fixture converges crossover_error_0 to ~0 for all 4 pairs, without adding new crossover_error columns even after 3 iterations."""
    filled, inters = _leveling_ready(BIAS, method="network")
    _data, new_inters = crossover_network_levelling(
        filled,
        inters,
        data_col="value",
        levelled_col="value_leveled",
        line_column="line",
        distance_column="dist_along_line",
        degree=1,
        relaxation_factor=0.5,
        max_iterations=3,
        plot_convergence=False,
        progressbar=False,
        raise_error_if_unchanged=False,
    )
    assert new_inters["crossover_error_0"].to_numpy() == pytest.approx(
        [0, 0, 0, 0], abs=1e-9
    )
    assert [c for c in new_inters.columns if "crossover_error_" in c] == [
        "crossover_error_0"
    ]


def test_crossover_network_levelling_default_raise_error_stops_cleanly():
    """The default raise_error_if_unchanged=True should not surface the internal 'Mistie hasn't changed' UserWarning to the caller once converged, and still return fully levelled data."""
    filled, inters = _leveling_ready(BIAS, method="network")
    data, new_inters = crossover_network_levelling(
        filled,
        inters,
        data_col="value",
        levelled_col="value_leveled",
        line_column="line",
        distance_column="dist_along_line",
        degree=1,
        relaxation_factor=0.5,
        max_iterations=3,
        plot_convergence=False,
        progressbar=False,
    )
    assert not data["value_leveled"].isna().any()
    assert new_inters["crossover_error_0"].to_numpy() == pytest.approx(
        [0, 0, 0, 0], abs=1e-9
    )
    assert [c for c in new_inters.columns if "crossover_error_" in c] == [
        "crossover_error_0"
    ]


def test_relaxation_factor_damps_single_iteration_correction():
    """With max_iterations=1, an undamped relaxation_factor=1.0 correction moves the mistie further from its starting value in a single iteration than a damped relaxation_factor=0.5 correction does.

    Empirically verified on this fixture (each line touches exactly 2 others,
    so both ends of a crossing move symmetrically): relaxation_factor=0.5
    lands exactly on 0 mistie in one shot (both lines meet halfway), while
    relaxation_factor=1.0 overshoots past the target and flips the mistie's
    sign (same magnitude, opposite sign) rather than cancelling it -- i.e. the
    undamped correction is a *larger* single-iteration change even though it
    does not converge as cleanly, matching the docstring's warning that
    values close to 1 "correct faster but are more likely to
    overshoot/oscillate."
    """
    filled, inters = _leveling_ready(BIAS, method="network")
    original_mistie = calculate_crossover_errors(
        filled, inters, data_col="value", line_column="line"
    )["crossover_error_0"].to_numpy()

    _data_full, result_full = crossover_network_levelling(
        filled,
        inters,
        data_col="value",
        levelled_col="value_leveled",
        line_column="line",
        distance_column="dist_along_line",
        degree=1,
        relaxation_factor=1.0,
        max_iterations=1,
        plot_convergence=False,
        progressbar=False,
        raise_error_if_unchanged=False,
    )
    _data_damped, result_damped = crossover_network_levelling(
        filled,
        inters,
        data_col="value",
        levelled_col="value_leveled",
        line_column="line",
        distance_column="dist_along_line",
        degree=1,
        relaxation_factor=0.5,
        max_iterations=1,
        plot_convergence=False,
        progressbar=False,
        raise_error_if_unchanged=False,
    )
    change_full = np.abs(result_full["crossover_error_0"].to_numpy() - original_mistie)
    change_damped = np.abs(
        result_damped["crossover_error_0"].to_numpy() - original_mistie
    )
    assert (change_full > change_damped).all()


def test_lines_to_level_subset_only_levels_named_lines():
    """Passing lines_to_level=["F1"] should only change F1's levelled values; F2/T1/T2 pass through unchanged."""
    filled, inters = _leveling_ready(BIAS, method="network")
    data, _inters = crossover_network_levelling(
        filled,
        inters,
        data_col="value",
        levelled_col="value_leveled",
        line_column="line",
        distance_column="dist_along_line",
        lines_to_level=["F1"],
        degree=1,
        relaxation_factor=0.5,
        max_iterations=1,
        plot_convergence=False,
        progressbar=False,
        raise_error_if_unchanged=False,
    )
    f1 = data[data.line == "F1"]
    assert not np.allclose(f1["value_leveled"], f1["value"])
    for line in ["F2", "T1", "T2"]:
        subset = data[data.line == line]
        assert subset["value_leveled"].to_numpy() == pytest.approx(
            subset["value"].to_numpy()
        )


def test_network_levelling_filter_type_runs_without_nans():
    """Using filter_type instead of degree should run without error and produce no NaNs in the levelled column."""
    filled, inters = _leveling_ready(BIAS, method="network")
    data, _inters = crossover_network_levelling(
        filled,
        inters,
        data_col="value",
        levelled_col="value_leveled",
        line_column="line",
        distance_column="dist_along_line",
        filter_type="g300",
        relaxation_factor=0.5,
        max_iterations=1,
        plot_convergence=False,
        progressbar=False,
        raise_error_if_unchanged=False,
    )
    assert not data["value_leveled"].isna().any()


def test_private_network_levelling_raises_when_both_degree_and_filter_type_given():
    """Passing both degree and filter_type should raise UserWarning from the private single-iteration helper."""
    filled, inters = _leveling_ready(BIAS, method="network")
    with pytest.raises(UserWarning, match="only provide either"):
        _crossover_network_levelling(
            filled,
            inters,
            data_col="value",
            levelled_col="value_leveled",
            line_column="line",
            distance_column="dist_along_line",
            degree=1,
            filter_type="g300",
        )


def test_private_network_levelling_raises_when_neither_degree_nor_filter_type_given():
    """Passing neither degree nor filter_type should raise UserWarning from the private single-iteration helper."""
    filled, inters = _leveling_ready(BIAS, method="network")
    with pytest.raises(UserWarning, match="must provide either"):
        _crossover_network_levelling(
            filled,
            inters,
            data_col="value",
            levelled_col="value_leveled",
            line_column="line",
            distance_column="dist_along_line",
        )


def test_isolated_line_without_intersections_passes_through_unchanged():
    """A line with no intersections at all should be silently excluded from levelling and pass through unchanged."""
    filled, inters = _leveling_ready_with_isolated_line(BIAS, method="network")
    data, _inters = crossover_network_levelling(
        filled,
        inters,
        data_col="value",
        levelled_col="value_leveled",
        line_column="line",
        distance_column="dist_along_line",
        degree=1,
        relaxation_factor=0.5,
        max_iterations=1,
        plot_convergence=False,
        progressbar=False,
        raise_error_if_unchanged=False,
    )
    iso = data[data.line == "ISO"]
    assert len(iso) > 0
    assert iso["value_leveled"].to_numpy() == pytest.approx(iso["value"].to_numpy())


def test_network_levelling_plot_convergence_smoke_test():
    """plot_convergence=True with enough iterations to trigger plotting should run without error."""
    filled, inters = _leveling_ready(BIAS, method="network")
    crossover_network_levelling(
        filled,
        inters,
        data_col="value",
        levelled_col="value_leveled",
        line_column="line",
        distance_column="dist_along_line",
        degree=1,
        relaxation_factor=0.5,
        max_iterations=5,
        rms_percent_change_tolerance=-1,
        plot_convergence=True,
        progressbar=False,
        raise_error_if_unchanged=False,
    )


def test_network_levelling_plot_dynamic_convergence_smoke_test(monkeypatch):
    """plot_dynamic_convergence=True should run without error when IPython's display is mocked."""
    monkeypatch.setattr(plotting_mod, "display", lambda *_a, **_k: _FakeDisplayHandle())
    filled, inters = _leveling_ready(BIAS, method="network")
    crossover_network_levelling(
        filled,
        inters,
        data_col="value",
        levelled_col="value_leveled",
        line_column="line",
        distance_column="dist_along_line",
        degree=1,
        relaxation_factor=0.5,
        max_iterations=5,
        rms_percent_change_tolerance=-1,
        plot_convergence=False,
        plot_dynamic_convergence=True,
        progressbar=False,
        raise_error_if_unchanged=False,
    )


def test_crossover_network_levelling_trend_found_array_error_zeroes_correction(
    monkeypatch,
):
    """When airbornegeo.trend raises a ValueError whose message contains 'Found array with'
    (sklearn's empty-array error), the network levelling except (ValueError, UserWarning)
    handler logs and falls back to a zero levelling_correction instead of propagating."""
    filled, inters = _leveling_ready(BIAS, method="network")

    def fake_trend(*_args, **_kwargs):
        msg = "Found array with 0 sample(s) (shape=(0, 1)) while a minimum of 1 is required by LinearRegression."
        raise ValueError(msg)

    monkeypatch.setattr(clv_mod.airbornegeo, "trend", fake_trend)

    data, _inters = _crossover_network_levelling(
        filled,
        inters,
        data_col="value",
        levelled_col="value_leveled",
        line_column="line",
        distance_column="dist_along_line",
        degree=1,
        lines_to_level=["F1"],
        raise_error_if_unchanged=False,
    )
    f1 = data[data.line == "F1"]
    assert f1.value_leveled.to_numpy() == pytest.approx(f1.value.to_numpy())


def test_crossover_network_levelling_trend_other_valueerror_is_reraised(monkeypatch):
    """A ValueError from airbornegeo.trend that isn't the 'Found array with' sklearn
    message should be re-raised (wrapped in a new ValueError) rather than silently
    swallowed, mirroring the same behaviour in the pair-levelling helper."""
    filled, inters = _leveling_ready(BIAS, method="network")

    def fake_trend(*_args, **_kwargs):
        msg = "some unrelated failure"
        raise ValueError(msg)

    monkeypatch.setattr(clv_mod.airbornegeo, "trend", fake_trend)

    # the source re-raises via `raise ValueError from e` with no message of its own
    with pytest.raises(ValueError):  # noqa: PT011
        _crossover_network_levelling(
            filled,
            inters,
            data_col="value",
            levelled_col="value_leveled",
            line_column="line",
            distance_column="dist_along_line",
            degree=1,
            lines_to_level=["F1"],
            raise_error_if_unchanged=False,
        )


def _mismatched_network_fixture():
    """Minimal hand-built data/inters for an F1/T1 crossing plus an unrelated, untouched
    F2/T2 crossing, bypassing the real create_intersection_table/interpolate_intersections
    pipeline. calculate_crossover_errors matches purely on (line, intersecting_line,
    easting, northing) so it computes a valid crossover_error_0 for both pairs, but the
    hand-set dist_along_line1/2 in `inters` for F1/T1 deliberately don't match the
    distance_column value on F1's is_intersection row in `data`, so the np.isclose
    distance-match inside the filter_type branch fails to find any match. The extra
    F2/T2 crossing (left out of lines_to_level, so passed through unchanged) keeps at
    least one non-NaN mistie in the mistie table, avoiding an all-NaN RMSE warning."""
    data = pd.DataFrame(
        {
            "line": ["F1", "F1", "T1", "T1", "F2", "F2", "T2", "T2"],
            "dist_along_line": [0.0, 50.0, 0.0, 30.0, 0.0, 50.0, 0.0, 30.0],
            "value": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
            "easting": [10.0, 100.0, 200.0, 100.0, 20.0, 300.0, 400.0, 300.0],
            "northing": [10.0, 100.0, 200.0, 100.0, 20.0, 300.0, 400.0, 300.0],
            "is_intersection": [False, True, False, True, False, True, False, True],
            "intersecting_line": [None, "T1", None, "F1", None, "T2", None, "F2"],
        }
    )
    inters = pd.DataFrame(
        {
            "line1": ["F1", "F2"],
            "line2": ["T1", "T2"],
            "easting": [100.0, 300.0],
            "northing": [100.0, 300.0],
            "dist_along_line1": [999.0, 50.0],
            "dist_along_line2": [999.0, 30.0],
        }
    )
    return data, inters


def test_crossover_network_levelling_filter_type_no_match_continues():
    """When one of a line's intersection rows has a distance_column value that doesn't
    np.isclose-match the dist_along_line1 stored in inters, the match lookup for that row
    returns empty and the loop 'continue's (skipping the mistie assignment for that row)
    without raising, while the line's other, correctly-matched crossing still gets a
    mistie value so the branch completes normally rather than falling into the
    zero-correction path."""
    filled, inters = _leveling_ready(BIAS, method="network")
    corrupt = inters.copy()
    row_mask = (corrupt.line1 == "F1") & (corrupt.line2 == "T1")
    corrupt.loc[row_mask, "dist_along_line1"] += 12345.0

    result, _inters = _crossover_network_levelling(
        filled,
        corrupt,
        data_col="value",
        levelled_col="value_leveled",
        line_column="line",
        distance_column="dist_along_line",
        filter_type="g300",
        lines_to_level=["F1"],
        raise_error_if_unchanged=False,
    )
    f1 = result[result.line == "F1"]
    assert not f1.value_leveled.isna().any()


def test_crossover_network_levelling_filter_type_matched_nan_mistie_zeroes_correction(
    caplog,
):
    """When a data row's distance value correctly matches its intersection's
    dist_along_line1/2, but the underlying data value at that crossing is NaN, the
    resulting network_mistie is NaN too, so `match` is found (no 'continue') but the
    single assigned mistie value is NaN. n_valid_misties then ends up 0, triggering the
    warning + zero-fill fallback rather than interpolating/filtering."""
    data, inters = _mismatched_network_fixture()
    # correct the F1/T1 distances so the row matches (only the mistie value is NaN)
    inters = inters.copy()
    inters.loc[inters.line1 == "F1", "dist_along_line1"] = 50.0
    inters.loc[inters.line1 == "F1", "dist_along_line2"] = 30.0
    data = data.copy()
    data.loc[(data.line == "F1") & (data.dist_along_line == 50.0), "value"] = np.nan

    result, _inters = _crossover_network_levelling(
        data,
        inters,
        data_col="value",
        levelled_col="value_leveled",
        line_column="line",
        distance_column="dist_along_line",
        filter_type="g300",
        lines_to_level=["F1"],
        raise_error_if_unchanged=False,
    )
    f1 = result[result.line == "F1"]
    assert f1.value_leveled.to_numpy() == pytest.approx(
        f1.value.to_numpy(), nan_ok=True
    )
    assert "no mistie values could be matched" in caplog.text


def test_crossover_network_levelling_wrapper_crossover_error_col_fallback(monkeypatch):
    """Regression coverage for the `except ValueError: current_crossover_error_col =
    'crossover_error_0'` fallback in the public crossover_network_levelling wrapper: if
    `intersection_weight_col` happens to equal the sole crossover_error_* column name,
    that column gets excluded from the max()-search list, so max() raises ValueError and
    the code falls back to the literal name 'crossover_error_0'."""
    filled, inters = _leveling_ready(BIAS, method="network")

    def fake_private(data, inters, **_kwargs):
        d = data.copy()
        d["value_leveled"] = d["value"]
        i = inters.copy()
        i["crossover_error_0"] = 0.0
        return d, i

    monkeypatch.setattr(clv_mod, "_crossover_network_levelling", fake_private)

    data, inters = crossover_network_levelling(
        filled,
        inters,
        data_col="value",
        levelled_col="value_leveled",
        line_column="line",
        distance_column="dist_along_line",
        degree=1,
        intersection_weight_col="crossover_error_0",
        max_iterations=1,
        plot_convergence=False,
        progressbar=False,
    )
    assert "crossover_error_0" in inters.columns
    assert not data["value_leveled"].isna().any()


# --------------------------------------------------------------------------- #
# alternating_iterative_line_levelling
# --------------------------------------------------------------------------- #


def test_alternating_iterative_line_levelling_single_pass_reduces_mistie():
    """A single alternating pass (one lines1->lines2 level, one lines2->lines1 level) should reduce the crossover mistie RMS on the exact-fit fixture."""
    filled, inters_valid = _leveling_ready(BIAS, method="groups")
    orig_ce = calculate_crossover_errors(
        filled, inters_valid, data_col="value", line_column="line"
    )
    orig_rms = np.sqrt(np.mean(orig_ce["crossover_error_0"].to_numpy() ** 2))

    _data, new_inters = alternating_iterative_line_levelling(
        filled,
        inters_valid,
        data_col="value",
        levelled_col="value_leveled",
        line_column="line",
        distance_column="dist_along_line",
        degree=1,
        max_iterations=1,
        plot_convergence=False,
        progressbar=False,
    )
    error_cols = [c for c in new_inters.columns if c.startswith("crossover_error_")]
    final_col = f"crossover_error_{max(int(c.split('_')[-1]) for c in error_cols)}"
    final_rms = np.sqrt(np.mean(new_inters[final_col].to_numpy() ** 2))
    assert final_rms < orig_rms


def test_alternating_iterative_line_levelling_missing_line_type_raises():
    """Data without a line_type column should raise AssertionError naming all required columns."""
    filled, inters_valid = _leveling_ready(BIAS, method="groups")
    no_type = filled.drop(columns="line_type")
    with pytest.raises(
        AssertionError,
        match=r"\['line', 'line_type', 'dist_along_line', 'value'\] must be in the dataframe",
    ):
        alternating_iterative_line_levelling(
            no_type,
            inters_valid,
            data_col="value",
            levelled_col="value_leveled",
            line_column="line",
            distance_column="dist_along_line",
            degree=1,
            max_iterations=1,
            plot_convergence=False,
            progressbar=False,
        )


def test_alternating_iterative_line_levelling_degree_and_filter_type_raises():
    """Providing both degree and filter_type should raise UserWarning before any iteration runs."""
    filled, inters_valid = _leveling_ready(BIAS, method="groups")
    with pytest.raises(
        UserWarning, match="only provide either `filter_type` or `degree`, not both"
    ):
        alternating_iterative_line_levelling(
            filled,
            inters_valid,
            data_col="value",
            levelled_col="value_leveled",
            line_column="line",
            distance_column="dist_along_line",
            degree=1,
            filter_type="g300",
            max_iterations=1,
            plot_convergence=False,
            progressbar=False,
        )


def test_alternating_iterative_line_levelling_no_degree_no_filter_type_raises():
    """Providing neither degree nor filter_type should raise UserWarning before any iteration runs."""
    filled, inters_valid = _leveling_ready(BIAS, method="groups")
    with pytest.raises(
        UserWarning, match="must provide either `filter_type` or `degree`"
    ):
        alternating_iterative_line_levelling(
            filled,
            inters_valid,
            data_col="value",
            levelled_col="value_leveled",
            line_column="line",
            distance_column="dist_along_line",
            max_iterations=1,
            plot_convergence=False,
            progressbar=False,
        )


def test_alternating_iterative_line_levelling_crossover_error_col_fallback(monkeypatch):
    """Regression coverage for the `except ValueError: current_crossover_error_col =
    'crossover_error_0'` fallback inside alternating_iterative_line_levelling: if the
    inner crossover_pair_levelling calls return inters whose only crossover_error_*
    column equals intersection_weight_col, that column is excluded from the max()-search
    list, max() raises ValueError, and the loop falls back to the literal name
    'crossover_error_0'."""
    filled, inters_valid = _leveling_ready(BIAS, method="groups")

    def fake_pair(data, inters, **_kwargs):
        d = data.copy()
        d["value_leveled"] = d["value"]
        i = inters.copy()
        i["crossover_error_0"] = 0.0
        return d, i

    monkeypatch.setattr(clv_mod, "crossover_pair_levelling", fake_pair)

    data, inters = alternating_iterative_line_levelling(
        filled,
        inters_valid,
        data_col="value",
        levelled_col="value_leveled",
        line_column="line",
        distance_column="dist_along_line",
        degree=1,
        intersection_weight_col="crossover_error_0",
        max_iterations=1,
        plot_convergence=False,
        progressbar=False,
    )
    assert "crossover_error_0" in inters.columns
    assert not data["value_leveled"].isna().any()


def test_alternating_iterative_line_levelling_lines_to_level_subset():
    """Restricting lines_to_level to one line per type should change only those two lines, leaving the rest untouched."""
    filled, inters_valid = _leveling_ready(BIAS, method="groups")
    data, _inters = alternating_iterative_line_levelling(
        filled,
        inters_valid,
        data_col="value",
        levelled_col="value_leveled",
        line_column="line",
        distance_column="dist_along_line",
        degree=1,
        lines_to_level=["F1", "T1"],
        max_iterations=1,
        plot_convergence=False,
        progressbar=False,
    )
    for line in ["F1", "T1"]:
        subset = data[data.line == line]
        assert not np.allclose(subset.value_leveled.to_numpy(), subset.value.to_numpy())
    for line in ["F2", "T2"]:
        subset = data[data.line == line]
        assert subset.value_leveled.to_numpy() == pytest.approx(subset.value.to_numpy())


def test_alternating_iterative_line_levelling_multi_iteration_converges_without_dividing_by_zero():
    """Regression test for a fixed bug: once the alternating passes converge closely enough
    between iterations that calculate_crossover_errors treats the mistie as "unchanged"
    (its default raise_error_if_unchanged=True on the internal crossover_pair_levelling
    calls then makes that inner call a silent no-op via its own except UserWarning: break),
    the outer loop's own levelling_correction becomes exactly zero, and
    `correction_rms_values[-1] / rms` used to divide by zero on the following iteration
    (crossover_levelling.py's alternating_iterative_line_levelling, around line 1005),
    which this project's pytest config (filterwarnings = ["error", ...]) turned into a
    hard failure. The function now guards that division on rms != 0 and falls back to
    np.inf, matching the same np.inf sentinel already used for the first iteration. This
    reproduces the previously-crashing scenario on the exact-fit fixture at
    max_iterations=2 and confirms it now completes cleanly, fully converged.
    """
    filled, inters_valid = _leveling_ready(BIAS, method="groups")
    data, new_inters = alternating_iterative_line_levelling(
        filled,
        inters_valid,
        data_col="value",
        levelled_col="value_leveled",
        line_column="line",
        distance_column="dist_along_line",
        degree=1,
        max_iterations=2,
        plot_convergence=False,
        progressbar=False,
    )
    assert not data["value_leveled"].isna().any()
    error_cols = [c for c in new_inters.columns if c.startswith("crossover_error_")]
    final_col = f"crossover_error_{max(int(c.split('_')[-1]) for c in error_cols)}"
    assert new_inters[final_col].to_numpy() == pytest.approx(
        [0.0, 0.0, 0.0, 0.0], abs=1e-9
    )


# --------------------------------------------------------------------------- #
# calculate_intersection_weights
# --------------------------------------------------------------------------- #


def _weights_fixture():
    """Minimal inters/gdf pair for calculate_intersection_weights, using its own
    line/tie (not line1/line2) and line/intersecting_line column convention."""
    inters = gpd.GeoDataFrame(
        {
            "line": ["A", "A", "B"],
            "tie": ["X", "Y", "X"],
            "max_dist": [10.0, 50.0, 5.0],
        },
        geometry=gpd.points_from_xy([0, 1, 2], [0, 1, 2]),
        crs=CRS,
    )
    gdf = gpd.GeoDataFrame(
        {
            "line": ["A", "X", "A", "Y", "B", "X"],
            "intersecting_line": ["X", "A", "Y", "A", "X", "B"],
            "height": [100.0, 110.0, 100.0, 120.0, 105.0, 108.0],
        },
        geometry=gpd.points_from_xy(range(6), range(6)),
        crs=CRS,
    )
    return gdf, inters


def test_max_dist_weight_weight_by_all():
    """max_dist_weight should be normalized across all rows, reversed so smaller distances score higher."""
    gdf, inters = _weights_fixture()
    result = calculate_intersection_weights(
        gdf, inters, weight_by="all", max_dist_weight=1.0
    )
    assert result["max_dist_weight"].to_numpy() == pytest.approx(
        [0.889, 0.001, 1.0], abs=1e-3
    )
    assert result["crossover_error_weight"].to_numpy() == pytest.approx(
        [0.889, 0.001, 1.0], abs=1e-3
    )


def test_max_dist_weight_floor_clamps_before_normalizing():
    """Distances below max_dist_floor should be clamped to the floor before normalization."""
    gdf, inters = _weights_fixture()
    result = calculate_intersection_weights(
        gdf, inters, weight_by="all", max_dist_weight=1.0, max_dist_floor=20.0
    )
    # raw max_dist [10, 50, 5] is clamped to [20, 50, 20] before normalizing
    assert result["max_dist_weight"].to_numpy() == pytest.approx(
        [1.0, 0.001, 1.0], abs=1e-3
    )


def test_height_difference_weight_weight_by_all():
    """height_difference_weight should compare the height at each line/tie crossing to the tie/line crossing."""
    gdf, inters = _weights_fixture()
    result = calculate_intersection_weights(
        gdf, inters, weight_by="all", height_difference_weight=1.0
    )
    # (A,X): |100-110|=10, (A,Y): |100-120|=20, (B,X): |105-108|=3
    assert result["height_difference"].to_numpy() == pytest.approx([10.0, 20.0, 3.0])
    assert result["height_difference_weight"].to_numpy() == pytest.approx(
        [0.589, 0.001, 1.0], abs=1e-3
    )
    assert result["crossover_error_weight"].to_numpy() == pytest.approx(
        [0.589, 0.001, 1.0], abs=1e-3
    )


def test_interpolation_type_weight_counts_extrapolated_columns():
    """interpolation_type_weight should be derived from the count of columns equal to 'extrapolated' per row."""
    gdf, inters = _weights_fixture()
    inters["line1_interpolation_type"] = ["interpolated", "extrapolated", "none"]
    inters["line2_interpolation_type"] = ["extrapolated", "extrapolated", "none"]
    result = calculate_intersection_weights(
        gdf, inters, weight_by="all", interpolation_type_weight=1.0
    )
    # (A,X): 1 extrapolated col, (A,Y): 2, (B,X): 0
    assert result["number_of_extrapolations"].to_numpy().tolist() == [1, 2, 0]
    assert result["interpolation_type_weight"].to_numpy() == pytest.approx(
        [0.5005, 0.001, 1.0], abs=1e-3
    )


def test_data_1st_derive_weight_without_col_name_raises():
    """data_1st_derive_weight without data_1st_derive_col_name should raise ValueError."""
    gdf, inters = _weights_fixture()
    with pytest.raises(ValueError, match="must provide 'data_1st_derive_col_name'"):
        calculate_intersection_weights(
            gdf, inters, weight_by="all", data_1st_derive_weight=1.0
        )


def test_height_1st_derive_weight_without_col_name_raises():
    """height_1st_derive_weight without height_1st_derive_col_name should raise ValueError."""
    gdf, inters = _weights_fixture()
    with pytest.raises(ValueError, match="must provide 'height_1st_derive_col_name'"):
        calculate_intersection_weights(
            gdf, inters, weight_by="all", height_1st_derive_weight=1.0
        )


def test_data_1st_derive_weight_with_col_name():
    """data_1st_derive should be the max absolute gradient of the line/tie pair at each crossing."""
    gdf, inters = _weights_fixture()
    gdf["grad"] = [1.0, -2.0, 3.0, -0.5, -4.0, 2.0]
    result = calculate_intersection_weights(
        gdf,
        inters,
        weight_by="all",
        data_1st_derive_weight=1.0,
        data_1st_derive_col_name="grad",
    )
    # (A,X): max(|1.0|,|-2.0|)=2.0, (A,Y): max(|3.0|,|-0.5|)=3.0, (B,X): max(|-4.0|,|2.0|)=4.0
    assert result["data_1st_derive"].to_numpy() == pytest.approx([2.0, 3.0, 4.0])
    assert result["data_1st_derive_weight"].to_numpy() == pytest.approx(
        [1.0, 0.5005, 0.001], abs=1e-3
    )


def test_max_dist_weight_weight_by_tie_normalizes_per_group():
    """weight_by='tie' should normalize max_dist_weight within each 'tie' group rather than across all rows."""
    gdf, inters = _weights_fixture()
    result = calculate_intersection_weights(
        gdf, inters, weight_by="tie", max_dist_weight=1.0
    )
    group_x = result[result.tie == "X"]
    # tie 'X' has two distinct distances (10.0 and 5.0), so they end up at opposite
    # ends of the range
    assert sorted(group_x["max_dist_weight"].to_numpy()) == pytest.approx(
        [0.001, 1.0], abs=1e-3
    )
    group_y = result[result.tie == "Y"]
    # tie 'Y' has a single row, a degenerate normalize_values case that falls back to
    # 'low' (1.0 for this intermediate column, reversed axis)
    assert group_y["max_dist_weight"].to_numpy() == pytest.approx([1.0], abs=1e-3)


def test_height_difference_weight_floor_clamps_before_normalizing_weight_by_line():
    """height_difference_floor should clamp small differences before per-'line'-group normalization."""
    gdf, inters = _weights_fixture()
    result = calculate_intersection_weights(
        gdf,
        inters,
        weight_by="line",
        height_difference_weight=1.0,
        height_difference_floor=15.0,
    )
    # raw height_difference [10, 20, 3] clamped to [15, 20, 15] before per-group
    # normalization; group A (rows 0,1) has two distinct clamped values so they end up
    # at opposite ends, group B (row 2) is a single-row degenerate case
    group_a = result[result.line == "A"]
    assert sorted(group_a["height_difference_weight"].to_numpy()) == pytest.approx(
        [0.001, 1.0], abs=1e-3
    )
    group_b = result[result.line == "B"]
    assert group_b["height_difference_weight"].to_numpy() == pytest.approx(
        [1.0], abs=1e-3
    )


def test_interpolation_type_weight_weight_by_line():
    """weight_by='line' should normalize interpolation_type_weight within each 'line' group."""
    gdf, inters = _weights_fixture()
    inters["line1_interpolation_type"] = ["interpolated", "extrapolated", "none"]
    inters["line2_interpolation_type"] = ["extrapolated", "extrapolated", "none"]
    result = calculate_intersection_weights(
        gdf, inters, weight_by="line", interpolation_type_weight=1.0
    )
    group_a = result[result.line == "A"]
    # group A extrapolation counts [1, 2] -> opposite ends of range within the group
    assert sorted(group_a["interpolation_type_weight"].to_numpy()) == pytest.approx(
        [0.001, 1.0], abs=1e-3
    )
    group_b = result[result.line == "B"]
    assert group_b["interpolation_type_weight"].to_numpy() == pytest.approx(
        [1.0], abs=1e-3
    )


def test_data_1st_derive_weight_floor_and_weight_by_line():
    """data_1st_derive_floor should clamp small gradients before per-'line'-group normalization."""
    gdf, inters = _weights_fixture()
    gdf["grad"] = [1.0, -2.0, 3.0, -0.5, -4.0, 2.0]
    result = calculate_intersection_weights(
        gdf,
        inters,
        weight_by="line",
        data_1st_derive_weight=1.0,
        data_1st_derive_floor=2.5,
        data_1st_derive_col_name="grad",
    )
    # raw data_1st_derive [2.0, 3.0, 4.0] clamped to [2.5, 3.0, 4.0]; group A (rows 0,1)
    # has two distinct clamped values, group B (row 2) is single-row degenerate
    group_a = result[result.line == "A"]
    assert sorted(group_a["data_1st_derive_weight"].to_numpy()) == pytest.approx(
        [0.001, 1.0], abs=1e-3
    )
    group_b = result[result.line == "B"]
    assert group_b["data_1st_derive_weight"].to_numpy() == pytest.approx(
        [1.0], abs=1e-3
    )


def test_height_1st_derive_weight_with_col_name_weight_by_all():
    """height_1st_derive should be the max absolute height-gradient of the line/tie pair at each crossing, mirroring data_1st_derive_weight."""
    gdf, inters = _weights_fixture()
    gdf["height_grad"] = [1.0, -2.0, 3.0, -0.5, -4.0, 2.0]
    result = calculate_intersection_weights(
        gdf,
        inters,
        weight_by="all",
        height_1st_derive_weight=1.0,
        height_1st_derive_col_name="height_grad",
    )
    # (A,X): max(|1.0|,|-2.0|)=2.0, (A,Y): max(|3.0|,|-0.5|)=3.0, (B,X): max(|-4.0|,|2.0|)=4.0
    assert result["height_1st_derive"].to_numpy() == pytest.approx([2.0, 3.0, 4.0])
    assert result["height_1st_derive_weight"].to_numpy() == pytest.approx(
        [1.0, 0.5005, 0.001], abs=1e-3
    )
    assert result["crossover_error_weight"].to_numpy() == pytest.approx(
        [1.0, 0.5005, 0.001], abs=1e-3
    )


def test_height_1st_derive_weight_floor_and_weight_by_line():
    """height_1st_derive_floor should clamp small gradients before per-'line'-group normalization, exercising the height_1st_derive_weight kwarg's 'else' (weight_by='line') branch."""
    gdf, inters = _weights_fixture()
    gdf["height_grad"] = [1.0, -2.0, 3.0, -0.5, -4.0, 2.0]
    result = calculate_intersection_weights(
        gdf,
        inters,
        weight_by="line",
        height_1st_derive_weight=1.0,
        height_1st_derive_floor=2.5,
        height_1st_derive_col_name="height_grad",
    )
    # raw height_1st_derive [2.0, 3.0, 4.0] clamped to [2.5, 3.0, 4.0]; group A (rows
    # 0,1) has two distinct clamped values so they land at opposite ends, group B (row
    # 2) is a single-row degenerate case
    group_a = result[result.line == "A"]
    assert sorted(group_a["height_1st_derive_weight"].to_numpy()) == pytest.approx(
        [0.001, 1.0], abs=1e-3
    )
    group_b = result[result.line == "B"]
    assert group_b["height_1st_derive_weight"].to_numpy() == pytest.approx(
        [1.0], abs=1e-3
    )


def test_weight_by_invalid_raises():
    """An unrecognized weight_by value should raise ValueError."""
    gdf, inters = _weights_fixture()
    with pytest.raises(ValueError, match="weight_by must be 'line', 'tie', or 'all'"):
        calculate_intersection_weights(
            gdf, inters, weight_by="bogus", max_dist_weight=1.0
        )


def test_max_dist_weight_weight_by_line_normalizes_per_group():
    """weight_by='line' should normalize max_dist_weight within each 'line' group rather than across all rows."""
    gdf, inters = _weights_fixture()
    result = calculate_intersection_weights(
        gdf, inters, weight_by="line", max_dist_weight=1.0
    )
    group_a = result[result.line == "A"]
    # group A has two distinct distances, so they end up at opposite ends of the range
    assert group_a["max_dist_weight"].to_numpy() == pytest.approx(
        [1.0, 0.001], abs=1e-3
    )
    group_b = result[result.line == "B"]
    # group B has a single row, a degenerate case for normalize_values: min == max, so
    # it falls back to returning the 'low' value. For this intermediate max_dist_weight
    # column low=1 (not 0.001, since the axis is reversed so small distances score high),
    # so the single-row group's max_dist_weight is 1.0, not 0.001.
    assert group_b["max_dist_weight"].to_numpy() == pytest.approx([1.0], abs=1e-3)
    # crossover_error_weight is then re-normalized per group with low=0.001, so the
    # same single-row group collapses to 0.001 at that second stage.
    assert group_b["crossover_error_weight"].to_numpy() == pytest.approx(
        [0.001], abs=1e-3
    )


def test_combining_two_weights_is_a_weighted_average():
    """crossover_error_weight should be a weighted average of the normalized component weights, renormalized."""
    gdf, inters = _weights_fixture()
    result = calculate_intersection_weights(
        gdf,
        inters,
        weight_by="all",
        max_dist_weight=1.0,
        height_difference_weight=2.0,
    )
    # already-verified component weights: max_dist_weight=[0.889,0.001,1.0],
    # height_difference_weight=[0.589,0.001,1.0], combined with factors 1 and 2
    assert result["crossover_error_weight"].to_numpy() == pytest.approx(
        [0.689, 0.001, 1.0], abs=1e-3
    )


def test_gdf_missing_line_column_raises():
    """gdf without a 'line' column should raise an AssertionError."""
    gdf, inters = _weights_fixture()
    gdf = gdf.drop(columns="line")
    with pytest.raises(AssertionError, match="gdf must have column 'line'"):
        calculate_intersection_weights(
            gdf, inters, weight_by="all", max_dist_weight=1.0
        )


def test_plot_true_calls_plotly_points(monkeypatch):
    """plot=True should call airbornegeo.plotly_points once and still return the weighted table."""
    gdf, inters = _weights_fixture()
    calls = []
    monkeypatch.setattr(
        "airbornegeo.plotly_points", lambda *_args, **kwargs: calls.append(kwargs)
    )
    result = calculate_intersection_weights(
        gdf, inters, weight_by="all", max_dist_weight=1.0, plot=True
    )
    assert len(calls) == 1
    assert len(result) == 3


# --------------------------------------------------------------------------- #
# plot_levelling_convergence (module-level, distinct from
# airbornegeo.plotting.plot_levelling_convergence)
# --------------------------------------------------------------------------- #


def test_plot_levelling_convergence_runs_and_plots_line():
    """A results table with three crossover_error_ columns should produce a figure with an 'Iteration' x-label, one xtick per column, and a plotted line."""
    results = pd.DataFrame(
        {
            "crossover_error_0": [1.0, -2.0, 3.0],
            "crossover_error_1": [0.5, -1.0, 1.5],
            "crossover_error_2": [0.2, -0.4, 0.6],
        }
    )
    plot_levelling_convergence(results)
    fig = clv_mod.plt.gcf()
    ax1 = fig.axes[0]
    assert ax1.get_xlabel() == "Iteration"
    assert len(ax1.get_xticks()) == 3
    assert len(ax1.lines) > 0


def test_plot_levelling_convergence_sets_ylabel():
    """The y-axis label should read 'Cross-over RMSE'."""
    results = pd.DataFrame({"crossover_error_0": [1.0, 2.0, 3.0]})
    plot_levelling_convergence(results)
    ax1 = clv_mod.plt.gcf().axes[0]
    assert ax1.get_ylabel() == "Cross-over RMSE"


def test_plot_levelling_convergence_logy_sets_log_scale():
    """logy=True should set a log y-scale, unlike the default linear scale."""
    results = pd.DataFrame({"crossover_error_0": [1.0, 2.0, 3.0]})
    plot_levelling_convergence(results, logy=True)
    ax1 = clv_mod.plt.gcf().axes[0]
    assert ax1.get_yscale() == "log"


def test_plot_levelling_convergence_default_is_linear_scale():
    """Without logy, the y-scale should stay linear."""
    results = pd.DataFrame({"crossover_error_0": [1.0, 2.0, 3.0]})
    plot_levelling_convergence(results)
    ax1 = clv_mod.plt.gcf().axes[0]
    assert ax1.get_yscale() == "linear"


def test_plot_levelling_convergence_sets_title():
    """The title kwarg should be readable back from the current axes' title."""
    results = pd.DataFrame({"crossover_error_0": [1.0, 2.0, 3.0]})
    plot_levelling_convergence(results, title="my custom title")
    ax1 = clv_mod.plt.gcf().axes[0]
    assert ax1.get_title() == "my custom title"


def test_plot_levelling_convergence_default_title():
    """The default title should be used when none is given."""
    results = pd.DataFrame({"crossover_error_0": [1.0, 2.0, 3.0]})
    plot_levelling_convergence(results)
    ax1 = clv_mod.plt.gcf().axes[0]
    assert ax1.get_title() == "Levelling convergence"


def test_plot_levelling_convergence_no_crossover_columns_does_not_crash():
    """A results table with no crossover_error_ columns should plot nothing and not raise."""
    results = pd.DataFrame({"other_col": [1.0, 2.0, 3.0]})
    plot_levelling_convergence(results)
    ax1 = clv_mod.plt.gcf().axes[0]
    assert len(ax1.get_xticks()) == 0
    assert ax1.lines[0].get_xdata().size == 0
    assert ax1.lines[0].get_ydata().size == 0


def test_plot_levelling_convergence_as_median_changes_plotted_values():
    """as_median=True should use the median-based RMSE, giving different plotted y-data than the mean-based default when an outlier is present."""
    results = pd.DataFrame({"crossover_error_0": [1.0, 1.0, 1.0, 100.0]})
    plot_levelling_convergence(results, as_median=False)
    mean_ydata = clv_mod.plt.gcf().axes[0].lines[0].get_ydata().copy()
    clv_mod.plt.close("all")

    plot_levelling_convergence(results, as_median=True)
    median_ydata = clv_mod.plt.gcf().axes[0].lines[0].get_ydata()

    assert mean_ydata[0] == pytest.approx(
        np.sqrt(np.mean(np.array([1.0, 1.0, 1.0, 100.0]) ** 2))
    )
    assert median_ydata[0] == pytest.approx(
        np.sqrt(np.median(np.array([1.0, 1.0, 1.0, 100.0]) ** 2))
    )
    assert mean_ydata[0] != pytest.approx(median_ydata[0])


def test_plot_levelling_convergence_uses_dataframe_column_order_not_numeric_order():
    """The function filters results.columns by substring match and never numerically sorts the extracted suffixes, so columns out of numeric order in the DataFrame are plotted in their DataFrame order, not iteration order (see crossover_error_cols in the source)."""
    results = pd.DataFrame(
        {
            "crossover_error_2": [10.0],
            "crossover_error_0": [1.0],
            "crossover_error_1": [5.0],
        }
    )
    plot_levelling_convergence(results)
    ydata = clv_mod.plt.gcf().axes[0].lines[0].get_ydata()
    expected = [
        np.sqrt(np.mean(np.array([10.0]) ** 2)),
        np.sqrt(np.mean(np.array([1.0]) ** 2)),
        np.sqrt(np.mean(np.array([5.0]) ** 2)),
    ]
    assert list(ydata) == pytest.approx(expected)
