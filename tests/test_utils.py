import logging

import numpy as np
import pandas as pd
import pytest

from airbornegeo.utils import (
    DuplicateFilter,
    _apply_grouped,
    _check_coord_columns,
    _iter_groups,
    get_min_max,
    largest_line_dimensions,
    median_line_spacing,
    normalize_values,
    rmse,
)


def test_duplicate_filter_suppresses_repeated_messages_within_context():
    """DuplicateFilter should suppress repeated log messages only while its context is active."""
    logger = logging.getLogger("test_duplicate_filter")
    logger.setLevel(logging.INFO)
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    logger.addHandler(handler)
    try:
        with DuplicateFilter(logger):
            logger.info("same message")
            logger.info("same message")
            logger.info("different message")
        assert [r.getMessage() for r in records] == [
            "same message",
            "different message",
        ]

        # filter is removed on exit - duplicates pass through again
        logger.info("same message")
        assert len(records) == 3
    finally:
        logger.removeHandler(handler)


def test_normalize_values_default_range():
    """With no low/high given, values should be scaled to the default [0, 1] range."""
    result = normalize_values(np.array([0.0, 5.0, 10.0]))
    assert result == pytest.approx([0.0, 0.5, 1.0])


def test_normalize_values_custom_range():
    """A custom low/high should scale values into that range instead of [0, 1]."""
    result = normalize_values(np.array([0.0, 5.0, 10.0]), low=-1, high=1)
    assert result == pytest.approx([-1.0, 0.0, 1.0])


def test_normalize_values_quantile_clipping():
    """Values above the given upper quantile should be clipped to that quantile before scaling."""
    x = np.array([0.0, 1.0, 2.0, 3.0, 100.0])
    result = normalize_values(x, quantiles=(0, 0.8))
    assert result[-1] == 1.0
    assert result[0] == 0.0


def test_normalize_values_all_equal_returns_low(caplog):
    """All-equal input should return an array of the low value and log a warning, rather than dividing by zero."""
    with caplog.at_level(logging.WARNING, logger="airbornegeo"):
        result = normalize_values(np.array([5.0, 5.0, 5.0]), low=2)
    assert result.tolist() == pytest.approx([2.0, 2.0, 2.0])
    assert "min and max values are equal" in caplog.text


def test_normalize_values_does_not_mutate_input():
    """normalize_values() should not modify the caller's input array in place."""
    x = np.array([1.0, 2.0, 3.0])
    x_before = x.copy()
    normalize_values(x)
    assert np.array_equal(x, x_before)


def test_rmse_basic():
    """rmse() should compute the root mean squared value of the input."""
    data = np.array([3.0, 4.0, 0.0])
    assert rmse(data) == pytest.approx(np.sqrt((9 + 16 + 0) / 3))


def test_rmse_as_median():
    """as_median=True should compute the root median squared value instead of the mean."""
    data = np.array([3.0, 4.0, 0.0])
    assert rmse(data, as_median=True) == pytest.approx(3.0)


def test_rmse_ignores_nan():
    """NaN values should be excluded from the RMSE calculation."""
    data = np.array([3.0, 4.0, np.nan])
    assert rmse(data) == pytest.approx(np.sqrt((9 + 16) / 2))


def test_get_min_max_basic():
    """get_min_max() should return the plain min and max of the input by default."""
    assert get_min_max(np.array([1.0, 5.0, 10.0, -3.0])) == (-3.0, 10.0)


def test_get_min_max_absolute():
    """absolute=True should return the max absolute value as a symmetric +/- range."""
    v_min, v_max = get_min_max(np.array([1.0, 5.0, 10.0, -3.0]), absolute=True)
    assert v_min == -10.0
    assert v_max == 10.0


def test_get_min_max_robust():
    """robust=True should return the default 2nd/98th percentile values instead of the true min/max."""
    v_min, v_max = get_min_max(np.arange(100, dtype=float), robust=True)
    assert v_min == pytest.approx(1.98)
    assert v_max == pytest.approx(97.02)


def test_get_min_max_custom_robust_percentiles():
    """Custom robust_percentiles should be used instead of the default 2nd/98th."""
    v_min, v_max = get_min_max(
        np.arange(100, dtype=float), robust=True, robust_percentiles=(0.1, 0.9)
    )
    assert v_min == pytest.approx(9.9)
    assert v_max == pytest.approx(89.1)


def test_median_line_spacing_two_parallel_lines():
    """Two parallel lines 100 m apart should give a median line spacing of ~100 m."""
    n = 10
    line_a = pd.DataFrame(
        {"easting": np.linspace(0, 900, n), "northing": np.zeros(n), "line": "A"}
    )
    line_b = pd.DataFrame(
        {"easting": np.linspace(0, 900, n), "northing": np.full(n, 100.0), "line": "B"}
    )
    data = pd.concat([line_a, line_b], ignore_index=True)
    assert median_line_spacing(data, line_column="line") == pytest.approx(100.0)


def test_median_line_spacing_missing_coord_columns_raises():
    """Missing easting/northing columns should raise AssertionError via _check_coord_columns."""
    data = pd.DataFrame({"x": [1.0], "y": [2.0], "line": ["A"]})
    with pytest.raises(AssertionError, match="Projected coordinates columns"):
        median_line_spacing(data, line_column="line")


def test_median_line_spacing_missing_line_column_raises():
    """A missing line_column should raise AssertionError naming it."""
    data = pd.DataFrame({"easting": [0.0, 1.0], "northing": [0.0, 1.0]})
    with pytest.raises(
        AssertionError, match=r"\['not_a_col'\] must be in the dataframe"
    ):
        median_line_spacing(data, line_column="not_a_col")


def test_largest_line_dimensions_straight_and_l_shaped_lines():
    """A straight line's dimension is its length; an L-shaped line's is its long leg."""
    n = 11
    straight = pd.DataFrame(
        {"easting": np.linspace(0, 1000, n), "northing": np.zeros(n), "line": "A"}
    )
    l_shaped = pd.DataFrame(
        {
            "easting": np.concatenate([np.linspace(0, 500, n), np.full(n, 500.0)]),
            "northing": np.concatenate([np.zeros(n), np.linspace(0, 200, n)]),
            "line": "B",
        }
    )
    data = pd.concat([straight, l_shaped], ignore_index=True)
    dims = largest_line_dimensions(data, line_column="line")
    assert dims["A"] == pytest.approx(1000.0)
    assert dims["B"] == pytest.approx(500.0)


def test_largest_line_dimensions_circular_line():
    """A circular line's largest dimension should be ~its diameter, not its perimeter."""
    theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    data = pd.DataFrame(
        {
            "easting": 500 * np.cos(theta),
            "northing": 500 * np.sin(theta),
            "line": "A",
        }
    )
    dims = largest_line_dimensions(data, line_column="line")
    assert dims["A"] == pytest.approx(1000.0, rel=0.01)


def test_largest_line_dimensions_ignores_nan_coordinates():
    """Rows with NaN coordinates should not affect the result."""
    data = pd.DataFrame(
        {
            "easting": [0.0, 50.0, 100.0, np.nan],
            "northing": [0.0, 0.0, 0.0, 1e6],
            "line": ["A"] * 4,
        }
    )
    dims = largest_line_dimensions(data, line_column="line")
    assert dims["A"] == pytest.approx(100.0)


def test_largest_line_dimensions_single_point_line():
    """A line with a single point should have a dimension of 0."""
    data = pd.DataFrame({"easting": [1.0], "northing": [2.0], "line": ["A"]})
    dims = largest_line_dimensions(data, line_column="line")
    assert dims["A"] == 0.0


def test_largest_line_dimensions_missing_columns_raises():
    """Missing coordinate or line columns should raise AssertionError."""
    with pytest.raises(AssertionError, match="Projected coordinates columns"):
        largest_line_dimensions(pd.DataFrame({"x": [1.0], "line": ["A"]}), "line")
    data = pd.DataFrame({"easting": [0.0], "northing": [0.0]})
    with pytest.raises(
        AssertionError, match=r"\['not_a_col'\] must be in the dataframe"
    ):
        largest_line_dimensions(data, line_column="not_a_col")


def test_check_coord_columns_passes_when_present():
    """_check_coord_columns() should not raise when easting/northing are both present."""
    _check_coord_columns(pd.DataFrame({"easting": [1.0], "northing": [2.0]}))


def test_check_coord_columns_missing_raises_with_rename_hint():
    """Missing easting/northing should raise AssertionError with a df.rename() hint in the message."""
    with pytest.raises(AssertionError, match=r"df\.rename\(columns="):
        _check_coord_columns(pd.DataFrame({"x": [1.0], "y": [2.0]}))


def test_iter_groups_preserves_first_appearance_order():
    """_iter_groups() should yield groups in first-appearance order, not sorted order."""
    data = pd.DataFrame({"line": ["B", "B", "A", "A"], "x": [1.0, 2.0, 3.0, 4.0]})
    keys = [key for key, _ in _iter_groups(data, "line", progressbar=False)]
    assert keys == ["B", "A"]


def test_iter_groups_progressbar_wraps_in_tqdm():
    """progressbar=True should wrap the group iterator in a tqdm object without changing its contents."""
    data = pd.DataFrame({"line": ["A", "B"], "x": [1.0, 2.0]})
    wrapped = _iter_groups(data, "line", progressbar=True)
    assert type(wrapped).__name__ == "tqdm"
    assert [key for key, _ in wrapped] == ["A", "B"]


def test_apply_grouped_no_groupby_single_output():
    """With groupby_column=None, a single-array-returning func should be applied once to the whole dataframe."""
    data = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    result = _apply_grouped(
        data, groupby_column=None, progressbar=False, func=lambda d: d.x.to_numpy() + 1
    )
    assert isinstance(result, np.ndarray)
    assert result.tolist() == pytest.approx([2.0, 3.0, 4.0])


def test_apply_grouped_no_groupby_tuple_output():
    """With groupby_column=None, a tuple-returning func should return a tuple of numpy arrays."""
    data = pd.DataFrame({"x": [1.0, 2.0]})
    result = _apply_grouped(
        data,
        groupby_column=None,
        progressbar=False,
        func=lambda d: (d.x.to_numpy(), d.x.to_numpy() * 2),
    )
    assert isinstance(result, tuple)
    assert result[0].tolist() == pytest.approx([1.0, 2.0])
    assert result[1].tolist() == pytest.approx([2.0, 4.0])


def test_apply_grouped_groupby_single_output_preserves_row_order():
    """Grouped results should be realigned to the dataframe's original row order, even with unsorted group labels."""
    data = pd.DataFrame({"line": ["B", "B", "A", "A"], "x": [1.0, 2.0, 3.0, 4.0]})
    result = _apply_grouped(
        data,
        groupby_column="line",
        progressbar=False,
        func=lambda d: d.x.to_numpy() * 2,
    )
    assert result.tolist() == pytest.approx([2.0, 4.0, 6.0, 8.0])


def test_apply_grouped_groupby_tuple_output():
    """A tuple-returning func applied per group should return a tuple of arrays aligned to the original rows."""
    data = pd.DataFrame({"line": ["A", "A", "B", "B"], "x": [1.0, 2.0, 3.0, 4.0]})
    result = _apply_grouped(
        data,
        groupby_column="line",
        progressbar=False,
        func=lambda d: (d.x.to_numpy(), d.x.to_numpy() * 10),
    )
    assert result[0].tolist() == pytest.approx([1.0, 2.0, 3.0, 4.0])
    assert result[1].tolist() == pytest.approx([10.0, 20.0, 30.0, 40.0])


def test_apply_grouped_realigns_results_regardless_of_custom_index():
    """Row realignment should work correctly even when the dataframe has a non-default index."""
    data = pd.DataFrame(
        {"line": ["B", "B", "A", "A"], "x": [1.0, 2.0, 3.0, 4.0]},
        index=[10, 11, 12, 13],
    )
    result = _apply_grouped(
        data,
        groupby_column="line",
        progressbar=False,
        func=lambda d: d.x.to_numpy() * 2,
    )
    assert result.tolist() == pytest.approx([2.0, 4.0, 6.0, 8.0])
