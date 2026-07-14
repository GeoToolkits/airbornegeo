import numpy as np
import pandas as pd
import pytest

from airbornegeo.grid_levelling import level_to_grid


def _survey_with_2d_trend(n=30, seed=0, offset=50.0):
    rng = np.random.default_rng(seed)
    easting = np.linspace(0, 1000, n)
    northing = np.linspace(0, 500, n)
    grid_vals = 2 * easting + 3 * northing + 10
    data_vals = grid_vals + rng.normal(0, 1, n) + offset
    return pd.DataFrame(
        {"data": data_vals, "grid": grid_vals, "easting": easting, "northing": northing}
    )


def test_level_to_grid_2d_trend_removes_constant_offset():
    """A 2D trend fit (no groupby) should remove a constant offset added on top of the grid."""
    data = _survey_with_2d_trend(offset=50.0)
    result = level_to_grid(
        data,
        data_column="data",
        grid_column="grid",
        fit_by_column=("easting", "northing"),
        degree=1,
        groupby_column=None,
    )
    # levelling should remove the constant ~50 offset added on top of the grid trend
    assert result.to_numpy() == pytest.approx(data.grid.to_numpy(), abs=5)


def test_level_to_grid_degree_and_filter_both_raises_valueerror():
    """Providing both degree and filter_kwargs should raise ValueError."""
    data = _survey_with_2d_trend()
    with pytest.raises(
        ValueError, match="only provide filter_kwargs or degree, not both"
    ):
        level_to_grid(
            data,
            data_column="data",
            grid_column="grid",
            fit_by_column=("easting", "northing"),
            degree=1,
            filter_kwargs={"filter_width": 1000},
        )


def test_level_to_grid_2d_filter_raises_not_implemented():
    """2D levelling (no groupby) only supports trend fitting, not filtering."""
    data = _survey_with_2d_trend()
    with pytest.raises(
        NotImplementedError, match="for 2D, only trend fitting supported"
    ):
        level_to_grid(
            data,
            data_column="data",
            grid_column="grid",
            fit_by_column=("easting", "northing"),
            filter_kwargs={"filter_width": 1000},
        )


def test_level_to_grid_missing_fit_by_column_raises():
    """A missing fit_by_column name should raise AssertionError with the exact column list."""
    data = _survey_with_2d_trend()
    with pytest.raises(
        AssertionError, match=r"\('easting', 'missing'\) must be in the dataframe"
    ):
        level_to_grid(
            data,
            data_column="data",
            grid_column="grid",
            fit_by_column=("easting", "missing"),
            degree=1,
        )


def test_level_to_grid_groupby_requires_string_fit_by_column():
    """When groupby_column is set, fit_by_column must be a single string, not a tuple."""
    data = _survey_with_2d_trend()
    data["line"] = "A"
    with pytest.raises(AssertionError, match="fit_by_column must be a string"):
        level_to_grid(
            data,
            data_column="data",
            grid_column="grid",
            fit_by_column=("easting", "northing"),
            degree=1,
            groupby_column="line",
        )


def _survey_with_lines(n_per_line=15, seed=0):
    rng = np.random.default_rng(seed)
    dfs = []
    for line, offset in [("A", 50.0), ("B", -30.0)]:
        easting = np.linspace(0, 1000, n_per_line)
        northing = np.linspace(0, 500, n_per_line)
        grid_vals = 2 * easting + 3 * northing + 10
        data_vals = grid_vals + rng.normal(0, 1, n_per_line) + offset
        dfs.append(
            pd.DataFrame(
                {
                    "data": data_vals,
                    "grid": grid_vals,
                    "distance_along_line": np.linspace(0, 100, n_per_line),
                    "line": line,
                }
            )
        )
    return pd.concat(dfs, ignore_index=True)


def test_level_to_grid_groupby_degree_fit_per_line():
    """A per-line degree trend fit should remove each line's constant offset independently."""
    data = _survey_with_lines()
    result = level_to_grid(
        data,
        data_column="data",
        grid_column="grid",
        fit_by_column="distance_along_line",
        degree=1,
        groupby_column="line",
    )
    assert len(result) == len(data)
    assert not result.isna().any()
    assert result.to_numpy() == pytest.approx(data.grid.to_numpy(), abs=5)


def test_level_to_grid_groupby_missing_columns_raises():
    """A missing groupby or fit_by column should raise AssertionError with the exact column list."""
    data = _survey_with_lines()
    with pytest.raises(
        AssertionError, match=r"\['line', 'missing_col'\] must be in the dataframe"
    ):
        level_to_grid(
            data,
            data_column="data",
            grid_column="grid",
            fit_by_column="missing_col",
            degree=1,
            groupby_column="line",
        )


def test_level_to_grid_groupby_filter_correction():
    """A per-line low-pass filter correction should remove each line's added noise/offset."""
    dfs = []
    for line, offset in [("A", 50.0), ("B", -30.0)]:
        n = 100
        dist = np.linspace(0, 5000, n)
        grid_vals = np.sin(2 * np.pi * dist / 5000) * 10 + 20
        data_vals = grid_vals + offset + 0.1 * np.sin(2 * np.pi * dist / 50)
        dfs.append(
            pd.DataFrame(
                {
                    "data": data_vals,
                    "grid": grid_vals,
                    "distance_along_line": dist,
                    "line": line,
                }
            )
        )
    data = pd.concat(dfs, ignore_index=True)

    result = level_to_grid(
        data,
        data_column="data",
        grid_column="grid",
        fit_by_column="distance_along_line",
        filter_kwargs={"filter_width": 500},
        groupby_column="line",
    )
    assert result.shape == (200,)
    assert not result.isna().any()
    interior = slice(10, -10)
    assert result.to_numpy()[interior] == pytest.approx(
        data.grid.to_numpy()[interior], abs=1
    )


def test_level_to_grid_drops_nan_grid_rows():
    """Rows with a NaN grid value should be dropped from the result."""
    data = _survey_with_lines()
    data.loc[0, "grid"] = np.nan
    result = level_to_grid(
        data,
        data_column="data",
        grid_column="grid",
        fit_by_column="distance_along_line",
        degree=1,
        groupby_column="line",
    )
    assert len(result) == len(data) - 1
