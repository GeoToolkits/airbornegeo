import numpy as np
import pandas as pd
import pytest

from airbornegeo.block_reduce import block_reduce


def test_block_reduce_tuple_reduce_by_single_data_column():
    """2D block reduction (reduce_by as a tuple of two columns) with a single data column."""
    rng = np.random.default_rng(0)
    n = 20
    data = pd.DataFrame(
        {
            "easting": np.linspace(0, 1000, n) + rng.random(n),
            "northing": np.linspace(0, 500, n) + rng.random(n),
            "value": rng.random(n) * 10,
        }
    )
    result = block_reduce(
        data,
        np.mean,
        spacing=200,
        reduce_by=("easting", "northing"),
        progressbar=False,
    )
    assert list(result.columns) == ["easting", "northing", "value"]
    assert result["easting"].iloc[0] == pytest.approx(79.188431, abs=1e-5)
    assert len(result) == 7


def test_block_reduce_string_reduce_by_two_data_columns():
    """1D block reduction (reduce_by as a single column name) with two data columns."""
    rng = np.random.default_rng(1)
    n = 15
    data = pd.DataFrame(
        {
            "dist": np.linspace(0, 1000, n) + rng.random(n),
            "value": rng.random(n) * 10,
            "value2": rng.random(n) * 5,
        }
    )
    result = block_reduce(
        data, np.mean, spacing=200, reduce_by="dist", progressbar=False
    )
    assert list(result.columns) == ["dist", "tmp", "value", "value2"]
    assert (result["tmp"] == 0.0).all()
    assert result["dist"].iloc[0] == pytest.approx(71.964053, abs=1e-5)
    assert result["value"].iloc[0] == pytest.approx(3.302175, abs=1e-5)
    assert result["value2"].iloc[0] == pytest.approx(2.092373, abs=1e-5)
    assert len(result) == 5


def test_block_reduce_reduce_by_1tuple_matches_string():
    """Passing reduce_by as a single-element tuple should give identical results to a plain string."""
    data = pd.DataFrame(
        {"dist": [0.0, 50.0, 100.0, 500.0], "value": [1.0, 2.0, 3.0, 4.0]}
    )
    result_str = block_reduce(
        data, np.mean, spacing=200, reduce_by="dist", progressbar=False
    )
    result_tuple = block_reduce(
        data, np.mean, spacing=200, reduce_by=("dist",), progressbar=False
    )
    pd.testing.assert_frame_equal(result_str, result_tuple)
    assert result_str["dist"].tolist() == pytest.approx([50.0, 500.0])
    assert result_str["value"].tolist() == pytest.approx([2.0, 4.0])


def test_block_reduce_median_hand_checkable_blocks():
    """np.median reduction should match hand-computed block medians for simple, well-separated blocks."""
    data = pd.DataFrame(
        {
            "dist": [0.0, 10.0, 20.0, 300.0, 310.0],
            "value": [1.0, 3.0, 5.0, 100.0, 200.0],
        }
    )
    result = block_reduce(
        data, np.median, spacing=100, reduce_by="dist", progressbar=False
    )
    assert result["dist"].tolist() == pytest.approx([10.0, 305.0])
    assert result["value"].tolist() == pytest.approx([3.0, 150.0])


def test_block_reduce_does_not_mutate_original_dataframe():
    """block_reduce() should not add columns to or otherwise mutate the caller's dataframe."""
    data = pd.DataFrame({"dist": [0.0, 50.0, 100.0], "value": [1.0, 2.0, 3.0]})
    original_columns = list(data.columns)
    block_reduce(data, np.mean, spacing=200, reduce_by="dist", progressbar=False)
    assert list(data.columns) == original_columns
    assert "tmp" not in data.columns


def test_block_reduce_groupby_preserves_first_appearance_order():
    """Grouped output should follow the groups' first-appearance order in the input, not sorted order."""
    rng = np.random.default_rng(2)
    data = pd.DataFrame(
        {
            "line": ["B"] * 5 + ["A"] * 5 + ["C"] * 5,
            "dist": list(np.linspace(0, 400, 5)) * 3,
            "value": rng.random(15) * 10,
        }
    )
    result = block_reduce(
        data,
        np.mean,
        spacing=150,
        reduce_by="dist",
        groupby_column="line",
        progressbar=False,
    )
    assert result["line"].tolist() == ["B", "B", "B", "A", "A", "A", "C", "C", "C"]


@pytest.mark.parametrize("progressbar", [True, False])
def test_block_reduce_groupby_progressbar(progressbar):
    """Grouped reduction should give the same result regardless of the progressbar setting."""
    data = pd.DataFrame(
        {
            "line": ["a", "a", "b", "b"],
            "dist": [0.0, 50.0, 0.0, 50.0],
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )
    result = block_reduce(
        data,
        np.mean,
        spacing=1000,
        reduce_by="dist",
        groupby_column="line",
        progressbar=progressbar,
    )
    assert result["value"].tolist() == pytest.approx([1.5, 3.5])
    assert result["line"].tolist() == ["a", "b"]


def test_block_reduce_numeric_groupby_column_dtype_overwrite_bug():
    """Regression test: a numeric groupby_column gets redundantly block-reduced then overwritten by the literal group key, so its dtype ends up matching the key (int64) not the reduction function's float64 output."""
    data = pd.DataFrame(
        {
            "line": [1, 1, 2, 2],
            "dist": [0.0, 50.0, 0.0, 50.0],
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )
    result = block_reduce(
        data,
        np.mean,
        spacing=1000,
        reduce_by="dist",
        groupby_column="line",
        progressbar=False,
    )
    assert list(result.columns) == ["dist", "tmp", "line", "value"]
    assert result["line"].tolist() == [1, 2]
    assert result["line"].dtype == np.int64
    assert result["value"].tolist() == pytest.approx([1.5, 3.5])


def test_block_reduce_groupby_single_row_group():
    """A group with a single point should pass through unreduced alongside a multi-point group."""
    data = pd.DataFrame(
        {"line": ["A", "B", "B"], "dist": [0.0, 0.0, 50.0], "value": [1.0, 2.0, 3.0]}
    )
    result = block_reduce(
        data,
        np.mean,
        spacing=1000,
        reduce_by="dist",
        groupby_column="line",
        progressbar=False,
    )
    assert result["dist"].tolist() == pytest.approx([0.0, 25.0])
    assert result["value"].tolist() == pytest.approx([1.0, 2.5])
    assert result["line"].tolist() == ["A", "B"]


def test_block_reduce_excludes_geometry_column_even_if_numeric():
    """A numeric 'geometry' column should be excluded from the reduced output."""
    rng = np.random.default_rng(3)
    n = 10
    data = pd.DataFrame(
        {
            "easting": np.linspace(0, 1000, n) + rng.random(n),
            "northing": np.linspace(0, 500, n) + rng.random(n),
            "value": rng.random(n) * 10,
            "geometry": rng.random(n) * 5,
        }
    )
    result = block_reduce(
        data,
        np.mean,
        spacing=300,
        reduce_by=("easting", "northing"),
        progressbar=False,
    )
    assert list(result.columns) == ["easting", "northing", "value"]
    assert "geometry" not in result.columns


def test_block_reduce_excludes_non_numeric_columns():
    """Non-numeric columns should be dropped from the reduced output entirely."""
    data = pd.DataFrame(
        {
            "dist": [0.0, 50.0, 100.0, 500.0],
            "value": [1.0, 2.0, 3.0, 4.0],
            "label": ["a", "b", "c", "d"],
        }
    )
    result = block_reduce(
        data, np.mean, spacing=200, reduce_by="dist", progressbar=False
    )
    assert "label" not in result.columns
    assert list(result.columns) == ["dist", "tmp", "value"]


def test_block_reduce_center_coordinates_kwarg_changes_coordinates():
    """The center_coordinates kwarg should be passed through to verde.BlockReduce and change the output coordinates."""
    rng = np.random.default_rng(4)
    n = 10
    data = pd.DataFrame(
        {
            "easting": np.linspace(0, 1000, n) + rng.random(n),
            "northing": np.linspace(0, 500, n) + rng.random(n),
            "value": rng.random(n) * 10,
        }
    )
    default_result = block_reduce(
        data,
        np.mean,
        spacing=300,
        reduce_by=("easting", "northing"),
        progressbar=False,
    )
    centered_result = block_reduce(
        data,
        np.mean,
        spacing=300,
        reduce_by=("easting", "northing"),
        center_coordinates=True,
        progressbar=False,
    )
    assert not np.allclose(
        default_result["easting"].to_numpy(), centered_result["easting"].to_numpy()
    )


def test_block_reduce_unsupported_kwarg_raises_typeerror():
    """An unrecognized kwarg forwarded to verde.BlockReduce should raise TypeError."""
    data = pd.DataFrame({"dist": [0.0, 50.0], "value": [1.0, 2.0]})
    with pytest.raises(TypeError):
        block_reduce(
            data,
            np.mean,
            spacing=200,
            reduce_by="dist",
            not_a_kwarg=True,
            progressbar=False,
        )


def test_block_reduce_missing_column_tuple_reduce_by_raises():
    """A missing column in a tuple reduce_by should raise AssertionError naming both columns."""
    data = pd.DataFrame({"easting": [0.0, 1.0], "value": [1.0, 2.0]})
    with pytest.raises(
        AssertionError, match=r"\('easting', 'missing_col'\) must be in the dataframe"
    ):
        block_reduce(
            data,
            np.mean,
            spacing=200,
            reduce_by=("easting", "missing_col"),
            progressbar=False,
        )


def test_block_reduce_missing_column_string_reduce_by_raises():
    """A missing string reduce_by column should raise AssertionError naming it and the dummy 'tmp' column."""
    data = pd.DataFrame({"other": [0.0, 1.0], "value": [1.0, 2.0]})
    with pytest.raises(
        AssertionError, match=r"\('missing_col', 'tmp'\) must be in the dataframe"
    ):
        block_reduce(
            data, np.mean, spacing=200, reduce_by="missing_col", progressbar=False
        )


def test_block_reduce_missing_groupby_column_raises():
    """A missing groupby_column should raise a bare AssertionError with no message."""
    data = pd.DataFrame({"dist": [0.0, 1.0], "value": [1.0, 2.0]})
    with pytest.raises(AssertionError) as exc_info:
        block_reduce(
            data,
            np.mean,
            spacing=200,
            reduce_by="dist",
            groupby_column="nope",
            progressbar=False,
        )
    assert str(exc_info.value) == ""


def test_block_reduce_no_data_columns_raises_valueerror():
    """A dataframe with no data columns beyond reduce_by should raise ValueError."""
    data = pd.DataFrame({"dist": [0.0, 50.0, 100.0, 500.0]})
    with pytest.raises(ValueError, match="No objects to concatenate"):
        block_reduce(data, np.mean, spacing=200, reduce_by="dist", progressbar=False)


def test_block_reduce_empty_dataframe_no_groupby_raises_valueerror():
    """An empty dataframe with no groupby_column should raise ValueError."""
    data = pd.DataFrame(
        {"dist": pd.Series(dtype=float), "value": pd.Series(dtype=float)}
    )
    with pytest.raises(ValueError, match="zero-size array to reduction operation"):
        block_reduce(data, np.mean, spacing=200, reduce_by="dist", progressbar=False)


def test_block_reduce_empty_dataframe_with_groupby_raises_valueerror():
    """An empty dataframe with a groupby_column should raise ValueError."""
    data = pd.DataFrame(
        {
            "line": pd.Series(dtype=object),
            "dist": pd.Series(dtype=float),
            "value": pd.Series(dtype=float),
        }
    )
    with pytest.raises(ValueError, match="No objects to concatenate"):
        block_reduce(
            data,
            np.mean,
            spacing=200,
            reduce_by="dist",
            groupby_column="line",
            progressbar=False,
        )


def test_block_reduce_non_numeric_reduce_by_raises_typeerror():
    """A non-numeric reduce_by column should raise TypeError."""
    data = pd.DataFrame({"dist": ["a", "b", "c"], "value": [1.0, 2.0, 3.0]})
    with pytest.raises(TypeError):
        block_reduce(data, np.mean, spacing=200, reduce_by="dist", progressbar=False)


def test_block_reduce_single_point():
    """A single-point dataframe should pass through unchanged."""
    data = pd.DataFrame({"dist": [0.0], "value": [1.0]})
    result = block_reduce(
        data, np.mean, spacing=200, reduce_by="dist", progressbar=False
    )
    assert result["dist"].tolist() == pytest.approx([0.0])
    assert result["value"].tolist() == pytest.approx([1.0])


def test_block_reduce_all_coordinates_identical():
    """All points sharing the same coordinate should reduce to a single block."""
    data = pd.DataFrame({"dist": [10.0, 10.0, 10.0], "value": [1.0, 2.0, 3.0]})
    result = block_reduce(
        data, np.mean, spacing=200, reduce_by="dist", progressbar=False
    )
    assert len(result) == 1
    assert result["value"].iloc[0] == pytest.approx(2.0)
