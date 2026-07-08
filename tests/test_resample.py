import numpy as np
import pandas as pd
import pytest

from airbornegeo.resample import resample, resample_as


def test_resample_maxdist_none_raises_typeerror():
    """Documents a real bug: maxdist=None is claimed to be a valid default, but verde.distance_mask always raises for it."""
    data = pd.DataFrame({"t": [0.0, 1.0, 2.0], "v": [0.0, 1.0, 2.0]})
    with pytest.raises(TypeError):
        resample(
            data,
            spacing=1.0,
            resample_by="t",
            maxdist=None,
            groupby_column=None,
            progressbar=False,
        )


def test_resample_linear_function_reproduced_exactly():
    """Resampling a linear function should reproduce it exactly via cubic interpolation."""
    n = 20
    t = np.linspace(0, 10, n)
    val = 2.0 * t + 1.0
    data = pd.DataFrame({"t": t, "val": val})
    result = resample(
        data,
        spacing=1.0,
        resample_by="t",
        maxdist=100.0,
        groupby_column=None,
        progressbar=False,
    )
    assert list(result.columns) == ["t", "val"]
    assert result.t.tolist() == pytest.approx(list(range(11)))
    assert result.val.to_numpy() == pytest.approx(2.0 * result.t.to_numpy() + 1.0)


def test_resample_preserves_integer_dtype():
    """An integer-dtype data column should keep its dtype after resampling."""
    n = 20
    t = np.linspace(0, 10, n)
    data = pd.DataFrame({"t": t, "val": 2.0 * t, "flag": np.arange(n)})
    result = resample(
        data,
        spacing=1.0,
        resample_by="t",
        maxdist=100.0,
        groupby_column=None,
        progressbar=False,
    )
    assert result.flag.dtype == np.int64


def test_resample_maxdist_masks_points_near_gaps():
    """A tighter maxdist should mask out more resampled points near a data gap than a looser one."""
    t_gap = np.concatenate([np.linspace(0, 5, 10), np.linspace(20, 25, 10)])
    data = pd.DataFrame({"t": t_gap, "val": t_gap * 2})
    loose = resample(
        data,
        spacing=1.0,
        resample_by="t",
        maxdist=100.0,
        groupby_column=None,
        progressbar=False,
    )
    tight = resample(
        data,
        spacing=1.0,
        resample_by="t",
        maxdist=1.0,
        groupby_column=None,
        progressbar=False,
    )
    assert len(tight) < len(loose)


def test_resample_string_groupby_column_raises_because_it_is_dropped_as_non_numeric():
    """A string-labeled groupby_column is silently dropped by the numeric-only column filter, so it raises AssertionError even though it was provided."""
    n = 10
    t = np.linspace(0, 9, n)
    data = pd.DataFrame({"t": t, "val": t * 2, "line": ["A"] * n})
    with pytest.raises(AssertionError, match="groupby_column must be in the dataframe"):
        resample(
            data,
            spacing=1.0,
            resample_by="t",
            maxdist=100.0,
            groupby_column="line",
            progressbar=False,
        )


def test_resample_missing_groupby_column_raises():
    """A groupby_column that doesn't exist at all should raise AssertionError."""
    data = pd.DataFrame({"t": [0.0, 1.0, 2.0], "v": [0.0, 1.0, 2.0]})
    with pytest.raises(AssertionError, match="groupby_column must be in the dataframe"):
        resample(
            data,
            spacing=1.0,
            resample_by="t",
            maxdist=100.0,
            groupby_column="nope",
            progressbar=False,
        )


def test_resample_missing_resample_by_raises():
    """A missing resample_by column should raise AssertionError."""
    data = pd.DataFrame({"other": [1.0, 2.0, 3.0]})
    with pytest.raises(AssertionError, match="t must be in the dataframe"):
        resample(
            data,
            spacing=1.0,
            resample_by="t",
            maxdist=100.0,
            groupby_column=None,
            progressbar=False,
        )


@pytest.mark.parametrize("progressbar", [True, False])
def test_resample_numeric_groupby_preserves_first_appearance_order(progressbar):
    """Grouped resampling should follow first-appearance order of group labels, not sorted order, and fit each group independently."""
    t = np.linspace(0, 9, 10)
    data = pd.DataFrame(
        {
            "t": np.concatenate([t, t]),
            "val": np.concatenate([2 * t + 1, 3 * t + 5]),
            "line": [2] * 10 + [1] * 10,  # line 2 appears first, sorts after 1
        }
    )
    result = resample(
        data,
        spacing=1.0,
        resample_by="t",
        maxdist=100.0,
        groupby_column="line",
        progressbar=progressbar,
    )
    assert result.line.unique().tolist() == [2, 1]
    line2 = result[result.line == 2]
    line1 = result[result.line == 1]
    assert line2.val.to_numpy() == pytest.approx(2 * line2.t.to_numpy() + 1)
    assert line1.val.to_numpy() == pytest.approx(3 * line1.t.to_numpy() + 5)


def test_resample_as_clips_values_outside_data_range():
    """resample_as() should drop requested values that fall outside the original data's range."""
    n = 20
    t = np.linspace(0, 10, n)
    val = 2.0 * t + 1.0
    data = pd.DataFrame({"t": t, "val": val})
    resample_values = np.array([-5.0, 0.0, 2.5, 5.0, 10.0, 15.0])
    result = resample_as(
        data,
        resample_by="t",
        resample_values=resample_values,
        groupby_column=None,
        progressbar=False,
    )
    # -5.0 and 15.0 fall outside [0, 10] and are dropped
    assert result.t.tolist() == pytest.approx([0.0, 2.5, 5.0, 10.0])
    assert result.val.to_numpy() == pytest.approx(2.0 * result.t.to_numpy() + 1.0)


def test_resample_as_missing_resample_by_raises():
    """A missing resample_by column should raise AssertionError."""
    data = pd.DataFrame({"other": [1.0, 2.0]})
    with pytest.raises(AssertionError, match="t must be in the dataframe"):
        resample_as(
            data,
            resample_by="t",
            resample_values=np.array([1.0]),
            groupby_column=None,
            progressbar=False,
        )


def test_resample_as_groupby():
    """Grouped resample_as() should interpolate each group independently at the requested values."""
    t = np.linspace(0, 9, 10)
    data = pd.DataFrame(
        {
            "t": np.concatenate([t, t]),
            "val": np.concatenate([2 * t + 1, 3 * t + 5]),
            "line": [1] * 10 + [2] * 10,
        }
    )
    result = resample_as(
        data,
        resample_by="t",
        resample_values=np.array([0.0, 3.0, 6.0, 9.0]),
        groupby_column="line",
        progressbar=False,
    )
    assert result.shape == (8, 3)
    line1 = result[result.line == 1]
    line2 = result[result.line == 2]
    assert line1.val.to_numpy() == pytest.approx(2 * line1.t.to_numpy() + 1)
    assert line2.val.to_numpy() == pytest.approx(3 * line2.t.to_numpy() + 5)
