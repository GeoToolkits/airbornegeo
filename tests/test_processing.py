import logging

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from airbornegeo.processing import detect_outliers, split_into_segments, unique_line_id


@pytest.fixture(autouse=True)
def _no_gui_plots(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *_a, **_k: None)
    yield
    plt.close("all")


def test_split_into_segments_basic():
    """Gaps larger than the threshold should split the data into sequential integer segments."""
    data = pd.DataFrame({"val": [0, 1, 2, 10, 11, 12, 20, 21, 22]})
    result = split_into_segments(data, threshold=5, column_name="val")
    assert result.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert result.dtype == np.int64


def test_split_into_segments_missing_column_raises():
    """A missing column_name should raise AssertionError naming it."""
    data = pd.DataFrame({"other": [1, 2, 3]})
    with pytest.raises(
        AssertionError, match=r"dataframe must contain columns \['val'\]"
    ):
        split_into_segments(data, threshold=5, column_name="val")


@pytest.mark.parametrize("segment_column", ["seg", "anything", "even_missing_col"])
def test_split_into_segments_segment_column_not_none_raises_not_implemented(
    segment_column,
):
    """Any non-None segment_column should raise NotImplementedError before other validation runs."""
    data = pd.DataFrame({"val": [0, 1, 2], "seg": [1.0, 1.0, 1.0]})
    with pytest.raises(NotImplementedError):
        split_into_segments(
            data, threshold=5, column_name="val", segment_column=segment_column
        )


def test_split_into_segments_angular_difference_wraps_large_jump():
    """With angular_difference=True, a jump from 0 to 350 degrees should wrap to -10, not split at a threshold of 100."""
    data = pd.DataFrame({"val": [0.0, 350.0]})
    result = split_into_segments(
        data, threshold=100, column_name="val", angular_difference=True
    )
    assert result.tolist() == [0, 0]


def test_split_into_segments_no_angular_difference_does_not_wrap():
    """Without angular_difference, the same 0-to-350 jump should be treated as a raw 350 diff and split."""
    data = pd.DataFrame({"val": [0.0, 350.0]})
    result = split_into_segments(
        data, threshold=100, column_name="val", angular_difference=False
    )
    assert result.tolist() == [0, 1]


def test_split_into_segments_angular_difference_values():
    """Angular-wrapped diffs should produce the exact expected segment boundaries for a sequence of degree values."""
    data = pd.DataFrame({"val": [350.0, 0.0, 10.0, 355.0, 5.0]})
    result = split_into_segments(
        data, threshold=9, column_name="val", angular_difference=True
    )
    assert result.tolist() == [0, 1, 2, 2, 3]


def test_split_into_segments_smoothing_window():
    """Smoothing the diff before thresholding can produce several segment boundaries around a single large gap."""
    data = pd.DataFrame({"val": [0, 2, 4, 6, 20, 22, 24, 26]})
    result = split_into_segments(
        data, threshold=5, column_name="val", smoothing_window=3
    )
    assert result.tolist() == [0, 0, 0, 0, 1, 2, 3, 3]


def test_split_into_segments_threshold_boundary_exact_equal_does_not_split():
    """A diff exactly equal to the threshold should not start a new segment."""
    data = pd.DataFrame({"val": [0.0, 5.0, 10.0]})
    result = split_into_segments(data, threshold=5, column_name="val")
    assert result.tolist() == [0, 0, 0]


def test_split_into_segments_threshold_boundary_just_above_splits():
    """A diff just above the threshold should start a new segment."""
    data = pd.DataFrame({"val": [0.0, 5.0001, 10.0]})
    result = split_into_segments(data, threshold=5, column_name="val")
    assert result.tolist() == [0, 1, 1]


def test_split_into_segments_min_points_per_segment_nans_small_segments():
    """Segments smaller than min_points_per_segment should be NaN'd out, preserving a custom index."""
    data = pd.DataFrame(
        {"val": [0, 1, 2, 3, 4, 20, 21, 40, 41, 42, 43]}, index=range(50, 61)
    )
    result = split_into_segments(
        data, threshold=5, column_name="val", min_points_per_segment=3
    )
    expected = [0.0, 0.0, 0.0, 0.0, 0.0, np.nan, np.nan, 2.0, 2.0, 2.0, 2.0]
    assert result.tolist() == pytest.approx(expected, nan_ok=True)
    assert result.dtype == np.float64
    assert list(result.index) == list(range(50, 61))


def test_split_into_segments_min_points_per_segment_log_message_all_dropped(caplog):
    """Regression test for an off-by-one bug: dropping all segments logs one fewer than the true dropped count."""
    data = pd.DataFrame(
        {"val": [0, 1, 2, 10, 11, 12, 20, 21, 22]}, index=range(100, 109)
    )
    with caplog.at_level(logging.INFO, logger="airbornegeo"):
        result = split_into_segments(
            data, threshold=5, column_name="val", min_points_per_segment=4
        )
    assert result.isna().all()
    assert list(result.index) == list(range(100, 109))
    assert "dropped 2 segments which contained less than 4 points." in caplog.text


def test_split_into_segments_min_points_per_segment_log_message_one_dropped(caplog):
    """Regression test: dropping exactly one segment (of three) logs a dropped count of 0, not 1."""
    data = pd.DataFrame(
        {
            "val": [
                0,
                1,
                2,
                3,
                4,  # segment 0, size 5
                20,
                21,  # segment 1, size 2 (should be dropped)
                40,
                41,
                42,
                43,  # segment 2, size 4
            ]
        }
    )
    with caplog.at_level(logging.INFO, logger="airbornegeo"):
        result = split_into_segments(
            data, threshold=5, column_name="val", min_points_per_segment=3
        )
    assert result.isna().sum() == 2
    assert "dropped 0 segments which contained less than 3 points." in caplog.text


@pytest.mark.parametrize("min_points_per_segment", [0, -1])
def test_split_into_segments_min_points_per_segment_non_positive_skips_filtering(
    min_points_per_segment, caplog
):
    """A non-positive min_points_per_segment should skip small-segment filtering entirely."""
    data = pd.DataFrame({"val": [0, 1, 2, 10, 11, 12, 20, 21, 22]})
    with caplog.at_level(logging.INFO, logger="airbornegeo"):
        result = split_into_segments(
            data,
            threshold=5,
            column_name="val",
            min_points_per_segment=min_points_per_segment,
        )
    assert result.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert result.dtype == np.int64
    assert "dropped" not in caplog.text


def test_split_into_segments_single_row():
    """A single-row dataframe should return a single segment 0."""
    data = pd.DataFrame({"val": [0.0]})
    result = split_into_segments(data, threshold=5, column_name="val")
    assert result.tolist() == [0]


def test_split_into_segments_empty_dataframe():
    """An empty dataframe should return an empty int64 segment series."""
    data = pd.DataFrame({"val": pd.Series(dtype=float)})
    result = split_into_segments(data, threshold=5, column_name="val")
    assert len(result) == 0
    assert result.dtype == np.int64


def test_unique_line_id_order_of_first_appearance():
    """Line labels should be mapped to integers in order of first appearance, not sorted order."""
    data = pd.DataFrame({"line": ["b", "a", "c", "a", "b", "d"]})
    result = unique_line_id(data, line_col_name="line")
    assert result.tolist() == [1, 2, 3, 2, 1, 4]
    assert result.dtype == np.int64


def test_unique_line_id_custom_column_name():
    """A custom line_col_name should be used instead of the default 'line' column."""
    data = pd.DataFrame({"flight": ["x", "y", "x"]})
    result = unique_line_id(data, line_col_name="flight")
    assert result.tolist() == [1, 2, 1]


def test_unique_line_id_numeric_labels():
    """Numeric line labels should be mapped to sequential integers the same way string labels are."""
    data = pd.DataFrame({"flight_line": [100, 200, 100, 300]})
    result = unique_line_id(data, line_col_name="flight_line")
    assert result.tolist() == [1, 2, 1, 3]


def test_unique_line_id_single_repeated_label():
    """A single repeated label should map to the same integer for every row."""
    data = pd.DataFrame({"line": ["x", "x", "x"]})
    result = unique_line_id(data, line_col_name="line")
    assert result.tolist() == [1, 1, 1]


def test_unique_line_id_preserves_custom_index():
    """The result should keep the input dataframe's original index."""
    data = pd.DataFrame({"line": ["a", "b", "a"]}, index=[10, 20, 30])
    result = unique_line_id(data, line_col_name="line")
    assert list(result.index) == [10, 20, 30]


def test_unique_line_id_empty_dataframe():
    """An empty dataframe should return an empty int64 series."""
    data = pd.DataFrame({"line": pd.Series([], dtype=object)})
    result = unique_line_id(data, line_col_name="line")
    assert len(result) == 0
    assert result.dtype == np.int64


def test_unique_line_id_missing_column_raises_keyerror():
    """A missing line_col_name should raise a plain KeyError."""
    data = pd.DataFrame({"other": [1, 2, 3]})
    with pytest.raises(KeyError):
        unique_line_id(data, line_col_name="line")


def test_unique_line_id_nan_in_labels_raises():
    """A NaN/None value in the line label column should raise IntCastingNaNError."""
    data = pd.DataFrame({"line": ["a", None, "a", "b"]})
    with pytest.raises(
        pd.errors.IntCastingNaNError, match="Cannot convert non-finite values"
    ):
        unique_line_id(data, line_col_name="line")


def test_detect_outliers_no_outliers_logs_info(caplog):
    """A column with no IQR outliers should log an info message and create no figure."""
    data = pd.DataFrame({"no_outliers": [1, 2, 3, 4, 5]})
    with caplog.at_level(logging.INFO, logger="airbornegeo"):
        result = detect_outliers(data)  # pylint: disable=assignment-from-no-return
    assert result is None
    assert "No outliers detected in column: no_outliers" in caplog.text
    assert plt.get_fignums() == []


def test_detect_outliers_with_outliers_creates_figure(caplog):
    """A column with an IQR outlier should create a boxplot figure and log no 'no outliers' message."""
    data = pd.DataFrame({"has_outliers": [1, 2, 3, 4, 1000]})
    with (
        caplog.at_level(logging.INFO, logger="airbornegeo"),
        pytest.warns(PendingDeprecationWarning, match="vert"),
    ):
        result = detect_outliers(data)  # pylint: disable=assignment-from-no-return
    assert result is None
    assert len(plt.get_fignums()) == 1
    assert "No outliers detected" not in caplog.text


def test_detect_outliers_non_numeric_column_skipped(caplog):
    """A non-numeric column should be skipped entirely - no log message, no figure."""
    data = pd.DataFrame({"label": ["a", "b", "c", "d", "e"]})
    with caplog.at_level(logging.INFO, logger="airbornegeo"):
        result = detect_outliers(data)  # pylint: disable=assignment-from-no-return
    assert result is None
    assert caplog.text == ""
    assert plt.get_fignums() == []


def test_detect_outliers_mixed_dataframe(caplog):
    """Only the outlier-containing numeric column should plot; the clean numeric column logs, and the label column is ignored."""
    data = pd.DataFrame(
        {
            "no_outliers": [1, 2, 3, 4, 5],
            "has_outliers": [1, 2, 3, 4, 1000],
            "label": ["a", "b", "c", "d", "e"],
        }
    )
    with (
        caplog.at_level(logging.INFO, logger="airbornegeo"),
        pytest.warns(PendingDeprecationWarning, match="vert"),
    ):
        detect_outliers(data)
    assert "No outliers detected in column: no_outliers" in caplog.text
    assert "has_outliers" not in caplog.text
    assert "label" not in caplog.text
    assert len(plt.get_fignums()) == 1


def test_detect_outliers_nan_containing_column_still_detects_outlier(caplog):
    """NaN values should be ignored by the IQR calculation, but a real outlier should still be detected."""
    data = pd.DataFrame({"col": [1.0, 2.0, np.nan, 4.0, 100.0]})
    with (
        caplog.at_level(logging.INFO, logger="airbornegeo"),
        pytest.warns(PendingDeprecationWarning, match="vert"),
    ):
        detect_outliers(data)
    assert "No outliers detected" not in caplog.text
    assert len(plt.get_fignums()) == 1


def test_detect_outliers_all_nan_column_reports_no_outliers(caplog):
    """An all-NaN numeric column should report no outliers rather than erroring."""
    data = pd.DataFrame({"all_nan": [np.nan, np.nan, np.nan]})
    with caplog.at_level(logging.INFO, logger="airbornegeo"):
        result = detect_outliers(data)  # pylint: disable=assignment-from-no-return
    assert result is None
    assert "No outliers detected in column: all_nan" in caplog.text
    assert plt.get_fignums() == []


def test_detect_outliers_empty_dataframe():
    """An empty dataframe should return None without error."""
    result = detect_outliers(pd.DataFrame())  # pylint: disable=assignment-from-no-return
    assert result is None


def test_detect_outliers_boolean_column_raises_typeerror():
    """Documents a real limitation: pandas treats bool columns as numeric, so IQR arithmetic on them raises TypeError."""
    data = pd.DataFrame({"boolean": [True, False, True, False, True]})
    with pytest.raises(TypeError):
        detect_outliers(data)
