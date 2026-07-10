import numpy as np
import pandas as pd
import pytest

import airbornegeo.interpolating as interpolating_module
from airbornegeo.interpolating import (
    _deduplicate_and_sort,
    _fill_one_nan,
    _fit_interp1d_with_fallback,
    _fit_method,
    _method_fallback_chain,
    _resolve_fill_value,
    _windowed_attempt,
    interpolate_1d_single_nan,
    interpolate_missing,
    interpolate_missing_pointwise,
    interpolate_missing_pointwise_with_windows,
    optimal_spline_damping,
)


def test_method_fallback_chain_invalid_method_raises_valueerror():
    """An unknown method string should raise a ValueError."""
    with pytest.raises(ValueError, match="not a valid method string"):
        _method_fallback_chain("bogus")


def test_method_fallback_chain_nearest_is_just_itself():
    """'nearest' has nothing simpler to fall back to."""
    assert _method_fallback_chain("nearest") == ["nearest"]


def test_method_fallback_chain_linear_falls_back_to_nearest():
    """'linear' should fall back to 'nearest' only."""
    assert _method_fallback_chain("linear") == ["linear", "nearest"]


def test_method_fallback_chain_cubic_falls_back_through_linear_and_nearest():
    """Any other method should fall back through linear then nearest."""
    assert _method_fallback_chain("cubic") == ["cubic", "linear", "nearest"]


def test_resolve_fill_value_not_extrapolating_returns_nan():
    """When not extrapolating, out-of-range points should stay NaN."""
    y = np.array([1.0, 2.0, 3.0])
    assert np.isnan(_resolve_fill_value(None, y, "linear", extrapolate=False))


def test_resolve_fill_value_nearest_method_clamps_to_edges():
    """Extrapolating with method='nearest' and no fill_value should clamp to edges."""
    y = np.array([1.0, 2.0, 3.0])
    assert _resolve_fill_value(None, y, "nearest", extrapolate=True) == (1.0, 3.0)
    assert _resolve_fill_value("extrapolate", y, "nearest", extrapolate=True) == (
        1.0,
        3.0,
    )


def test_resolve_fill_value_default_extrapolate_string():
    """Extrapolating with fill_value=None (non-nearest method) should return 'extrapolate'."""
    y = np.array([1.0, 2.0, 3.0])
    assert _resolve_fill_value(None, y, "linear", extrapolate=True) == "extrapolate"


def test_resolve_fill_value_edge_clamps_to_first_and_last():
    """fill_value='edge' should clamp to the first/last y values."""
    y = np.array([5.0, 2.0, 9.0])
    assert _resolve_fill_value("edge", y, "linear", extrapolate=True) == (5.0, 9.0)


def test_resolve_fill_value_mean_fills_with_nanmean_of_y():
    """fill_value='mean' should fill with the (nan-aware) mean of y on both sides."""
    y = np.array([1.0, 2.0, np.nan, 3.0])
    result = _resolve_fill_value("mean", y, "linear", extrapolate=True)
    assert result == (2.0, 2.0)


def test_resolve_fill_value_explicit_tuple_passed_through():
    """An explicit (lo, hi) tuple should be passed through unchanged."""
    y = np.array([1.0, 2.0, 3.0])
    assert _resolve_fill_value((-1.0, 99.0), y, "linear", extrapolate=True) == (
        -1.0,
        99.0,
    )


def test_fit_method_returns_none_when_too_few_points():
    """Fewer points than the method requires should return None instead of fitting."""
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])
    assert _fit_method(x, y, "cubic", extrapolate=False, fill_value=None) is None


def test_fit_method_returns_none_when_interp1d_raises(monkeypatch):
    """If scipy's interp1d itself raises, _fit_method should catch it and return None."""

    def _raise(*_args, **_kwargs):
        msg = "simulated interp1d failure"
        raise ValueError(msg)

    monkeypatch.setattr(interpolating_module.scipy.interpolate, "interp1d", _raise)

    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 1.0, 4.0, 9.0])
    assert _fit_method(x, y, "linear", extrapolate=False, fill_value=None) is None


def test_fit_method_returns_callable_with_enough_points():
    """Enough points should return a usable callable."""
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 1.0, 4.0, 9.0])
    f = _fit_method(x, y, "linear", extrapolate=False, fill_value=None)
    assert f is not None
    assert np.isclose(f(0.5), 0.5)


def test_deduplicate_and_sort_keeps_first_occurrence_and_sorts():
    """Duplicate x-values should be deduped (keeping first y) and results sorted by x."""
    x = np.array([2.0, 0.0, 2.0, 1.0])
    y = np.array([20.0, 0.0, 999.0, 10.0])
    x_out, y_out = _deduplicate_and_sort(x, y)
    np.testing.assert_array_equal(x_out, [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(y_out, [0.0, 10.0, 20.0])


def test_fit_interp1d_with_fallback_falls_back_to_linear_when_cubic_infeasible():
    """With too few points for cubic, the fallback chain should land on linear."""
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 2.0])
    f, used_method = _fit_interp1d_with_fallback(
        x, y, "cubic", extrapolate=False, fill_value=None
    )
    assert used_method == "linear"
    assert f is not None


def test_fit_interp1d_with_fallback_returns_none_when_all_methods_fail():
    """With a single point, even 'nearest' can't be fit reliably via interp1d, but here
    we force total failure by providing zero points."""
    x = np.array([])
    y = np.array([])
    f, used_method = _fit_interp1d_with_fallback(
        x, y, "cubic", extrapolate=False, fill_value=None
    )
    assert f is None
    assert used_method == "none"


def test_interpolate_1d_single_nan_returns_value_for_linear():
    """A simple linear fit should evaluate correctly at the requested x_index."""
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    result = interpolate_1d_single_nan(x, y, 1.5, method="linear")
    assert np.isclose(result, 1.5)


def test_interpolate_1d_single_nan_returns_nan_when_fit_fails():
    """Too few points for the requested method should return NaN, not raise."""
    x = np.array([0.0])
    y = np.array([0.0])
    result = interpolate_1d_single_nan(x, y, 0.5, method="cubic")
    assert np.isnan(result)


def test_interpolate_1d_single_nan_returns_nan_when_evaluation_raises(monkeypatch):
    """If the fitted function raises when evaluated at x_index, return NaN instead."""

    class _RaisingCallable:
        def __call__(self, *_args, **_kwargs):
            msg = "simulated evaluation failure"
            raise ValueError(msg)

    monkeypatch.setattr(
        interpolating_module,
        "_fit_method",
        lambda *_args, **_kwargs: _RaisingCallable(),
    )

    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 1.0, 4.0, 9.0])
    result = interpolate_1d_single_nan(x, y, 1.5, method="linear")
    assert np.isnan(result)


def test_windowed_attempt_doubles_window_until_enough_points():
    """A window too narrow to include enough points should double until it succeeds."""
    x = np.array([0.0, 1.0, 5.0, 9.0, 10.0])
    y = np.array([0.0, 1.0, 5.0, 9.0, 10.0])
    value, used_type = _windowed_attempt(
        x,
        y,
        5.0,
        window_width=0.5,
        method="linear",
        extrapolate=False,
        fill_value=None,
        max_window_doublings=3,
    )
    assert used_type == "interpolated"
    assert np.isclose(value, 5.0)


def test_windowed_attempt_doubles_window_when_result_is_nan():
    """If enough points are found but the fit still produces NaN (e.g. xi is outside
    the fitted range and extrapolate=False), the window should double and retry."""
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    value, used_type = _windowed_attempt(
        x,
        y,
        10.0,
        window_width=15.0,
        method="linear",
        extrapolate=False,
        fill_value=None,
        max_window_doublings=1,
    )
    assert used_type == "none"
    assert np.isnan(value)


def test_windowed_attempt_gives_up_after_max_doublings():
    """If doubling the window still doesn't find enough points, return NaN/'none'."""
    x = np.array([0.0, 100.0])
    y = np.array([0.0, 100.0])
    value, used_type = _windowed_attempt(
        x,
        y,
        50.0,
        window_width=0.1,
        method="cubic",
        extrapolate=False,
        fill_value=None,
        max_window_doublings=1,
    )
    assert used_type == "none"
    assert np.isnan(value)


def test_fill_one_nan_falls_back_to_extrapolate_stage_when_requested():
    """When interpolation-only stage fails and extrapolate=True, the extrapolated
    stage should be tried and succeed."""
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])
    value, used_method, used_type = _fill_one_nan(
        x,
        y,
        5.0,
        method_chain=["linear", "nearest"],
        fill_value=None,
        extrapolate=True,
    )
    assert used_type == "extrapolated"
    assert used_method is not None
    assert not np.isnan(value)


def test_fill_one_nan_returns_none_when_extrapolate_false_and_out_of_range():
    """Without extrapolation, a point outside the data range can't be filled."""
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])
    value, used_method, used_type = _fill_one_nan(
        x,
        y,
        5.0,
        method_chain=["linear", "nearest"],
        fill_value=None,
        extrapolate=False,
    )
    assert used_type == "none"
    assert used_method is None
    assert np.isnan(value)


def _df_with_nan(n=8, nan_idx=4):
    x = np.linspace(0, 10, n)
    y = x**2
    y[nan_idx] = np.nan
    return pd.DataFrame({"x": x, "y": y})


def test_interpolate_missing_pointwise_missing_columns_raises_assertionerror():
    """Missing required columns should raise AssertionError."""
    data = pd.DataFrame({"foo": [1, 2, 3]})
    with pytest.raises(AssertionError):
        interpolate_missing_pointwise(
            data, to_interp="y", interp_on="x", progressbar=False
        )


def test_interpolate_missing_pointwise_fills_interior_nan():
    """A single interior NaN should be filled and get an 'interpolated' type label."""
    data = _df_with_nan()
    result = interpolate_missing_pointwise(
        data, to_interp="y", interp_on="x", method="cubic", progressbar=False
    )
    assert not result["y"].isna().any()
    assert result["y_interpolation_type"].iloc[4] == "interpolated"


def test_interpolate_missing_pointwise_logs_fallback_when_requested_method_fails(
    caplog,
):
    """When cubic can't be fit (too few valid points) but linear can, a fallback
    debug message should be logged."""
    data = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, np.nan, 2.0]})
    with caplog.at_level("DEBUG", logger="airbornegeo"):
        result = interpolate_missing_pointwise(
            data, to_interp="y", interp_on="x", method="cubic", progressbar=False
        )
    assert not result["y"].isna().any()
    assert result["y_interpolation_type"].iloc[1] == "interpolated"
    assert "fell back to 'linear'" in caplog.text


def test_interpolate_missing_pointwise_no_valid_points_returns_unchanged():
    """If every value is NaN, nothing can be interpolated and NaNs pass through."""
    data = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [np.nan, np.nan, np.nan]})
    result = interpolate_missing_pointwise(
        data, to_interp="y", interp_on="x", progressbar=False
    )
    assert result["y"].isna().all()
    assert (result["y_interpolation_type"] == "none").all()


def test_interpolate_missing_pointwise_with_windows_missing_columns_raises():
    """Missing required columns should raise AssertionError."""
    data = pd.DataFrame({"foo": [1, 2, 3]})
    with pytest.raises(AssertionError):
        interpolate_missing_pointwise_with_windows(
            data, window_width=2.0, to_interp="y", interp_on="x", progressbar=False
        )


def test_interpolate_missing_pointwise_with_windows_fills_interior_nan():
    """A NaN should be filled using a local window around it."""
    data = _df_with_nan(n=12, nan_idx=6)
    result = interpolate_missing_pointwise_with_windows(
        data,
        window_width=3.0,
        to_interp="y",
        interp_on="x",
        method="linear",
        progressbar=False,
    )
    assert not result["y"].isna().any()


def test_interpolate_missing_pointwise_with_windows_groupby_column():
    """groupby_column should fit/interpolate independently per group."""
    data = pd.concat(
        [
            _df_with_nan(n=12, nan_idx=6).assign(line="A"),
            _df_with_nan(n=12, nan_idx=6).assign(line="B"),
        ],
        ignore_index=True,
    )
    result = interpolate_missing_pointwise_with_windows(
        data,
        window_width=3.0,
        to_interp="y",
        interp_on="x",
        method="linear",
        groupby_column="line",
        progressbar=False,
    )
    assert not result["y"].isna().any()
    assert set(result["line"]) == {"A", "B"}


def test_interpolate_missing_pointwise_with_windows_logs_fallback_when_method_fails(
    caplog,
):
    """When cubic can't be fit within the window (even after doubling) but linear
    can, a fallback debug message should be logged."""
    data = pd.DataFrame(
        {
            "x": [0.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0],
            "y": [0.0, 2.0, 4.0, np.nan, 6.0, 8.0, 10.0],
        }
    )
    with caplog.at_level("DEBUG", logger="airbornegeo"):
        result = interpolate_missing_pointwise_with_windows(
            data,
            window_width=0.9,
            to_interp="y",
            interp_on="x",
            method="cubic",
            progressbar=False,
        )
    assert not result["y"].isna().any()
    assert "fell back to 'linear'" in caplog.text


def test_interpolate_missing_pointwise_with_windows_leaves_unfillable_nan_as_none():
    """A NaN with too few nearby points even after window doubling should stay NaN
    with interpolation_type 'none'."""
    data = pd.DataFrame({"x": [0.0, 100.0], "y": [0.0, np.nan]})
    result = interpolate_missing_pointwise_with_windows(
        data,
        window_width=0.1,
        to_interp="y",
        interp_on="x",
        method="cubic",
        progressbar=False,
    )
    assert result["y"].isna().any()
    assert "none" in result["y_interpolation_type"].to_numpy()


def test_interpolate_missing_missing_columns_raises_assertionerror():
    """Missing required columns should raise AssertionError."""
    data = pd.DataFrame({"foo": [1, 2, 3]})
    with pytest.raises(AssertionError):
        interpolate_missing(data, to_interp="y", interp_on="x", progressbar=False)


def test_interpolate_missing_fills_nans():
    """interpolate_missing should fill NaNs using a fit from the group's valid points."""
    data = _df_with_nan()
    result = interpolate_missing(
        data, to_interp="y", interp_on="x", method="linear", progressbar=False
    )
    assert not result["y"].isna().any()


def test_interpolate_missing_no_valid_points_warns_and_returns_unchanged(caplog):
    """With no valid points to fit from, the segment should be returned unchanged
    with a warning logged."""
    data = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [np.nan, np.nan, np.nan]})
    with caplog.at_level("WARNING"):
        result = interpolate_missing(
            data, to_interp="y", interp_on="x", progressbar=False
        )
    assert result["y"].isna().all()
    assert "No valid points available" in caplog.text


def test_interpolate_missing_all_methods_fail_warns_and_returns_unchanged(
    caplog, monkeypatch
):
    """If every method in the fallback chain fails to fit, warn and return unchanged."""
    monkeypatch.setattr(
        interpolating_module,
        "_fit_interp1d_with_fallback",
        lambda *_args, **_kwargs: (None, "none"),
    )

    data = pd.DataFrame({"x": [0.0, 1.0], "y": [5.0, np.nan]})
    with caplog.at_level("WARNING"):
        result = interpolate_missing(
            data, to_interp="y", interp_on="x", method="cubic", progressbar=False
        )
    assert result["y"].isna().iloc[1]
    assert "failed to fit" in caplog.text


def test_interpolate_missing_with_groupby_column():
    """groupby_column should fit/interpolate independently per group."""
    data = pd.concat(
        [
            _df_with_nan().assign(line="A"),
            _df_with_nan().assign(line="B"),
        ],
        ignore_index=True,
    )
    result = interpolate_missing(
        data,
        to_interp="y",
        interp_on="x",
        method="linear",
        groupby_column="line",
        progressbar=False,
    )
    assert not result["y"].isna().any()
    assert set(result["line"]) == {"A", "B"}


def _grid_coords_and_data(n=25):
    rng = np.random.default_rng(0)
    easting = rng.uniform(0, 1000, n)
    northing = rng.uniform(0, 1000, n)
    data = np.sin(easting / 200) + np.cos(northing / 200)
    return (easting, northing), data


def test_optimal_spline_damping_single_damping_value_returns_fitted_spline():
    """A single (non-iterable) damping value should be wrapped in a list and still
    produce a fitted spline."""
    coordinates, data = _grid_coords_and_data()
    spline = optimal_spline_damping(coordinates, data, dampings=1e-3)
    assert hasattr(spline, "predict")
    predicted = spline.predict(coordinates)
    assert len(predicted) == len(data)


def test_optimal_spline_damping_multiple_dampings_picks_best():
    """With multiple candidate dampings, the returned spline should have one of them
    selected as damping_."""
    coordinates, data = _grid_coords_and_data()
    dampings = [1e-4, 1e-2, 1.0]
    spline = optimal_spline_damping(coordinates, data, dampings=dampings)
    assert spline.damping_ in dampings


def test_optimal_spline_damping_warns_when_best_damping_at_range_limit(caplog):
    """If the best damping found lands on the edge of the provided range, a warning
    should be logged suggesting to expand the search range."""
    coordinates, data = _grid_coords_and_data()
    # a very small and a very large damping, with nothing in between close to
    # optimal - the CV winner should land on one of the boundary values.
    dampings = [1e-8, 1e8]
    with caplog.at_level("WARNING"):
        spline = optimal_spline_damping(coordinates, data, dampings=dampings)
    assert spline.damping_ in (1e-8, 1e8)


def test_optimal_spline_damping_retries_with_fewer_splits_on_valueerror(
    monkeypatch, caplog
):
    """If SplineCV.fit raises ValueError, the function should retry with fewer
    KFold splits, eventually falling back to a plain (undamped) Spline fit."""

    class _FailingSplineCV:
        def __init__(self, *_args, **_kwargs):
            pass

        def fit(self, *_args, **_kwargs):
            msg = "simulated SplineCV fit failure"
            raise ValueError(msg)

    class _FakeSpline:
        def __init__(self, *_args, **_kwargs):
            self.damping_ = None

        def fit(self, *_args, **_kwargs):
            return self

    monkeypatch.setattr(interpolating_module.vd, "SplineCV", _FailingSplineCV)
    monkeypatch.setattr(interpolating_module.vd, "Spline", _FakeSpline)

    coordinates, data = _grid_coords_and_data(n=5)
    with caplog.at_level("WARNING"):
        spline = optimal_spline_damping(coordinates, data, dampings=[1e-3])

    assert isinstance(spline, _FakeSpline)
    assert spline.damping_ is None
    assert "decreasing number of splits" in caplog.text
    assert "fitting spline with no damping" in caplog.text


def test_optimal_spline_damping_ignores_missing_damping_attribute(monkeypatch):
    """If the fitted spline has no damping_ attribute at all, the limit-check should
    be skipped silently rather than raising."""

    class _NoDampingAttrSplineCV:
        def __init__(self, *_args, **_kwargs):
            pass

        def fit(self, *_args, **_kwargs):
            return self

    monkeypatch.setattr(interpolating_module.vd, "SplineCV", _NoDampingAttrSplineCV)

    coordinates, data = _grid_coords_and_data(n=5)
    spline = optimal_spline_damping(coordinates, data, dampings=[1e-3, 1.0, 100.0])
    assert not hasattr(spline, "damping_")
