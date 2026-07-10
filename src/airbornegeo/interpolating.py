import copy
import typing
import warnings

import numpy as np
import pandas as pd
import scipy
import sklearn
import verde as vd
from tqdm.autonotebook import tqdm

import airbornegeo
from airbornegeo import logger

_METHOD_MIN_POINTS = {
    "cubic": 4,
    "quadratic": 3,
    "slinear": 2,
    "linear": 2,
    "zero": 1,
    "previous": 1,
    "next": 1,
    "nearest": 1,
    "nearest-up": 1,
}


def _method_fallback_chain(method: str) -> list[str]:
    """
    Build the fallback hierarchy `method -> linear -> nearest`, keeping only stages at
    or below the requested method's point requirement. 'nearest' has nothing simpler
    to fall back to, so its chain is just itself.
    """
    if method not in _METHOD_MIN_POINTS:
        msg = "not a valid method string"
        raise ValueError(msg)

    if method == "nearest":
        return ["nearest"]
    if method == "linear":
        return ["linear", "nearest"]
    return [method, "linear", "nearest"]


def _apply_over_groups(
    data: pd.DataFrame,
    *,
    groupby_column: str | None,
    progressbar: bool,
    interp_func: typing.Callable[..., pd.DataFrame],
    interp_func_kwargs: dict,
    pass_group_name: bool = False,
) -> pd.DataFrame:
    """
    Shared driver for `interpolate_missing_pointwise`,
    `interpolate_missing_pointwise_with_windows`, and `interpolate_missing`: applies
    `interp_func` to the whole dataframe, or to each group in turn if
    `groupby_column` is provided.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing the data to interpolate.
    groupby_column : str | None
        Column name to group by before interpolating.
    progressbar : bool
        Show progress bar for each group.
    interp_func : Callable
        The interpolation function to apply, e.g.
        `_interpolate_missing_pointwise`.
    interp_func_kwargs : dict
        Keyword arguments (other than `data`) to pass through to `interp_func`.
    pass_group_name : bool, optional
        If True, also pass the group's key through as `segment_name=` (only used
        when `groupby_column` is set), so `interp_func` can reference it in log
        messages. By default False.

    Returns
    -------
    pd.DataFrame
        Dataframe with the interpolated column.
    """
    data = data.copy()

    if groupby_column is None:
        return interp_func(data, **interp_func_kwargs)

    groups = data.groupby(groupby_column, sort=False)
    pbar = tqdm(groups, desc="Interpolating segments") if progressbar else groups

    filled_segments = [
        interp_func(segment_data, segment_name=segment_name, **interp_func_kwargs)
        if pass_group_name
        else interp_func(segment_data, **interp_func_kwargs)
        for segment_name, segment_data in pbar
    ]

    return pd.concat(filled_segments, ignore_index=True)


def interpolate_missing_pointwise(
    data: pd.DataFrame,
    *,
    to_interp: str,
    interp_on: str,
    method: str = "cubic",
    extrapolate: bool = False,
    fill_value: tuple[float, float] | str | None = None,
    groupby_column: str | None = None,
    progressbar: bool = True,
) -> pd.DataFrame:
    """
    Interpolate NaN's in the "to_interp" column, based on values from "interp_on". If
    groupby_column is provided, the dataframe will first be grouped by this so only
    data from the group containing the NaN is used to interpolate.

    To interpolate multiple columns, call this function once per column.

    Falls back through the hierarchy `method` -> 'linear' -> 'nearest' per-NaN if the
    requested method can't be fit (insufficient points, or an actual fit failure). See
    `_interpolate_missing_pointwise` for details.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing the data to interpolate
    to_interp : str
        Column to interpolate
    interp_on : str
        Column to interpolate on
    method : str, optional
        Interpolation method to use, by default "cubic"
    extrapolate : bool, optional
        Whether to extrapolate beyond the data range, by default False
    fill_value : tuple[float, float] | str | None, optional
        Value to use for filling gaps, by default None
    groupby_column : str | None, optional
        Column name to group by before interpolating, by default None
    progressbar : bool, optional
        Show progress bar for each group, by default True

    Returns
    -------
    pd.DataFrame
        Dataframe with interpolated column
    """
    cols = [interp_on, to_interp]
    if groupby_column:
        cols.append(groupby_column)
    assert all(c in data.columns for c in cols), (
        f"dataframe must contain columns {cols}"
    )

    return _apply_over_groups(
        data,
        groupby_column=groupby_column,
        progressbar=progressbar,
        interp_func=_interpolate_missing_pointwise,
        interp_func_kwargs={
            "to_interp": to_interp,
            "interp_on": interp_on,
            "method": method,
            "extrapolate": extrapolate,
            "fill_value": fill_value,
        },
    )


def interpolate_missing_pointwise_with_windows(
    data: pd.DataFrame,
    *,
    window_width: float,
    to_interp: str,
    interp_on: str,
    method: str = "cubic",
    extrapolate: bool = False,
    fill_value: tuple[float, float] | str | None = None,
    groupby_column: str | None = None,
    progressbar: bool = True,
) -> pd.DataFrame:
    """
    Interpolate NaN's in the "to_interp" column, based on values from "interp_on"
    using only values within a window around the NaN. If groupby_column is provided,
    the dataframe will first be grouped by this so only data from the group
    containing the NaN is used to interpolate.

    To interpolate multiple columns, call this function once per column.

    For each NaN, the requested `method` is tried first, expanding the window if
    there aren't enough points, before falling back through the hierarchy `method` ->
    'linear' -> 'nearest'. See `_interpolate_missing_pointwise_with_windows`
    for details.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing the data to interpolate
    window_width : float
        width of data window around NaN value to use in the interpolation, in units of
        the data provided in the column interp_on
    to_interp : str
        Column to interpolate
    interp_on : str
        Column to interpolate on
    method : str, optional
        Interpolation method to use, by default "cubic"
    extrapolate : bool, optional
        Whether to extrapolate beyond the data range, by default False
    fill_value : tuple[float, float] | str | None, optional
        Value to use for filling gaps, by default None
    groupby_column : str | None, optional
        Column name to group by before interpolating, by default None
    progressbar : bool, optional
        Show progress bar for each group, by default True

    Returns
    -------
    pd.DataFrame
        Dataframe with interpolated column
    """
    cols = [interp_on, to_interp]
    if groupby_column:
        cols.append(groupby_column)
    assert all(c in data.columns for c in cols), (
        f"dataframe must contain columns {cols}"
    )

    with airbornegeo.utils.DuplicateFilter(logger):
        return _apply_over_groups(
            data,
            groupby_column=groupby_column,
            progressbar=progressbar,
            interp_func=_interpolate_missing_pointwise_with_windows,
            interp_func_kwargs={
                "to_interp": to_interp,
                "interp_on": interp_on,
                "window_width": window_width,
                "method": method,
                "extrapolate": extrapolate,
                "fill_value": fill_value,
            },
        )


def _resolve_fill_value(
    fill_value: tuple[float, float] | str | None,
    y: np.ndarray,
    method: str,
    extrapolate: bool,
) -> tuple[float, float] | str | float:
    """
    Translate the user-facing `fill_value` option into the concrete value scipy's
    `interp1d` expects, given whether extrapolation is enabled and which method is
    being used.

    - Not extrapolating -> np.nan (out-of-range points stay NaN).
    - Extrapolating with `fill_value` unset -> 'extrapolate', except for
      'nearest', where scipy's own extrapolation is unreliable across versions, so
      we clamp to the edge values manually instead.
    - Extrapolating with 'edge' -> clamp to the first/last y values.
    - Extrapolating with 'mean' -> fill with the mean of y on both sides.
    - Anything else -> passed through as-is (e.g. an explicit (lo, hi) tuple).
    """
    if not extrapolate:
        return np.nan
    if method == "nearest" and fill_value in (None, "extrapolate"):
        return (y[0], y[-1])
    if fill_value is None:
        return "extrapolate"
    if fill_value == "edge":
        return (y[0], y[-1])
    if fill_value == "mean":
        m = np.nanmean(y)
        return (m, m)
    return fill_value


def _fit_method(
    x: np.ndarray,
    y: np.ndarray,
    method: str,
    *,
    extrapolate: bool,
    fill_value: tuple[float, float] | str | None,
) -> typing.Callable | None:
    """
    Fit a single scipy `interp1d` for `method` on deduped, sorted (x, y). Returns the
    fitted callable, or None if there aren't enough points, or the fit itself fails.
    """
    min_points = _METHOD_MIN_POINTS.get(method, 2)
    if len(x) < min_points:
        return None

    local_fill_value = _resolve_fill_value(fill_value, y, method, extrapolate)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The number of derivatives at boundaries does not match:",
            )
            f = scipy.interpolate.interp1d(
                x,
                y,
                kind=method,
                bounds_error=False,
                fill_value=local_fill_value,
                assume_sorted=True,
            )
    except Exception as e:  # noqa: BLE001 # pylint: disable=broad-exception-caught
        logger.debug("Method '%s' failed to fit (%s points): %s", method, len(x), e)
        return None

    return f


def _deduplicate_and_sort(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    De-duplicate x (keeping the first occurrence of each value) and sort by x, since
    spline-based methods (cubic, quadratic, slinear) require strictly increasing x.
    """
    x, unique_idx = np.unique(x, return_index=True)
    return x, y[unique_idx]


def _fit_interp1d_with_fallback(
    x: np.ndarray,
    y: np.ndarray,
    method: str,
    *,
    extrapolate: bool,
    fill_value: tuple[float, float] | str | None,
) -> tuple[typing.Callable | None, str]:
    """
    Try each method in the fallback chain (`method` -> 'linear' -> 'nearest'),
    building a scipy interp1d function on (x, y). A fit is only accepted if it also
    produces finite output on its own training data. Returns (callable,
    used_method), or (None, "none") if every method in the chain fails.
    """
    x, y = _deduplicate_and_sort(x, y)

    for candidate_method in _method_fallback_chain(method):
        f = _fit_method(
            x,
            y,
            candidate_method,
            extrapolate=extrapolate,
            fill_value=fill_value,
        )
        if f is None:
            continue

        if candidate_method != method:
            logger.debug("Falling back from '%s' to '%s'", method, candidate_method)
        return f, candidate_method

    return None, "none"


def interpolate_missing(
    data: pd.DataFrame,
    *,
    to_interp: str,
    interp_on: str,
    method: str = "cubic",
    extrapolate: bool = False,
    fill_value: tuple[float, float] | str | None = None,
    groupby_column: str | None = None,
    progressbar: bool = True,
) -> pd.DataFrame:
    """
    Interpolate NaN's in a dataframe's "to_interp" column, based on values from the
    "interp_on" column. Falls back through `method` -> 'linear' -> 'nearest' if the
    requested method can't be fit (insufficient points, or an actual fit failure),
    trying each in turn rather than deciding once from point count alone.

    To interpolate multiple columns, call this function once per column.

    method:
        'linear', 'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic',
        'previous', 'next'

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing the data to interpolate
    to_interp : str
        Column to interpolate
    interp_on : str
        Column to interpolate on
    method : str, optional
        Interpolation method to use, by default "cubic"
    extrapolate : bool, optional
        Whether to extrapolate beyond the data range, by default False
    fill_value : tuple[float, float] | str | None, optional
        Value to use for filling gaps, by default None
    groupby_column : str | None, optional
        Column name to group by before interpolating, by default None
    progressbar : bool, optional
        Show progress bar for each group, by default True

    Returns
    -------
    pd.DataFrame
        Dataframe with interpolated column
    """
    data = data.copy()

    cols = [to_interp, interp_on]
    if groupby_column is not None:
        cols.append(groupby_column)
    assert all(x in data.columns for x in cols), (
        f"dataframe must contain columns {cols} "
    )

    def _interp_segment(
        segment_data: pd.DataFrame, segment_name: typing.Any = None
    ) -> pd.DataFrame:
        segment_data = segment_data.copy()

        # drop NaN's
        segment_no_nans = segment_data.dropna(subset=[to_interp, interp_on], how="any")
        n_valid = len(segment_no_nans)

        # nothing to interpolate from at all: leave as-is (still NaN)
        if n_valid == 0:
            logger.warning(
                "No valid points available to interpolate '%s'%s; returning "
                "unchanged (still NaN)",
                to_interp,
                f" for group '{segment_name}'" if segment_name is not None else "",
            )
            return segment_data

        f, _used_method = _fit_interp1d_with_fallback(
            segment_no_nans[interp_on].to_numpy(),
            segment_no_nans[to_interp].to_numpy(),
            method,
            extrapolate=extrapolate,
            fill_value=fill_value,
        )

        if f is None:
            logger.warning(
                "All methods (%s) failed to fit '%s'%s; returning unchanged",
                _method_fallback_chain(method),
                to_interp,
                f" for group '{segment_name}'" if segment_name is not None else "",
            )
            return segment_data

        nan_mask = segment_data[to_interp].isna()
        values = f(segment_data.loc[nan_mask, interp_on])
        segment_data.loc[nan_mask, to_interp] = values

        return segment_data

    return _apply_over_groups(
        data,
        groupby_column=groupby_column,
        progressbar=progressbar,
        interp_func=_interp_segment,
        interp_func_kwargs={},
        pass_group_name=True,
    )


def _interpolate_missing_pointwise(
    data: pd.DataFrame,
    *,
    to_interp: str,
    interp_on: str,
    method: str = "cubic",
    extrapolate: bool = False,
    fill_value: tuple[float, float] | str | None = None,
) -> pd.DataFrame:
    """
    Interpolate NaN's in "to_interp" column, based on value(s) from "interp_on"
    column(s).

    For each NaN, tries each method in the fallback hierarchy `method` -> 'linear' ->
    'nearest' in turn (without extrapolation first, then with extrapolation if
    `extrapolate` is True and nothing succeeded), using whichever is the first
    method that actually produces a valid result -- not just whichever the point
    count alone suggests should work (e.g. cubic needs >= 4 points, linear needs >=
    2, nearest needs >= 1, but a method can still fail for other reasons, such as
    numerical instability during extrapolation).

    method:
        'linear', 'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic',
        'cubic', 'previous', 'next'
    """
    data = data.copy()

    col_list = [to_interp, interp_on]
    assert all(x in data.columns for x in col_list), (
        f"dataframe must contain columns {col_list} "
    )

    data = data.sort_values(interp_on)

    x = data[interp_on].to_numpy()
    y = data[to_interp].to_numpy()

    out = y.copy()
    interp_type = np.full(len(y), "none", dtype=object)

    valid_mask = ~np.isnan(y)
    n_valid = int(valid_mask.sum())

    # no valid points at all: nothing can be interpolated, return as-is
    if n_valid == 0:
        data[to_interp] = out
        data[f"{to_interp}_interpolation_type"] = interp_type
        return data

    xs_all = x[valid_mask]
    ys_all = y[valid_mask]

    method_chain = _method_fallback_chain(method)

    # iterate through NaNs
    for idx in np.where(np.isnan(y))[0]:
        xi = x[idx]

        value, used_method, used_type = _fill_one_nan(
            xs_all,
            ys_all,
            xi,
            method_chain=method_chain,
            fill_value=fill_value,
            extrapolate=extrapolate,
        )

        if used_method is not None and used_method != method:
            logger.debug(
                "'%s' failed for value at %s=%s; fell back to '%s' (%s)",
                method,
                interp_on,
                xi,
                used_method,
                used_type,
            )

        out[idx] = value
        interp_type[idx] = used_type

    data[to_interp] = out
    data[f"{to_interp}_interpolation_type"] = interp_type

    return data


def _windowed_attempt(
    x: np.ndarray,
    y: np.ndarray,
    xi: float,
    *,
    window_width: float,
    method: str,
    extrapolate: bool,
    fill_value: tuple[float, float] | str | None,
    max_window_doublings: int = 1,
) -> tuple[float, str]:
    """
    Attempt to fill a single NaN at `xi` using a window of `window_width` either
    side, doubling the window up to `max_window_doublings` times if there aren't
    enough points for `method`. Returns (value, used_type), where used_type is
    'interpolated', 'extrapolated', or 'none'.
    """
    min_points = _METHOD_MIN_POINTS.get(method, 2)

    win = window_width
    for _ in range(max_window_doublings + 1):
        llim, ulim = xi - win, xi + win
        left = np.searchsorted(x, llim, side="left")
        right = np.searchsorted(x, ulim, side="right")

        xs = x[left:right]
        ys = y[left:right]

        # remove other NaNs in window (only interpolate 1 at a time)
        m = ~np.isnan(ys)
        xs = xs[m]
        ys = ys[m]

        if len(xs) < min_points:
            win *= 2
            logger.debug(
                "Not enough points in window for '%s' (need >= %s, have %s); "
                "doubling window size to %s",
                method,
                min_points,
                len(xs),
                win,
            )
            continue

        try:
            value = interpolate_1d_single_nan(
                xs,
                ys,
                xi,
                method=method,
                extrapolate=extrapolate,
                fill_value=fill_value,
            )
            if np.isnan(value):
                msg = "filled value is NaN"
                raise ValueError(msg)  # noqa: TRY301
        except Exception:  # noqa: BLE001 # pylint: disable=broad-exception-caught
            win *= 2
            logger.debug(
                "Error during '%s' interpolation; doubling window size to %s",
                method,
                win,
            )
            continue

        return value, ("extrapolated" if extrapolate else "interpolated")

    return np.nan, "none"


def _fill_one_nan(
    x: np.ndarray,
    y: np.ndarray,
    xi: float,
    *,
    method_chain: list[str],
    fill_value: tuple[float, float] | str | None,
    extrapolate: bool,
    window_width: float | None = None,
    max_window_doublings: int = 1,
) -> tuple[float, str | None, str]:
    """
    Shared driver for filling a single NaN at `xi`, used by both
    `_interpolate_missing_pointwise` (window_width=None) and
    `_interpolate_missing_pointwise_with_windows` (window_width set).

    Tries each method in `method_chain` in turn without extrapolation; if none
    succeed and `extrapolate` is True, retries the whole chain with extrapolation
    allowed. When `window_width` is given, each attempt uses only points within that
    window of `xi`, expanding the window (up to `max_window_doublings` times) if
    there aren't enough points before moving to the next method in the chain.

    Returns (value, used_method, used_type), where used_type is 'interpolated',
    'extrapolated', or 'none', and used_method is None if every method failed.
    """
    stages = [(False, "interpolated")]
    if extrapolate:
        stages.append((True, "extrapolated"))

    for do_extrapolate, type_label in stages:
        for candidate_method in method_chain:
            if window_width is None:
                value = interpolate_1d_single_nan(
                    x,
                    y,
                    xi,
                    method=candidate_method,
                    extrapolate=do_extrapolate,
                    fill_value=fill_value,
                )
                succeeded = not np.isnan(value)
            else:
                value, kind = _windowed_attempt(
                    x,
                    y,
                    xi,
                    window_width=window_width,
                    method=candidate_method,
                    extrapolate=do_extrapolate,
                    fill_value=fill_value,
                    max_window_doublings=max_window_doublings,
                )
                succeeded = kind != "none"

            if succeeded:
                return value, candidate_method, type_label

    return np.nan, None, "none"


def _interpolate_missing_pointwise_with_windows(
    data: pd.DataFrame,
    *,
    window_width: float,
    to_interp: str,
    interp_on: str,
    method: str = "cubic",
    extrapolate: bool = False,
    fill_value: tuple[float, float] | str | None = None,
) -> pd.DataFrame:
    """
    Create a window of data either side of NaN's based on the interp_on column
    and interpolate the value. Useful when NaN's are sparse, or lines are long.

    For each NaN, the requested `method` is tried first, expanding the window (up to
    2x `window_width`) if there aren't enough points. Only once window expansion has
    been fully exhausted for `method` does it fall back to a simpler method in the
    hierarchy `method` -> 'linear' -> 'nearest', repeating the same window-expansion
    process for each fallback method in turn.
    """
    data = data.copy()

    col_list = [interp_on, to_interp]
    assert all(x in data.columns for x in col_list), (
        f"dataframe must contain columns {col_list} "
    )

    data = data.sort_values(interp_on)

    x = data[interp_on].to_numpy()
    y = data[to_interp].to_numpy()

    out = y.copy()
    interp_type = np.full(len(y), "none", dtype=object)

    method_chain = _method_fallback_chain(method)

    # iterate through NaNs
    for idx in np.where(np.isnan(y))[0]:
        xi = x[idx]

        value, used_method, used_type = _fill_one_nan(
            x,
            y,
            xi,
            method_chain=method_chain,
            fill_value=fill_value,
            extrapolate=extrapolate,
            window_width=window_width,
            max_window_doublings=1,
        )

        if used_type == "none":
            logger.debug(
                "All methods (%s) and window expansions failed for value at %s=%s; "
                "returning NaN",
                method_chain,
                interp_on,
                xi,
            )
        elif used_method != method:
            logger.debug(
                "'%s' failed for value at %s=%s even after window expansion; fell "
                "back to '%s' (%s)",
                method,
                interp_on,
                xi,
                used_method,
                used_type,
            )

        out[idx] = value
        interp_type[idx] = used_type

    data[to_interp] = out
    data[f"{to_interp}_interpolation_type"] = interp_type

    return data


def interpolate_1d_single_nan(
    x,
    y,
    x_index,
    method: str = "cubic",
    extrapolate: bool = False,
    fill_value: tuple[float, float] | str | None = None,
):
    """
    Fit an interpolator to (x, y) and evaluate it at `x_index`, returning a single
    scalar value (or NaN if the method can't be fit given the available points, or
    the fit fails for some other reason).
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # remove NaNs once (important)
    mask = ~np.isnan(y)
    x = x[mask]
    y = y[mask]

    # remove duplicate x-values, which break strictly-increasing requirements for
    # spline-based methods (cubic, quadratic, slinear) -- keep the first occurrence.
    # Without this, cubic spline construction can raise or silently produce a
    # singular/garbage fit even though the point count looks sufficient.
    x, y = _deduplicate_and_sort(x, y)

    f = _fit_method(
        x,
        y,
        method,
        extrapolate=extrapolate,
        fill_value=fill_value,
    )
    if f is None:
        logger.debug(
            "Only %s unique valid point(s) available for method '%s' (needs >= %s), "
            "or fit failed; returning NaN",
            len(x),
            method,
            _METHOD_MIN_POINTS.get(method, 2),
        )
        return np.nan

    try:
        return f(x_index).item()
    except Exception as e:  # noqa: BLE001 # pylint: disable=broad-exception-caught
        logger.debug(
            "Evaluating method '%s' at x_index=%s failed: %s; returning NaN",
            method,
            x_index,
            e,
        )
        return np.nan


def optimal_spline_damping(
    coordinates: tuple[pd.Series | np.ndarray, pd.Series | np.ndarray],
    data: pd.Series | np.ndarray,
    weights: pd.Series | np.ndarray | None = None,
    **kwargs: typing.Any,
) -> vd.Spline:
    """
    Find the best damping parameter for a verde.SplineCV() fit. All kwargs are passed to
    the verde.SplineCV class.

    Parameters
    ----------
    coordinates : tuple[pandas.Series  |  numpy.ndarray, pandas.Series  |  \
            numpy.ndarray]
        easting and northing coordinates of the data
    data : pandas.Series | numpy.ndarray
        data for fitting the spline to
    weights : pandas.Series | numpy.ndarray | None, optional
        if not None, then the weights assigned to each data point. Typically, this
        should be 1 over the data uncertainty squared, by default None

    Keyword Arguments
    -----------------
    dampings : float | None
        The positive damping regularization parameter. Controls how much smoothness is
        imposed on the estimated forces. If None, no regularization is used, by default
        None
    force_coords : bool
        The easting and northing coordinates of the point forces. If None (default),
        then will be set to the data coordinates.
    cv : None | cross-validation generator
        Any scikit-learn cross-validation generator. If not given, will use the
        default set by :func:`verde.cross_val_score`.
    delayed : bool
        If True, will use :func:`dask.delayed.delayed` to dispatch computations and
        allow :mod:`dask` to execute the grid search in parallel (see note
        above).
    scoring : None | str | Callable
        The scoring function (or name of a function) used for cross-validation.
        Must be known to scikit-learn. See the description of *scoring* in
        :func:`sklearn.model_selection.cross_val_score` for details. If None,
        will fall back to the :meth:`verde.Spline.score` method.

    Returns
    -------
    verde.Spline
        the spline which best fits the data
    """
    kwargs = copy.deepcopy(kwargs)

    dampings = kwargs.pop("dampings", None)

    # if single damping value provided, convert to list
    if isinstance(dampings, typing.Iterable):
        pass
    else:
        dampings = [dampings]

    n_splits = 5
    while n_splits > 0:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r".*mindist parameter of verde\.Spline.*",
                    category=FutureWarning,
                )
                spline = vd.SplineCV(
                    dampings=dampings,
                    cv=sklearn.model_selection.KFold(
                        n_splits=n_splits,
                        shuffle=True,
                        random_state=0,
                    ),
                    scoring="neg_root_mean_squared_error",
                    **kwargs,
                )
                spline.fit(
                    coordinates,
                    data,
                    weights=weights,
                )
            break
        except ValueError as e:
            logger.error(e)
            msg = "decreasing number of splits by 1 until ValueError is resolved"
            logger.warning(msg)
        if n_splits == 1:
            msg = "ValueError not resolved, fitting spline with no damping"
            logger.warning(msg)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r".*mindist parameter of verde\.Spline.*",
                    category=FutureWarning,
                )
                spline = vd.Spline(
                    damping=None,
                    scoring="neg_root_mean_squared_error",
                    **kwargs,
                )
                spline.fit(
                    coordinates,
                    data,
                    weights=weights,
                )
        n_splits -= 1

    # if len(dampings) > 1:
    # try:
    # logger.info("Best SplineCV score: %s", spline.scores_.max())
    # except AttributeError:
    # logger.info("Best SplineCV score: %s", max(dask.compute(spline.scores_)[0]))

    # logger.info("Best damping: %s", spline.damping_)

    dampings_without_none = [i for i in dampings if i is not None]

    try:
        if spline.damping_ is None:
            pass
        elif len(dampings) > 2 and spline.damping_ in [
            np.min(dampings_without_none),
            np.max(dampings_without_none),
        ]:
            logger.warning(
                "Best damping value (%s) is at the limit of provided values (%s, %s) "
                "and thus is likely not a global minimum, expand the range of values "
                "test to ensure the best parameter value value is found.",
                spline.damping_,
                np.nanmin(dampings_without_none),
                np.nanmax(dampings_without_none),
            )
    except AttributeError:
        pass

    return spline
