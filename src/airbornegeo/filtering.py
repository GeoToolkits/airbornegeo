import itertools
import typing
import warnings

import harmonica as hm
import numpy as np
import pandas as pd
import verde as vd
import xarray as xr
import xrft
from numpy.typing import NDArray
from scipy import fft, ndimage

from airbornegeo.utils import _iter_groups

try:
    import pygmt

    _HAS_PYGMT = True
except ImportError:
    _HAS_PYGMT = False

try:
    import rioxarray  # noqa: F401 # pylint: disable=unused-import

    _HAS_RIOXARRAY = True
except ImportError:
    _HAS_RIOXARRAY = False


def pad1d(
    data: pd.DataFrame,
    *,
    data_column: str,
    independent_column: str,
    width_percentage: float,
    mode: str = "reflect",
    **kwargs: typing.Any,
) -> pd.DataFrame:
    """
    Pad a dataframe in the front and back, which reduces edge effects for 1D filtering.
    The pad width is given by a percentage of the range of values in column given by
    independent_column. For this column, the pad values are extrapolation of the values.
    For example, if independent_column is along track distance in meters, from 0 to 100,
    and width_percentage is 10, than a 10 m pad would be added to the beginning and
    end of the data, with the same spacing as the median spacing of the along track
    distances. The pad values for the data column are chosen based on the supplied mode.

    Parameters
    ----------
    data : pd.DataFrame
        _description_
    data_column : str
        _description_
    independent_column : str
        _description_
    width_percentage : float
        The width of the pad to add before and after the data in percentage of the
        range of values provided by independent_column, by default 10.
    mode : str, optional
        The mode to use for padding, by default is "reflect".
    kwargs : Any, optional
        Keyword arguments to pass directly to np.pad, such as stat_length and
        constant_values.

    Returns
    -------
    pd.DataFrame
        A dataframe with a column 'true_index' which can be used to reset the dataframe
        to the index it had before padding, and the padded data and independent variable
        columns.
    """
    data = data.copy()

    data = data[[independent_column, data_column]]

    # get data spacing
    filter_by = data[independent_column].to_numpy()

    spacing = np.median(np.diff(filter_by))

    # pad as percentage of filter_by range
    pad_dist = (filter_by.max() - filter_by.min()) * (width_percentage / 100)
    pad_dist = round(pad_dist / spacing) * spacing

    # get the number of points to pad
    n_pad = int(pad_dist / spacing)

    # add pad points to filter_by values
    lower_pad = np.linspace(
        filter_by.min() - pad_dist,
        filter_by.min() - spacing,
        n_pad,
    )
    upper_pad = np.linspace(
        filter_by.max(),
        filter_by.max() + pad_dist,
        n_pad,
    )

    vals = np.concatenate((lower_pad, upper_pad))
    new_dist = pd.DataFrame({independent_column: vals})

    # pad the line in the front and back
    padded = (
        pd.concat(
            [data.reset_index(), new_dist],
        )
        .sort_values(by=independent_column)
        .set_index("index")
    ).reset_index()
    padded = padded.rename(columns={"index": "true_index"})

    # get unpadded data
    unpadded_data = data[data_column].to_numpy()

    # pad this with numpy
    padded_data = np.pad(
        unpadded_data,
        pad_width=n_pad,
        mode=mode,
        **kwargs,
    )

    # add padded data to padded dataframe
    padded[data_column] = padded_data

    return padded


_GMT_FILTER_SHAPE_CODES = {
    "gaussian": "g",
    "boxcar": "b",
    "cosine": "c",
    "median": "m",
}


def _build_gmt_filter_type(
    filter_shape: str, filter_width: float, filter_type: str, robust: bool
) -> str:
    """
    Build a GMT filter1d filter_type string (e.g. "g1000+h") from filter_shape,
    filter_width, filter_type, and robust. GMT's convention: low-pass needs no suffix,
    high-pass appends "+h", and a robust filter uses an uppercase filter code.
    """
    code = _GMT_FILTER_SHAPE_CODES[filter_shape]
    if robust:
        code = code.upper()
    suffix = "+h" if filter_type == "highpass" else ""
    return f"{code}{filter_width}{suffix}"


def filter_line(
    data: pd.DataFrame,
    *,
    filter_width: float,
    data_column: str,
    filter_by_column: str,
    filter_type: str = "lowpass",
    filter_shape: str = "gaussian",
    engine: str = "scipy",
    groupby_column: str | None = None,
    progressbar: bool = True,
    pad_width_percentage: float = 10,
    pad_mode: str = "reflect",
    max_gap: float | None = None,
    robust: bool = False,
    robust_threshold: float = 2.5,
    **kwargs: typing.Any,
) -> pd.Series:
    """
    Apply a 1D filter to a column of a pandas DataFrame along values of another column.
    The filter_by_column would typically be either distance along track for a spatial
    filter, or a time column, for a temporal filter. The dataframe can be grouped by the
    groupby_column before applying the filter. This column could contain flight names,
    or lines names. filter_width is the filter width in the same units as
    filter_by_column (e.g. if filter_by_column is "distance_along_line" in meters, a
    filter_width of 1000 gives a 1000 m filter; if filter_by_column is a time column in
    seconds, filter_width is in seconds). Ends of lines are automatically padded to
    avoid edge effects.

    Two engines are available, selected with the engine parameter:

    - "scipy" (the default): applies the filter directly in Python via scipy, with no
      external dependencies. It additionally splits each line at gaps in
      filter_by_column larger than max_gap and filters the segments separately, so
      anomalies are not smeared across genuine breaks in the data (see max_gap). This
      engine is usually substantially faster than "gmt", especially when filtering
      many lines/segments individually with groupby_column, since GMT's filter1d has
      a large fixed per-call overhead.
    - "gmt": applies the filter with GMT's filter1d, via the optional pygmt package
      (raises an ImportError with installation instructions if pygmt is not
      installed). This is a mature, independently-validated reference implementation;
      use it if you need results that exactly reproduce GMT's own processing.

    Both engines aim to give equivalent (though not always numerically identical)
    results for the same filter_shape/filter_width/filter_type/robust combination.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing the data points to filter.
    filter_width : float
        The filter width, in the same units as filter_by_column. For "gaussian", this
        is the full width (6 standard deviations); for the others, the full window
        width.
    data_column : str
        The data to filter.
    filter_by_column : str, optional
        The independent variable to filter against, typically either a time or distance
        along track values.
    filter_type : str, optional
        Either "lowpass" or "highpass", by default "lowpass".
    filter_shape : str, optional
        One of "gaussian", "boxcar", "cosine", or "median", by default "gaussian". See
        the "scipy" engine's implementation for what each of these represents; "gmt"
        reproduces the equivalent pygmt.filter1d filter type ("g", "b", "c", "m"
        respectively).
    engine : str, optional
        Either "scipy" or "gmt", by default "scipy".
    groupby_column : str | None, optional
        Column name to group by before filtering, by default None.
    progressbar : bool, optional
        Show progress bar for each group, by default True
    pad_width_percentage : float, optional
        The width of the pad to add before and after the data in percentage of the
        range of values provided by filter_by_column, by default 10.
    pad_mode : str, optional
        The mode to use for padding, by default is "reflect".
    max_gap : float | None, optional
        Only used by engine "scipy": split lines where consecutive filter_by_column
        values differ by more than this (in the units of filter_by_column) and filter
        each segment separately. By default (None) gaps larger than 10 times the median
        spacing are split; use numpy.inf to disable splitting.
    robust : bool, optional
        Use a robust variant of the filter that resists outliers, by default False.
        For engine "scipy", this despikes the data before filtering (replacing samples
        far from the local median with that median) and is not supported for
        filter_shape "median", which is already robust by construction. For engine
        "gmt", this uses GMT's own uppercase (robust) filter codes.
    robust_threshold : float, optional
        Only used by engine "scipy" when robust is True: samples deviating from the
        local median by more than robust_threshold times the local (MAD-based) robust
        standard deviation are replaced by that median before filtering, by default
        2.5 (matching GMT's default for its own robust filters).
    kwargs : Any, optional
        Keyword arguments to pass to np.pad, such as stat_length and
        constant_values.

    Returns
    -------
    pd.Series
        The filtered data values
    """
    if filter_type not in ("lowpass", "highpass"):
        msg = f"filter_type must be 'lowpass' or 'highpass', got '{filter_type}'"
        raise ValueError(msg)
    if filter_shape not in ("gaussian", "boxcar", "cosine", "median"):
        msg = (
            "filter_shape must be 'gaussian', 'boxcar', 'cosine', or 'median', got "
            f"'{filter_shape}'"
        )
        raise ValueError(msg)
    if engine not in ("scipy", "gmt"):
        msg = f"engine must be 'scipy' or 'gmt', got '{engine}'"
        raise ValueError(msg)
    if engine == "scipy" and robust and filter_shape == "median":
        msg = (
            "robust=True has no effect for filter_shape='median', which is already "
            "robust by construction; use robust=False"
        )
        raise ValueError(msg)

    data = data.copy()

    if engine == "scipy":
        if groupby_column is None:
            filtered_values = _filter_values_scipy(
                data[data_column].to_numpy(),
                data[filter_by_column].to_numpy(),
                filter_width=filter_width,
                filter_type=filter_type,
                filter_shape=filter_shape,
                pad_width_percentage=pad_width_percentage,
                pad_mode=pad_mode,
                max_gap=max_gap,
                robust=robust,
                robust_threshold=robust_threshold,
                **kwargs,
            )
            return pd.Series(filtered_values, index=data.index, name=data_column)

        for segment_name, segment_data in _iter_groups(
            data, groupby_column, progressbar
        ):
            data.loc[data[groupby_column] == segment_name, data_column] = (
                _filter_values_scipy(
                    segment_data[data_column].to_numpy(),
                    segment_data[filter_by_column].to_numpy(),
                    filter_width=filter_width,
                    filter_type=filter_type,
                    filter_shape=filter_shape,
                    pad_width_percentage=pad_width_percentage,
                    pad_mode=pad_mode,
                    max_gap=max_gap,
                    robust=robust,
                    robust_threshold=robust_threshold,
                    **kwargs,
                )
            )
        return data[data_column]

    # engine == "gmt"
    if not _HAS_PYGMT:
        msg = (
            "engine='gmt' requires the optional dependency 'pygmt'. Install it with "
            "`pip install pygmt` or `mamba install pygmt`, or use engine='scipy' "
            "instead (the default)."
        )
        raise ImportError(msg)

    gmt_filter_type = _build_gmt_filter_type(
        filter_shape, filter_width, filter_type, robust
    )

    if groupby_column is None:
        # pad the data with pad_mode, and the filter_by_column by extrapolation
        padded = pad1d(
            data,
            data_column=data_column,
            independent_column=filter_by_column,
            width_percentage=pad_width_percentage,
            mode=pad_mode,
            **kwargs,
        )

        # filter the padded data
        filtered = pygmt.filter1d(
            padded[[filter_by_column, data_column]],
            end=True,
            time_col=0,
            filter_type=gmt_filter_type,
        )

        filtered = filtered.rename(columns={0: filter_by_column, 1: data_column})

        # un-pad the data
        filtered["original_index"] = padded.true_index
        filtered = filtered.set_index("original_index")
        filtered = filtered[filtered.index.isin(data.index)]

        return filtered[data_column]

    for segment_name, segment_data in _iter_groups(data, groupby_column, progressbar):
        # pad the data with pad_mode, and the filter_by_column by extrapolation
        padded = pad1d(
            segment_data,
            data_column=data_column,
            independent_column=filter_by_column,
            width_percentage=pad_width_percentage,
            mode=pad_mode,
            **kwargs,
        )

        # filter the padded data
        # this resets the index, and drops and rows with NaNs
        filtered = pygmt.filter1d(
            padded[[filter_by_column, data_column]],
            end=True,
            time_col=0,
            filter_type=gmt_filter_type,
        )
        filtered.columns = [filter_by_column, data_column]

        # un-pad the data
        filtered["original_index"] = padded.true_index
        filtered = filtered.set_index("original_index")
        filtered = filtered[filtered.index.isin(segment_data.index)]

        # replace original data with filtered data
        data.loc[data[groupby_column] == segment_name, data_column] = filtered[
            data_column
        ]

    return data[data_column]


def _despike(values: NDArray, window: int, threshold: float) -> NDArray:
    """
    Replace samples more than threshold robust standard deviations (estimated from the
    median absolute deviation, MAD) from the local median with that median. This is
    the same outlier rule GMT's robust filters use, applied as a despiking
    pre-processing step rather than per-window during the (linear) FFT filter itself.
    """
    window = min(window, len(values))
    if window % 2 == 0:
        window += 1
    if window < 3:
        return values

    local_median = ndimage.median_filter(values, size=window, mode="reflect")
    deviation = np.abs(values - local_median)
    mad = ndimage.median_filter(deviation, size=window, mode="reflect")
    outliers = deviation > threshold * 1.4826 * mad

    despiked = values.copy()
    despiked[outliers] = local_median[outliers]
    return despiked


def _fft_filter_segment(
    values: NDArray,
    coords: NDArray,
    filter_width: float,
    pad_width_percentage: float,
    pad_mode: str,
    robust: bool,
    robust_threshold: float,
    filter_shape: str,
    **kwargs: typing.Any,
) -> NDArray:
    """
    Low-pass filter of a single gap-free segment. Returns the low-pass filtered values
    at the original (possibly irregular) coords.
    """
    # too few points to estimate anything below the segment scale; the best available
    # low-pass is the data itself
    if len(values) < 4:
        return values.copy()

    # resample onto a regular grid at the median spacing, since the FFT (and the
    # median window filter) require evenly spaced data
    spacing = np.median(np.diff(coords))
    if spacing <= 0:
        msg = "coords must be strictly increasing within a segment"
        raise ValueError(msg)
    n_grid = round(float((coords[-1] - coords[0]) / spacing)) + 1
    grid = np.linspace(coords[0], coords[-1], n_grid)
    grid_values = np.interp(grid, coords, values)

    # despike before filtering: replace outliers with the local median so they can't
    # smear into the smooth result, using the same window as the filter width
    if robust:
        window = round(filter_width / spacing)
        grid_values = _despike(grid_values, window=window, threshold=robust_threshold)

    # pad to reduce edge effects
    n_pad = round(n_grid * pad_width_percentage / 100)
    if pad_mode in ("reflect", "symmetric", "wrap"):
        n_pad = min(n_pad, n_grid - 1)
    if n_pad > 0:
        grid_values = np.pad(grid_values, n_pad, mode=pad_mode, **kwargs)

    if filter_shape == "median":
        # median is a non-linear filter and can't be expressed as a spectral transfer
        # function, so apply it directly as a sliding window instead
        window = max(3, round(filter_width / spacing))
        if window % 2 == 0:
            window += 1
        window = min(window, len(grid_values))
        low = ndimage.median_filter(grid_values, size=window, mode="reflect")
    else:
        # multiply the spectrum by each shape's (exact, analytic) transfer function,
        # using GMT's convention that the filter width is the full convolution width
        # (6 standard deviations for the gaussian)
        freqs = fft.rfftfreq(len(grid_values), d=spacing)
        width_freq = filter_width * freqs
        if filter_shape == "gaussian":
            sigma = filter_width / 6
            transfer = np.exp(-2 * np.pi**2 * sigma**2 * freqs**2)
        elif filter_shape == "boxcar":
            # the Fourier transform of a normalized boxcar of width w is sinc(w f)
            transfer = np.sinc(width_freq)
        else:  # cosine
            # the Fourier transform of a normalized raised-cosine (Hann) window
            transfer = (
                np.sinc(width_freq)
                + 0.5 * np.sinc(width_freq - 1)
                + 0.5 * np.sinc(width_freq + 1)
            )
        low = fft.irfft(fft.rfft(grid_values) * transfer, len(grid_values))

    # un-pad and sample the smooth curve back at the original coords
    if n_pad > 0:
        low = low[n_pad:-n_pad]
    return np.interp(coords, grid, low)


def _filter_values_scipy(
    values: NDArray,
    coords: NDArray,
    *,
    filter_width: float,
    filter_type: str,
    filter_shape: str,
    pad_width_percentage: float,
    pad_mode: str,
    max_gap: float | None,
    robust: bool,
    robust_threshold: float,
    **kwargs: typing.Any,
) -> NDArray:
    """
    Low-pass or high-pass filter 1D data in the frequency domain with scipy.fft (or,
    for filter_shape "median", a direct sliding-window filter). The data does not need
    to be evenly sampled; it is resampled onto a regular grid at the median spacing of
    coords, padded to reduce edge effects, filtered, and interpolated back onto the
    original coords. If coords contains gaps larger than max_gap, the data is split at
    the gaps and each segment is filtered separately, so anomalies are not smeared
    across the gaps. The high-pass filtered data is the data minus the low-pass
    filtered data. This is filter_line's engine="scipy" implementation; validation of
    filter_type/filter_shape/robust is expected to already have happened in the
    caller.
    """
    values = np.asarray(values, dtype=float)
    coords = np.asarray(coords, dtype=float)

    if values.shape != coords.shape:
        msg = "values and coords must have the same length"
        raise ValueError(msg)

    steps = np.diff(coords)
    if np.any(steps < 0):
        msg = "filter_by_column must be sorted in ascending order to use engine='scipy'"
        raise ValueError(msg)

    # find gaps larger than max_gap and filter each segment separately
    if len(values) > 1:
        if max_gap is None:
            max_gap = 10 * np.median(steps)
        segment_starts = np.flatnonzero(steps > max_gap) + 1
    else:
        segment_starts = np.array([], dtype=int)
    bounds = [0, *segment_starts.tolist(), len(values)]

    low = np.empty_like(values)
    for start, stop in itertools.pairwise(bounds):
        low[start:stop] = _fft_filter_segment(
            values[start:stop],
            coords[start:stop],
            filter_width,
            pad_width_percentage,
            pad_mode,
            robust,
            robust_threshold,
            filter_shape,
            **kwargs,
        )

    if filter_type == "highpass":
        return values - low
    return low


def _nearest_grid_fill(
    grid: xr.DataArray,
    method: str = "verde",
    crs: str | None = None,
) -> xr.DataArray:
    """
    fill missing values in a grid with the nearest value.

    Parameters
    ----------
    grid : xarray.DataArray
        grid with missing values
    method : str, optional
        choose method of filling, by default "verde"
    crs : str | None, optional
        if method is 'rioxarray', provide the crs of the grid, in format 'epsg:xxxx',
        by default None
    Returns
    -------
    xarray.DataArray
        filled grid
    """

    # TODO: also check out rasterio fillnodata() https://rasterio.readthedocs.io/en/latest/api/rasterio.fill.html#rasterio.fill.fillnodata
    # uses https://gdal.org/en/stable/api/gdal_alg.html#_CPPv414GDALFillNodata15GDALRasterBandH15GDALRasterBandHdiiPPc16GDALProgressFuncPv
    # can fill with nearest neighbor or inverse distance weighting

    # get coordinate names
    original_dims = list(grid.sizes.keys())

    # get original grid name
    original_name = grid.name

    if method == "rioxarray":
        if not _HAS_RIOXARRAY:
            msg = (
                "The 'rioxarray' method requires the optional dependency 'rioxarray'. "
                "Install it with `pip install rioxarray` or `mamba install rioxarray`, "
                "or use method='verde' instead."
            )
            raise ImportError(msg)
        filled: xr.DataArray = (
            grid.rio.write_crs(crs)
            .rio.set_spatial_dims(original_dims[1], original_dims[0])
            .rio.write_nodata(np.nan)
            .rio.interpolate_na(method="nearest")
            .rename(original_name)
        )
    elif method == "verde":
        df = vd.grid_to_table(grid)
        df_dropped = df[df[grid.name].notna()]
        coords = (df_dropped[grid.dims[1]], df_dropped[grid.dims[0]])
        region = vd.get_region((df[grid.dims[1]], df[grid.dims[0]]))
        filled = (
            vd.KNeighbors()
            .fit(coords, df_dropped[grid.name])
            .grid(
                region=region,
                shape=grid.shape,
                data_names=original_name,
                dims=(original_dims[1], original_dims[0]),
            )[original_name]
        )
    else:
        msg = "method must be 'rioxarray', or 'verde'"
        raise ValueError(msg)

    # reset coordinate names if changed
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="rename '")
        return filled.rename(
            {
                next(iter(filled.dims)): original_dims[0],
                list(filled.dims)[1]: original_dims[1],
            }
        )


def filter_grid(
    grid: xr.DataArray,
    filter_width: float | None = None,
    height_displacement: float | None = None,
    filter_type: str = "lowpass",
    pad_width_factor: int = 3,
    pad_mode: str = "linear_ramp",
    pad_constant: float | None = None,
    pad_end_values: float | None = None,
) -> xr.DataArray:
    """
    Apply a spatial filter to a grid.

    Parameters
    ----------
    grid : xarray.DataArray
        grid to filter the values of
    filter_width : float, optional
        width of the filter in meters, by default None
    height_displacement : float, optional
        height displacement for upward continuation, relative to observation height, by
        default None
    filter_type : str, optional
        type of filter to use from 'lowpass', 'highpass' 'up_deriv', 'easting_deriv',
        'northing_deriv', 'up_continue', 'horizontal_gradient' or 'total_gradient', by
        default "lowpass"
    pad_width_factor : int, optional
        factor of grid width to pad the grid by, by default 3, which equates to a pad
        with a width of 1/3 of the grid width.
    pad_mode : str, optional
        mode of padding, can be "linear", by default "linear_ramp"
    pad_constant : float | None, optional
        constant value to use for padding, by default None
    pad_end_values : float | None, optional
        value to use for end of padding if pad_mode is "linear_ramp", by default None

    Returns
    -------
    xarray.DataArray
        a filtered grid
    """
    # get coordinate names
    original_dims = list(grid.sizes.keys())

    # get original grid name
    original_name = grid.name

    # if there are nan's, fill them with nearest neighbor
    if grid.isnull().any():  # noqa: PD003
        filled = _nearest_grid_fill(grid, method="verde")
    else:
        filled = grid.copy()

    # reset coordinate names if changed
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="rename '")
        filled = filled.rename(
            {
                next(iter(filled.dims)): original_dims[0],
                list(filled.dims)[1]: original_dims[1],
            }
        )

    # define width of padding in each direction
    pad_width = {
        original_dims[1]: grid[original_dims[1]].size // pad_width_factor,
        original_dims[0]: grid[original_dims[0]].size // pad_width_factor,
    }

    if pad_mode == "constant":
        if pad_constant is None:
            pad_constant = filled.median()
        pad_end_values = None

    if (pad_mode == "linear_ramp") and (pad_end_values is None):
        pad_end_values = filled.median()

    if pad_mode != "constant":
        pad_constant = (
            None  # needed until https://github.com/xgcm/xrft/issues/211 is fixed
        )

    # apply padding
    pad_kwargs = {
        **pad_width,
        "mode": pad_mode,
        "constant_values": pad_constant,
        "end_values": pad_end_values,
    }

    padded = xrft.pad(
        filled,
        **pad_kwargs,
    )

    if filter_type == "lowpass":
        if filter_width is None:
            msg = "filter_width must be provided if filter_type is 'lowpass'"
            raise ValueError(msg)
        filt = hm.gaussian_lowpass(
            padded,
            wavelength=filter_width,
        ).rename("filt")
    elif filter_type == "highpass":
        if filter_width is None:
            msg = "filter_width must be provided if filter_type is 'highpass'"
            raise ValueError(msg)
        filt = hm.gaussian_highpass(
            padded,
            wavelength=filter_width,
        ).rename("filt")
    elif filter_type == "up_deriv":
        filt = hm.derivative_upward(padded).rename("filt")
    elif filter_type == "easting_deriv":
        filt = hm.derivative_easting(padded).rename("filt")
    elif filter_type == "northing_deriv":
        filt = hm.derivative_northing(padded).rename("filt")
    elif filter_type == "up_continue":
        if height_displacement is None:
            msg = "height_displacement must be provided if filter_type is 'up_continue'"
            raise ValueError(msg)
        filt = hm.upward_continuation(
            padded, height_displacement=height_displacement
        ).rename("filt")
    elif filter_type == "total_gradient":
        filt = hm.total_gradient_amplitude(padded).rename("filt")
    elif filter_type == "horizontal_gradient":
        east_deriv = hm.derivative_easting(padded).rename("filt")
        north_deriv = hm.derivative_northing(padded).rename("filt")
        filt = np.sqrt(east_deriv**2 + north_deriv**2)
    else:
        msg = (
            "filter_type must be 'lowpass', 'highpass' 'up_deriv', 'easting_deriv', "
            "'northing_deriv', 'up_continue', or 'total_gradient'"
        )
        raise ValueError(msg)

    unpadded = xrft.unpad(filt, pad_width)

    # reset coordinate values to original (avoid rounding errors)
    unpadded = unpadded.assign_coords(
        {
            original_dims[0]: grid[original_dims[0]].to_numpy(),
            original_dims[1]: grid[original_dims[1]].to_numpy(),
        }
    )

    if grid.isnull().any():  # noqa: PD003
        result: xr.DataArray = xr.where(grid.notnull(), unpadded, grid)  # noqa: PD004
    else:
        result = unpadded.copy()

    # reset coordinate names if changed
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="rename '")
        result = result.rename(
            {
                next(iter(result.dims)): original_dims[0],
                # list(result.dims)[0]: original_dims[0],
                list(result.dims)[1]: original_dims[1],
            }
        )

    return result.rename(original_name)
