import typing
import warnings

import harmonica as hm
import numpy as np
import pandas as pd
import pygmt
import verde as vd
import xarray as xr
import xrft
from tqdm.autonotebook import tqdm


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


def filter_line(
    data: pd.DataFrame,
    *,
    filter_type: str,
    data_column: str,
    filter_by_column: str,
    groupby_column: str | None = None,
    pad_width_percentage: float = 10,
    pad_mode: str = "reflect",
    **kwargs: typing.Any,
) -> pd.DataFrame:
    """
    Apply a 1D filter to a column of a pandas DataFrame along values of another column.
    The filter_by_column would typically be either distance along track for a spatial
    filter, or a time column, for a temporal filter. The dataframe can be grouped by the
    groupby_column before applying the filter. This column could contain flight names,
    or lines names. The type of filter is supplied by filter_type, which uses the GMT
    string format for filters. For example, if filter_by_column is "distance_along_line"
    in meters, and filter_type is "g1000+l", then this would give a 1000 m low pass
    gaussian filter. If filter_by_column is "time" in seconds, then the filter width
    would be 1000 seconds. Ends of lines are automatically padded to avoid edge effects.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing the data points to filter.
    filter_type : str
        A string with format "<type><width>+h" where type is GMT filter type, width is
        the filter width in same units as filter_by_column, and optional +h switches
        from low-pass to high-pass filter; e.g. "g10+h" is a 10m high-pass Gaussian
        filter.
    data_column : str
        The data to filter.
    filter_by_column : str, optional
        The independent variable to filter against, typically either a time or distance
        along track values.
    groupby_column : str | None, optional
        Column name to group by before filtering, by default None.
    pad_width_percentage : float, optional
        The width of the pad to add before and after the data in percentage of the
        range of values provided by filter_by_column, by default 10.
    pad_mode : str, optional
        The mode to use for padding, by default is "reflect".
    kwargs : Any, optional
        Keyword arguments to pass to np.pad, such as stat_length and
        constant_values.

    Returns
    -------
    pd.Series
        The filtered data values
    """
    data = data.copy()

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
            filter_type=filter_type,
        )

        filtered = filtered.rename(columns={0: filter_by_column, 1: data_column})

        # un-pad the data
        filtered["original_index"] = padded.true_index
        filtered = filtered.set_index("original_index")
        filtered = filtered[filtered.index.isin(data.index)]

        return filtered[data_column]

    for segment_name, segment_data in tqdm(
        data.groupby(groupby_column), desc="Segments"
    ):
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
        filtered = pygmt.filter1d(
            padded[[filter_by_column, data_column]],
            end=True,
            time_col=0,
            filter_type=filter_type,
        ).rename(columns={0: filter_by_column, 1: data_column})

        # un-pad the data
        filtered["original_index"] = padded.true_index
        filtered = filtered.set_index("original_index")
        filtered = filtered[filtered.index.isin(segment_data.index)]

        # replace original data with filtered data
        data.loc[data[groupby_column] == segment_name, data_column] = filtered[
            data_column
        ]

    return data[data_column]


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
    # elif method == "pygmt":
    #     filled = pygmt.grdfill(grid, mode="n", verbose="q").rename(original_name)
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
        'northing_deriv', 'up_continue', or 'total_gradient', by default "lowpass"
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
        filt = hm.gaussian_lowpass(padded, wavelength=filter_width).rename("filt")
    elif filter_type == "highpass":
        if filter_width is None:
            msg = "filter_width must be provided if filter_type is 'highpass'"
            raise ValueError(msg)
        filt = hm.gaussian_highpass(padded, wavelength=filter_width).rename("filt")
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
