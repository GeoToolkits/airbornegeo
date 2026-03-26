import copy
import typing
import warnings

import harmonica as hm
import numpy as np
import pandas as pd
import pygmt
import verde as vd
import xarray as xr
import xrft
from numpy.typing import NDArray

from airbornegeo import logger


class DuplicateFilter:
    """
    Filters away duplicate log messages.
    Adapted from https://stackoverflow.com/a/60462619/18686384
    """

    def __init__(self, log):  # type: ignore[no-untyped-def]
        self.msgs = set()
        self.log = log

    def filter(self, record):  # type: ignore[no-untyped-def] # pylint: disable=missing-function-docstring
        msg = str(record.msg)
        is_duplicate = msg in self.msgs
        if not is_duplicate:
            self.msgs.add(msg)
        return not is_duplicate

    def __enter__(self):  # type: ignore[no-untyped-def]
        self.log.addFilter(self)

    def __exit__(self, exc_type, exc_val, exc_tb):  # type: ignore[no-untyped-def]
        self.log.removeFilter(self)


def normalize_values(
    x: NDArray,
    low: float = 0,
    high: float = 1,
    quantiles: tuple[float, float] = (0, 1),
) -> NDArray:
    """
    Normalize a list of numbers by scaling the min and max values to  be between low and
    high. The min and max values can changed to be any quatile of the data.

    Parameters
    ----------
    x : NDArray
        numbers to normalize
    low : float, optional
        lower value for normalization, by default 0
    high : float, optional
        higher value for normalization, by default 1
    quantiles : tuple[float, float], optional
        quantiles to use for min and max values, by default (0, 1)

    Returns
    -------
    NDArray
        a normalized list of numbers
    """
    x = copy.deepcopy(x)
    x = np.array(x)

    floor = np.nanquantile(x, quantiles[0])
    ciel = np.nanquantile(x, quantiles[1])

    x = np.where(x < floor, floor, x)
    x = np.where(x > ciel, ciel, x)

    min_val = np.min(x)
    max_val = np.max(x)

    if min_val == max_val:
        logger.warning("min and max values are equal, returning list of low values")
        return np.full_like(x, low)

    norm = (x - min_val) / (max_val - min_val)

    return norm * (high - low) + low


def rmse(data: typing.Any, as_median: bool = False) -> float:
    """
    function to give the root mean/median squared error (RMSE) of data

    Parameters
    ----------
    data : numpy.ndarray[typing.Any, typing.Any]
        input data
    as_median : bool, optional
        choose to give root median squared error instead, by default False

    Returns
    -------
    float
        RMSE value
    """
    if as_median:
        value: float = np.sqrt(np.nanmedian(data**2).item())
    else:
        value = np.sqrt(np.nanmean(data**2).item())

    return value


def get_min_max(
    values: pd.Series | NDArray,
    robust: bool = False,
    absolute: bool = False,
    robust_percentiles: tuple[float, float] = (0.02, 0.98),
) -> tuple[float, float]:
    """
    Get a grids max and min values.

    Parameters
    ----------
    values : pandas.Series or numpy.ndarray
        values to find min or max for
    robust: bool, optional
        choose whether to return the 2nd and 98th percentile values, instead of the
        min/max
    absolute : bool, optional
        return the absolute min and max values, by default False
    robust_percentiles : tuple[float, float], optional
        decimal percentiles to use for robust min and max, by default (0.02, 0.98)

    Returns
    -------
    tuple[float, float]
        returns the min and max values.
    """

    if robust:
        v_min, v_max = np.nanquantile(values, robust_percentiles)
    else:
        v_min, v_max = np.nanmin(values), np.nanmax(values)

    if absolute is True:
        v_min, v_max = -vd.maxabs([v_min, v_max]), vd.maxabs([v_min, v_max])  # pylint: disable=used-before-assignment

    assert v_min <= v_max, "min value should be less than or equal to max value"  # pylint: disable=possibly-used-before-assignment
    return (v_min, v_max)


def sample_grid(
    data: pd.DataFrame,
    grid: xr.DataArray,
    coord_names: tuple[str, str],
) -> NDArray:
    """
    Sample grid values at supplied points in dataframe.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe containing columns 'easting', 'northing', or columns with names
        defined by kwarg "coord_names".
    grid : xarray.DataArray
        Grid to sample
    coord_names : tuple[str, str],
        Names of the coordinates, in order x then y.

    Returns
    -------
    NDArray
        sample values, which can be added to the dataframe
    """
    data = data.copy()

    data["original_index"] = data.index

    data = data.reset_index(drop=True)

    assert all(c in data.columns for c in coord_names), (
        f"{coord_names} must be in the dataframe"
    )

    # get points to sample at
    points = data[list(coord_names)]

    # sample the grid at all x,y points
    sampled = pygmt.grdtrack(
        points=points,
        grid=grid,
        newcolname="tmp",
        no_skip=True,  # if false causes issues
        verbose="warning",
        interpolation="c",
    )

    data["tmp"] = sampled.tmp

    return data.set_index("original_index").tmp


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
