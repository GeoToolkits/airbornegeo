import copy
import typing

import numpy as np
import pandas as pd
import verde as vd
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
