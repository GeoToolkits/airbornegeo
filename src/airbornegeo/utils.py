import copy
import typing

import numpy as np
import pandas as pd
import scipy.spatial
import verde as vd
from numpy.typing import NDArray
from tqdm.autonotebook import tqdm

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


def median_line_spacing(
    data: pd.DataFrame,
    line_column: str,
) -> float:
    """
    Estimate the flight line spacing by first, for each point on a specific line,
    finding the distances to the nearest point on another line and repeating this for
    all lines. Then take the median of these nearest neighbor distances. If you want
    to find flight lines spacing versus tie line spacing, only supply one line type at a
    time.

    Parameters
    ----------
    data : pd.DataFrame
        The dataframe containing columns 'easting', 'northing', and the column given by
        line_column, to compute the median line spacing for.
    line_column : str
        name of the column containing the line number/name

    Returns
    -------
    float
        the median of the distances to the nearest point on another line for each data
        point, as an approximation of flight lines spacing.

    """
    # check columns are present
    _check_coord_columns(data)
    cols = [line_column]

    assert all(col in data.columns for col in cols), f"{cols} must be in the dataframe"

    data = data.copy()

    min_distances = []
    for line_name in data[line_column].unique():
        # get line data
        line_df = data[data[line_column] == line_name]

        # get rest of survey data
        survey_df = data[data[line_column] != line_name]

        # for each line point, find the distance to the nearest non-line point
        kdtree = scipy.spatial.KDTree(survey_df[["easting", "northing"]].to_numpy())
        min_dist, _ = kdtree.query(line_df[["easting", "northing"]].to_numpy(), k=1)

        min_distances.append(min_dist)

    min_distances = np.concat(min_distances)

    return np.median(min_distances)


def _check_coord_columns(
    data: pd.DataFrame,
) -> None:
    """
    Check that the dataframe contains the columns 'easting' and 'northing' and if not
    inform users about `.rename()`.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe which should contain the columns
    """

    # check columns are present
    cols = ["easting", "northing"]

    assert all(col in data.columns for col in cols), (
        f"Projected coordinates columns ({cols}) must be in the dataframe. If you have them with other names (e.g. 'x','y'), you can rename your dataframe (df) with df = df.rename(columns={{'x':'easting','y':'northing'}})"
    )


def _init_iteration_progressbar(
    max_iterations: int,
    progressbar: bool,
) -> tuple[bool, typing.Iterable[int]]:
    """
    Build the iterator to loop over iterations 1..max_iterations, optionally wrapped
    in a progress bar, disabling the progress bar when there's only 1 iteration.

    Parameters
    ----------
    max_iterations : int
        Number of iterations to loop over.
    progressbar : bool
        Whether to show a progress bar over the iterations.

    Returns
    -------
    tuple[bool, Iterable[int]]
        The (possibly disabled) progressbar flag, and the iterable of iteration
        numbers to loop over.
    """
    if max_iterations == 1:
        progressbar = False

    iterations = range(1, max_iterations + 1)
    return progressbar, (tqdm(iterations) if progressbar else iterations)


def _iter_groups(
    data: pd.DataFrame,
    groupby_column: str | list[str],
    progressbar: bool,
    desc: str = "Segments",
) -> typing.Iterable[tuple[typing.Any, pd.DataFrame]]:
    """
    Group `data` by `groupby_column`, preserving the order groups first appear in
    `data` (instead of pandas' default sorted-by-key order), and optionally wrap the
    resulting iterator in a progress bar.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe to group.
    groupby_column : str | list[str]
        Column name(s) to group by.
    progressbar : bool
        Show a progress bar while iterating over groups.
    desc : str, optional
        Progress bar description, by default "Segments".

    Returns
    -------
    Iterable[tuple[Any, pd.DataFrame]]
        Iterable of (group key, group dataframe) pairs.
    """
    groups = data.groupby(groupby_column, sort=False)
    return tqdm(groups, desc=desc) if progressbar else groups


def _apply_grouped(
    data: pd.DataFrame,
    *,
    groupby_column: str | list[str] | None,
    progressbar: bool,
    func: typing.Callable[[pd.DataFrame], typing.Any],
    desc: str = "Segments",
) -> typing.Any:
    """
    Apply `func` to `data` as a whole, or independently to each group defined by
    `groupby_column`, and combine the results in `data`'s original row order.

    `func` is given a dataframe (all of `data`, or one group's rows) and must return
    either a single array-like, or a tuple of array-likes, each with one value per
    row of the dataframe it was given. Results are combined by their original row
    index rather than by group-iteration order, so the output stays aligned with
    `data` even when the group labels don't sort into the order they first appear in
    `data`.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe to apply `func` to.
    groupby_column : str | list[str] | None
        Column name(s) to group by before applying `func`, or None to apply it to
        the whole dataframe at once.
    progressbar : bool
        Show a progress bar for each group.
    func : Callable[[pd.DataFrame], Any]
        Function to apply to `data`, or to each group's dataframe.
    desc : str, optional
        Progress bar description, by default "Segments".

    Returns
    -------
    Any
        A numpy array (or tuple of numpy arrays, matching what `func` returns), with
        one value per row of `data`.
    """
    if groupby_column is None:
        result = func(data)
        if isinstance(result, tuple):
            return tuple(np.asarray(r) for r in result)
        return np.asarray(result)

    segments = list(_iter_groups(data, groupby_column, progressbar, desc=desc))
    results = [func(segment) for _, segment in segments]
    indices = [segment.index for _, segment in segments]

    is_tuple = isinstance(results[0], tuple)
    if not is_tuple:
        results = [(r,) for r in results]

    combined = tuple(
        pd.concat(
            [
                pd.Series(np.asarray(r[i]), index=idx)
                for r, idx in zip(results, indices, strict=True)
            ]
        )
        .loc[data.index]
        .to_numpy()
        for i in range(len(results[0]))
    )
    return combined if is_tuple else combined[0]
