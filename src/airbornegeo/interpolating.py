import warnings

import numpy as np
import pandas as pd
import scipy
from tqdm.autonotebook import tqdm

import airbornegeo
from airbornegeo import logger


def interpolate_missing(
    data: pd.DataFrame,
    *,
    to_interp: list[str] | str,
    interp_on: str,
    method: str = "cubic",
    extrapolate: bool = False,
    fill_value: tuple[float, float] | str | None = None,
    groupby_column: str | None = None,
) -> pd.DataFrame:
    """
    Interpolate NaN's in "to_interp" column(s), based on value(s) from "interp_on". If
    groupby_column is provided, the dataframe will first be grouped by this so only
    data from the group containing the NaN is used to interpolate.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing the data to interpolate
    to_interp : list[str] | str
        Column(s) to interpolate
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

    Returns
    -------
    pd.DataFrame
        Dataframe with interpolated columns
    """
    data = data.copy()

    if isinstance(to_interp, str):  # type: ignore [unreachable]
        to_interp = [to_interp]  # type: ignore [unreachable]

    col_list = [interp_on, *to_interp]
    if groupby_column is not None:
        col_list.append(groupby_column)
    assert all(x in data.columns for x in col_list), (
        f"dataframe must contain columns {col_list} "
    )

    if groupby_column is None:
        # iterate through columns
        for col in to_interp:
            filled = interpolate_missing_single_column(
                data,
                to_interp=col,
                interp_on=interp_on,
                method=method,
                extrapolate=extrapolate,
                fill_value=fill_value,
            )
            data[col] = filled[col]
        return data

    filled_segments = []
    for _segment_name, segment_data in tqdm(
        data.groupby(groupby_column), desc="Segments"
    ):
        for col in to_interp:
            filled = interpolate_missing_single_column(
                segment_data,
                to_interp=col,
                interp_on=interp_on,
                method=method,
                extrapolate=extrapolate,
                fill_value=fill_value,
            )
            data[col] = filled[col]
            filled_segments.append(data)
    return pd.concat(filled_segments)


def interpolate_missing_with_windows(
    data: pd.DataFrame,
    *,
    window_width: float,
    to_interp: str | list[str],
    interp_on: str,
    method: str = "cubic",
    extrapolate: bool = False,
    fill_value: tuple[float, float] | str | None = None,
    groupby_column: str | None = None,
) -> pd.DataFrame:
    """
    Interpolate NaN's in "to_interp" column(s), based on value(s) from "interp_on" using
    only values within a window around the NaN. If groupby_column is provided, the
    dataframe will first be grouped by this so only data from the group containing the
    NaN is used to interpolate.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing the data to interpolate
    window_width : float
        width of data window around NaN value to use in the interpolation, in units of
        the data provided in the column interp_on
    to_interp : list[str] | str
        Column(s) to interpolate
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

    Returns
    -------
    pd.DataFrame
        Dataframe with interpolated columns
    """

    data = data.copy()

    if isinstance(to_interp, str):
        to_interp = [to_interp]

    col_list = [interp_on, *to_interp]
    if groupby_column is not None:
        col_list.append(groupby_column)
    assert all(x in data.columns for x in col_list), (
        f"dataframe must contain columns {col_list} "
    )

    if groupby_column is None:
        # iterate through columns
        with airbornegeo.utils.DuplicateFilter(logger):
            for col in to_interp:
                logger.debug("Interpolating column: %s", col)
                filled = interpolate_missing_with_windows_single_column(
                    data,
                    to_interp=col,
                    interp_on=interp_on,
                    window_width=window_width,
                    method=method,
                    extrapolate=extrapolate,
                    fill_value=fill_value,
                )
                data = filled
        return data

    filled_segments = []
    for _segment_name, segment_data in tqdm(
        data.groupby(groupby_column), desc="Segments"
    ):
        for col in to_interp:
            filled = interpolate_missing_with_windows_single_column(
                segment_data,
                to_interp=col,
                interp_on=interp_on,
                window_width=window_width,
                method=method,
                extrapolate=extrapolate,
                fill_value=fill_value,
            )
            # segment_data[col] = filled[col]
            filled_segments.append(filled)
    return pd.concat(filled_segments)


def scipy_interpolate_missing(
    data: pd.DataFrame,
    *,
    to_interp: str,
    interp_on: str,
    method: str = "linear",
    extrapolate: bool = False,
    fill_value: tuple[float, float] | str | None = None,
) -> pd.DataFrame:
    """
    interpolate NaN's in "to_interp" column, based on values from "interp_on" column
    method:
        'linear', 'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic',
        'previous', 'next'
    """
    data = data.copy()

    col_list = [to_interp, interp_on]
    assert all(x in data.columns for x in col_list), (
        f"dataframe must contain columns {col_list} "
    )

    # drop NaN's
    data_no_nans = data.dropna(subset=[to_interp, interp_on], how="any")

    if extrapolate is True:
        bounds_error = False
        if fill_value is None:
            fill_value = "extrapolate"
        elif fill_value == "edge":
            fill_value = (
                data_no_nans[to_interp].iloc[0],
                data_no_nans[to_interp].iloc[-1],
            )
        elif fill_value == "mean":
            fill_value = (np.nanmean(data[to_interp]), np.nanmean(data[to_interp]))
        logger.debug("extrapolating with fill_value: %s", fill_value)
    else:
        bounds_error = False
        fill_value = np.nan

    # define interpolation function
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The number of derivatives at boundaries does not match:",
        )
        # this is legacy! Info here https://docs.scipy.org/doc/scipy/tutorial/interpolate/1D.html#tutorial-interpolate-1dsection
        f = scipy.interpolate.interp1d(
            data_no_nans[interp_on],
            data_no_nans[to_interp],
            kind=method,
            bounds_error=bounds_error,
            fill_value=fill_value,
        )

    # get interpolated values at points with NaN's
    values = f(data[data[to_interp].isna()][interp_on])

    # fill NaN's  with values
    data.loc[data[to_interp].isna(), to_interp] = values

    return data


def interpolate_missing_single_column(
    data: pd.DataFrame,
    *,
    to_interp: str,
    interp_on: str,
    method: str = "cubic",
    extrapolate: bool = False,
    fill_value: tuple[float, float] | str | None = None,
) -> pd.DataFrame:
    """
    interpolate NaN's in "to_interp" column, based on value(s) from "interp_on"
    column(s).
    method:
        'linear', 'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic',
        'cubic', 'previous', 'next'
    """

    col_list = [to_interp, interp_on]
    assert all(x in data.columns for x in col_list), (
        f"dataframe must contain columns {col_list} "
    )

    args = {
        "data": data,
        "to_interp": to_interp,
        "interp_on": interp_on,
        "method": method,
        "extrapolate": extrapolate,
        "fill_value": fill_value,
    }

    return scipy_interpolate_missing(**args)


def interpolate_missing_with_windows_single_column(
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
    Create a window of data either side of NaN's based on "distance_along_line" column and
    interpolate the value. Useful when NaN's are sparse, or lines are long.
    """
    data = data.copy()

    col_list = [interp_on, to_interp]
    assert all(x in data.columns for x in col_list), (
        f"dataframe must contain columns {col_list} "
    )

    data[f"{to_interp}_interpolation_type"] = "none"

    # iterate through NaNs
    for i in data[data[to_interp].isna()].index:  # pylint: disable=too-many-nested-blocks
        # get value to interpolate on (e.g. distance along line) for NaN
        dist_at_nan = data[interp_on].loc[i]

        # try interpolation with set window width, if there's not enough data (bounds
        # error), double the width up to 2 times.
        # if 2 attempts fail and extrapolate is True, allow extrapolation, if False, return NaN
        # if extrapolating, start with original window width and double up to 2 times, if fails, return NaN
        win = window_width
        while win <= window_width * 2:
            try:
                # get data inside window
                llim, ulim = dist_at_nan - win, dist_at_nan + win
                data_inside = data[data[interp_on].between(llim, ulim)]

                if len(data_inside) <= 1:
                    win += win
                    logger.debug(
                        "Error during interpolation, doubling window size to %s",
                        win,
                    )
                    continue

                # may be multiple NaN's within window (some outside of bounds)
                # but we only extract the fill value for loc[i]
                filled = interpolate_missing(
                    data_inside,
                    to_interp=[to_interp],
                    interp_on=interp_on,
                    method=method,
                    extrapolate=False,
                    fill_value=fill_value,
                )
                # extract just the filled value
                value = filled[to_interp].loc[i]
                if np.isnan(value):
                    msg = "filled value is NaN"
                    raise ValueError(msg)  # noqa: TRY301

                interp_type = "interpolated"

            except Exception:  # noqa: BLE001 pylint: disable=broad-exception-caught
                # logger.error(e)
                win += win
                logger.debug(
                    "Error with interpolation, doubling window size to %s",
                    win,
                )
                # # error messages for too few points in window
                # few_points_errors = [
                #     "cannot reshape array of",
                #     "Found array with",
                #     "The number of derivatives at boundaries does not match:",
                # ]
                # # error message for bounds error
                # bounds_errors = [
                #     "in x_new is above the interpolation range",
                #     "in x_new is below the interpolation range",
                # ]
                # if any(item in str(e) for item in few_points_errors):
                #     win += win
                #     logger.warning(
                #         "too few points in window,"
                #         "doubling window size to %s",
                #         win,
                #     )
                # elif any(item in str(e) for item in bounds_errors):
                #     win += win
                #     logger.warning(
                #         "bounds error for interpolation,"
                #         "doubling window size to %s",
                #         win,
                #     )
                # else:  # raise other errors
                #     win += win
                #     logger.error(e)
                #     logger.warning(
                #         "Error for interpolation, "
                #         "doubling window size to %s",
                #         win,
                #     )
                continue
            break
        else:
            if extrapolate:
                # try extrapolation with set window width, if there's not enough data, double the width up to 2 times.
                win = window_width
                while win <= window_width * (2):
                    try:
                        # get data inside window
                        llim, ulim = dist_at_nan - win, dist_at_nan + win
                        data_inside = data[data[interp_on].between(llim, ulim)]

                        if len(data_inside) <= 1:
                            win += win
                            logger.debug(
                                "Error with interpolation, doubling window size to %s",
                                win,
                            )
                            continue

                        # may be multiple NaN's within window (some outside of bounds)
                        # but we only extract the fill value for loc[i]
                        filled = interpolate_missing(
                            data_inside,
                            to_interp=[to_interp],
                            interp_on=interp_on,
                            method=method,
                            extrapolate=True,
                            fill_value=fill_value,
                        )
                        # extract just the filled value
                        value = filled[to_interp].loc[i]
                        if np.isnan(value):
                            msg = "filled value is NaN"
                            raise ValueError(msg)  # noqa: TRY301

                        interp_type = "extrapolated"

                    except Exception:  # noqa: BLE001 pylint: disable=broad-exception-caught
                        win += win
                        logger.debug(
                            "Error with interpolation, doubling window size to %s",
                            win,
                        )
                        continue
                    break
                else:
                    logger.debug(
                        "Extrapolation failed after window expanded 2 times, to %s "
                        "returning NaN for interpolated value",
                        win,
                    )
                    value = np.nan
                    interp_type = "none"
            else:
                logger.debug(
                    "Window expanded 2 times, to %s, without success and `extrapolate` "
                    "set to False, returning NaN for interpolated value",
                    win,
                )
                value = np.nan
                interp_type = "none"

        # add values into dataframe
        data.at[i, to_interp] = value  # noqa: PD008
        data.at[i, f"{to_interp}_interpolation_type"] = interp_type  # noqa: PD008

    return data
