import typing

import pandas as pd
import seaborn as sns
import verde as vd

import airbornegeo

sns.set_theme()


def level_to_grid(
    data: pd.DataFrame,
    *,
    data_column: str,
    grid_column: str,
    fit_by_column: str | tuple[str],
    degree: int | None = None,
    filter_kwargs: dict[str, typing.Any] | None = None,
    groupby_column: str | None = None,
) -> pd.Series:
    """
    Level flight lines based on the misfit between the data_column and grid_column
    values. The levelling correction is determined from the misfit values, by either
    fitting a trend with a specified order to the misfits, or by spatially or temporally
    filtering the misfits. This fitting or filtering use the data in fit_by_column as
    the independent variable. The levelling correction is the subtracted from the line
    data. The grid values can be sampled into the dataframe with ::func`sample_grid`.
    If groupby_column is provided, the trend will be fit a line-by-line basis in 1D
    using the column distance_along_line. If groupby_column is not provided, the trend
    will be fit to the entire survey in 2D using the cartesian coordinate columns
    provided by groupby_column.

    Parameters
    ----------
    data : pd.DataFrame
        dataframe with the data and grid values.
    data_column : str
        the column name for the data to fit.
    grid_column : str
        the column name with the sample grid values.
    fit_by_column : str or None, optional
        the column(s) to filter or fit a trend against, for example a column called
        'unixtime' which contains sequentially second values, if filtering temporally,
        or a column called 'distance_along_line' with sequential distance values in
        meters, if filtering spatially, or if groupby_column is not used, then you must
        provide two column names to fit a trend / filter in 2D, for each 'easting' and
        'northing'.
    groupby_column : str | None
        Column name to group by, by default None
    degree : int or None, optional
        the degree order to fit to the misfit values.
    filter_kwargs : dict | None, optional
        Alternative to `degree`: keyword arguments forwarded to
        `airbornegeo.filter_line` to low-pass filter the misfit values (besides
        `data_column` and `filter_by_column`, which are set for you). Must include
        `filter_width`; anything else you don't provide falls back to
        `filter_line`'s own defaults (e.g. `filter_shape="gaussian"`,
        `engine="scipy"`).

    Returns
    -------
    pd.Series
        The levelled data
    """
    data = data.copy()

    data = data.dropna(subset=grid_column)

    data["tmp_misfit"] = data[data_column] - data[grid_column]

    if (degree is not None) & (filter_kwargs is not None):
        msg = "only provide filter_kwargs or degree, not both"
        raise ValueError(msg)

    if groupby_column is None:
        assert all(col in data.columns for col in fit_by_column), (
            f"{fit_by_column} must be in the dataframe"
        )
        if degree is not None:
            # calculate correction by fitting trend to misfit values
            vdtrend = vd.Trend(degree=degree).fit(
                (data[fit_by_column[0]], data[fit_by_column[1]]),
                data.tmp_misfit,
            )
            correction = vdtrend.predict(
                coordinates=((data[fit_by_column[0]], data[fit_by_column[1]]))
            )
        elif filter_kwargs is not None:
            msg = "for 2D, only trend fitting supported, not filtering"
            raise NotImplementedError(msg)
        return data[data_column] - correction  # pylint: disable=possibly-used-before-assignment

    assert isinstance(fit_by_column, str), "fit_by_column must be a string"

    cols = [groupby_column, fit_by_column]
    assert all(col in data.columns for col in cols), f"{cols} must be in the dataframe"

    # fit a trend to the misfits on line-by-line basis
    for _segment_name, segment_data in data.groupby(groupby_column):
        if degree is not None:
            # calculate correction by fitting trend to misfit values
            correction = airbornegeo.trend(
                data_to_fit=segment_data,
                cols_to_fit=[fit_by_column, "tmp_misfit"],
                data_to_predict=segment_data,
                cols_to_predict=[fit_by_column, "correction"],
                degree=degree,
            ).correction
        elif filter_kwargs is not None:
            # calculate levelling correction by low pass filtering the misfits values
            correction = airbornegeo.filter_line(
                segment_data,
                data_column="tmp_misfit",
                filter_by_column=fit_by_column,
                **filter_kwargs,
            )
        # add correction values to the main dataframe
        data.loc[data.line == _segment_name, "levelling_correction"] = correction

    return data[data_column] - data.levelling_correction
