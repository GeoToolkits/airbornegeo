import typing

import numpy as np
import pandas as pd
import pygmt
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


def filter1d(
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
        Column name to group by before sorting by time, by default None.
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
