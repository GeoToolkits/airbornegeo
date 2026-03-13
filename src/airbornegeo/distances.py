import geopandas as gpd
import numpy as np
import pandas as pd
import seaborn as sns
import shapely
from numpy.typing import NDArray
from shapely.geometry import LineString

import airbornegeo

sns.set_theme()


def _relative_distance(
    x: NDArray,
    y: NDArray,
) -> NDArray:
    """
    calculate the relative distance between each successive set of points

    Parameters
    ----------
    x : NDArray
        array of the first coordinate values
    y : NDArray
        array of the second coordinate values

    Returns
    -------
    NDArray
        the distance between each set of points
    """
    assert len(x) == len(y)

    # shift the arrays by 1
    x_lag = np.empty_like(x)
    x_lag[:1] = np.nan
    x_lag[1:] = x[:-1]

    y_lag = np.empty_like(y)
    y_lag[:1] = np.nan
    y_lag[1:] = y[:-1]

    # compute distance between each set of coordinates
    rel_dist = np.sqrt((x - x_lag) ** 2 + (y - y_lag) ** 2)

    # set first row distance to 0
    rel_dist[0] = 0
    return rel_dist


def relative_distance(
    data: pd.DataFrame,
) -> pd.Series:
    """
    Calculate distance between successive points in a dataframe. This assumes the data
    have been sorted by time, and that there are not flights which overlap in time. If
    there are, you can first sort by flight, then by time. For example, if you have
    columns 'flight' and 'unixtime', you can accomplish this with
    `df = df.sort_values(["flight", "unixtime"])`.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe containing columns 'easting' and 'northing in meters.

    Returns
    -------
    pandas.Series
        Returns a pandas Series of the relative distances which can be assigned to a new
        column.
    """
    return _relative_distance(data.easting.values, data.northing.values)


def cumulative_distance(
    data: pd.DataFrame,
) -> pd.Series:
    """
    Calculate the cumulative distance along track in a dataframe. This assumes the data
    have been sorted by time, and that there are not flights which overlap in time. If
    there are, you can first sort by flight, then by time. For example, if you have
    columns 'flight' and 'unixtime', you can accomplish this with
    `df = df.sort_values(["flight", "unixtime"])`.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe containing columns 'easting' and 'northing in meters.

    Returns
    -------
    pandas.Series
        Returns a pandas Series of the relative distances which can be assigned to a new
        column.
    """
    return relative_distance(data).cumsum()


def along_track_distance(
    data: gpd.GeoDataFrame,
    groupby_column: str | None = None,
    guess_start_position: bool = False,
) -> pd.Series:
    """
    Calculate the distances along track in meters. The dataframe will be grouped by
    column `groupby_column`, and the distance will start from the first row of each
    group. For example, a group can be a survey, a flight, or an individual line. This
    function will sort each group by time, via the column `unixtime` so distance will be
    0 at the lowest timestamp of each group. If you don't have time information, you can
    make a fake `unixtime` column based on the index of the rows.

    Parameters
    ----------
    data : gpd.GeoDataFrame
        Dataframe containing the data points to calculate the distance along each line,
        must have a set geometry column.
    groupby_column : str | None, optional
        Column name to group by before sorting by time, by default None
    guess_start_position: bool, optional
        If True, this will determine the start of the line, not by the first row, but by
        finding the leftmost corner of the line. This is useful if you don't have a time
        column and your data is not sorted by time.

    Returns
    -------
    pd.Series
        The along track distance in meters
    """
    if guess_start_position:
        if groupby_column is None:
            # turn point data into line
            line = gpd.GeoSeries(LineString(data.geometry.tolist()))

            # find minimum rotated rectangle around line
            rect = line.iloc[0].minimum_rotated_rectangle

            # get angle of rotation
            angle = airbornegeo.processing.azimuth(rect)
            if 90 < angle <= 180:
                angle = angle - 180

            # rotate the line to be horizontal
            line_horizontal = line.rotate(angle, origin=shapely.centroid(rect))
            horizontal_df = line_horizontal.get_coordinates(
                index_parts=True,
                ignore_index=True,
            )
            horizontal_df["original_index"] = data.index
            horizontal_df = horizontal_df.sort_values("x").reset_index(drop=True)
            horizontal_df = horizontal_df.rename(
                columns={"x": "easting", "y": "northing"}
            )
            horizontal_df["tmp"] = cumulative_distance(horizontal_df)
            horizontal_df = horizontal_df.sort_values("original_index").set_index(
                "original_index"
            )
            return horizontal_df.tmp
        data = data.copy()
        for _segment_name, segment_data in data.groupby(groupby_column):
            # turn point data into line
            line = gpd.GeoSeries(LineString(segment_data.geometry.tolist()))

            # find minimum rotated rectangle around line
            rect = line.iloc[0].minimum_rotated_rectangle

            # get angle of rotation
            angle = airbornegeo.processing.azimuth(rect)
            if 90 < angle <= 180:
                angle = angle - 180

            # rotate the line to be horizontal
            line_horizontal = line.rotate(angle, origin=shapely.centroid(rect))
            horizontal_df = line_horizontal.get_coordinates(
                index_parts=True,
                ignore_index=True,
            )
            horizontal_df["original_index"] = segment_data.index
            horizontal_df = horizontal_df.sort_values("x").reset_index(drop=True)
            horizontal_df = horizontal_df.rename(
                columns={"x": "easting", "y": "northing"}
            )
            horizontal_df["tmp"] = cumulative_distance(horizontal_df)
            horizontal_df = horizontal_df.sort_values("original_index").set_index(
                "original_index"
            )
            data.loc[data[groupby_column] == _segment_name, "tmp"] = horizontal_df.tmp
        return data.tmp

    if groupby_column is None:
        return cumulative_distance(data)

    # iterate through groups, append distances, and concat
    distances = []
    for _segment_name, segment_data in data.groupby(groupby_column):
        dist = cumulative_distance(segment_data)
        distances.append(dist)
    return np.concatenate(distances)
