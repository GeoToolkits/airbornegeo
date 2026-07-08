import math

import geopandas as gpd
import numpy as np
import pandas as pd
import seaborn as sns
import shapely
from geographiclib.geodesic import Geodesic
from numpy.typing import NDArray
from shapely.geometry import LineString

from airbornegeo.utils import _apply_grouped, _check_coord_columns, _iter_groups

sns.set_theme()


def directional_velocity(
    data: pd.DataFrame,
    *,
    time_column: str = "unixtime",
    coordinate_column: str = "easting",
    groupby_column: str | None = None,
    progressbar: bool = True,
) -> NDArray:
    """
    Calculate one component of velocity, which is the change in coordinate values
    divided by the change in time between each successive row in the dataframe. For
    example, if latitude in decimal degrees are provided via column `coordinate_column`
    and time in seconds is provided to `time_column`, this would return the latitudinal
    component of velocity in degrees per second. This assumes the data have been sorted
    by time, and that there are not flights which overlap in time. If there are, you can
    first sort by flight, then by time. For example, if you have columns 'flight' and
    'unixtime', you can accomplish this with
    `data = data.sort_values(["flight", "unixtime"])`. If groupby_column is provided,
    the dataframe will first be grouped by this. The returned units are based on the
    units provided by the coordinate_column and time_column.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing the data points to calculate the ground speed for,
        must have columns 'unixtime' and 'relative_distance'
    time_column : str
        name of the column containing the time in seconds
    coordinate_column : str
        name of the column containing the coordinates to use for calculating velocity
    groupby_column : str | None, optional
        Column name to group by before calculation, by default None
    progressbar : bool, optional
        Show progress bar for each group, by default True

    Returns
    -------
    NDArray
        The velocity component in units of the provided coordinate and time columns
    """
    data = data.copy()

    col_list = [time_column, coordinate_column]
    if groupby_column is not None:
        col_list.append(groupby_column)
    assert all(x in data.columns for x in col_list), (
        f"dataframe must contain columns {col_list} "
    )

    return _apply_grouped(
        data,
        groupby_column=groupby_column,
        progressbar=progressbar,
        func=lambda d: np.gradient(d[coordinate_column], d[time_column]),
    )


def ground_speed(
    data: pd.DataFrame,
    *,
    time_column: str,
    groupby_column: str | None = None,
    progressbar: bool = True,
) -> NDArray:
    """
    TODO: do calculation forward for 1st points so they aren't 0
    Calculate the ground speed in meters per second. This is change in distance divided by the
    change in time between each successive row in the dataframe. This assumes the data
    have been sorted by time, and that there are not flights which overlap in time. If
    there are, you can first sort by flight, then by time. For example, if you have
    columns 'flight' and 'unixtime', you can accomplish this with
    `data = data.sort_values(["flight", "unixtime"])`. If groupby_column is provided,
    the dataframe will first be grouped by this.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing the data points to calculate the ground speed for, must
        have columns 'easting' and 'northing'.
    time_column : str
        name of the column containing the time in seconds
    groupby_column : str | None, optional
        Column name to group by before calculation, by default None
    progressbar : bool, optional
        Show progress bar for each group, by default True

    Returns
    -------
    NDArray
        The groundspeed in units of meters per second
    """
    data = data.copy()
    _check_coord_columns(data)

    col_list = [time_column]
    if groupby_column is not None:
        col_list.append(groupby_column)
    assert all(x in data.columns for x in col_list), (
        f"dataframe must contain columns {col_list} "
    )

    data["cumulative_distance"] = cumulative_distance(
        data,
        groupby_column=groupby_column,
        progressbar=progressbar,
    )

    return _apply_grouped(
        data,
        groupby_column=groupby_column,
        progressbar=progressbar,
        func=lambda d: np.gradient(d.cumulative_distance, d[time_column]),
    )


# def _vertical_acceleration(
#     time: NDArray,
#     height: NDArray,
# ) -> NDArray:
#     """
#     Calculate the vertical acceleration between each successive set of points

#     Parameters
#     ----------
#     time : NDArray
#         array of the time values
#     height : NDArray
#         array of the height values

#     Returns
#     -------
#     NDArray
#         the vertical acceleration between each set of points
#     """
#     assert len(time) == len(height)

#     # shift the arrays by 1
#     time_lag = np.empty_like(time)
#     time_lag[:1] = np.nan
#     time_lag[1:] = time[:-1]

#     height_lag = np.empty_like(height)
#     height_lag[:1] = np.nan
#     height_lag[1:] = height[:-1]

#     # compute vertical velocity
#     vertical_vel = (height - height_lag) / (time - time_lag)

#     # shift arrays by 1
#     vertical_vel_lag = np.empty_like(vertical_vel)
#     vertical_vel_lag[:1] = np.nan
#     vertical_vel_lag[1:] = vertical_vel[:-1]

#     # compute vertical acceleration
#     return (vertical_vel - vertical_vel_lag) / (time - time_lag)


def vertical_acceleration(
    data: pd.DataFrame,
    *,
    time_column: str,
    height_column: str,
    groupby_column: str | None = None,
    progressbar: bool = True,
    time_threshold: float | None = None,
    smoothing_window: int | None = None,
) -> NDArray:
    """
    Calculate the 2nd derivative of height change with respect to time for each line.
    If there is a gap between points greater than time_threshold in seconds, the line
    will be split at the gap and the acceleration will be NaN for the points on
    either side of the gap.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing the data points and must have columns set from time_column
        and height_column.
    time_column : str
        Column name to containing the time in seconds
    height_column : str
        Column name to containing the flight height in meters
    groupby_column : str | None, optional
        Column name to group by before calculation, by default None
    progressbar : bool, optional
        Show progress bar for each group, by default True
    time_threshold : float
        Threshold in seconds for determining gaps in the data, where acceleration will be set to NaN
    smoothing_window : int, optional
        Window size in number of data points for smoothing each derivative after it's
        been calculated, by default None

    Returns
    -------
    NDArray
        Array containing the vertical acceleration in m/s^2 for each point in the
        dataframe.
    """
    data = data.copy()

    col_list = [time_column, height_column]
    if groupby_column is not None:
        col_list.append(groupby_column)
    assert all(x in data.columns for x in col_list), (
        f"dataframe must contain columns {col_list} "
    )

    if (groupby_column is None) and (time_threshold is not None):
        # split data into segments where there is a gap in time greater than
        # time_threshold
        # Calculate time difference between each point
        data["tmp_time_diff"] = pd.to_timedelta(data[time_column].diff(), unit="s")

        # Create new subline when gap > time_threshold
        data["tmp_new_group"] = (
            data["tmp_time_diff"] > pd.Timedelta(seconds=time_threshold)
        ).astype(int)

        # Cumulative sum per Line to generate subline number
        data["tmp_segment"] = data["tmp_new_group"].cumsum()

        # Drop helper columns if not needed
        data = data.drop(columns=["tmp_time_diff", "tmp_new_group"])

        groupby_column = "tmp_segment"

    elif (groupby_column is not None) and (time_threshold is not None):
        # split data into segments where there is a gap in time greater than
        # time_threshold
        # Calculate time difference between each point
        data["tmp_time_diff"] = pd.to_timedelta(
            data.groupby(groupby_column)[time_column].diff(), unit="s"
        )

        # Create new subline when gap > time_threshold
        data["tmp_new_group"] = (
            data["tmp_time_diff"] > pd.Timedelta(seconds=time_threshold)
        ).astype(int)

        # Cumulative sum per Line to generate subline number
        data["tmp_segment"] = data.groupby(groupby_column)["tmp_new_group"].cumsum()

        # Drop helper columns if not needed
        data = data.drop(columns=["tmp_time_diff", "tmp_new_group"])

        groupby_column = [groupby_column, "tmp_segment"]

    def _accel(segment: pd.DataFrame) -> pd.Series:
        times = segment[time_column]
        heights = segment[height_column]
        vertical_vel = np.gradient(heights, times)
        vertical_accel = pd.Series(np.gradient(vertical_vel, times))

        if smoothing_window is not None:
            return vertical_accel.rolling(window=smoothing_window, min_periods=1).mean()
        return vertical_accel

    return _apply_grouped(
        data,
        groupby_column=groupby_column,
        progressbar=progressbar,
        func=_accel,
    )


def relative_track_ellipsoid(
    lat: NDArray,
    lon: NDArray,
) -> NDArray:
    """
    Calculate the track between each successive set of points in degrees clockwise from
    geographic north in the range 0 to 360. This uses the WGS84 ellipsoid, make the
    results more accurate the ::func:`relative_track_sheroid`.

    Parameters
    ----------
    lat : NDArray
        array of the latitude coordinate values in decimal degrees
    lon : NDArray
        array of the longitude coordinate values in decimal degrees

    Returns
    -------
    NDArray
        the track between each set of points in decimal degrees from geographic north
    """

    geod = Geodesic.WGS84
    tracks = []
    # Calculate tracks for all segments (N-1)
    for i in range(len(lat) - 1):
        line = geod.Inverse(lat[i], lon[i], lat[i + 1], lon[i + 1])
        tracks.append(line["azi1"])

    # Duplicate the last calculated track if the list isn't empty
    if tracks:
        tracks.append(tracks[-1])
    elif len(lat) == 1:
        # Handle single point case (no direction possible)
        tracks.append(None)

    # make tracks in range of 0 to 360
    return np.array(tracks) % 360


def relative_track_spheroid(
    lat: NDArray,
    lon: NDArray,
) -> NDArray:
    """
    Calculate the track between each successive set of points in degrees clockwise from
    geographic north in the range 0 to 360. This assumes the Earth is a sphere, which
    is less accurate the using an ellipsoid model.

    Parameters
    ----------
    lat : NDArray
        array of the latitude coordinate values in decimal degrees
    lon : NDArray
        array of the longitude coordinate values in decimal degrees

    Returns
    -------
    NDArray
        the track between each set of points in decimal degrees from geographic north
    """

    # ensure 1D
    lat = np.asarray(lat).ravel()
    lon = np.asarray(lon).ravel()

    assert len(lat) == len(lon)

    # convert degrees to radians
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)

    # git difference of each successive row
    delta_lon = np.diff(lon)

    # start points (all except last)
    lat_start = lat[:-1]
    # end points (all except first)
    lat_end = lat[1:]

    y = np.sin(delta_lon) * np.cos(lat_end)
    x = np.cos(lat_start) * np.sin(lat_end) - np.sin(lat_start) * np.cos(
        lat_end
    ) * np.cos(delta_lon)

    # calculate track
    track_rad = np.atan2(y, x)
    track_deg = np.rad2deg(track_rad)

    # duplicate the last computed track for the final point, consistent
    # with relative_track_ellipsoid
    track_deg = np.append(track_deg, track_deg[-1])

    # make tracks in range of 0 to 360
    return track_deg % 360


def track(
    data: pd.DataFrame,
    *,
    latitude_column: str,
    longitude_column: str,
    groupby_column: str | None = None,
    progressbar: bool = True,
    ellipsoid: bool = True,
) -> NDArray:
    """
    Calculate the track between each successive row in a dataframe. Track is the angle
    from geographic north (positive clockwise) that and aircraft travels over the
    ground. This is different to the heading or bearing, which is the angle the nose of
    the plane points, which is affected by wind. If groupby_column is provided, the
    dataframe will first be grouped by this.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing the data points and must have columns 'easting' and
        'northing'.
    groupby_column : str | None, optional
        Column name to group by before calculation, by default None
    progressbar : bool, optional
        Show progress bar for each group, by default True

    Returns
    -------
    NDArray
        The track in degrees, in the range 0 to 360, positive clockwise from geographic
        north
    """
    track_func = relative_track_ellipsoid if ellipsoid else relative_track_spheroid

    return _apply_grouped(
        data,
        groupby_column=groupby_column,
        progressbar=progressbar,
        func=lambda d: track_func(
            lat=d[latitude_column].to_numpy(),
            lon=d[longitude_column].to_numpy(),
        ),
    )


def _relative_distance(
    x: NDArray,
    y: NDArray,
) -> NDArray:
    """
    Calculate the relative distance between each successive set of points

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

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

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
    *,
    groupby_column: str | None = None,
    progressbar: bool = True,
) -> NDArray:
    """
    Calculate distance between successive points in a dataframe. This assumes the data
    have been sorted by time, and that there are not flights which overlap in time. If
    there are, you must first sort by flight, then by time. For example, if you have
    columns 'flight' and 'unixtime', you can accomplish this with
    `data = data.sort_values(["flight", "unixtime"])`. If groupby_column is provided,
    the dataframe will first be grouped by this.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe containing the data, must have columns 'easting' and 'northing'.
    groupby_column : str | None, optional
        Column name to group by before calculation, by default None
    progressbar : bool, optional
        Show progress bar for each group, by default True

    Returns
    -------
    NDArray
        Returns an array of the relative distances which can be assigned to a new
        column.
    """
    _check_coord_columns(data)
    col_list = []
    if groupby_column is not None:
        col_list.append(groupby_column)
    assert all(x in data.columns for x in col_list), (
        f"dataframe must contain columns {col_list} "
    )

    return _apply_grouped(
        data,
        groupby_column=groupby_column,
        progressbar=progressbar,
        func=lambda d: _relative_distance(
            d["easting"].to_numpy(), d["northing"].to_numpy()
        ),
    )


def cumulative_distance(
    data: pd.DataFrame,
    *,
    groupby_column: str | None = None,
    progressbar: bool = True,
) -> NDArray:
    """
    Calculate the cumulative distance along track in a dataframe. This assumes the data
    have been sorted by time, and that there are not flights which overlap in time. If
    there are, you can first sort by flight, then by time. For example, if you have
    columns 'flight' and 'unixtime', you can accomplish this with
    `data = data.sort_values(["flight", "unixtime"])`. If groupby_column is provided,
    the dataframe will first be grouped by this.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe containing the data, must have columns 'easting' and 'northing'.
    groupby_column : str | None, optional
        Column name to group by before calculation, by default None
    progressbar : bool, optional
        Show progress bar for each group, by default True

    Returns
    -------
    NDArray
        Returns an array of the cumulative distances which can be assigned to a new
        column.
    """
    _check_coord_columns(data)
    distances = relative_distance(
        data,
        groupby_column=groupby_column,
        progressbar=progressbar,
    )

    if groupby_column is None:
        return np.cumsum(distances)

    # reset the cumulative sum at the start of each group, rather than continuing
    # to accumulate across groups
    return (
        pd.Series(distances, index=data.index)
        .groupby(data[groupby_column])
        .cumsum()
        .to_numpy()
    )


def along_track_distance(
    data: pd.DataFrame,
    *,
    groupby_column: str | None = None,
    progressbar: bool = True,
    guess_start_position: bool = False,
) -> NDArray:
    """
    Calculate the distances along track in meters. This assumes the data have been
    sorted by time, and that there are not flights which overlap in time. If there are,
    you can first sort by flight, then by time. For example, if you have columns
    'flight' and 'unixtime', you can accomplish this with
    `data = data.sort_values(["flight", "unixtime"])`. If groupby_column is provided,
    the dataframe will first be grouped by this.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing the data points to calculate the distance along each line,
        must have columns 'easting' and 'northing', and (if guess_start_position is
        True) a set geometry column.
    groupby_column : str | None, optional
        Column name to group by before calculation, by default None
    progressbar : bool, optional
        Show progress bar for each group, by default True
    guess_start_position: bool, optional
        If True, this will determine the start of the line, not by the first row, but by
        finding the leftmost corner of the line. This is useful if you don't have a time
        column and your data is not sorted by time.

    Returns
    -------
    NDArray
        The along track distance in meters
    """
    _check_coord_columns(data)
    if guess_start_position:
        assert isinstance(data, gpd.GeoDataFrame), (
            "if `guess_start_position` is True, `data` must be a geopandas geodataframe."
        )
        if groupby_column is None:
            # turn point data into line
            line = gpd.GeoSeries(LineString(data.geometry.tolist()))

            # find minimum rotated rectangle around line
            rect = line.iloc[0].minimum_rotated_rectangle

            # get angle of rotation
            angle = azimuth(rect)
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
            horizontal_df["tmp"] = cumulative_distance(
                horizontal_df,
                groupby_column=None,
                progressbar=False,
            )
            horizontal_df = horizontal_df.sort_values("original_index").set_index(
                "original_index"
            )
            return horizontal_df.tmp.loc[data.index].to_numpy()

        data = data.copy()
        for _segment_name, segment_data in _iter_groups(
            data, groupby_column, progressbar
        ):
            # turn point data into line
            line = gpd.GeoSeries(LineString(segment_data.geometry.tolist()))

            # find minimum rotated rectangle around line
            rect = line.iloc[0].minimum_rotated_rectangle

            # get angle of rotation
            angle = azimuth(rect)
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
            horizontal_df["tmp"] = cumulative_distance(
                horizontal_df,
                groupby_column=None,
                progressbar=False,
            )
            horizontal_df = horizontal_df.sort_values("original_index").set_index(
                "original_index"
            )
            data.loc[data[groupby_column] == _segment_name, "tmp"] = horizontal_df.tmp
        return data.tmp.to_numpy()

    return cumulative_distance(
        data,
        groupby_column=groupby_column,
        progressbar=progressbar,
    )


def _azimuth_between_points(
    point1: tuple[float, float],
    point2: tuple[float, float],
) -> float:
    """azimuth between 2 points (interval 0 - 180)"""

    angle = np.arctan2(point2[1] - point1[1], point2[0] - point1[0])
    return np.degrees(angle) if angle > 0 else np.degrees(angle) + 180  # type: ignore[no-any-return]


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    """distance between points"""
    return math.hypot(b[0] - a[0], b[1] - a[1])


def azimuth(mrr) -> float:  # type: ignore[no-untyped-def]
    """azimuth of minimum_rotated_rectangle"""
    bbox = list(mrr.exterior.coords)
    axis1 = _dist(bbox[0], bbox[3])
    axis2 = _dist(bbox[0], bbox[1])

    if axis1 <= axis2:
        az = _azimuth_between_points(bbox[0], bbox[1])
    else:
        az = _azimuth_between_points(bbox[0], bbox[3])

    return az
