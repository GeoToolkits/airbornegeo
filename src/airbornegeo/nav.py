import math

import geopandas as gpd
import numpy as np
import pandas as pd
import seaborn as sns
import shapely
from geographiclib.geodesic import Geodesic
from numpy.typing import NDArray
from shapely.geometry import LineString
from tqdm.autonotebook import tqdm

sns.set_theme()


def directional_velocity(
    data: pd.DataFrame,
    *,
    time_column: str = "unixtime",
    coordinate_column: str = "easting",
    groupby_column: str | None = None,
) -> pd.Series:
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

    Returns
    -------
    pd.Series
        The velocity component in units of the provided coordinate and time columns
    """
    data = data.copy()

    if groupby_column is None:
        return np.gradient(data[coordinate_column], data[time_column])

    # iterate through groups, append speeds, and concat
    groups = []
    for _segment_name, segment_data in tqdm(
        data.groupby(groupby_column), desc="Segments"
    ):
        groups.append(
            np.gradient(segment_data[coordinate_column], segment_data[time_column])
        )

    return np.concatenate(groups)


def ground_speed(
    data: pd.DataFrame,
    *,
    time_column: str = "unixtime",
    easting_column: str = "easting",
    northing_column: str = "northing",
    groupby_column: str | None = None,
) -> pd.Series:
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
        Dataframe containing the data points to calculate the ground speed for,
        must have columns 'unixtime' and 'relative_distance'.
    time_column : str
        name of the column containing the time in seconds
    easting_column : str
        name of the column containing the easting coordinates in meters
    northing_column : str
        name of the column containing the northing coordinates in meters
    groupby_column : str | None, optional
        Column name to group by before calculation, by default None

    Returns
    -------
    pd.Series
        The groundspeed in units of meters per second
    """
    data = data.copy()

    data["cumulative_distance"] = cumulative_distance(
        data,
        easting_column=easting_column,
        northing_column=northing_column,
        groupby_column=groupby_column,
    )

    if groupby_column is None:
        return np.gradient(data.cumulative_distance, data[time_column])

    # iterate through groups, append speeds, and concat
    groups = []
    for _segment_name, segment_data in tqdm(
        data.groupby(groupby_column), desc="Segments"
    ):
        groups.append(
            np.gradient(segment_data.cumulative_distance, segment_data[time_column])
        )
    return np.concatenate(groups)


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
    time_threshold: float | None = None,
    smoothing_window: int | None = None,
) -> pd.Series:
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
    time_threshold : float
        Threshold in seconds for determining gaps in the data, where acceleration will be set to NaN
    smoothing_window : int, optional
        Window size in number of data points for smoothing each derivative after it's
        been calculated, by default None

    Returns
    -------
    pd.Series
        Series containing the vertical acceleration in m/s^2 for each point in the
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

    if (groupby_column is not None) and (time_threshold is not None):
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

    if groupby_column is None:
        times = data[time_column]
        heights = data[height_column]
        vertical_vel = np.gradient(heights, times)
        vertical_accel = pd.Series(np.gradient(vertical_vel, times))

        if smoothing_window is not None:
            return vertical_accel.rolling(window=smoothing_window, min_periods=1).mean()
        return vertical_accel

    # iterate through groups, append accelerations, and concat
    accels = []
    for _segment_name, segment_data in tqdm(
        data.groupby(groupby_column), desc="Segments"
    ):
        times = segment_data[time_column]
        heights = segment_data[height_column]
        vertical_vel = np.gradient(heights, times)
        vertical_accel = pd.Series(np.gradient(vertical_vel, times))

        if smoothing_window is not None:
            vertical_accel = vertical_accel.rolling(
                window=smoothing_window, min_periods=1
            ).mean()
        accels.append(vertical_accel)

    return np.concatenate(accels)


def relative_track_ellipsoid(
    lat: NDArray,
    lon: NDArray,
) -> NDArray:
    """
    Calculate the track between each successive set of points in degrees clockwise from
    geographic north in the range 180 to -180. This uses the WGS84 ellipsoid, make the
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

    return np.array(tracks)


def relative_track_spheroid(
    lat: NDArray,
    lon: NDArray,
) -> NDArray:
    """
    Calculate the track between each successive set of points in degrees clockwise from
    geographic north in the range 180 to -180. This assumes the Earth is a sphere, which
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
    return np.insert(track_deg, 0, np.nan)


def track(
    data: pd.DataFrame,
    *,
    latitude_column: str,
    longitude_column: str,
    groupby_column: str | None = None,
    ellipsoid: bool = True,
) -> pd.Series:
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

    Returns
    -------
    pd.Series
        The track in degrees, -180 to 180, positive clockwise from geographic north
    """
    track_func = relative_track_ellipsoid if ellipsoid else relative_track_spheroid

    if groupby_column is None:
        return track_func(
            lat=data[latitude_column].values,
            lon=data[longitude_column].values,
        )

    # iterate through groups, append tracks, and concat
    tracks = []
    for _segment_name, segment_data in tqdm(
        data.groupby(groupby_column), desc="Segments"
    ):
        calc_track = track_func(
            lat=segment_data[latitude_column].values,
            lon=segment_data[longitude_column].values,
        )
        tracks.append(calc_track)
    return np.concatenate(tracks)


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
    easting_column: str = "easting",
    northing_column: str = "northing",
    groupby_column: str | None = None,
) -> pd.Series:
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
        Dataframe containing the data.
    easting_column : str
        name of the column containing the easting coordinates in meters
    northing_column : str
        name of the column containing the northing coordinates in meters
    groupby_column : str | None, optional
        Column name to group by before calculation, by default None

    Returns
    -------
    pandas.Series
        Returns a pandas Series of the relative distances which can be assigned to a new
        column.
    """
    col_list = [easting_column, northing_column]
    if groupby_column is not None:
        col_list.append(groupby_column)
    assert all(x in data.columns for x in col_list), (
        f"dataframe must contain columns {col_list} "
    )

    if groupby_column is None:
        return _relative_distance(
            data[easting_column].values, data[northing_column].values
        )

    # iterate through groups, append distances, and concat
    dists = []
    for _segment_name, segment_data in tqdm(
        data.groupby(groupby_column), desc="Segments"
    ):
        distances = _relative_distance(
            segment_data[easting_column].values, segment_data[northing_column].values
        )
        dists.append(distances)
    return np.concatenate(dists)


def cumulative_distance(
    data: pd.DataFrame,
    *,
    easting_column: str = "easting",
    northing_column: str = "northing",
    groupby_column: str | None = None,
) -> pd.Series:
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
        Dataframe containing the data.
    easting_column : str
        name of the column containing the easting coordinates in meters
    northing_column : str
        name of the column containing the northing coordinates in meters
    groupby_column : str | None, optional
        Column name to group by before calculation, by default None

    Returns
    -------
    pandas.Series
        Returns a pandas Series of the relative distances which can be assigned to a new
        column.
    """
    return relative_distance(
        data,
        easting_column=easting_column,
        northing_column=northing_column,
        groupby_column=groupby_column,
    ).cumsum()


def along_track_distance(
    data: pd.DataFrame,
    *,
    easting_column: str = "easting",
    northing_column: str = "northing",
    groupby_column: str | None = None,
    guess_start_position: bool = False,
) -> pd.Series:
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
        must have a set geometry column.
    easting_column : str
        name of the column containing the easting coordinates in meters
    northing_column : str
        name of the column containing the northing coordinates in meters
    groupby_column : str | None, optional
        Column name to group by before calculation, by default None
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
            horizontal_df["tmp"] = cumulative_distance(
                horizontal_df,
                easting_column="x",
                northing_column="y",
                groupby_column=None,
            )
            horizontal_df = horizontal_df.sort_values("original_index").set_index(
                "original_index"
            )
            return horizontal_df.tmp

        data = data.copy()
        for _segment_name, segment_data in tqdm(
            data.groupby(groupby_column), desc="Segments"
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
            horizontal_df["tmp"] = cumulative_distance(
                horizontal_df,
                easting_column="x",
                northing_column="y",
                groupby_column=None,
            )
            horizontal_df = horizontal_df.sort_values("original_index").set_index(
                "original_index"
            )
            data.loc[data[groupby_column] == _segment_name, "tmp"] = horizontal_df.tmp
        return data.tmp

    if groupby_column is None:
        return cumulative_distance(
            data,
            easting_column=easting_column,
            northing_column=northing_column,
            groupby_column=None,
        )

    # iterate through groups, append distances, and concat
    groups = []
    for _segment_name, segment_data in data.groupby(groupby_column):
        groups.append(
            cumulative_distance(
                segment_data,
                easting_column=easting_column,
                northing_column=northing_column,
                groupby_column=None,
            )
        )
    return np.concatenate(groups)


# def eastward_velocity(lat_deg, lon_deg, time):
#     """
#     Compute eastward velocity (m/s) from latitude, longitude, and time.

#     Parameters
#     ----------
#     lat_deg : array-like
#         Latitude in degrees
#     lon_deg : array-like
#         Longitude in degrees
#     time : array-like
#         Time in seconds (difference between consecutive measurements should be ~1-10 s)

#     Returns
#     -------
#     v_east : ndarray
#         Eastward velocity in m/s
#     """
#     lat_rad = np.radians(np.asarray(lat_deg))
#     lon_rad = np.unwrap(np.radians(np.asarray(lon_deg)))  # avoid ±180 jumps

#     time = np.asarray(time)
#     dt = np.diff(time, prepend=np.nan)
#     dt[dt == 0] = np.nan  # avoid division by zero

#     dlon = np.diff(lon_rad, prepend=np.nan)

#     r = 6371000  # Earth radius in meters
#     return r * np.cos(lat_rad) * dlon / dt


# def northward_velocity(lat_deg, lon_deg, time):
#     """
#     Compute northward velocity (m/s) from latitude, longitude, and time.

#     Parameters
#     ----------
#     lat_deg : array-like
#         Latitude in degrees
#     lon_deg : array-like
#         Longitude in degrees
#     time : array-like
#         Time in seconds (difference between consecutive measurements should be ~1-10 s)

#     Returns
#     -------
#     v_north : ndarray
#         Northward velocity in m/s
#     """
#     lat_rad = np.radians(np.asarray(lat_deg))
#     _lon_rad = np.unwrap(np.radians(np.asarray(lon_deg)))  # avoid ±180 jumps

#     time = np.asarray(time)
#     dt = np.diff(time, prepend=np.nan)
#     dt[dt == 0] = np.nan  # avoid division by zero

#     dlat = np.diff(lat_rad, prepend=np.nan)

#     r = 6371000  # Earth radius in meters
#     return r * dlat / dt


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
