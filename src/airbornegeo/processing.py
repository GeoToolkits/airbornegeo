import math

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from shapely.geometry import LineString
from tqdm.autonotebook import tqdm

from airbornegeo import logger

sns.set_theme()


def split_into_segments(
    data: pd.DataFrame,
    threshold: float,
    column_name: str,
    min_points_per_segment: int = 0,
) -> pd.Series:
    """
    Split dataframe into segments where there is a gap in the supplied values greater
    than the threshold. Data are sorted by column 'unixtime'. The values are chosen
    with column_name, and could be quantities such as time in seconds, cumulative
    distances, or aircraft bearings.

    Parameters
    ----------
    data : pd.DataFrame
        _description_
    threshold : float
        _description_, by default None
    column_name: str
        Name of column supplying to data.
    min_points_per_segment: int or None
        Segments with fewer points are giving a segment id of NaN, by default 0

    Returns
    -------
    pd.Series
        A series with new new segments identified with integers
    """
    df = data.copy()

    col_list = ["unixtime", column_name]
    assert all(x in df.columns for x in col_list), (
        f"dataframe must contain columns {col_list} "
    )

    # save index, sort by time and reset index
    df = df.reset_index(names="tmp_index")
    df = df.sort_values(by="unixtime").reset_index(drop=True)

    # Calculate difference between each point
    df["diff"] = df[column_name].diff()

    # Create new segment when gap > distance_threshold
    df["segment"] = (df["diff"] > threshold).cumsum()

    # remove segments which are less than specified number of points
    if min_points_per_segment > 0:
        groups = df.groupby("segment")
        prior_len = len(df.segment.unique())
        # make segment ID nan for small segments
        small_segments = groups.filter(lambda x: len(x) < min_points_per_segment)
        df.loc[small_segments.index, "segment"] = np.nan
        post_len = len(df.segment.unique())
        logger.info(
            "dropped %s segments which contained less than %s points.",
            prior_len - post_len,
            min_points_per_segment,
        )

    # Reset index and sort
    df = df.set_index("tmp_index").sort_values("tmp_index")

    return df.segment


def vertical_acceleration(
    df: pd.DataFrame,
    time_threshold: float,
    flight_col_name: str = "flight",
    time_col_name: str | None = "unixtime",
    height_col_name: str = "height",
    smoothing_window: int = 3,
) -> pd.Series:
    """
    Calculate the 2nd derivative of height change with respect to time for each line.
    If there is a gap between points greater than time_threshold in seconds, the line
    will be split at the gap and the acceleration will be NaN for the points on
    either side of the gap.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the flight data.
    time_threshold : float
        Threshold in seconds for determining gaps in the data, where acceleration will be set to NaN
    flight_col_name : str, optional
        Column name for the flight, by default "flight"
    time_col_name : str | None, optional
        Column name for the time, by default "unixtime"
    height_col_name : str, optional
        Column name for the height, by default "height"
    smoothing_window : int, optional
        Window size in number of data points for smoothing the height data before
        calculating acceleration, and smoothing teach derivative after it's been
        calculated, by default 3
    Returns
    -------
    pd.Series
        Series containing the vertical acceleration in m/s^2 for each point in the dataframe.
    """
    df = df.copy()

    # split flights into segments where there is a gap in time greater than
    # time_threshold
    # Calculate time difference within each Line
    df["time_diff"] = pd.to_timedelta(
        df.groupby(flight_col_name)[time_col_name].diff(), unit="s"
    )

    # Create new subline when gap > time_threshold
    df["new_group"] = (df["time_diff"] > pd.Timedelta(seconds=time_threshold)).astype(
        int
    )

    # Cumulative sum per Line to generate subline number
    df["segment"] = df.groupby(flight_col_name)["new_group"].cumsum()

    # Drop helper columns if not needed
    df = df.drop(columns=["time_diff", "new_group"])

    # # Smooth the height data before calculating derivatives
    # df[height_col_name] = (
    #     df.groupby([flight_col_name, 'segment'])[height_col_name]
    #     .rolling(smoothing_window, center=True, min_periods=1)
    #     .mean()
    #     .reset_index(level=[0,1], drop=True)
    # )

    # df['first_derivative'] = (
    #     df.groupby([flight_col_name, 'segment'])
    #     .apply(lambda g: g[height_col_name].diff() / g[time_col_name].diff())
    #     .reset_index(level=[0,1], drop=True)
    # )

    # df['second_derivative'] = (
    #     df.groupby([flight_col_name, 'segment'])
    #     .apply(lambda g: g['first_derivative'].diff() / g[time_col_name].diff())
    #     .reset_index(level=[0,1], drop=True)
    # )

    df["dt"] = df.groupby([flight_col_name, "segment"])[time_col_name].diff()
    df["dv"] = df.groupby([flight_col_name, "segment"])[height_col_name].diff()

    df["first_derivative"] = df["dv"] / df["dt"]

    # # smooth the first derivative
    # df['first_derivative'] = (
    #     df.groupby([flight_col_name, 'segment'])['first_derivative']
    #     .rolling(smoothing_window, center=True, min_periods=1)
    #     .mean()
    #     .reset_index(level=[0,1], drop=True)
    # )

    df["d_first"] = df.groupby([flight_col_name, "segment"])["first_derivative"].diff()

    df["second_derivative"] = df["d_first"] / df["dt"]

    # smooth the second derivative
    df["second_derivative"] = (
        df.groupby([flight_col_name, "segment"])["second_derivative"]
        .rolling(smoothing_window, center=True, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    df = df.drop(columns=["dt", "dv", "d_first"])

    return df.second_derivative

    # dfs = []
    # for _name, segment in df.groupby([flight_col_name, "segment"]):
    #     dt = segment.unixtime.diff().dt.seconds.values
    #     first_derivative = segment.height.diff().div(dt, axis=0)
    #     second_derivative = first_derivative.diff().div(dt, axis=0)

    #     segment["first_derivative"] = first_derivative
    #     segment["second_derivative"] = second_derivative

    #     dfs.append(segment_sorted)

    # df = pd.concat(dfs).reset_index(drop=True).sort_values(by=[flight_col_name])

    # return df.dist_along_flight


def eastward_velocity(lat_deg, lon_deg, time):
    """
    Compute eastward velocity (m/s) from latitude, longitude, and time.

    Parameters
    ----------
    lat_deg : array-like
        Latitude in degrees
    lon_deg : array-like
        Longitude in degrees
    time : array-like
        Time in seconds (difference between consecutive measurements should be ~1-10 s)

    Returns
    -------
    v_east : ndarray
        Eastward velocity in m/s
    """
    lat_rad = np.radians(np.asarray(lat_deg))
    lon_rad = np.unwrap(np.radians(np.asarray(lon_deg)))  # avoid ±180 jumps

    time = np.asarray(time)
    dt = np.diff(time, prepend=np.nan)
    dt[dt == 0] = np.nan  # avoid division by zero

    dlon = np.diff(lon_rad, prepend=np.nan)

    r = 6371000  # Earth radius in meters
    return r * np.cos(lat_rad) * dlon / dt


def northward_velocity(lat_deg, lon_deg, time):
    """
    Compute northward velocity (m/s) from latitude, longitude, and time.

    Parameters
    ----------
    lat_deg : array-like
        Latitude in degrees
    lon_deg : array-like
        Longitude in degrees
    time : array-like
        Time in seconds (difference between consecutive measurements should be ~1-10 s)

    Returns
    -------
    v_north : ndarray
        Northward velocity in m/s
    """
    lat_rad = np.radians(np.asarray(lat_deg))
    _lon_rad = np.unwrap(np.radians(np.asarray(lon_deg)))  # avoid ±180 jumps

    time = np.asarray(time)
    dt = np.diff(time, prepend=np.nan)
    dt[dt == 0] = np.nan  # avoid division by zero

    dlat = np.diff(lat_rad, prepend=np.nan)

    r = 6371000  # Earth radius in meters
    return r * dlat / dt


def unique_line_id(
    df: pd.DataFrame,
    line_col_name: str = "line",
) -> pd.Series:
    """
    Convert supplied lines names into integers.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Dataframe containing the data points and the line labels.
        must have a set geometry column.
    line_col_name : str, optional
        Column name specifying the line number, by default "line"

    Returns
    -------
    pd.Series
        The line names for each point in the GeoDataFrame
    """
    df1 = df.copy()

    line_names = list(df1[line_col_name].unique())

    line_series = df1[line_col_name]
    df1[line_col_name] = np.nan

    for i, n in enumerate(line_names):
        df1.loc[line_series == n, line_col_name] = int(i + 1)

    return df1[line_col_name].astype(int)


def line_bearing(
    data: gpd.GeoDataFrame,
    groupby_column: str = "line",
) -> pd.Series:
    """
    Calculate the average bearing of each line in a GeoDataFrame. The bearing is
    calculated by finding the minimum rotated rectangle around the line, and then
    calculating the angle of the rectangle.

    Parameters
    ----------
    data : gpd.GeoDataFrame
        Dataframe containing the data points and the line labels.
        must have a set geometry column.
    groupby_column: str, optional
        Column to group the dataframe by, by default "line"

    Returns
    -------
    pd.Series
        The bearing of each line in degrees
    """

    data = data.copy()

    assert isinstance(data, gpd.GeoDataFrame), "gdf must be a GeoDataFrame"
    assert data.geometry.geom_type.isin(["Point"]).all(), "geometry must be points"
    assert groupby_column in data.columns, "line column must be in dataframe"

    data["bearing"] = np.nan

    for segment_name, segment_data in tqdm(
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

        data.loc[data[groupby_column] == segment_name, "bearing"] = angle

    return data.bearing


def bearing(data: gpd.GeoDataFrame, window_width: float) -> pd.Series:
    """
    Calculate the average bearing of a moving window in a GeoDataFrame. The bearing is
    calculated by finding the minimum rotated rectangle around the window, and then
    calculating the angle of the rectangle.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Dataframe containing the data points and must have a set geometry column.

    Returns
    -------
    pd.Series
        The bearing in degrees
    """
    data = data.copy()

    # save index, sort by time and reset index
    data = data.reset_index(names="tmp_index")

    # # do with pandas, much slower
    # def angle(series):
    #     view = data.iloc[series.index]
    #     delta_x = view.easting.iloc[-1] - view.easting.iloc[0]
    #     delta_y = view.northing.iloc[-1] - view.northing.iloc[0]
    #     return np.rad2deg(np.arctan2(delta_y, delta_x))
    # return data[['easting','northing']].rolling(window_width).apply(angle)['easting']

    # do with numpy, much faster
    # create sliding windows of data
    windows = np.lib.stride_tricks.sliding_window_view(
        data[["easting", "northing"]].values,
        window_width,
        axis=0,
    )

    def angle(window):
        delta_x = window[0][-1] - window[0][0]
        delta_y = window[1][-1] - window[1][0]
        return np.rad2deg(np.arctan2(delta_y, delta_x))

    angles = []
    for x in windows:
        # get angle of rotation
        ang = angle(x)
        # if 90 < ang <= 180:
        #     ang = ang - 180
        angles.append(ang)

    data["tmp_bearing"] = np.pad(
        angles, (0, len(data) - len(windows)), constant_values=np.nan
    )

    # Reset index and sort
    data = data.set_index("tmp_index").sort_values("tmp_index")

    return data.tmp_bearing


def detect_outliers(df: pd.DataFrame) -> None:
    """
    Detects outliers in each column of a Pandas DataFrame using the IQR method
    and visualizes them using box plots.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    """
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            q1 = df[column].quantile(0.25)
            q3 = df[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = (df[column] < lower_bound) | (df[column] > upper_bound)

            if outliers.any():
                plt.figure(figsize=(8, 6))
                sns.boxplot(x=df[column])
                plt.title(f"Boxplot of {column} (Outliers Detected)")
                plt.show()
            else:
                logger.info("No outliers detected in column: %s", column)


# def detect_outliers(
#     df,
#     zscore_threshold: float = 4,
# ):
#     df = df.copy()

#     # df['index'] = df.index
#     df = df.dropna(axis=1, how="any")

#     stats_df = df.apply(scipy.stats.zscore)

#     outliers = stats_df[(stats_df > zscore_threshold).any(axis=1)]
#     # cols = outliers.columns.tolist()
#     # outliers = outliers.merge(df[['index']], left_index=True, right_index=True, how="left")
#     # outliers = outliers[["index"]+cols]

#     return outliers


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
