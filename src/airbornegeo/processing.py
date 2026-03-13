import math
import typing

import geopandas as gpd
import harmonica as hm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shapely
import verde as vd
from numpy.typing import NDArray
from shapely.geometry import LineString, Point
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


def reduce_by_line(
    df: gpd.GeoDataFrame | pd.DataFrame,
    reduction: typing.Callable[..., float | int],
    spacing: float,
    reduce_by: str | tuple[str, str],
    **kwargs,
) -> pd.DataFrame:
    """
    Reduce data by line based on the column(s) provided by reduce_by and the reduction
    function. For example, if 'reduce_by' is 'distance_along_line', 'spacing' is 1000,
    and 'reduction' is np.mean', then the data will be reduced by taking the mean value
    along every 1 km. If 'reduce_by' is 'unixtime', this would take the mean of every
    1000 seconds. If 'reduce_by' is ('easting', 'northing'), this would take the mean of
    every point in a 1x1 km block. The reduction function can be any function that takes
    an array of values and  returns a single value, such as np.mean, np.median, np.max,
    etc. The function will return a new dataframe with the reduced data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to be reduced.
    reduction : typing.Callable
        function to use in reduction, e.g. np.mean
    spacing : float,
        The spacing to reduce the data by, in the same units as whatever
        column is specified by reduce_by
    reduce_by : str or tuple of str
        Column name(s) to reduce by. If a single column name is provided, the data will
        be reduced by only column. If a tuple of two column names are provided, the data
        will be reduced by both columns, e.g. in 1x1 km blocks if the columns are
        'easting' and 'northing'.
    kwargs : typing.Any
        Any additional keyword arguments to pass to verde.BlockReduce.

    Returns
    -------
    pd.DataFrame
        DataFrame with reduced data.
    """

    df = df.copy()

    # get only numeric columns
    df = df.select_dtypes(include="number")

    if isinstance(reduce_by, str):
        reduce_by = (reduce_by,)

    if len(reduce_by) == 1:
        reduce_by = (reduce_by[0], "tmp")  # add dummy column for second coordinate
        df["tmp"] = 0.0

    assert "line" in df.columns, "'line' column must be in the dataframe"
    assert all(col in df.columns for col in reduce_by), (
        f"{reduce_by} must be in the dataframe"
    )

    # define verde reducer function
    reducer = vd.BlockReduce(
        reduction,
        spacing=spacing,
        **kwargs,
    )

    # get list of data columns to reduce
    input_data_names = tuple(
        df.columns.drop([*list(reduce_by), "line", "geometry"], errors="ignore")
    )

    blocked_dfs = []
    for name, line_df in tqdm(df.groupby("line"), desc="Lines"):
        # get tuples of pd.Series
        input_coords = tuple([line_df[col].to_numpy() for col in reduce_by])  # pylint: disable=consider-using-generator
        input_data = tuple([line_df[col].to_numpy() for col in input_data_names])  # pylint: disable=consider-using-generator

        # apply reduction
        coordinates, data = reducer.filter(
            coordinates=input_coords,
            data=input_data,
        )

        # add reduced coordinates to a dictionary
        coord_cols = dict(zip(reduce_by, coordinates, strict=False))

        # add reduced data to a dictionary
        if len(input_data_names) < 2:
            data_cols = {input_data_names[0]: data}
        else:
            data_cols = dict(zip(input_data_names, data, strict=False))

        # merge dicts and create dataframe
        blocked = pd.DataFrame(data=coord_cols | data_cols)

        blocked["line"] = name
        blocked = blocked.drop(columns=["tmp"], errors="ignore")
        blocked_dfs.append(blocked)

    return pd.concat(blocked_dfs).reset_index(drop=True)


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

    # Create new subline when gap > 1 minute
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


def eotvos_correction_full(
    coordinates: tuple[NDArray, NDArray, NDArray],
    time: NDArray,
) -> pd.Series:
    # ell = bl.WGS84
    # angular_velocity = ell.angular_velocity # radians per second
    # semi_major_axis = ell.semimajor_axis
    # flattening = ell.flattening
    a = 6378137.0
    b = 6356752.3142

    lon, lat, ht = coordinates

    omega = 0.00007292115  # siderial rotation rate, radians/sec
    ecc = (a - b) / a

    latr = np.deg2rad(lat)
    lonr = np.deg2rad(lon)
    lonr = np.unwrap(lonr)

    # get time derivatives of position
    dt = np.diff(time, prepend=np.nan)
    dlat = np.diff(latr, prepend=np.nan) / dt
    dlon = np.diff(lonr, prepend=np.nan) / dt
    dht = np.diff(ht, prepend=np.nan) / dt
    ddlat = np.diff(dlat, prepend=np.nan) / dt
    ddlon = np.diff(dlon, prepend=np.nan) / dt
    ddht = np.diff(dht, prepend=np.nan) / dt

    # sines and cosines etc
    slat = np.sin(latr)
    clat = np.cos(latr)
    s2lat = np.sin(2 * latr)
    c2lat = np.cos(2 * latr)

    # calculate r' and its derivatives
    rp = a * (1 - ecc * slat * slat)
    drp = -a * dlat * ecc * s2lat
    ddrp = -a * ddlat * ecc * s2lat - 2 * a * dlat * dlat * ecc * c2lat

    # calculate deviation from normal and derivatives
    d = np.arctan(ecc * s2lat)
    dd = 2 * dlat * ecc * c2lat
    ddd = 2 * ddlat * ecc * c2lat - 4 * dlat * dlat * ecc * s2lat

    # define r and its derivatives
    r = np.vstack((-rp * np.sin(d), np.zeros(len(rp)), -rp * np.cos(d) - ht)).T
    rdot = np.vstack(
        (
            -drp * np.sin(d) - rp * dd * np.cos(d),
            np.zeros(len(rp)),
            -drp * np.cos(d) + rp * dd * np.sin(d) - dht,
        )
    ).T
    ci = (
        -ddrp * np.sin(d)
        - 2.0 * drp * dd * np.cos(d)
        - rp * (ddd * np.cos(d) - dd * dd * np.sin(d))
    )
    ck = (
        -ddrp * np.cos(d)
        + 2.0 * drp * dd * np.sin(d)
        + rp * (ddd * np.sin(d) + dd * dd * np.cos(d) - ddht)
    )
    rdotdot = np.vstack((ci, np.zeros(len(ci)), ck)).T

    # define w and derivative
    w = np.vstack(((dlon + omega) * clat, -dlat, -(dlon + omega) * slat)).T
    wdot = np.vstack(
        (
            dlon * clat - (dlon + omega) * dlat * slat,
            -ddlat,
            -ddlon * slat - (dlon + omega) * dlat * clat,
        )
    ).T

    w2xrdot = np.cross(2 * w, rdot)
    wdotxr = np.cross(wdot, r)
    wxr = np.cross(w, r)
    wxwxr = np.cross(w, wxr)

    # calculate wexwexre, centrifugal acceleration due to the Earth
    re = np.vstack((-rp * np.sin(d), np.zeros(len(rp)), -rp * np.cos(d))).T
    we = np.vstack((omega * clat, np.zeros(len(slat)), -omega * slat)).T
    wexre = np.cross(we, re)
    _wexwexre = np.cross(we, wexre)
    wexr = np.cross(we, r)
    wexwexr = np.cross(we, wexr)

    # calculate total acceleration for the aircraft
    accel = rdotdot + w2xrdot + wdotxr + wxwxr

    # Eotvos correction is the vertical component of the total acceleration of
    # the aircraft minus the centrifugal acceleration of the Earth, convert to mGal
    return (accel[:, 2] - wexwexr[:, 2]) * 100e3


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


def eotvos_correction(lat_deg, lon_deg, time):
    """
    Compute simple Eötvös correction (mGal) from latitude, longitude, and time.

    Parameters
    ----------
    lat_deg : array-like
        Latitude in degrees
    lon_deg : array-like
        Longitude in degrees
    time : array-like
        Time in seconds

    Returns
    -------
    E : ndarray
        Eötvös correction in mGal
    """
    v_east = eastward_velocity(lat_deg, lon_deg, time)
    lat_rad = np.radians(lat_deg)

    omega = 7.292115e-5  # Earth's rotation rate, rad/s
    r = 6371000  # Earth radius in meters

    # Classical Eötvös formula
    e = 2 * omega * v_east * np.cos(lat_rad) + (v_east**2) / r

    # Convert to mGal
    return e * 1e5


def relative_distance(
    df: pd.DataFrame,
    reverse: bool = False,
) -> pd.DataFrame:
    """
    calculate distance between x,y points in a dataframe, relative to the previous row.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing columns x and y in meters.
    reverse : bool, optional,
        choose whether to reverse the profile, by default is False

    Returns
    -------
    pandas.DataFrame
        Returns original dataframe with additional column rel_dist
    """
    df = df.copy()
    if reverse is True:
        df1 = df[::-1].reset_index(drop=True)
    elif reverse is False:
        df1 = df.copy()  # .reset_index(drop=True)

    # from https://stackoverflow.com/a/75824992/18686384
    df1["x_lag"] = df1.easting.shift(1)  # pylint: disable=used-before-assignment
    df1["y_lag"] = df1.northing.shift(1)
    df1["rel_dist"] = np.sqrt(
        (df1.easting - df1["x_lag"]) ** 2 + (df1.northing - df1["y_lag"]) ** 2
    )
    # set first row distance to 0
    df1.loc[0, "rel_dist"] = 0
    df1 = df1.drop(["x_lag", "y_lag"], axis=1)
    return df1.dropna(subset=["rel_dist"])


def distance_along_flight(
    df: pd.DataFrame,
    flight_col_name: str = "flight",
    time_col_name: str | None = "unixtime",
    **kwargs: typing.Any,
) -> pd.Series:
    """
    Calculate the distance along each flight in meters using the time column to sort the
    data points.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the flight data.
    flight_col_name : str, optional
        Column name for the flight, by default "flight"
    time_col_name : str | None, optional
        Column name for the time, by default "unixtime"

    Returns
    -------
    pd.Series
        Series containing the distance along each flight in meters
    """
    reverse = kwargs.get("reverse", False)
    df = df.copy()

    groups = df.groupby(flight_col_name)

    dfs = []
    for _name, flight in groups:
        flight_sorted = flight.sort_values(by=[time_col_name]).reset_index(drop=True)
        flight_sorted = relative_distance(flight_sorted, reverse=reverse)
        dist = flight_sorted.rel_dist.cumsum()
        flight_sorted["dist_along_flight"] = dist
        # df.loc[df[flight_col_name] == i, "dist_along_flight"] = dist
        dfs.append(flight_sorted)

    df = pd.concat(dfs).reset_index(drop=True).sort_values(by=[flight_col_name])

    return df.dist_along_flight


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
    gdf: gpd.GeoDataFrame,
) -> pd.Series:
    """
    Calculate the average bearing of each line in a GeoDataFrame. The bearing is
    calculated by finding the minimum rotated rectangle around the line, and then
    calculating the angle of the rectangle.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Dataframe containing the data points and the line labels.
        must have a set geometry column.

    Returns
    -------
    pd.Series
        The bearing of each line in degrees
    """

    gdf = gdf.copy()

    assert isinstance(gdf, gpd.GeoDataFrame), "gdf must be a GeoDataFrame"
    assert gdf.geometry.geom_type.isin(["Point"]).all(), "geometry must be points"
    assert "line" in gdf.columns, "line column must be in dataframe"

    grouped = gdf.groupby("line")

    gdf["bearing"] = np.nan

    for name, data in grouped:
        # turn point data into line
        line = gpd.GeoSeries(LineString(data.geometry.tolist()))

        # find minimum rotated rectangle around line
        rect = line.iloc[0].minimum_rotated_rectangle

        # get angle of rotation
        bearing = azimuth(rect)
        if 90 < bearing <= 180:
            bearing = bearing - 180

        gdf.loc[gdf.line == name, "bearing"] = bearing

    return gdf.bearing


def distance_along_line(
    gdf: gpd.GeoDataFrame,
    time_col_name: str | None = None,
) -> pd.Series:
    """
    Calculate the distances along each flight line in meters. If 'time_col_name' is
    provided, this will inform which end of the line is the beginning. If not, the line
    will be rotate horizontally, and the left-most side will be used as the start.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Dataframe containing the data points to calculate the distance along each line,
        must have a set geometry column.
    time_col_name : str | None, optional
        Column name containing time in seconds for each datapoint, by default None

    Returns
    -------
    pd.Series
        The distance along each line in meters
    """

    gdf = gdf.copy()

    assert "line" in gdf.columns, "line column must be in dataframe"

    grouped = gdf.groupby("line")

    gdf["dist_along_line"] = np.nan

    for name, data in grouped:
        # turn point data into line
        line = gpd.GeoSeries(LineString(data.geometry.tolist()))
        df = line.get_coordinates(
            index_parts=True,
            ignore_index=True,
        )

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
        horizontal_df["original_index"] = horizontal_df.index
        horizontal_df = horizontal_df.sort_values("x").reset_index(drop=True)

        # it time provided, use to determine which side is start of line
        # if not, use minimum x (left) as start
        if time_col_name is not None:
            start_arg = data[time_col_name].argmin()
            end_arg = data[time_col_name].argmax()
            start_coords = shapely.get_coordinates(data.iloc[start_arg].geometry)[0]
            end_coords = shapely.get_coordinates(data.iloc[end_arg].geometry)[0]
            logger.debug(
                "From time column, line starts at %s and ends at %s",
                start_coords,
                end_coords,
            )
            # determine which side of rotated line is the start
            start_arg = df[(df.x == start_coords[0]) & (df.y == start_coords[1])].index[
                0
            ]

            horizontal_start_arg = horizontal_df[
                horizontal_df.original_index == start_arg
            ].index

            if horizontal_start_arg > len(df) / 2:
                h_start_arg = horizontal_df.x.argmax()
                h_end_arg = horizontal_df.x.argmin()
            else:
                h_start_arg = horizontal_df.x.argmin()
                h_end_arg = horizontal_df.x.argmax()
            start_arg = int(horizontal_df.iloc[h_start_arg].original_index)
            end_arg = int(horizontal_df.iloc[h_end_arg].original_index)
            start_coords = df.iloc[start_arg][["x", "y"]].to_numpy()
            end_coords = df.iloc[end_arg][["x", "y"]].to_numpy()

            logger.debug("Line starts at %s and ends at %s", start_coords, end_coords)
        else:
            # get start and end points of line
            start_arg = horizontal_df.x.argmin()
            end_arg = horizontal_df.x.argmax()
            start_coords = df.iloc[start_arg][["x", "y"]].to_numpy()
            end_coords = df.iloc[end_arg][["x", "y"]].to_numpy()
            logger.debug(
                "Assuming line starts at %s and ends at %s", start_coords, end_coords
            )

        # calculate distance along line from starting point
        dist = data.distance(Point(*start_coords))
        gdf.loc[gdf.line == name, "dist_along_line"] = dist

    if time_col_name is not None:
        grouped = gdf.groupby(["line"])
        for name, data in grouped:
            df = (
                data.sort_values(by=[time_col_name])
                .dropna(subset=[time_col_name])
                .reset_index(drop=True)
            )
            assert df.iloc[0].dist_along_line < df.iloc[-1].dist_along_line, (
                f"line {name} is not in order; {df}"
            )
            df = (
                data.sort_values(by="dist_along_line")
                .dropna(subset=[time_col_name])
                .reset_index(drop=True)
            )
            assert df.iloc[0][time_col_name] < df.iloc[-1][time_col_name]

    return gdf.dist_along_line


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
