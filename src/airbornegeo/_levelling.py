from __future__ import annotations

import copy
import itertools
import math
import warnings

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import pygmt
import scipy
import seaborn as sns
import shapely
import verde as vd
from invert4geom import utils as invert4geom_utils
from IPython.display import clear_output
from polartoolkit import utils
from shapely.geometry import LineString, Point
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from tqdm.autonotebook import tqdm
from sklearn.pipeline import Pipeline
from airbornegeo import logger

sns.set_theme()


# def points_in_polygon(
#     df: pd.DataFrame | gpd.GeoDataFrame,
#     polygon: gpd.GeoDataFrame,
#     coord_cols: tuple[str, str] = ('easting', 'northing')
# ):
#      df = df.copy()
#     if isinstance(df, pd.DataFrame):
#         df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[[coord_cols]]))




def detect_outliers(df):
    """
    Detects outliers in each column of a Pandas DataFrame using the IQR method
    and visualizes them using box plots.
    Args:
        df (pd.DataFrame): The input DataFrame.
    """
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = (df[column] < lower_bound) | (df[column] > upper_bound)

            if outliers.any():
              plt.figure(figsize=(8, 6))
              sns.boxplot(x=df[column])
              plt.title(f'Boxplot of {column} (Outliers Detected)')
              plt.show()
            else:
              print(f'No outliers detected in column: {column}')


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


def calculate_intersection_weights(
    inters,
    gdf,
    weight_by: str,
    max_dist_weight: float | None = None,
    max_dist_floor: float | None = None,
    interpolation_type_weight: float | None = None,
    height_difference_weight: float | None = None,
    height_difference_floor: float | None = None,
    data_1st_derive_weight: float | None = None,
    data_1st_derive_floor: float | None = None,
    data_1st_derive_col_name: str | None = None,
    data_2nd_derive_weight: float | None = None,
    data_2nd_derive_floor: float | None = None,
    data_2nd_derive_col_name: str | None = None,
    height_1st_derive_weight: float | None = None,
    height_1st_derive_floor: float | None = None,
    height_1st_derive_col_name: str | None = None,
    height_2nd_derive_weight: float | None = None,
    height_2nd_derive_floor: float | None = None,
    height_2nd_derive_col_name: str | None = None,
    line_col_name: str = "line",
    height_col_name: str = "height",
    plot=False,
):
    inters = inters.copy()
    gdf = gdf.copy()

    # get list of lines from inters
    lines = [*inters.line.unique(), *inters.tie.unique()]

    # subset data based on lines
    gdf = gdf[gdf[line_col_name].isin(lines)]


    if weight_by == "line":
        pass
    elif weight_by == "tie":
        pass
    elif weight_by == "all":
        pass
    else:
        raise ValueError("weight_by must be 'line', 'tie', or 'all'")

    weights_cols = []
    weights_dict = {}
    plot_cols = []
    if max_dist_weight is not None:
        weight_vals = inters.max_dist
        if max_dist_floor is not None:
            weight_vals = np.where(
                weight_vals < max_dist_floor,
                max_dist_floor,
                weight_vals,
            )
        inters["max_dist_weight"] = weight_vals

        if weight_by == "all":
            inters["max_dist_weight"] = normalize_values(
                inters["max_dist_weight"],
                low=1, high=0.001, # reversed so large distances are bad
                # quantiles=(0.02, 0.98),
            )
        else:
            inters["max_dist_weight"] = inters.groupby(weight_by)["max_dist_weight"].transform(
                lambda x: normalize_values(
                    x,
                    low=1, high=0.001, # reversed so large distances are bad
                    # quantiles=(0.02, 0.98),
                )
            )

        weights_cols.append("max_dist_weight")
        weights_dict["max_dist_weight"] = max_dist_weight
        plot_cols.append("max_dist")

    if height_difference_weight is not None:
        # find height at intersection for line and tie
        for ind, row in inters.iterrows():
            # search data for values at intersecting lines
            line_value = gdf[
                (gdf[line_col_name] == row.line) & (gdf.intersecting_line == row.tie)
            ][height_col_name].values[0]
            tie_value = gdf[
                (gdf[line_col_name] == row.tie) & (gdf.intersecting_line == row.line)
            ][height_col_name].values[0]
            inters.loc[ind, "flight_height"] = line_value
            inters.loc[ind, "tie_height"] = tie_value
        inters["height_difference"] = np.abs(inters.flight_height - inters.tie_height)

        weight_vals = inters.height_difference

        if height_difference_floor is not None:
            weight_vals = np.where(
                weight_vals < height_difference_floor,
                height_difference_floor,
                weight_vals,
            )
        inters["height_difference_weight"] = weight_vals

        if weight_by == "all":
            inters["height_difference_weight"] = normalize_values(
                inters["height_difference_weight"],
                low=1, high=0.001, # reversed so large differences are bad
                # quantiles=(0.02, 0.98),
            )
        else:
            inters["height_difference_weight"] = inters.groupby(weight_by)["height_difference_weight"].transform(
                lambda x: normalize_values(x,
                low=1, high=0.001, # reversed so large differences are bad
                # quantiles=(0.02, 0.98),
                )
            )

        weights_cols.append("height_difference_weight")
        weights_dict["height_difference_weight"] = height_difference_weight
        plot_cols.append("height_difference")

    if interpolation_type_weight is not None:
        cond = inters == "extrapolated"
        inters["number_of_extrapolations"] = cond.sum(axis=1)
        inters["interpolation_type_weight"] = inters.number_of_extrapolations

        if weight_by == "all":
            inters["interpolation_type_weight"] = normalize_values(
                inters["interpolation_type_weight"],
                low=1, high=0.001, # reversed so large numbers of extrapolations are bad
                # quantiles=(0.02, 0.98),
            )
        else:
            inters["interpolation_type_weight"] = inters.groupby(weight_by)["interpolation_type_weight"].transform(
                lambda x: normalize_values(x,
                low=1, high=0.001, # reversed so large numbers of extrapolations are bad
                # quantiles=(0.02, 0.98),
                )
            )

        weights_cols.append("interpolation_type_weight")
        weights_dict["interpolation_type_weight"] = interpolation_type_weight
        plot_cols.append("number_of_extrapolations")

    if data_1st_derive_weight is not None:
        if data_1st_derive_col_name is None:
            raise ValueError(
                f"must provide 'data_1st_derive_col_name'"
            )
        # find data gradient at intersection for line and tie
        for ind, row in inters.iterrows():
            # search data for values at intersecting lines
            line_value = gdf[
                (gdf[line_col_name] == row.line) & (gdf.intersecting_line == row.tie)
            ][data_1st_derive_col_name].values[0]
            tie_value = gdf[
                (gdf[line_col_name] == row.tie) & (gdf.intersecting_line == row.line)
            ][data_1st_derive_col_name].values[0]
            inters.loc[ind, "data_1st_derive"] = np.mean(np.abs([line_value, tie_value]))
        weight_vals = inters.data_1st_derive
        if data_1st_derive_floor is not None:
            weight_vals = np.where(
                weight_vals < data_1st_derive_floor,
                data_1st_derive_floor,
                weight_vals,
            )
        inters["data_1st_derive_weight"] = weight_vals

        if weight_by == "all":
            inters["data_1st_derive_weight"] = normalize_values(
                inters["data_1st_derive_weight"],
                low=1, high=0.001, # reversed so large gradients are bad
                # quantiles=(0.02, 0.98),
            )
        else:
            inters["data_1st_derive_weight"] = inters.groupby(weight_by)["data_1st_derive_weight"].transform(
                lambda x: normalize_values(x,
                low=1, high=0.001, # reversed so large gradients are bad
                # quantiles=(0.02, 0.98),
                )
            )

        weights_cols.append("data_1st_derive_weight")
        weights_dict["data_1st_derive_weight"] = data_1st_derive_weight
        plot_cols.append("data_1st_derive")

    if data_2nd_derive_weight is not None:
        if data_2nd_derive_col_name is None:
            raise ValueError(
                f"must provide 'data_2nd_derive_col_name'"
            )
        # find data gradient at intersection for line and tie
        for ind, row in inters.iterrows():
            # search data for values at intersecting lines
            line_value = gdf[
                (gdf[line_col_name] == row.line) & (gdf.intersecting_line == row.tie)
            ][data_2nd_derive_col_name].values[0]
            tie_value = gdf[
                (gdf[line_col_name] == row.tie) & (gdf.intersecting_line == row.line)
            ][data_2nd_derive_col_name].values[0]
            inters.loc[ind, "data_2nd_derive"] = np.mean(np.abs([line_value, tie_value]))
        weight_vals = inters.data_2nd_derive
        if data_2nd_derive_floor is not None:
            weight_vals = np.where(
                weight_vals < data_2nd_derive_floor,
                data_2nd_derive_floor,
                weight_vals,
            )
        inters["data_2nd_derive_weight"] = weight_vals

        if weight_by == "all":
            inters["data_2nd_derive_weight"] = normalize_values(
                inters["data_2nd_derive_weight"],
                low=1, high=0.001, # reversed so large gradients are bad
                # quantiles=(0.02, 0.98),
            )
        else:
            inters["data_2nd_derive_weight"] = inters.groupby(weight_by)["data_2nd_derive_weight"].transform(
                lambda x: normalize_values(x,
                low=1, high=0.001, # reversed so large gradients are bad
                # quantiles=(0.02, 0.98),
                )
            )

        weights_cols.append("data_2nd_derive_weight")
        weights_dict["data_2nd_derive_weight"] = data_2nd_derive_weight
        plot_cols.append("data_2nd_derive")

    if height_1st_derive_weight is not None:
        if height_1st_derive_col_name is None:
            raise ValueError(
                f"must provide 'height_1st_derive_col_name'"
            )
        # find height gradient at intersection for line and tie
        for ind, row in inters.iterrows():
            # search data for values at intersecting lines
            line_value = gdf[
                (gdf[line_col_name] == row.line) & (gdf.intersecting_line == row.tie)
            ][height_1st_derive_col_name].values[0]
            tie_value = gdf[
                (gdf[line_col_name] == row.tie) & (gdf.intersecting_line == row.line)
            ][height_1st_derive_col_name].values[0]
            inters.loc[ind, "height_1st_derive"] = np.mean(np.abs([line_value, tie_value]))
        weight_vals = inters.height_1st_derive
        if height_1st_derive_floor is not None:
            weight_vals = np.where(
                weight_vals < height_1st_derive_floor,
                height_1st_derive_floor,
                weight_vals,
            )
        inters["height_1st_derive_weight"] = weight_vals

        if weight_by == "all":
            inters["height_1st_derive_weight"] = normalize_values(
                inters["height_1st_derive_weight"],
                low=1, high=0.001, # reversed so large gradients are bad
                # quantiles=(0.02, 0.98),
            )
        else:
            inters["height_1st_derive_weight"] = inters.groupby(weight_by)["height_1st_derive_weight"].transform(
                lambda x: normalize_values(x,
                low=1, high=0.001, # reversed so large gradients are bad
                # quantiles=(0.02, 0.98),
                )
            )

        weights_cols.append("height_1st_derive_weight")
        weights_dict["height_1st_derive_weight"] = height_1st_derive_weight
        plot_cols.append("height_1st_derive")

    if height_2nd_derive_weight is not None:
        if height_2nd_derive_col_name is None:
            raise ValueError(
                f"must provide 'height_2nd_derive_col_name'"
            )
        # find height gradient at intersection for line and tie
        for ind, row in inters.iterrows():
            # search data for values at intersecting lines
            line_value = gdf[
                (gdf[line_col_name] == row.line) & (gdf.intersecting_line == row.tie)
            ][height_2nd_derive_col_name].values[0]
            tie_value = gdf[
                (gdf[line_col_name] == row.tie) & (gdf.intersecting_line == row.line)
            ][height_2nd_derive_col_name].values[0]
            inters.loc[ind, "height_2nd_derive"] = np.mean(np.abs([line_value, tie_value]))
        weight_vals = inters.height_2nd_derive
        if height_2nd_derive_floor is not None:
            weight_vals = np.where(
                weight_vals < height_2nd_derive_floor,
                height_2nd_derive_floor,
                weight_vals,
            )
        inters["height_2nd_derive_weight"] = weight_vals

        if weight_by == "all":
            inters["height_2nd_derive_weight"] = normalize_values(
                inters["height_2nd_derive_weight"],
                low=1, high=0.001, # reversed so large gradients are bad
                # quantiles=(0.02, 0.98),
            )
        else:
            inters["height_2nd_derive_weight"] = inters.groupby(weight_by)["height_2nd_derive_weight"].transform(
                lambda x: normalize_values(x,
                low=1, high=0.001, # reversed so large gradients are bad
                # quantiles=(0.02, 0.98),
                )
            )

        weights_cols.append("height_2nd_derive_weight")
        weights_dict["height_2nd_derive_weight"] = height_2nd_derive_weight
        plot_cols.append("height_2nd_derive")

    logger.info("combining individual weight cols with following factors: %s", weights_dict)
    # calcualted weighted mean of the weights
    def weighted_average(df, weights):
        return df[list(weights)].mul(weights).sum(axis=1) / sum(weights.values())

    # inters["mistie_weight"] = weighted_average(inters, weights_dict)
    # inters["mistie_weights"] = inters[weights_cols].mean(axis=1)

    if weight_by == "all":
        inters["mistie_weight"] = weighted_average(inters, weights_dict)
        inters["mistie_weight"] = normalize_values(
            inters["mistie_weight"],
            low=0.001, high=1,
        )
    else:
        inters["mistie_weight"] = inters.groupby(weight_by).apply(
            lambda x: pd.Series(weighted_average(x, weights_dict), index=x.index),
            include_groups=False,
        ).reset_index(drop=True)
        # inters["mistie_weight"] = inters.groupby(weight_by).transform(
        #     lambda x: weighted_average(x, weights_dict),
        # )
        inters["mistie_weight"] = inters.groupby(weight_by)["mistie_weight"].transform(
            lambda x: normalize_values(x,
            low=0.001, high=1,
            )
        )

    if plot:
        plotly_points(
            inters,
            color_col="mistie_weight",
            hover_cols=[
                "line",
                "tie",
                "mistie_weight",
                *weights_cols,
                *plot_cols,
            ],
            cmap="matter_r",
            # robust=False,
            cmap_lims=(0, 1),
            # point_width=0,
            point_size=6,
            theme=None,
        )
    return inters


def plot_levelling_convergence(
    results,
    logy=False,
    title="Levelling convergence",
    as_median=False,
):
    sns.set_theme()

    # get mistie columns
    cols = [s for s in results.columns.to_list() if s.startswith("mistie")]
    cols = [c for c in cols if c[-1].isdigit()]

    iters = len(cols)

    mistie_rmses = [
        utils.rmse(
            results[i],
            as_median=as_median,
        )
        for i in cols
    ]

    fig, ax1 = plt.subplots(figsize=(5, 3.5))
    plt.title(title)
    ax1.plot(range(iters), mistie_rmses, "bo-")
    ax1.set_xlabel("Iteration")
    if logy:
        ax1.set_yscale("log")
    if as_median:
        label = "Cross-over root median squared error (mGal)"
    else:
        label = "Cross-over root mean squared error (mGal)"
    ax1.set_ylabel(label, color="k")
    ax1.tick_params(axis="y", colors="k", which="both")

    ax1.set_xticks(range(iters))
    # plt.tight_layout()

    return fig


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
        df1 = df.copy()#.reset_index(drop=True)

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
    **kwargs: typing.Any
) -> pd.Series:
    """
    """
    reverse = kwargs.get("reverse", False)
    df = df.copy()

    df = df.groupby(flight_col_name)

    dfs = []
    for name, flight in df:
        flight = flight.sort_values(by=[time_col_name]).reset_index(drop=True)
        flight = relative_distance(flight, reverse=reverse)
        dist = flight.rel_dist.cumsum()
        flight["dist_along_flight"] = dist
        # df.loc[df[flight_col_name] == i, "dist_along_flight"] = dist
        dfs.append(flight)

    df = pd.concat(dfs).reset_index(drop=True).sort_values(by=[flight_col_name])

    return df.dist_along_flight



def distance_along_line(
    gdf: gpd.GeoDataFrame,
    line_col_name: str = "line",
    time_col_name: str | None = "unixtime",
) -> pd.Series:
    """
    Calculate the distances along each flight line in meters. If 'time_col_name' is
    provided, this will inform which end of the line is the beginning. If not, the line
    will be rotate horizontally, and the left-most side will be used as the start.

    Parameters
    ----------
    data : gpd.GeoDataFrame
        Dataframe containing the data points to calculate the distance along each line,
        must have a set geometry column.
    line_col_name : str, optional
        Column name specifying the line number, by default "line"
    time_col_name : str | None, optional
        Column name containing time in seconds for each datapoint, by default "unixtime"

    Returns
    -------
    pd.Series
        The distance along each line in meters
    """

    gdf = gdf.copy()

    grouped = gdf.groupby([line_col_name])

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
        angle = _azimuth(rect)
        if angle > 90 and angle <= 180:
            angle = angle - 180
        # print(angle)

        # rotate the line to be horizontal
        line_horizontal = line.rotate(angle, origin=shapely.centroid(rect))
        horizontal_df = line_horizontal.get_coordinates(
            index_parts=True,
            ignore_index=True,
        )
        horizontal_df["original_index"] = horizontal_df.index
        horizontal_df = horizontal_df.sort_values("x").reset_index(drop=True)

        # print(df)
        # print(horizontal_df)
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
            # print(start_arg)
            # print(df.iloc[start_arg])
            # print(horizontal_df[horizontal_df.original_index==start_arg])
            horizontal_start_arg = horizontal_df[
                horizontal_df.original_index == start_arg
            ].index
            # print(horizontal_start_arg)
            # if start_arg < len(df) / 2:
            #     start_arg = horizontal_df.x.argmax()
            #     end_arg = horizontal_df.x.argmin()
            # else:
            #     start_arg = horizontal_df.x.argmin()
            #     end_arg = horizontal_df.x.argmax()
            # start_coords = df.iloc[start_arg][["x","y"]].values
            # end_coords = df.iloc[end_arg][["x","y"]].values

            if horizontal_start_arg > len(df) / 2:
                h_start_arg = horizontal_df.x.argmax()
                h_end_arg = horizontal_df.x.argmin()
            else:
                h_start_arg = horizontal_df.x.argmin()
                h_end_arg = horizontal_df.x.argmax()
            start_arg = int(horizontal_df.iloc[h_start_arg].original_index)
            end_arg = int(horizontal_df.iloc[h_end_arg].original_index)
            start_coords = df.iloc[start_arg][["x", "y"]].values
            end_coords = df.iloc[end_arg][["x", "y"]].values

            logger.debug("Line starts at %s and ends at %s", start_coords, end_coords)
        else:
            # get start and end points of line
            start_arg = horizontal_df.x.argmin()
            end_arg = horizontal_df.x.argmax()
            start_coords = df.iloc[start_arg][["x", "y"]].values
            end_coords = df.iloc[end_arg][["x", "y"]].values
            logger.debug(
                "Assuming line starts at %s and ends at %s", start_coords, end_coords
            )

        # calculate distance along line from starting point
        dist = data.distance(Point(*start_coords))

        gdf.loc[gdf[line_col_name] == name, "dist_along_line"] = dist

    if time_col_name is not None:
        grouped = gdf.groupby([line_col_name])
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

    # gdf["dist_along_line"] = np.nan
    # for i in gdf[line_col_name].unique():
    #     line = gdf[gdf[line_col_name] == i]
    #     dist = line.distance(line.sort_values(
    #         by=time_col_name,
    #         na_position='last',
    #     ).geometry.iloc[0]).values
    #     gdf.loc[gdf[line_col_name] == i, "dist_along_line"] = dist

    return gdf.dist_along_line


def create_intersection_table(
    flight_lines: gpd.GeoDataFrame,
    tie_lines: gpd.GeoDataFrame,
    line_col_name: str = "line",
    exclude_ints: list[tuple[int]] | None = None,
    cutoff_dist: float | None = None,
    buffer_dist: float | None = None,
    grid_size: float = 100,
    plot: bool = True,
) -> gpd.GeoDataFrame:
    """
    create a dataframe which contains the intersections between provided flight and tie
    lines. For each intersection point, find the distance to the closest data point of
    each line. If the further of these two distances is greater than "cutoff_dist", the
    intersection is excluded. The intersections are calculated by
    representing the point data as lines, and finding the hypothetical crossover.
    By default crossovers will only be between the first and last point of a line. If
    there is an expected crossover just beyond the end of a line which should be
    included, use the `buffer_dist` arg to extend the line representation of the data,
    but note that extrapolation of data at these points will likely be inaccurate if
    buffer distance is too large.

    Parameters
    ----------
    flight_lines : gpd.GeoDataFrame
        Flight line data which must be a geodataframe with a registered geometry column
        and a column set by `line_col_name` specifying the line number.
    tie_lines : gpd.GeoDataFrame
        Tie line data which must be a geodataframe with a registered geometry column
        and a column set by `line_col_name` specifying the line number.
    line_col_name : str, optional
        Column name specifying the line numbers, by default "line"
    exclude_ints : list[tuple[int]] | None, optional
        List of tuples where each tuple is either a single line number to exclude from
        all intersections, or a pair of line numbers specifying specific intersections
        to exclude, by default None
    cutoff_dist : float, optional
        The maximum allowed distance from a theoretical intersection to the further of
        nearest data point of each intersecting line, by default None
    buffer_dist : float, optional
        The distance to extend the line representation of the data points, usefull for
        creating intersection which are just beyond the end of a line, by default None
    plot : bool, optional
        Plot a map of the resulting intersection points colored by distance to the
        further of the two nearest data points, by default True

    Returns
    -------
    gpd.GeoDataFrame
        An intersection table containing the locations of the theoretical intersections,
        the line and tie numbers, and the distance to the further of the two nearest
        datapoints of each line, and a geometry column.
    """

    lines_df = flight_lines.copy()
    ties_df = tie_lines.copy()

    # if is_intersection column exists, delete it and rows where it's true
    if "is_intersection" in lines_df.columns:
        rows_to_drop = lines_df[lines_df.is_intersection]
        lines_df = lines_df.drop(index=rows_to_drop.index)
    lines_df = lines_df.drop(columns="is_intersection", errors="ignore")
    if "is_intersection" in ties_df.columns:
        rows_to_drop = ties_df[ties_df.is_intersection]
        ties_df = ties_df.drop(index=rows_to_drop.index)
    ties_df = ties_df.drop(columns="is_intersection", errors="ignore")

    # get intersection points
    inters = get_line_tie_intersections(
        lines_gdf=lines_df,
        ties_gdf=ties_df,
        grid_size=grid_size,
        buffer_dist=buffer_dist,
    )
    # redo with buffer_dist to get extrapolated points
    # if buffer_dist is not None:
    #     inters_buffer = get_line_tie_intersections(
    #         lines_gdf=lines_df,
    #         ties_gdf=ties_df,
    #         grid_size=grid_size,
    #         buffer_dist=buffer_dist,
    #     )

    # get the largest of the two distance to each lines' nearest data point to the
    # theoretical intersection
    inters["max_dist"] = inters[["line_dist", "tie_dist"]].max(axis=1)
    # if buffer_dist is not None:
    #     inters_buffer["max_dist"] = inters_buffer[["line_dist", "tie_dist"]].max(axis=1)

    # keep only the closest of duplicated intersections
    a = len(inters)
    inters = (
        inters.sort_values(
            "max_dist",
            ascending=False,
        )
        .drop_duplicates(
            subset=["line", "tie"],
            keep="last",
        )
        .sort_index()
    )
    b = len(inters)
    if a != b:
        logger.info("Dropped %s duplicate intersections", a - b)
    # if buffer_dist is not None:
    #     inters_buffer = (
    #     inters_buffer.sort_values(
    #             "max_dist",
    #             ascending=False,
    #         )
    #         .drop_duplicates(
    #             subset=["line", "tie"],
    #             keep="last",
    #         )
    #         .sort_index()
    #     )

    # # get points which were extrapolated
    # inters["interpolation_type"] = "interpolated"
    # if buffer_dist is not None:
    #     inters_buffer["interpolation_type"] = 'extrapolated'
    #     df = inters_buffer[
    #         inters_buffer.set_index(['line', 'tie']).index.isin(list(zip(inters.line, inters.tie)))
    #     ].copy()
    #     inters_buffer.loc[df.index, "interpolation_type"] = "interpolated"
    #     inters = inters_buffer.copy()

    # if buffer_dist is not None:
    #     logger.info(
    #         "found %s intersections, of which %s were extrapolated",
    #         len(inters),
    #         len(inters[inters.interpolation_type == "extrapolated"]),
    #     )
    # else:
    #     logger.info("found %s intersections", len(inters))
    logger.info("found %s intersections", len(inters))

    # if intersection is not within cutoff_dist, remove rows
    if cutoff_dist is not None:
        prior_len = len(inters)
        inters = inters[inters.max_dist < cutoff_dist]
        logger.info(
            "removed %s intersections points which were further than %s km from "
            "nearest data point",
            prior_len - len(inters),
            int(cutoff_dist / 1000),
        )

    # get coords from geometry column
    inters["easting"] = inters.geometry.x
    inters["northing"] = inters.geometry.y

    if exclude_ints is not None:
        prior_len = len(inters)
        exclude_inds = []
        for i in exclude_ints:
            if isinstance(i, int | float):
                msg = (
                    "exclude_ints must be a list of tuples of individual or pairs of "
                    "line numbers"
                )
                raise ValueError(msg)
            # if pair of lines numbers given, get those indices
            if len(i) == 2:
                ind = inters[(inters.line == i[0]) & (inters.tie == i[1])].index.values
                exclude_inds.extend(ind)
                ind = inters[(inters.tie == i[0]) & (inters.line == i[1])].index.values
                exclude_inds.extend(ind)
            # if single line number, get all intersections of that line
            elif len(i) == 1:
                ind = inters[(inters.line == i[0]) | (inters.tie == i[0])].index.values
                exclude_inds.extend(ind)
        inters = inters.drop(index=exclude_inds).copy()
        logger.info(
            "manually ommited %s intersections points",
            prior_len - len(inters),
        )

    if plot is True:
        plotly_points(
            inters,
            color_col="max_dist",
            hover_cols=["line", "tie", "max_dist", "line_dist", "tie_dist"],
            robust=True,
            point_size=10,
            theme=None,
            cmap="matter",
            title="Distance from intersection to nearest data point",
        )
        plt.hist(inters.max_dist)
        plt.xlabel("Max distance between intersection and line/tie data (m)")

    return inters.drop(columns=["line_dist", "tie_dist"])


def add_intersections(
    df: gpd.GeoDataFrame,
    intersections: gpd.GeoDataFrame,
    line_col_name: str = "line",
    time_col_name: str = "unixtime",
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Add new rows to the dataframe for each intersection point and columns
    `is_intersection` and `intersection_line` to identify these intersections. All of
    the data column for these rows will have NaNs and should be filled with the
    function `interp1d_all_lines()`. Add columns to the intersections table for the
    distance along each line (flight and tie) to the intersection point. During
    levelling, levelling corrections are calculated using mistie values at intersections
    and interpolated along the entire lines based on these distances. Distances are
    calculate using the geometry column, and the time column informs which end of the
    line is the start.

    Parameters
    ----------
    df : gpd.GeoDataFrame
        Flight survey dataframe containing the data points to add intersections to.
        Must contain a geometry column and columns set by `line_col_name` and
        `time_col_name`
    intersections : gpd.GeoDataFrame
        Intersections table created by `create_intersection_table()`
    line_col_name : str, optional
        Column name specifying the line and tie names, by default "line"
    time_col_name : str, optional
        Column name specifying the time of each datapoints collection, by default
        "unixtime"

    Returns
    -------
    tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
        The updated flight survey dataframe and intersections table.
    """
    gdf = df.copy()
    inters = intersections.copy()

    # if is_intersection column exists, delete it and rows where it's true
    if "is_intersection" in gdf.columns:
        rows_to_drop = gdf[gdf.is_intersection]
        gdf = gdf.drop(index=rows_to_drop.index)
    gdf = gdf.drop(
        columns=["is_intersection", "intersecting_line", "interpolation_type"],
        errors="ignore",
    )

    prior_length = len(gdf)

    # add boolean column for whether point is an intersection
    gdf["is_intersection"] = False
    gdf["intersecting_line"] = np.nan

    # collect intersections to be added
    dfs = []
    for _, row in inters.iterrows():
        for i in list(gdf[line_col_name].unique()):
            if i in (row.line, row.tie):
                df = pd.DataFrame(
                    {
                        line_col_name: [i],
                        "easting": row.geometry.x,
                        "northing": row.geometry.y,
                        "is_intersection": True,
                    }
                )
                if i == row.line:
                    df["intersecting_line"] = row.tie
                else:
                    df["intersecting_line"] = row.line
                df["geometry"] = gpd.points_from_xy(df.easting, df.northing)
                dfs.append(df)

    # add intersections
    gdf = pd.concat([gdf, *dfs])

    # check correct number of intersections were added
    assert len(gdf) == prior_length + (2 * len(inters))

    # sort by lines
    gdf = gdf.sort_values(by=line_col_name)

    # print(gdf[(gdf.line==1070) & (gdf.intersecting_line==160)])
    # get distance along each line
    gdf["dist_along_line"] = distance_along_line(
        gdf,
        line_col_name=line_col_name,
        time_col_name=time_col_name,
    )
    # print(gdf[(gdf.line==1070) & (gdf.intersecting_line==160)])

    # sort by distance and reset index
    gdf = gdf.sort_values(by=[line_col_name, "dist_along_line"])
    gdf = gdf.reset_index(drop=True)

    # add dist along line to intersections dataframe
    # iterate through intersections
    for ind, row in inters.iterrows():
        # search data for values at intersecting lines
        line_value = gdf[
            (gdf[line_col_name] == row.line) & (gdf.intersecting_line == row.tie)
        ].dist_along_line.values[0]
        tie_value = gdf[
            (gdf[line_col_name] == row.tie) & (gdf.intersecting_line == row.line)
        ].dist_along_line.values[0]

        inters.loc[ind, "dist_along_flight_line"] = line_value
        inters.loc[ind, "dist_along_flight_tie"] = tie_value

    return gdf, inters


def _azimuth_between_points(point1, point2):
    """azimuth between 2 points (interval 0 - 180)"""

    angle = np.arctan2(point2[1] - point1[1], point2[0] - point1[0])
    return np.degrees(angle) if angle > 0 else np.degrees(angle) + 180


def _dist(a, b):
    """distance between points"""
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _azimuth(mrr):
    """azimuth of minimum_rotated_rectangle"""
    bbox = list(mrr.exterior.coords)
    axis1 = _dist(bbox[0], bbox[3])
    axis2 = _dist(bbox[0], bbox[1])

    if axis1 <= axis2:
        az = _azimuth_between_points(bbox[0], bbox[1])
    else:
        az = _azimuth_between_points(bbox[0], bbox[3])

    return az


def extend_line(line, distance, plot=False):
    """extend line in either direction by distance"""
    # find minimum rotated rectangle around line
    rect = line.minimum_rotated_rectangle
    angle = _azimuth(rect)
    # logger.debug("rotated angle:", angle)

    rect_center = shapely.centroid(rect).x, shapely.centroid(rect).y
    # print("rectangle center:", rect_center)

    # get length of long edge
    x, y = rect.exterior.coords.xy
    length = max(
        (
            Point(x[0], y[0]).distance(Point(x[1], y[1])),
            Point(x[1], y[1]).distance(Point(x[2], y[2])),
        )
    )
    # print("length:", length)

    # make new start and end points extended by distance
    start = Point(rect_center[0] - (length / 2) - distance, rect_center[1])
    end = Point(rect_center[0] + (length / 2) + distance, rect_center[1])

    # turn into a line and rotate back
    extended_line = shapely.affinity.rotate(
        LineString([start, end]), angle, origin=rect_center
    )

    # add new endpoints to original line
    # extended_line = shapely.unary_union()
    extended_line = LineString(
        [extended_line.coords[0], *line.coords, extended_line.coords[-1]]
    )

    if plot:
        l_coords = list(line.coords)
        x = [p[0] for p in l_coords]
        y = [p[1] for p in l_coords]
        plt.plot(x, y, "r.", markersize=10, label="original line")

        longl_coords = list(extended_line.coords)
        x = [p[0] for p in longl_coords]
        y = [p[1] for p in longl_coords]
        plt.plot(x, y, "g.", markersize=2, label="extended line")

        plt.legend()

        # make plot aspect same
        x_range = plt.xlim()[1] - plt.xlim()[0]
        plt.ylim(np.mean(y) - x_range / 2, np.mean(y) + x_range / 2)

    return extended_line


# def extend_lines(
#     gdf,
#     max_interp_dist,
# ):
#     """
#     WIP attempt to extend lines to intersect nearby lines
#     """
#     grouped = gdf[(gdf.line == 1040) | (gdf.line == 20)].groupby(
#         "line", as_index=False
#     )["geometry"]
#     gdf2 = gdf[(gdf.line == 1040) | (gdf.line == 20)]
#     # grouped = grouped.apply(lambda x: LineString(x.tolist()))
#     # lines = grouped.iloc[0:2].geometry.copy()

#     # for name1, name2 in itertools.combinations(list(grouped.groups.keys()), 2):
#     for name1, name2 in itertools.combinations(gdf2.line.unique(), 2):
#         line = LineString(grouped.get_group(name1).tolist())
#         tie = LineString(grouped.get_group(name2).tolist())

#         # get line endpoints
#         # line_endpoints = MultiPoint(
#         #   [Point(list(line.coords)[0]), Point(list(line.coords)[-1])])
#         # tie_endpoints = MultiPoint(
#         #   [Point(list(tie.coords)[0]), Point(list(tie.coords)[-1])])
#         line_endpoints = MultiPoint([list(line.coords)[0], list(line.coords)[-1]])
#         tie_endpoints = MultiPoint([list(tie.coords)[0], list(tie.coords)[-1]])

#         # logger.info(line_endpoints)
#         # logger.info(tie_endpoints)

#         # get nearest points on each line to the closest of the other lines endpoints
#         nearest_line_point_to_tie_endpoints = shapely.nearest_points(
#             line, tie_endpoints
#         )[0]
#         nearest_tie_point_to_line_endpoints = shapely.nearest_points(
#             tie, line_endpoints
#         )[0]

#         # logger.info(nearest_tie_point_to_line_endpoints)
#         # logger.info(nearest_line_point_to_tie_endpoints)

#         # get distances between nearest points on line with closest endpoint of
#         # other line
#         distance_tie_endpoint_to_line = np.min(
#             [x.distance(nearest_line_point_to_tie_endpoints) for x in tie_endpoints]
#         )
#         distance_line_endpoint_to_tie = np.min(
#             [x.distance(nearest_tie_point_to_line_endpoints) for x in line_endpoints]
#         )

#         # logger.info(distance_tie_endpoint_to_line)
#         # logger.info(distance_line_endpoint_to_tie)

#         # if distance is lower than cutoff, add intersection points to extend lines
#         if distance_line_endpoint_to_tie <= max_interp_dist:
#             tie_new = LineString(
#                 list(tie.coords) + list(nearest_line_point_to_tie_endpoints.coords)
#             )
#             assert len(list(tie.coords)) + 1 == len(list(tie_new.coords))
#             logger.info("extended line: %s", name1)
#         else:
#             tie_new = tie

#         # repeat for tie
#         if distance_tie_endpoint_to_line <= max_interp_dist:
#             line_new = LineString(
#                 list(line.coords) + list(nearest_tie_point_to_line_endpoints.coords)
#             )
#             assert len(list(line.coords)) + 1 == len(list(line_new.coords))
#             logger.info("extended line: %s", name2)
#         else:
#             line_new = line

#         # logger.info(len(list(line.coords)))
#         # logger.info(len(list(tie.coords)))
#         # logger.info(len(list(line_new.coords)))
#         # logger.info(len(list(tie_new.coords)))


def get_line_tie_intersections(
    lines_gdf: gpd.GeoSeries,
    ties_gdf: gpd.GeoSeries,
    grid_size: float = 100,
    buffer_dist: float | None = None,
    line_col_name: str = "line",
):
    """
    adapted from https://gis.stackexchange.com/questions/137909/intersecting-lines-to-get-crossings-using-python-with-qgis
    """
    # group by lines/ties
    grouped_lines = lines_gdf.groupby([line_col_name], as_index=False)["geometry"]
    grouped_ties = ties_gdf.groupby([line_col_name], as_index=False)["geometry"]

    # from points to lines
    grouped_lines = grouped_lines.apply(lambda x: LineString(x.tolist()))
    grouped_ties = grouped_ties.apply(lambda x: LineString(x.tolist()))

    # entend ends of lines by buffer_dist to account for expected intersections just
    # beyond lines
    if buffer_dist is not None:
        grouped_lines["geometry"] = grouped_lines.geometry.apply(
            lambda x: extend_line(x, buffer_dist)
        )
        grouped_ties["geometry"] = grouped_ties.geometry.apply(
            lambda x: extend_line(x, buffer_dist)
        )

    combos_names = list(itertools.product(grouped_lines.line, grouped_ties.line))
    combos_lines = list(
        itertools.product(grouped_lines.geometry, grouped_ties.geometry)
    )
    pbar = zip(
        tqdm(
            combos_lines,
            desc="Line/tie combinations",
        ),
        combos_names,
    )
    inters = []
    line_names = []
    tie_names = []
    for (line, tie), (l_name, t_name) in pbar:
        # determine intersection of line and tie
        # inter = line.intersection(tie)
        inter = shapely.intersection(line, tie, grid_size=grid_size)
        # inter = line.unary_union.intersection(tie)
        points = [Point(i) for i in shapely.get_coordinates(inter)]
        inters.extend(points)

        line_names.extend([l_name] * len(points))
        tie_names.extend([t_name] * len(points))

        # # if intersection is a point, add to list
        # if inter.type == "Point":
        #     inters.append(inter)
        # elif inter.type == "MultiPoint":
        #     inters.extend(list(inter.geoms))
        # elif inter.type == "MultiLineString":
        #     inters.extend(shapely.get_coordinates(inter))
        #     # multi_line = list(inter.geoms)
        #     # first_coords = multi_line[0].coords[0]
        #     # last_coords = multi_line[len(multi_line) - 1].coords[1]
        #     # inters.append(Point(first_coords[0], first_coords[1]))
        #     # inters.append(Point(last_coords[0], last_coords[1]))
        # elif inter.type == "LineString":
        #     inters.extend(shapely.get_coordinates(inter))
        #     # first_coords = inter.coords[0]
        #     # last_coords = inter.coords[1]
        #     # inters.append(Point(first_coords[0], first_coords[1]))
        #     # inters.append(Point(last_coords[0], last_coords[1]))
        # elif inter.type == "GeometryCollection":
        #     for geom in inter.geoms:
        #         if geom.type == "Point":
        #             inters.append(geom)
        #         elif geom.type == "MultiPoint":
        #             inters.extend(list(geom))
        #         elif geom.type == "multi_lineString":
        #             multi_line = list(geom)
        #             first_coords = multi_line[0].coords[0]
        #             last_coords = multi_line[len(multi_line) - 1].coords[1]
        #             inters.append(Point(first_coords[0], first_coords[1]))
        #             inters.append(Point(last_coords[0], last_coords[1]))

    gdf = gpd.GeoDataFrame(geometry=inters, data={"line": line_names, "tie": tie_names})

    # get nearest 2 lines to each intersection point
    # and nearest data point on each line to the intersection point
    line_names = []
    tie_names = []
    line_dists = []
    tie_dists = []

    pbar = tqdm(
        gdf.geometry,
        desc="Potential intersections",
        total=len(gdf.geometry),
    )
    for p in pbar:
        # for _i, p in enumerate(inters.geometry):
        # look into shapely.interpolate() to get points based on distance along line
        # look into shapely.project() to get distance along line which is closest point
        # to tie
        # shapely.crosses or shapely.intersects for if lines cross or not
        # shapely.nearest_points()

        # find nearest line/tie to intersection point using LineString's
        grouped_lines["dist"] = grouped_lines.geometry.distance(p)
        grouped_ties["dist"] = grouped_ties.geometry.distance(p)
        nearest_line = grouped_lines.sort_values(by="dist")[[line_col_name]].iloc[0]
        nearest_tie = grouped_ties.sort_values(by="dist")[[line_col_name]].iloc[0]

        # get line/tie names
        line = nearest_line[line_col_name]
        tie = nearest_tie[line_col_name]

        # append names to lists
        line_names.append(line)
        tie_names.append(tie)

        # get actual datapoints for each line (not LineString representation)
        line_points = lines_gdf[lines_gdf[line_col_name] == line]
        tie_points = ties_gdf[ties_gdf[line_col_name] == tie]

        # get nearest data point on each line/tie to intersection point
        nearest_datapoint_line = line_points.geometry.distance(p).sort_values().iloc[0]
        nearest_datapoint_tie = tie_points.geometry.distance(p).sort_values().iloc[0]

        # add distance to nearest data point on each line to lists
        line_dists.append(nearest_datapoint_line)
        tie_dists.append(nearest_datapoint_tie)

    # add names and distances as columns
    gdf["line"] = line_names
    gdf["tie"] = tie_names
    gdf["line_dist"] = line_dists
    gdf["tie_dist"] = tie_dists

    return gdf


def get_line_intersections(
    lines,
):
    """
    adapted from https://gis.stackexchange.com/questions/137909/intersecting-lines-to-get-crossings-using-python-with-qgis
    """

    inters = []
    for line1, line2 in itertools.combinations(lines, 2):
        if line1.intersects(line2):
            inter = line1.intersection(line2)

            if inter.type == "Point":
                inters.append(inter)
            elif inter.type == "MultiPoint":
                inters.extend(list(inter.geoms))
            elif inter.type == "MultiLineString":
                multi_line = list(inter.geoms)
                first_coords = multi_line[0].coords[0]
                last_coords = multi_line[len(multi_line) - 1].coords[1]
                inters.append(Point(first_coords[0], first_coords[1]))
                inters.append(Point(last_coords[0], last_coords[1]))
            elif inter.type == "GeometryCollection":
                for geom in inter:
                    if geom.type == "Point":
                        inters.append(geom)
                    elif geom.type == "MultiPoint":
                        inters.extend(list(geom))
                    elif geom.type == "multi_lineString":
                        multi_line = list(geom)
                        first_coords = multi_line[0].coords[0]
                        last_coords = multi_line[len(multi_line) - 1].coords[1]
                        inters.append(Point(first_coords[0], first_coords[1]))
                        inters.append(Point(last_coords[0], last_coords[1]))
    return inters


def scipy_interp1d(
    df,
    to_interp=None,
    interp_on=None,
    method=None,
    extrapolate=False,
    fill_value: tuple(float, float) | str | None = None,
):
    """
    interpolate NaN's in "to_interp" column, based on values from "interp_on" column
    method:
        'linear', 'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic',
        'previous', 'next'
    use kwargs to pass other arguments to scipy.interpolate.interp1d()
    """
    df = df.copy()

    # drop NaN's
    df_no_nans = df.dropna(subset=[to_interp, interp_on], how="any")

    if extrapolate is True:
        bounds_error = False
        if fill_value is None:
            fill_value = "extrapolate"
        elif fill_value == "edge":
            fill_value = (df_no_nans[to_interp].iloc[0], df_no_nans[to_interp].iloc[-1])
        elif fill_value == "mean":
            fill_value = (np.nanmean(df[to_interp]), np.nanmean(df[to_interp]))
        logger.debug("extrapolating with fill_value: %s", fill_value)
    else:
        bounds_error = True
        fill_value = np.nan

    # print(df_no_nans)
    # define interpolation function
    f = scipy.interpolate.interp1d(
        df_no_nans[interp_on],
        df_no_nans[to_interp],
        kind=method,
        bounds_error=bounds_error,
        fill_value=fill_value,
    )

    # get interpolated values at points with NaN's
    values = f(df[df[to_interp].isnull()][interp_on])

    # fill NaN's  with values
    df.loc[df[to_interp].isnull(), to_interp] = values

    return df


def verde_interp1d(
    df,
    to_interp=None,
    interp_on=None,
    method=None,
):
    """
    interpolate NaN's in "to_interp" column, based on coordinates from "interp_on"
    columns,
    method: vd.Spline(), vd.SplineCV(), vd.KNeighbors(), vd.Linear(), vd.Cubic()
    """
    df1 = df.copy()

    # drop NaN's
    df_no_nans = df1.dropna(subset=[to_interp, *interp_on], how="any")

    # fit interpolator to data
    method.fit(
        (df_no_nans[interp_on[0]], df_no_nans[interp_on[1]]), df_no_nans[to_interp]
    )

    # predict at NaN's
    values = method.predict(
        (
            df1[df1[to_interp].isnull()][interp_on[0]],
            df1[df1[to_interp].isnull()][interp_on[1]],
        ),
    )

    # fill NaN's  with values
    df1.loc[df1[to_interp].isnull(), to_interp] = values

    return df1


def interp1d_single_col(
    df,
    to_interp,
    interp_on,
    engine="scipy",
    method="cubic",
    extrapolate=False,
    fill_value=None,
    plot=False,
    line_col_name="line",
):
    """
    interpolate NaN's in "to_interp" column, based on value(s) from "interp_on"
    column(s).
    engine: "verde" or "scipy"
    method:
        for "verde": vd.Spline(), vd.SplineCV(), vd.KNeighbors(), vd.Linear(),
        vd.Cubic()
        for "scipy": 'linear', 'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic',
        'cubic', 'previous', 'next'
    """
    args = {
        "df": df,
        "to_interp": to_interp,
        "interp_on": interp_on,
        "method": method,
        "extrapolate": extrapolate,
        "fill_value": fill_value,
    }

    if engine == "verde":
        filled = verde_interp1d(**args)
    elif engine == "scipy":
        # try:
        filled = scipy_interp1d(**args)
        # except ValueError as e:
        #     # logger.error(args)
        #     filled = np.nan
    else:
        msg = "invalid string for engine type"
        raise ValueError(msg)

    if plot is True:
        plot_line_and_crosses(
            filled,
            line=filled[line_col_name].iloc[0],
            x=interp_on,
            y=[to_interp],
            y_axes=[i + 1 for i in range(len([to_interp]))],
        )

    return filled


def interp1d_windows_single_col(
    df,
    window_width=None,
    dist_col_name="dist_along_line",
    line_col_name="line",
    to_interp=None,
    plot_windows=False,
    plot_line=False,
    **kwargs,
):
    """
    Create a window of data either side of NaN's based on "dist_along_line" column and
    interpolate the value. Useful when NaN's are sparse, or lines are long. All kwargs
    are based to function "interp1d"
    """
    df = df.copy()
    kwargs = copy.deepcopy(kwargs)

    extrapolate = kwargs.pop("extrapolate", False)

    # iterate through NaNs
    # values = []
    for i in df[df[to_interp].isnull()].index:
        # get distance along line of NaN
        dist_at_nan = df[dist_col_name].loc[i]

        # try interpolation with set window width, if there's not enough data (bounds
        # error), double the width up to 4 times.
        # if 4 attempts fail and extrapolate is True, allow extrapolation, if False, return NaN
        # if extrapolating, start with original window width and double up to 4 times, if fails, return NaN
        win = window_width
        while win <= window_width * (2**4):
            # logger.info(
            #     "trying interpolation with increase window widths for intersection %s/%s",
            #     df.intersecting_line.loc[i],
            #     df[line_col_name].loc[i],
            # )
            try:
                # get data inside window
                llim, ulim = dist_at_nan - win, dist_at_nan + win
                df_inside = df[df[dist_col_name].between(llim, ulim)]

                # allow extrapolation if nan to fill is at limit of data and extrapolate is True
                # else keep expanding window
                # if (dist_at_nan <= df[dist_col_name].min()) or (dist_at_nan+1 >= df[dist_col_name].max()):
                #     if extrapolate is False:
                #         extrap = False
                #     if extrapolate is True:
                #         logger.info(
                #             "Line/Tie: %s/%s; NaN at limit of data, extrapolating",
                #             df.intersecting_line.loc[i], df[line_col_name].loc[i])
                #         extrap = True
                # else:
                #     extrap = False
                # extrap=True
                # print(df.intersecting_line.loc[i], df[line_col_name].loc[i])
                # if (df.intersecting_line.loc[i] == 410) and (df[line_col_name].loc[i] == 1080):
                #     print("YES")
                #     print(dist_at_nan)
                #     print(df[dist_col_name].min(), df[dist_col_name].max())
                #     # extrap=True
                #     print("extrapolate:", extrap)
                #     print(df_inside[df_inside[to_interp].isnull()])
                #     print(to_interp)

                if len(df_inside) <= 1:
                    win += win
                    logger.warn(
                        "Error with inter: %s/%s doubling window size to %s",
                        df.intersecting_line.loc[i],
                        df[line_col_name].loc[i],
                        win,
                    )
                    continue

                # if (df.intersecting_line.loc[i],df[line_col_name].loc[i]) == (160, 1040):
                # print(f"heere {df.intersecting_line.loc[i],df[line_col_name].loc[i]}")
                # print(df_inside.dropna(subset=[to_interp, "dist_long_line"], how="any"))

                # may be multiple NaN's within window (some outside of bounds)
                # but we only extract the fill value for loc[i]
                filled = interp1d(
                    df_inside,
                    to_interp=[to_interp],
                    extrapolate=False,
                    **kwargs,
                )
                # extract just the filled value
                value = filled[to_interp].loc[i]
                if value == np.nan:
                    raise ValueError("filled value is NaN")
                # if (df.intersecting_line.loc[i],df[line_col_name].loc[i]) == (160, 1040):
                #     print(f"heere {value}")
                # save value to a list
                # values.append(value)
                interp_type = "interpolated"

            except Exception:
                win += win
                logger.warn(
                    "Error with inter: %s/%s doubling window size to %s",
                    df.intersecting_line.loc[i],
                    df[line_col_name].loc[i],
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
                #         "too few points in window for intersection of lines %s & %s "
                #         "doubling window size to %s",
                #         df.intersecting_line.loc[i],
                #         df[line_col_name].loc[i],
                #         win,
                #     )
                # elif any(item in str(e) for item in bounds_errors):
                #     win += win
                #     logger.warning(
                #         "bounds error for interpolation of intersection of lines %s "
                #         "and %s, doubling window size to %s",
                #         df.intersecting_line.loc[i],
                #         df[line_col_name].loc[i],
                #         win,
                #     )
                # else:  # raise other errors
                #     win += win
                #     logger.error(e)
                #     logger.warning(
                #         "Error for interpolation of intersection of lines %s and %s, "
                #         "doubling window size to %s",
                #         df.intersecting_line.loc[i],
                #         df[line_col_name].loc[i],
                #         win,
                #     )
                continue
            break
        else:
            if extrapolate:
                # try extrapolation with set window width, if there's not enough data, double the width up to 4 times.
                win = window_width
                while win <= window_width * (2**4):
                    # logger.info(
                    #     "trying extrapolation with increase window widths for intersection %s/%s",
                    #     df.intersecting_line.loc[i],
                    #     df[line_col_name].loc[i],)
                    try:
                        # get data inside window
                        llim, ulim = dist_at_nan - win, dist_at_nan + win
                        df_inside = df[df[dist_col_name].between(llim, ulim)]

                        if len(df_inside) <= 1:
                            win += win
                            logger.warn(
                                "Error with inter: %s/%s doubling window size to %s",
                                df.intersecting_line.loc[i],
                                df[line_col_name].loc[i],
                                win,
                            )
                            continue

                        # may be multiple NaN's within window (some outside of bounds)
                        # but we only extract the fill value for loc[i]
                        filled = interp1d(
                            df_inside,
                            to_interp=[to_interp],
                            extrapolate=True,
                            **kwargs,
                        )
                        # extract just the filled value
                        value = filled[to_interp].loc[i]
                        if value == np.nan:
                            raise ValueError("filled value is NaN")

                        # save value to a list
                        # values.append(value)
                        interp_type = "extrapolated"

                    except Exception:
                        win += win
                        logger.warn(
                            "Error with inter: %s/%s doubling window size to %s",
                            df.intersecting_line.loc[i],
                            df[line_col_name].loc[i],
                            win,
                        )
                        continue
                    logger.info(
                        "Extrapolated value for inter: %s/%s",
                        df.intersecting_line.loc[i],
                        df[line_col_name].loc[i],
                    )
                    break
                else:
                    logger.error(
                        "Extrapolation failed after window expanded 4 times, to %s "
                        "returning NaN for intersection values",
                        win,
                    )
                    # values.append(np.nan)
                    value = np.nan
                    interp_type = None
            else:
                logger.error(
                    "Window expanded 4 times, to %s, without success and `extrapolate` "
                    "set to False, returning NaN for intersection values",
                    win,
                )
                # values.append(np.nan)
                value = np.nan
                interp_type = None

            if plot_windows is True:
                plot_line_and_crosses(
                    filled,
                    line=filled[line_col_name].iloc[0],
                    x=dist_col_name,
                    y=[to_interp],
                    y_axes=[i + 1 for i in range(len([to_interp]))],
                )
        # add values into dataframe
        df.loc[i, to_interp] = value
        df.loc[i, "interpolation_type"] = interp_type

    # # add values into dataframe
    # df.loc[df[to_interp].isnull(), to_interp] = values

    if plot_line is True:
        plot_line_and_crosses(
            df,
            line=df[line_col_name].iloc[0],
            x=dist_col_name,
            y=[to_interp],
            y_axes=[i + 1 for i in range(len([to_interp]))],
        )

    return df


def interp1d_windows(
    df,
    to_interp=None,
    plot_line=False,
    line_col_name="line",
    dist_col_name="dist_along_line",
    **kwargs,
):
    if line_col_name is not None:
        assert len(df[line_col_name].unique()) <= 1, (
            "Warning: provided more than 1 flight line"
        )

    if isinstance(to_interp, str):
        to_interp = [to_interp]

    df = df.copy()
    kwargs = copy.deepcopy(kwargs)

    # iterate through columns
    with invert4geom_utils.DuplicateFilter(logger):
        for col in to_interp:
            logger.debug(f"Interpolating column: {col}")
            filled = interp1d_windows_single_col(
                df,
                to_interp=col,
                line_col_name=line_col_name,
                **kwargs,
            )
            df = filled
            # df[col] = filled[col]

    if plot_line is True:
        plot_line_and_crosses(
            df,
            line=df[line_col_name].iloc[0],
            x=dist_col_name,
            y=to_interp,
            y_axes=[i + 1 for i in range(len(to_interp))],
        )

    return df


def interp1d(
    df,
    to_interp,
    interp_on,
    engine="scipy",
    method="cubic",
    extrapolate=False,
    fill_value=None,
    plot_line=False,
    line_col_name="line",
):
    """ """
    if line_col_name is not None:
        assert len(df[line_col_name].unique()) <= 1, (
            "Warning: provided more than 1 flight line"
        )

    if isinstance(to_interp, str):
        to_interp = [to_interp]

    df1 = df.copy()

    # iterate through columns
    # with invert4geom_utils.DuplicateFilter(logger):
    for col in to_interp:
        filled = interp1d_single_col(
            df1,
            to_interp=col,
            interp_on=interp_on,
            engine=engine,
            method=method,
            extrapolate=extrapolate,
            fill_value=fill_value,
        )
        # try:
        df1[col] = filled[col]
        # except:
        #     logger.error(f"Error with filling nans in column: {col}")

    if plot_line is True:
        plot_line_and_crosses(
            df1,
            line=df1[line_col_name].iloc[0],
            x=interp_on,
            y=to_interp,
            y_axes=[i + 1 for i in range(len(to_interp))],
        )

    return df1


def interp1d_all_lines(
    df: pd.DataFrame | gpd.GeoDataFrame,
    intersections: pd.DataFrame | gpd.GeoDataFrame,
    to_interp: list[str] | None = None,
    interp_on: str = "dist_along_line",
    method="cubic",
    engine="scipy",
    extrapolate=False,
    fill_value=None,
    line_col_name: str = "line",
    time_col_name: str = "unixtime",
    window_width: float | None = None,
    plot: bool = False,
    plot_variable: str = None,
    wait_for_input: bool = False,
) -> pd.DataFrame | gpd.GeoDataFrame:
    """
    _summary_

    Parameters
    ----------
    df : pd.DataFrame | gpd.GeoDataFrame
        Dataframe containing the data to interpolate
    intersections : pd.DataFrame | gpd.GeoDataFrame
        Dataframe containing the intersection points
    to_interp : list[str] | None, optional
        specify which column to interpolate NaNs for, by default is all columns except
        "is_intersection" and "intersecting_line"
    interp_on : str, optional
        Decide which column interpolation is based on, by default "dist_along_line"
    method : str, optional
        Decide between interpolation methods of 'linear', 'nearest', 'nearest-up',
        'zero', 'slinear', 'quadratic','cubic', 'previous', or 'next' if engine is
        "scipy", or vd.Spline(), vd.SplineCV(), vd.KNeighbors(), vd.Linear(), or
        vd.Cubic() if engine is "verde", by default "cubic"
    engine : str, optional
        Decide between "scipy" and "verde" for performing the interpolation, by default
        "scipy"
    line_col_name : str, optional
        Column name specifying the line numbers, by default "line"
    time_col_name : str, optional
        Column name specifying the time values, by default "unixtime"
    window_width : float, optional
        window width around each NaN to use for interpolation fitting, by default None
    plot : bool, optional
        plot the lines and interpolated points at intersections, by default False
    wait_for_input : bool, optional
        if true, will pause after each plot to allow inspection, by default False

    Returns
    -------
    pd.DataFrame | gpd.GeoDataFrame
        the survey dataframe with NaN's filled in the specified columns
    """
    df = df.copy()
    inters = intersections.copy()

    # add empty rows at each intersection to the df
    df, inters = add_intersections(
        df,
        inters,
        line_col_name=line_col_name,
        time_col_name=time_col_name,
    )

    if to_interp is None:
        to_interp = df.columns.drop(["is_intersection", "intersecting_line"])

    lines = df.groupby(line_col_name)
    filled_lines = []
    pbar = tqdm(lines, desc="Lines")
    for line, line_df in pbar:
        pbar.set_description(f"Line {line}")

        if window_width is None:
            filled = interp1d(
                line_df,
                to_interp=to_interp,
                interp_on=interp_on,
                engine=engine,
                method=method,
                extrapolate=extrapolate,
                fill_value=fill_value,
            )
        else:
            filled = interp1d_windows(
                line_df,
                to_interp=to_interp,
                window_width=window_width,
                interp_on=interp_on,
                engine=engine,
                method=method,
                extrapolate=extrapolate,
                fill_value=fill_value,
            )
        filled_lines.append(filled)

    filled_lines = pd.concat(filled_lines)

    # add whether intersection was interpolated or extrapolated with respect to both lines and ties
    for ind, row in inters.iterrows():
        # search data for values at intersecting lines
        flight_interp_type = filled_lines[
            (filled_lines[line_col_name] == row.line)
            & (filled_lines.intersecting_line == row.tie)
        ].interpolation_type.values[0]
        inters.loc[ind, "flight_interpolation_type"] = flight_interp_type
        tie_interp_type = filled_lines[
            (filled_lines[line_col_name] == row.tie)
            & (filled_lines.intersecting_line == row.line)
        ].interpolation_type.values[0]
        inters.loc[ind, "tie_interpolation_type"] = tie_interp_type

    if plot is True:
        for l in filled_lines.line.unique():
            if plot_variable is None:
                raise ValueError("need to supply variable name to plot")
            fig = plot_line_and_crosses(
                filled_lines,
                line=l,
                x=interp_on,
                y=plot_variable,
                y_axes=[i + 1 for i in range(len(plot_variable))],
                # plot_inters=[True]+[False for i in range(len(plot_variable)-1)],
                plot_inters=True,
            )
            fig.show()
            if wait_for_input is True:
                input("Press key to continue...")
            clear_output(wait=True)

    return filled_lines, inters


def calculate_misties(
    intersections: gpd.GeoDataFrame,
    data: gpd.GeoDataFrame,
    data_col: str,
    line_col_name: str = "line",
    plot: bool = False,
    robust: bool = True,
) -> gpd.GeoDataFrame:
    """
    Calculate mistie values for all intersections. For each intersection, find the data
    values for the line and tie from the survey dataframe and add those values to the
    intersection table as `line_value` and `tie_value`. If they exist, overwrite them.
    Calculate the mistie value as line_value - tie_value, and save this to a column
    `mistie_0`. If `mistie_0` exists, make a new column `mistie_1`, etc. If the new
    mistie values exactly match previous, don't make a new column. This allow to run
    the function multiple times without changing anything and not building up a large
    number of mistie columns.

    Parameters
    ----------
    intersections : gpd.GeoDataFrame
        Intersections table created by `create_intersection_table()`, then supplied to
        `add_intersections()`.
    data : gpd.GeoDataFrame
        Survey dataframe with intersection rows added by `add_intersections()` and
        interpolated with `interp1d_all_lines()`.
    data_col : str
        Column name for data values to calculate misties for
    line_col_name : str, optional
        Column name specifying the line numbers, by default "line"
    plot : bool, optional
        Plot the resulting mistie points on a map, by default False
    robust : bool, optional
        Use robust color limits for the map, by default True

    Returns
    -------
    gpd.GeoDataFrame
        An intersections table with new columns `line_value`, `tie_value` and `mistie_x`
        where x is incremented each time a new mistie is calculated.
    """

    inters = intersections.copy()
    df = data.copy()

    # save previous mistie columns by adding an integer to the end
    # if "mistie" in inters.columns:
    #     version = 0
    #     col_name = f"mistie_{0}"
    #     while col_name in inters.columns:
    #         version += 1
    #         col_name = f"mistie_{version}"
    #     inters = inters.rename(columns={"mistie": col_name})
    # get list of columns starting with "mistie"

    # iterate through intersections
    misties = []
    for ind, row in inters.iterrows():
        # search data for values at intersecting lines
        line_value = df[
            (df[line_col_name] == row.line) & (df.intersecting_line == row.tie)
        ][data_col].values[0]
        tie_value = df[
            (df[line_col_name] == row.tie) & (df.intersecting_line == row.line)
        ][data_col].values[0]

        assert line_value != np.nan
        assert tie_value != np.nan

        # mistie is line - tie
        misties.append(line_value - tie_value)
        # inters.loc[ind, "line_value"] = line_value
        # inters.loc[ind, "tie_value"] = tie_value

        # add misties to rows of data df which are intersection points
        # conditions = (df[line_col_name] == row.line) & (df.intersecting_line == row.tie)
        # df.loc[conditions, "mistie"] = line_value - tie_value

        # conditions = (df[line_col_name] == row.tie) & (df.intersecting_line == row.line)
        # df.loc[conditions, "mistie"] = line_value - tie_value

    # misties are defined as line - tie
    # misties = inters.line_value - inters.tie_value
    misties = pd.Series(misties)
    logger.info(f"mistie RMSE: {utils.rmse(misties)}")

    # if len(cols) == 0:
    #     mistie_col = "mistie_0"
    #     past_mistie_col = None
    #     logger.info("Created initial mistie column: %s", past_mistie_col)
    # else:
    #     mistie_col = f"mistie_{len(cols)}"
    #     past_mistie_col = f"mistie_{len(cols)-1}"
    # if len(cols) == 0:
    #     logger.info("Created initial mistie column: `mistie")
    # else:
    #     mistie_col = f"mistie_{len(cols)}"
    #     past_mistie_col = f"mistie_{len(cols)-1}"

    # get the latest mistie column
    # mistie_col = [int(col.split("_")[-1]) for col in inters.columns if "mistie" in col]
    # mistie_col = f"mistie_{max(mistie_col)}"

    # print("1",inters.mistie - misties)

    cols = [c for c in inters.columns if "mistie_" in c]
    cols = [c for c in cols if c[-1].isdigit()]

    # no mistie columns
    if "mistie" not in inters.columns:
        logger.info("Created initial mistie column: `mistie`")
        inters["mistie"] = misties
    else:
        next_mistie_col = f"mistie_{len(cols)}"
        # check if new misties are identical to old misties
        try:
            pd.testing.assert_series_equal(
                inters.mistie,
                misties,
                check_names=False,
            )

            logger.info("Mistie values are unchanged")
        except AssertionError:
            logger.info(
                "mistie column exists, replacing with new misties values and renaming old column `%s`",
                next_mistie_col,
            )
            inters = inters.rename(columns={"mistie": next_mistie_col})
            inters["mistie"] = misties

        # try:
        #     pd.testing.assert_series_equal(
        #         inters.mistie,
        #         inters[next_mistie_col],
        #         check_names=False,
        #     )
        #     logger.info("Mistie values are unchanged, using past mistie colum: %s", past_mistie_col)
        #     mistie_col = past_mistie_col
        # except AssertionError:
        #     inters[mistie_col] = misties
        #     logger.info("Previous mistie column: %s", past_mistie_col)
        #     logger.info("New mistie column: %s", mistie_col)

    if plot is True:
        plotly_points(
            inters,
            color_col="mistie",
            hover_cols=["line", "tie"],
            robust=robust,
            absolute=True,
            cmap="balance",
            point_size=10,
        )
        plt.hist(inters.mistie)
        plt.xlabel("Mistie value")
        plt.title("Histogram of misties")

    return inters


def verde_predict_trend(
    data_to_fit: pd.DataFrame,
    cols_to_fit: list,
    data_to_predict: pd.DataFrame,
    cols_to_predict: list,
    degree: int,
):
    """
    data_to_fit: pd.DataFrame with at least 3 columns: x, y, and data
    cols_to_fit: column names representing x, y, and data
    data_to_predict: pd.DataFrame with at least 2 columns: x, y
    cols_to_predict: column names representing x and y, and 3rd value for new column
        with predicted data
    """
    fit_df = data_to_fit.copy()
    predict_df = data_to_predict.copy()

    # fit a polynomial trend through the lines mistie values
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Under-determined problem")
        # fit a trend to the data values
        fit_coords = (fit_df[cols_to_fit[0]], fit_df[cols_to_fit[1]])
        trend = vd.Trend(degree=degree).fit(fit_coords, fit_df[cols_to_fit[2]])

        # predict the trend on the new values
        predict_coords = (
            predict_df[cols_to_predict[0]],
            predict_df[cols_to_predict[1]],
        )
        predicted = trend.predict(predict_coords)
        predict_df[cols_to_predict[2]] = predicted

    return predict_df


def skl_predict_trend(
    data_to_fit: pd.DataFrame,
    cols_to_fit: list,
    data_to_predict: pd.DataFrame,
    cols_to_predict: list,
    degree: int,
    sample_weight_col: str | None = None,
) -> pd.DataFrame:
    """
    data_to_fit: pd.DataFrame with at least 2 columns: distance, and data
    cols_to_fit: column names representing distance and data
    data_to_predict: pd.DataFrame with at least 1 columns: distance
    cols_to_predict: column names representing distance and new column
        with predicted data
    sample_weight_col: column name for sample weights
    """
    fit_df = data_to_fit.copy()
    predict_df = data_to_predict.copy()

    # fit a polynomial trend through the lines mistie values
    polynomial_features = PolynomialFeatures(
        degree=degree,
        include_bias=True,
    )
    linear_regression = LinearRegression()

    if sample_weight_col is not None:
        sample_weight = fit_df[sample_weight_col].to_numpy()
    else:
        sample_weight = None

    # x_poly = polynomial_features.fit_transform(
    #     fit_df[cols_to_fit[0]].to_numpy()[:, np.newaxis]
    # )
    # linear_regression.fit(
    #     x_poly,
    #     fit_df[cols_to_fit[1]].to_numpy(),
    #     sample_weight=sample_weight,
    # )
    # predicted = linear_regression.predict(x_poly)
    # print(predicted)
    # predict_df[cols_to_predict[1]] = predicted

    pipeline = Pipeline(
        [
            ("polynomial_features", polynomial_features),
            ("linear_regression", linear_regression),
        ]
    )
    pipeline.fit(
        fit_df[cols_to_fit[0]].to_numpy()[:, np.newaxis],
        fit_df[cols_to_fit[1]].to_numpy(),
        linear_regression__sample_weight=sample_weight,
    )
    predicted = pipeline.predict(
        predict_df[cols_to_predict[0]].to_numpy()[:, np.newaxis]
    )
    predict_df[cols_to_predict[1]] = predicted

    return predict_df


def level_survey_lines_to_grid(
    data: pd.DataFrame | gpd.GeoDataFrame,
    grid_col: str,
    degree: int,
    data_col: str,
    line_col_name: str = "line",
    distance_col: str = "dist_along_line",
    levelled_col: str = "levelled",
    plot: bool = False,
) -> pd.DataFrame | gpd.GeoDataFrame:
    """
    With grid values sampled along survey flight lines (grid_col), fit a trend
    or specified order to the misfit values (data_col - grid_col) and
    apply the correction to the data. The levelled data is saved in a new column
    specified by levelled_col.

    Parameters
    ----------
    df : pd.DataFrame | gpd.GeoDataFrame
        _description_
    grid_col : str
        _description_
    degree : int
        _description_
    data_col : str
        _description_
    line_col_name : str, optional
        _description_, by default "line"
    distance_col : str, optional
        _description_, by default "dist_along_line"
    levelled_col : str, optional
        _description_, by default "levelled"

    Returns
    -------
    pd.DataFrame | gpd.GeoDataFrame
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    df = data.copy()

    # calculate misfit survey data and sampled grid values
    if "misfit" in df.columns:
        msg = "Column 'misfit' already exists in dataframe, dropping it"
        # raise ValueError(msg)
        logger.warn(msg)
        df = df.drop(columns="misfit")

    if levelled_col in df.columns:
        logger.info("supplied levelled_col already in dataframe")

    df["misfit"] = df[data_col] - df[grid_col]

    # fit a trend to the misfits on line-by-line basis
    for line in df[line_col_name].unique():
        # subset a line
        line_df = df[df[line_col_name] == line]

        # calculate correction by fitting trend to misfit values
        correction = skl_predict_trend(
            data_to_fit=line_df,
            cols_to_fit=[distance_col, "misfit"],
            data_to_predict=line_df,
            cols_to_predict=[distance_col, "correction"],
            degree=degree,
        ).correction

        # add correction values to the main dataframe
        df.loc[df[line_col_name] == line, "levelling_correction"] = correction

        # apply correction to the data
        # df.loc[df[line_col_name]==line, levelled_col] = line_df[data_col] - correction
    logger.info("RMS of correction: %s", utils.rmse(correction))
    # apply correction to the data
    df[levelled_col] = df[data_col] - df.levelling_correction

    # # add correction to existing correction column if it exists
    # if correction_column_name in df.columns:
    #     df[correction_column_name] += df[f"trend_{degree}_correction"]
    # else:
    #     df[correction_column_name] = df[f"trend_{degree}_correction"]

    if plot:
        # plot old misties
        plotly_points(
            df,
            color_col="misfit",
            hover_cols=[line_col_name, data_col, "misfit"],
            cmap="balance",
            robust=False,
            absolute=True,
            point_width=0,
            point_size=5,
        )

        plotly_points(
            df,
            color_col="levelling_correction",
            hover_cols=[line_col_name, data_col, levelled_col],
            cmap="balance",
            robust=True,
            absolute=True,
            point_width=0,
            point_size=5,
        )
    return df.drop(columns=["misfit", "levelling_correction"])


def level_lines(
    inters: gpd.GeoDataFrame | pd.DataFrame,
    data: gpd.GeoDataFrame | pd.DataFrame,
    lines_to_level: list[float],
    data_col: str,
    levelled_col: str,
    cols_to_fit: str | None = None,
    cols_to_predict: str = "dist_along_line",
    degree: int | None = None,
    line_col_name: str = "line",
    sample_weight_col: str | None = None,
    plot=False,
):
    """
    Level lines based on intersection misties values. Fit a trend of specified order to
    intersection misties, and apply the correction to the `data_col` column.
    """
    df = data.copy()
    inters = inters.copy()

    # check if levelling lines to ties or vice versa
    levelling_ties = False
    levelling_lines = False
    for j in lines_to_level:
        if j in inters.tie.unique():
            levelling_ties = True
        if j in inters.line.unique():
            levelling_lines = True
    if (levelling_ties is True) & (levelling_lines is True):
        msg = "Supplied both lines and ties to be levelled!"
        raise ValueError(msg)
    if levelling_lines is True:
        logger.debug("Levelling lines to ties")
    elif levelling_ties is True:
        logger.debug("Levelling ties to lines")

    if cols_to_fit is None:
        # if levelling to ties, fit to dist_along_flight_tie
        # if lines_to_level[0] in inters.tie.unique():
        if levelling_ties is True:
            cols_to_fit = "dist_along_flight_tie"
            logger.debug("using column: %s", cols_to_fit)
        # elif lines_to_level[0] in inters.line.unique():
        elif levelling_lines is True:
            cols_to_fit = "dist_along_flight_line"
            logger.debug("using column: %s", cols_to_fit)

    # convert columns to fit on into a list if its a string
    if isinstance(cols_to_fit, str):
        cols_to_fit = [cols_to_fit]
    if isinstance(cols_to_predict, str):
        cols_to_predict = [cols_to_predict]

    # df levelled_col = f"{data_col}_levelled"
    # df[levelled_col] = np.nan
    df["levelling_correction"] = np.nan

    # get the latest mistie column
    inters2 = calculate_misties(
        inters,
        df,
        data_col=data_col,
        plot=False,
    )
    logger.info("mistie before levelling: %s mGal", utils.rmse(inters2.mistie))
    # mistie_col = [int(col.split("_")[-1]) for col in inters.columns if "mistie" in col]
    # mistie_col = f"mistie_{max(mistie_col)}"
    # logger.debug("most recent mistie column: %s", mistie_col)

    # fit a trend to the misfits on line-by-line basis
    # iterate through the chosen lines
    logger.debug("levelling the data")
    for line in lines_to_level:
        # subset a line
        line_df = df[df[line_col_name] == line].copy()

        # get intersections of line of interest
        ints = inters2[(inters2.line == line) | (inters2.tie == line)]

        # fit a polynomial trend through the lines mistie values
        # if predicting on 2 variables (easting and northing) use verde
        if len(cols_to_fit) > 1:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Under-determined problem")
                try:
                    line_df = verde_predict_trend(
                        data_to_fit=ints,
                        cols_to_fit=cols_to_fit + ["mistie"],  # noqa: RUF005
                        data_to_predict=line_df,
                        cols_to_predict=cols_to_predict + ["levelling_correction"],  # noqa: RUF005
                        degree=degree,
                    )
                except ValueError as e:
                    if "zero-size array to reduction operation minimum which" in str(e):
                        logger.error("Issue with line %s, skipping", line)
                        # if issues, correction is 0
                        line_df["levelling_correction"] = 0
                    else:
                        raise e
        # if predicting on 1 variable (distance along line) use scikitlearn
        elif len(cols_to_fit) == 1:
            try:
                line_df = skl_predict_trend(
                    data_to_fit=ints,  # df with mistie values
                    cols_to_fit=cols_to_fit  # noqa: RUF005
                    + ["mistie"],  # column names for distance/mistie
                    data_to_predict=line_df,  # df with line data
                    cols_to_predict=cols_to_predict  # noqa: RUF005
                    + [
                        "levelling_correction"
                    ],  # column names for distance/ levelling correction
                    degree=degree,  # degree order for fitting line to misties
                    sample_weight_col=sample_weight_col,
                )
            except ValueError as e:
                if "Found array with " in str(e):
                    logger.error("Issue with line %s, skipping", line)
                    # if issues, correction is 0
                    line_df["levelling_correction"] = 0
                else:
                    raise e

        # if levelling tie lines, negate the correction
        if levelling_ties is True:
            line_df["levelling_correction"] *= -1
        else:
            pass

        # remove the trend from the gravity
        values = line_df[data_col] - line_df.levelling_correction

        # update main df
        df.loc[df[line_col_name] == line, levelled_col] = values
        df.loc[df[line_col_name] == line, "levelling_correction"] = (
            line_df.levelling_correction
        )

    # add unchanged values for lines not included
    for line in df[line_col_name].unique():
        if line not in lines_to_level:
            df.loc[df[line_col_name] == line, levelled_col] = df.loc[
                df[line_col_name] == line, data_col
            ]
    # print(df)
    # try:
    #     pd.testing.assert_series_equal(
    #         df[levelled_col],
    #         df[data_col],
    #         check_names=False,
    #     )
    #     logger.error("Levelled col data identical to data to level, return unchanged dataframes")
    #     return data, inters
    # except AssertionError:
    #     pass

    # update mistie with levelled data
    inters = calculate_misties(
        inters,
        df,
        data_col=levelled_col,
        plot=False,
    )
    logger.info("mistie after levelling: %s mGal", utils.rmse(inters.mistie))

    if plot is True:
        # plot old misties
        ints = inters2[
            inters2.line.isin(lines_to_level) | inters2.tie.isin(lines_to_level)
        ]
        plotly_points(
            ints,
            color_col="mistie",
            hover_cols=[
                "line",
                "tie",
                "mistie",
            ],
            cmap="balance",
            absolute=True,
            # point_width=0,
            point_size=10,
            theme=None,
        )
        # plotly_points(
        #     ints,
        #     color_col=mistie_col,
        #     point_size=4,
        #     hover_cols=[
        #         "line",
        #         "tie",
        #         "line_value",
        #         "tie_value",
        #         mistie_col,
        #     ],
        # )

        plotly_points(
            df[df[line_col_name].isin(lines_to_level)],
            color_col="levelling_correction",
            hover_cols=[line_col_name, data_col, levelled_col],
            cmap="balance",
            absolute=True,
            point_width=0,
            point_size=5,
            theme=None,
            robust=True,
        )

    return df.drop(columns=["levelling_correction"]), inters


def iterative_line_levelling(
    inters: gpd.GeoDataFrame | pd.DataFrame,
    data: gpd.GeoDataFrame | pd.DataFrame,
    lines_to_level: list[float],
    data_col: str,
    levelled_col: str,
    cols_to_fit: str | None = None,
    cols_to_predict: str = "dist_along_line",
    degree: int | None = None,
    line_col_name: str = "line",
    sample_weight_col: str | None = None,
    iterations: int = 5,
    plot_results=False,
    plot_convergence=False,
    **kwargs,
):
    df = data.copy()
    ints = inters.copy()

    for i in range(1, iterations + 1):
        df, ints = level_lines(
            ints,
            df,
            lines_to_level=lines_to_level,
            cols_to_fit=cols_to_fit,
            cols_to_predict=cols_to_predict,
            degree=degree,
            data_col=data_col,
            levelled_col=levelled_col,
            line_col_name=line_col_name,
            sample_weight_col=sample_weight_col,
        )
        data_col = levelled_col

    if plot_convergence is True:
        plot_levelling_convergence(
            ints,
            logy=kwargs.get("logy", False),
            title=kwargs.get("title", "Levelling convergence"),
            as_median=kwargs.get("as_median", False),
        )
    if plot_results is True:
        # plot flight lines
        plotly_points(
            df[df.line.isin(lines_to_level)],
            color_col="levelling_correction",
            point_size=4,
            hover_cols=[
                line_col_name,
                f"{levelled_data_prefix}_{i}",
            ],
        )

    return df, ints


# def iterative_line_levelling(
#     inters: gpd.GeoDataFrame | pd.DataFrame,
#     data: gpd.GeoDataFrame | pd.DataFrame,
#     lines_to_level: list[float],
#     data_col: str,
#     levelled_col: str,
#     cols_to_fit: str | None = None,
#     cols_to_predict: str = "dist_along_line",
#     degree: int | None = None,
#     line_col_name: str = "line",
#     iterations : int = 5,
#     plot_iterations=False,
#     plot_results=False,
#     plot_convergence=False,
#     **kwargs,
# ):
#     df = data.copy()
#     ints = inters.copy()

#     if mistie_prefix is None:
#         mistie_prefix = f"mistie_trend{degree}"
#     if levelled_data_prefix is None:
#         levelled_data_prefix = f"levelled_data_trend{degree}"

#     rmse = utils.rmse(ints[starting_mistie_col])
#     logger.info("Starting RMS mistie: %s mGal", rmse)

#     for i in range(1, iterations + 1):
#         if i == 1:
#             data_col = starting_data_col
#             mistie_col = starting_mistie_col
#         else:
#             data_col = f"{levelled_data_prefix}_{i-1}"
#             mistie_col = f"{mistie_prefix}_{i-1}"

#         # with inv_utils.HiddenPrints():
#         df, ints = level_lines(
#             ints,
#             df,
#             lines_to_level=lines_to_level,
#             cols_to_fit=cols_to_fit,
#             cols_to_predict=cols_to_predict,
#             degree=degree,
#             data_col=data_col,
#             levelled_col=f"{levelled_data_prefix}_{i}",
#             mistie_col=mistie_col,
#             new_mistie_col=f"{mistie_prefix}_{i}",
#             line_col_name=line_col_name,
#         )
#         rmse = utils.rmse(ints[f"{mistie_prefix}_{i}"])
#         logger.info("RMS mistie after iteration %s: %s mGal", i, rmse)
#         rmse_corr = utils.rmse(df[(df.line.isin(lines_to_level))].levelling_correction)
#         logger.info("RMS correction to lines: %s mGal", rmse_corr)

#         if plot_iterations is True:
#             # plot flight lines
#             plotly_points(
#                 df[df.line.isin(lines_to_level)],
#                 color_col="levelling_correction",
#                 point_size=2,
#                 hover_cols=[
#                     line_col_name,
#                     f"{levelled_data_prefix}_{i}",
#                 ],
#             )

#     levelled_col = list(df.columns)[-1]

#     df["levelling_correction"] = df[starting_data_col] - df[levelled_col]
#     if plot_convergence is True:
#         plot_levelling_convergence(
#             ints,
#             mistie_prefix=mistie_prefix,
#             logy=kwargs.get("logy", False),
#         )
#     if plot_results is True:
#         # plot flight lines
#         plotly_points(
#             df[df.line.isin(lines_to_level)],
#             color_col="levelling_correction",
#             point_size=4,
#             hover_cols=[
#                 line_col_name,
#                 f"{levelled_data_prefix}_{i}",
#             ],
#         )

#     return (
#         df,
#         ints,
#     )


def iterative_levelling_alternate(
    inters: gpd.GeoDataFrame | pd.DataFrame,
    data: gpd.GeoDataFrame | pd.DataFrame,
    tie_line_names: list[float],
    flight_line_names: list[float],
    data_col: str,
    levelled_col: str,
    cols_to_fit: str | None = None,
    cols_to_predict: str = "dist_along_line",
    degree: int | None = None,
    line_col_name: str = "line",
    sample_weight_col: str | None = None,
    iterations: int = 5,
    plot_results=False,
    plot_convergence=False,
    **kwargs,
):
    df = data.copy()
    ints = inters.copy()

    for i in range(1, iterations + 1):
        # level lines to ties
        ints = calculate_misties(
            ints,
            df,
            data_col=data_col,
        )
        prior_mistie = utils.rmse(ints.mistie)
        df, ints = level_lines(
            ints,
            df,
            lines_to_level=flight_line_names,
            cols_to_fit=cols_to_fit,
            cols_to_predict=cols_to_predict,
            degree=degree,
            data_col=data_col,
            levelled_col=levelled_col,
            line_col_name=line_col_name,
            sample_weight_col=sample_weight_col,
        )
        # level ties to lines
        df, ints = level_lines(
            ints,
            df,
            lines_to_level=tie_line_names,
            cols_to_fit=cols_to_fit,
            cols_to_predict=cols_to_predict,
            degree=degree,
            data_col=levelled_col,
            levelled_col=levelled_col,
            line_col_name=line_col_name,
            sample_weight_col=sample_weight_col,
        )
        data_col = levelled_col
        post_mistie = utils.rmse(ints.mistie)
        if post_mistie > prior_mistie:
            logger.warn("Mistie increased, ending iterations")
            break

    if plot_convergence is True:
        plot_levelling_convergence(
            ints,
            logy=kwargs.get("logy", False),
            title=kwargs.get("title", "Levelling convergence"),
            as_median=kwargs.get("as_median", False),
        )
    # if plot_results is True:
    #     # plot flight lines
    #     plotly_points(
    #         df[df.line.isin(flight_line_names)],
    #         color_col="levelling_correction",
    #         point_size=2,
    #         hover_cols=[
    #             "line",
    #             f"{levelled_data_prefix}_{i}l",
    #         ],
    #     )
    #     # plot tie lines
    #     plotly_points(
    #         df[df.line.isin(tie_line_names)],
    #         color_col="levelling_correction",
    #         point_size=4,
    #         hover_cols=[
    #             "line",
    #             f"{levelled_data_prefix}_{i}t",
    #         ],
    #     )

    return df, ints


def plotly_points(
    df,
    coord_names=None,
    color_col=None,
    hover_cols=None,
    point_size=4,
    point_width=1,
    cmap=None,
    cmap_middle=None,
    cmap_lims=None,
    robust=True,
    absolute=False,
    theme=None,
    title=None,
):
    """
    Create a scatterplot of spatial data. By default, coordinates are extracted from
    geopandas geometry column, or from user specified columns given by 'coord_names'.
    """
    data = df[df[color_col].notna()].copy()

    if coord_names is None:
        try:
            x = data.geometry.x
            y = data.geometry.y
        except AttributeError:
            try:
                x = data["easting"]
                y = data["northing"]
            except AttributeError:
                try:
                    x = data["x"]
                    y = data["y"]
                except AttributeError:
                    pass
        coord_names = (x, y)

    # either
    if cmap_lims is None:
        vmin, vmax = utils.get_min_max(
            data[color_col],
            robust=robust,
            absolute=absolute,
        )
        cmap_lims = (vmin, vmax)
    else:
        vmin, vmax = cmap_lims

    if cmap is None:
        if (cmap_lims[0] < 0) and (cmap_lims[1] > 0):
            cmap = "balance"
            cmap_middle = 0
        else:
            cmap = None
            cmap_middle = None
    else:
        cmap_middle = None

    if cmap_middle == 0:
        max_abs = vd.maxabs((vmin, vmax))
        cmap_lims = (-max_abs, max_abs)

    fig = px.scatter(
        data,
        x=coord_names[0],
        y=coord_names[1],
        color=data[color_col],
        color_continuous_scale=cmap,
        color_continuous_midpoint=cmap_middle,
        range_color=cmap_lims,
        hover_data=hover_cols,
        template=theme,
    )
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )

    fig.update_layout(
        title_text=title,
        autosize=False,
        width=800,
        height=800,
    )

    fig.update_traces(
        marker={"size": point_size, "line": {"color": "black", "width": point_width}}
    )

    fig.show()


def plotly_profiles(
    data,
    y: tuple[str],
    x="dist_along_line",
    y_axes=None,
    x_lims=None,
    y_lims=None,
    **kwargs,
):
    """
    plot data profiles with plotly
    currently only allows 3 separate y axes, set with "y_axes", starting with 1
    """
    df = data.copy()

    # turn y column name into list
    if isinstance(y, str):
        y = [y]

    # list of y axes to use, if none, all will be same
    y_axes = ["" for _ in y] if y_axes is None else [str(x) for x in y_axes]
    assert "0" not in y_axes, "No '0' or 0 allowed, axes start with 1"
    # convert y axes to plotly expected format: "y", "y2", "y3" ...
    y_axes = [s.replace("1", "") for s in y_axes]
    y_axes = [f"y{x}" for x in y_axes]

    # # lim x and y ranges
    # if xlims is not None:
    #     df = df[df[x].between(*xlims)]
    # if ylims is not None:
    #     df = df[df[y].between(*ylims)]

    if y_lims is not None:
        for i in y_lims:
            if isinstance(i, list | tuple):
                assert len(y_lims) == len(y), "y_lims must be same length as y"
            elif isinstance(i, int | float):
                y_lims = [y_lims for _ in y]

    # set plotting mode
    modes = kwargs.get("modes")
    if modes is None:
        modes = ["markers" for _ in y]

    # set marker properties
    marker_sizes = kwargs.get("marker_sizes")
    marker_symbols = kwargs.get("marker_symbols")
    if marker_sizes is None:
        marker_sizes = [2 for _ in y]
    if marker_symbols is None:
        marker_symbols = ["circle" for _ in y]

    fig = go.Figure()

    # iterate through data columns
    for i, col in enumerate(y):
        fig.add_trace(
            go.Scatter(
                mode=modes[i],
                x=df[x],
                y=df[col],
                name=col,
                marker_size=marker_sizes[i],
                marker_symbol=marker_symbols[i],
                marker_color=plotly.colors.DEFAULT_PLOTLY_COLORS[i],
                yaxis=y_axes[i],
            )
        )

    unique_axes = len(pd.Series(y_axes).unique())
    if unique_axes >= 1:
        y_axes_args = dict(yaxis=dict(title=y[y_axes.index("y")]))
        x_domain = [0, 1]
    if unique_axes >= 2:
        y_axes_args["yaxis2"] = dict(
            title=y[y_axes.index("y2")], overlaying="y", side="right"
        )
        x_domain = [0, 1]
    if unique_axes >= 3:
        y_axes_args["yaxis3"] = dict(
            title=y[y_axes.index("y3")],
            anchor="free",
            overlaying="y",
        )
        x_domain = [0.15, 1]
    else:
        pass

    if x_lims is not None:
        fig.update_layout(xaxis=dict(range=x_lims))
    for i, col in enumerate(y):
        if y_lims is not None:
            if y_axes[i] == "y":
                y_axes_args["yaxis"]["range"] = y_lims[i]
            elif y_axes[i] == "y2":
                y_axes_args["yaxis2"]["range"] = y_lims[i]
            elif y_axes[i] == "y3":
                y_axes_args["yaxis3"]["range"] = y_lims[i]
            else:
                y_axes_args["yaxis3"]["range"] = y_lims[i]

    fig.update_layout(
        title_text=kwargs.get("title"),
        xaxis=dict(
            title=x,
            domain=x_domain,
        ),
        **y_axes_args,
    )

    return fig


def plot_line_and_crosses(
    df,
    y: tuple[str],
    line=None,
    line_col_name="line",
    x="dist_along_line",
    plot_inters=False,
    use_intersection_y=True,
    y_axes=None,
    x_lims=None,
    y_lims=None,
    plot_type="plotly",
    **kwargs,
):
    """
    plot lines and crosses
    """

    # turn y column name into list
    if isinstance(y, str):
        y = [y]

    # list of y axes to use, if none, all will be same
    if y_axes is None:
        y_axes = ["1" for _ in y]

    if len(df[line_col_name].unique()) <= 1:
        line = df[line_col_name].iloc[0]

    try:
        line_df = df[df[line_col_name] == line].sort_values(
            by=["line", "intersecting_line"]
        )
    except KeyError:
        line_df = df[df[line_col_name] == line].sort_values(by=["line"])

    # if xlims is not None:
    #     line_df = line_df[line_df[x].between(*xlims)]
    # if ylims is not None:
    #     line_df = line_df[line_df[x].between(*ylims)]

    # list of which dataset to plot intersections for
    if plot_inters is True:
        plot_inters = []
        for i in y:
            if len(line_df[i].dropna()) == len(line_df[line_df.is_intersection]):
                plot_inters.append(False)
            else:
                plot_inters.append(True)

    if plot_type == "plotly":
        fig = plotly_profiles(
            line_df,
            x=x,
            y=y,
            y_axes=y_axes,
            y_lims=y_lims,
            x_lims=x_lims,
            title=f"Line: {line}",
            **kwargs,
        )
        # convert numbers to strings
        y_axes = [str(i) for i in y_axes]
        assert "0" not in y_axes, "No '0' or 0 allowed, axes start with 1"
        # convert y axes to plotly expected format: "y", "y2", "y3" ...
        y_axes = [s.replace("1", "") for s in y_axes]
        y_axes = [f"y{i}" for i in y_axes]

        if plot_inters is not False:
            for i, z in enumerate(y):
                if plot_inters[i] is True:
                    intersections = df[df.intersecting_line == line].sort_values(
                        by=["line", "intersecting_line"]
                    )
                    # in no intersection in database yet, plot point with y value of line
                    if len(intersections) == 0:
                        # logger.info("using y value from line")
                        y_val = line_df[line_df.is_intersection][z]
                        text = line
                    # if intersections exists, use y value of intersecting line
                    else:
                        # logger.info("using y value from crossing line")
                        y_val = intersections[z]
                        text = intersections.line
                    if use_intersection_y is False:
                        y_val = line_df[line_df.is_intersection][z]

                    fig.add_trace(
                        go.Scatter(
                            mode="markers+text",
                            x=line_df[line_df.is_intersection][x],
                            y=y_val,
                            yaxis=y_axes[i],
                            marker_size=5,
                            marker_symbol="diamond",
                            marker_color=plotly.colors.DEFAULT_PLOTLY_COLORS[i],
                            name=f"{z} intersections",
                            text=text,
                            textposition="top center",
                        ),
                    )
                else:
                    pass

        # fig.show()

    elif plot_type == "mpl":
        fig, ax1 = plt.subplots(figsize=(9, 6))
        plt.grid()
        plot_elements = []
        for i, z in enumerate(y):
            if i > 0:
                ax2 = ax1.twinx()
                axis = ax2
                color = kwargs.get("point_color", "orangered")
            else:
                axis = ax1
                color = kwargs.get("point_color", "mediumslateblue")

            plotted_line = axis.plot(
                line_df[x],
                line_df[z],
                linewidth=0.5,
                color=color,
                marker=".",
                markersize=kwargs.get("point_size", 0.1),
                label=z,
            )
            plot_elements.append(plotted_line[0])
            axis.set_ylabel(z)

            if plot_inters is not False:
                intersections = df[df.intersecting_line == line]
                # in no intersection in database yet, plot point with y value of line
                if len(intersections) == 0:
                    y = line_df[line_df.is_intersection][z]
                    text = line
                # if intersections exists, use y value of intersecting line
                else:
                    y = intersections[z]
                    text = intersections.line
                plotted_point = axis.scatter(
                    x=line_df[line_df.is_intersection][x],
                    y=y,
                    s=kwargs.get("point_size", 20),
                    c=color,
                    marker="x",
                    zorder=2,
                    label=f"{z} intersections",
                )
                plot_elements.append(plotted_point)
                for j, dist in enumerate(line_df[line_df.is_intersection][x]):
                    axis.text(
                        dist,
                        y.values[j],
                        s=text.values[j],
                        fontsize="x-small",
                    )

        labels = [x.get_label() for x in plot_elements]
        plt.legend(plot_elements, labels)

        ax1.set_xlabel(x)

        plt.title(f"Line number: {line}")
        # plt.show()
    return fig


def plot_flightlines(
    fig: pygmt.Figure,
    df: pd.DataFrame,
    direction: str = "EW",
    plot_labels: bool = True,
    plot_lines: bool = True,
    **kwargs,
):
    # group lines by their line number
    lines = [v for _, v in df.groupby("line")]

    # plot lines
    if plot_lines is True:
        for i in list(range(len(lines))):
            fig.plot(
                x=lines[i].easting,
                y=lines[i].northing,
                # pen=kwargs.get("pen", "0.3p,white"),
                style=kwargs.get("style", "p.5p"),
                fill=kwargs.get("fill", "gray"),
            )

    # plot labels
    if plot_labels is True:
        for i in list(range(len(lines))):
            # switch label locations for every other line
            if (i % 2) == 0:
                if direction == "EW":
                    offset = "0.25c/0c"
                    # plot label at max value of x-coord
                    x_or_y = "easting"
                    # angle of label
                    angle = 0
                elif direction == "NS":
                    offset = "0c/0.25c"
                    # plot label at max value of y-coord
                    x_or_y = "northing"
                    # angle of label
                    angle = 90
                else:
                    msg = "invalid direction string"
                    raise ValueError(msg)
                # plot label
                fig.text(
                    x=lines[i].easting.loc[lines[i][x_or_y].idxmax()],
                    y=lines[i].northing.loc[lines[i][x_or_y].idxmax()],
                    text=str(int(lines[i].line.iloc[0])),
                    justify="CM",
                    font=kwargs.get("font", "5p,black"),
                    fill="white",
                    offset=offset,
                    angle=angle,
                )
            else:
                if direction == "EW":
                    offset = "-0.25c/0c"
                    # plot label at max value of x-coord
                    x_or_y = "easting"
                    # angle of label
                    angle = 0
                elif direction == "NS":
                    offset = "0c/-0.25c"
                    # plot label at max value of y-coord
                    x_or_y = "northing"
                    # angle of label
                    angle = 90
                else:
                    msg = "invalid direction string"
                    raise ValueError(msg)
                # plot label
                fig.text(
                    x=lines[i].easting.loc[lines[i][x_or_y].idxmin()],
                    y=lines[i].northing.loc[lines[i][x_or_y].idxmin()],
                    text=str(int(lines[i].line.iloc[0])),
                    justify="CM",
                    font=kwargs.get("font", "5p,black"),
                    fill="white",
                    offset=offset,
                    angle=angle,
                )
