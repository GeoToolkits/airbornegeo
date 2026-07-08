import itertools  # pylint: disable=too-many-lines
import typing

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import seaborn as sns
import shapely
from IPython.display import clear_output
from shapely.geometry import LineString, Point
from tqdm.autonotebook import tqdm

import airbornegeo
from airbornegeo import logger

sns.set_theme()


def extend_line(
    line: LineString,
    distance: float,
    plot: bool = False,
) -> LineString:
    """extend line in either direction by distance"""
    coords = np.asarray(line.coords)

    # remove consecutive duplicate points
    mask = np.any(np.diff(coords, axis=0) != 0, axis=1)
    coords = np.vstack([coords[0], coords[1:][mask]])

    if len(coords) < 2:
        return line

    # first non-zero segment
    start_vec = coords[0] - coords[1]
    start_vec /= np.linalg.norm(start_vec)

    # last non-zero segment
    end_vec = coords[-1] - coords[-2]
    end_vec /= np.linalg.norm(end_vec)

    start = coords[0] + distance * start_vec
    end = coords[-1] + distance * end_vec

    extended_line = LineString(np.vstack([start, coords, end]))

    if plot:
        l_coords = list(line.coords)
        x = [p[0] for p in l_coords]
        y = [p[1] for p in l_coords]
        plt.plot(x, y, "r.", markersize=10, label="original line")

        longl_coords = list(extended_line.coords)
        x = [p[0] for p in longl_coords]
        y = [p[1] for p in longl_coords]
        plt.plot(x, y, "g.", markersize=2, label="extended line")
        plt.gca().set_aspect("equal")
        plt.legend()

    return extended_line


# def extend_line_alternative_method(
#     line: LineString,
#     distance: float,
#     plot: bool = False,
# ) -> LineString:
#     """extend line in either direction by distance"""
#     # find minimum rotated rectangle around line
#     rect = line.minimum_rotated_rectangle
#     angle = airbornegeo.nav.azimuth(rect)

#     rect_center = shapely.centroid(rect).x, shapely.centroid(rect).y

#     # get length of long edge
#     x, y = rect.exterior.coords.xy
#     length = max(
#         (
#             Point(x[0], y[0]).distance(Point(x[1], y[1])),
#             Point(x[1], y[1]).distance(Point(x[2], y[2])),
#         )
#     )

#     # make new start and end points extended by distance
#     start = Point(rect_center[0] - (length / 2) - distance, rect_center[1])
#     end = Point(rect_center[0] + (length / 2) + distance, rect_center[1])

#     # turn into a line and rotate back
#     extended_line = shapely.affinity.rotate(
#         LineString([start, end]), angle, origin=rect_center
#     )

#     # add new endpoints to original line
#     # extended_line = shapely.unary_union()
#     extended_line = LineString(
#         [extended_line.coords[0], *line.coords, extended_line.coords[-1]]
#     )

#     if plot:
#         l_coords = list(line.coords)
#         x = [p[0] for p in l_coords]
#         y = [p[1] for p in l_coords]
#         plt.plot(x, y, "r.", markersize=10, label="original line")

#         longl_coords = list(extended_line.coords)
#         x = [p[0] for p in longl_coords]
#         y = [p[1] for p in longl_coords]
#         plt.plot(x, y, "g.", markersize=2, label="extended line")

#         plt.legend()
#         plt.gca().set_aspect("equal")

#     return extended_line


def get_line_intersections(
    lines1_gdf: gpd.GeoDataFrame,
    lines2_gdf: gpd.GeoDataFrame | None = None,
    *,
    line_column: str,
    grid_size: float = 1,
    buffer_dist: float | None = None,
) -> gpd.GeoDataFrame:
    """
    Find intersections between flight lines.

    If ``lines2_gdf`` is supplied, intersections are calculated between every
    line in ``lines1_gdf`` and every line in ``lines2_gdf`` (traditional
    line/tie levelling).

    If ``lines2_gdf`` is None, intersections are calculated between every
    unique pair of lines in ``lines1_gdf`` (network levelling).

    Effect of buffer_dist:
    Original (no intersection):
        A ----------------

                    |
                    |
                    B

    With buffer (1 intersection):
                    |
        A ----------x----------
                    |
                    |
                    |
                    B

    Parameters
    ----------
    lines1_gdf : GeoDataFrame
        First set of flight lines as point observations.
    lines2_gdf : GeoDataFrame | None, optional
        Second set of flight lines. If None, all unique line pairs are used.
    line_column : str
        Column containing the line identifiers.
    grid_size : float, optional
        Precision used when calculating intersections.
    buffer_dist : float | None, optional
        Distance to extend the ends of each line before calculating
        intersections.

    Returns
    -------
    GeoDataFrame
        Columns:
            line1
            line2
            line1_dist
            line2_dist
            geometry
    """

    assert line_column in lines1_gdf.columns

    network = lines2_gdf is None

    if network:
        lines2_gdf = lines1_gdf
    else:
        assert line_column in lines2_gdf.columns

    # convert points to LineStrings
    grouped1 = lines1_gdf.groupby(line_column, as_index=False)["geometry"].apply(
        lambda x: LineString(x.tolist())
    )
    grouped1["original_geometry"] = grouped1.geometry.copy()
    if network:
        grouped2 = grouped1
    else:
        grouped2 = lines2_gdf.groupby(line_column, as_index=False)["geometry"].apply(
            lambda x: LineString(x.tolist())
        )
        grouped2["original_geometry"] = grouped2.geometry.copy()

    # extend ends of lines by buffer_dist to account for expected intersections just
    # beyond lines
    if buffer_dist is not None:
        grouped1["geometry"] = grouped1.geometry.apply(
            lambda g: extend_line(g, buffer_dist)
        )

        if not network:
            grouped2["geometry"] = grouped2.geometry.apply(
                lambda g: extend_line(g, buffer_dist)
            )

    # dictionaries of original points for nearest-point distance calculation
    point_groups1 = {  # noqa: C416 # pylint: disable=unnecessary-comprehension
        name: df for name, df in lines1_gdf.groupby(line_column)
    }

    if network:
        point_groups2 = point_groups1
    else:
        point_groups2 = {  # noqa: C416 # pylint: disable=unnecessary-comprehension
            name: df for name, df in lines2_gdf.groupby(line_column)
        }

    # create iterator over line pairs
    if network:
        iterator = itertools.combinations(
            grouped1.itertuples(index=False),
            2,
        )
        total = len(grouped1) * (len(grouped1) - 1) // 2
    else:
        iterator = itertools.product(
            grouped1.itertuples(index=False),
            grouped2.itertuples(index=False),
        )
        total = len(grouped1) * len(grouped2)

    # dicts used to keep track if intersections were only find via buffer_dist
    original_lines1 = dict(
        zip(grouped1[line_column], grouped1["original_geometry"], strict=True)
    )
    original_lines2 = dict(
        zip(
            grouped2[line_column],
            grouped2["original_geometry"],
            strict=True,
        )
    )

    # calculate intersections
    intersections = []
    for line1, line2 in tqdm(iterator, total=total, desc="Line pairs"):
        line1_name = getattr(line1, line_column)
        line2_name = getattr(line2, line_column)
        inter = shapely.intersection(
            line1.geometry,
            line2.geometry,
            grid_size=grid_size,
        )

        if inter.geom_type == "Point":
            points = [inter]
        elif inter.geom_type == "MultiPoint":
            points = list(inter.geoms)
        else:
            if inter.is_empty:
                continue
            points = [inter.representative_point()]

        coords = shapely.get_coordinates(points)

        if len(coords) == 0:
            continue

        for coord in coords:
            point = Point(coord)
            # point = shapely.set_precision(
            #     Point(coord),
            #     grid_size,
            # )
            # Determine whether the intersection is only possible because of buffering
            is_buffered = False

            if buffer_dist is not None:
                # is_buffered = (
                #     not original_lines1[line1_name].intersects(point)
                #     or not original_lines2[line2_name].intersects(point)
                # )
                tol = grid_size / 2
                is_buffered = (
                    original_lines1[line1_name].distance(point) > tol
                    or original_lines2[line2_name].distance(point) > tol
                )
            intersections.append(
                {
                    "line1": line1_name,
                    "line2": line2_name,
                    "line1_dist": (
                        point_groups1[line1_name].geometry.distance(point).min()
                    ),
                    "line2_dist": (
                        point_groups2[line2_name].geometry.distance(point).min()
                    ),
                    "is_buffered": is_buffered,
                    "geometry": point,
                }
            )

    return gpd.GeoDataFrame(
        intersections,
        geometry="geometry",
        crs=lines1_gdf.crs,
    )


def create_intersection_table(
    data: gpd.GeoDataFrame,
    *,
    line_column: str,
    method: str,
    exclude_ints: list[list[float, float] | list[float] | float] | None = None,
    cutoff_dist: float | None = None,
    buffer_dist: float | None = None,
    block_size: float | None = None,
    grid_size: float = 1,
) -> gpd.GeoDataFrame:
    """
    Create a dataframe which contains the intersections between all combinations of
    lines. For each intersection point, find the distance to the closest data point of
    each line. If the further of these two distances is greater than "cutoff_dist", the
    intersection is excluded. The intersections are calculated by representing the point
    data as lines, and finding the hypothetical crossover.
    By default crossovers will only be within the endpoints of each line. If
    there is an expected crossover just beyond the end of a line which should be
    included, use the `buffer_dist` arg to extend the line representation of the data,
    but note that extrapolation of data at these points will likely be inaccurate if
    buffer distance is too large. If lines intersection very often, duplicate
    intersections within a window, specified by grid_size, can be excluded.

    Effect of buffer_dist:
    Original (no intersection):
        A ----------------

                    |
                    |
                    B

    With buffer (1 intersection):
                    |
        A ----------x----------
                    |
                    |
                    |
                    B

    Parameters
    ----------
    data : gpd.GeoDataFrame
        Dataframe with survey data.
    line_column : str
        Column name containing the line / flight / segment names
    method : str
        Choose between "groups" where two categories of lines are supplied, such as
        survey lines and tie lines, and intersections are only found between these two
        groups, or 'network' where all intersections between any supplied lines are
        found. If "groups", column 'line_type' must be provided with possible values 0,
        1, or 2, where 0 denotes the first category, 1 denotes the second category, and
        2 denotes lines to be excluded.
    exclude_ints : list[tuple[int]] | None, optional
        List of tuples where each tuple is either a single line number to exclude from
        all intersections, or a pair of line numbers specifying specific intersections
        to exclude, by default None
    cutoff_dist : float, optional
        The maximum allowed distance from a theoretical intersection to the further of
        nearest data point of each intersecting line, by default None
    buffer_dist : float, optional
        The distance to extend the line representation of the data points, useful for
        creating intersection which are just beyond the end of a line, by default None
    block_size : float, optional
        For a spatial block of this width, intersection within with the same pairs of
        lines (duplicates) will be dropped, except the intersection with the lowest
        min_dist. This is useful for when lines cross very often.
    grid_size : float, optional
        The resolution to snap the intersection coordinates to.  by default 1

    Returns
    -------
    gpd.GeoDataFrame
        An intersection table containing the locations of the theoretical intersections,
        the intersecting lines numbers, and the distance to the further of the two
        nearest datapoints of each line, and a geometry column.
    """
    data = data.copy()

    cols = [line_column]
    if method == "groups":
        cols.append("line_type")

    assert all(col in data.columns for col in cols), f"{cols} must be in the dataframe"

    assert isinstance(data, gpd.GeoDataFrame), "data must be a GeoDataFrame"
    assert data.geometry.geom_type.isin(["Point"]).all(), "geometry must be points"

    # if is_intersection column exists, delete it and rows where it's true
    if "is_intersection" in data.columns:
        rows_to_drop = data[data.is_intersection]
        data = data.drop(index=rows_to_drop.index)
    data = data.drop(columns="is_intersection", errors="ignore")

    if method == "groups":
        # get intersection points just between two line types
        inters = get_line_intersections(
            lines1_gdf=data[data.line_type == 0],
            lines2_gdf=data[data.line_type == 1],
            line_column=line_column,
            grid_size=grid_size,
            buffer_dist=buffer_dist,
        )
    elif method == "network":
        # get intersection points of all lines
        inters = get_line_intersections(
            lines1_gdf=data,
            lines2_gdf=None,
            line_column=line_column,
            grid_size=grid_size,
            buffer_dist=buffer_dist,
        )
    else:
        msg = "method must be either 'groups' or 'network'"
        raise ValueError(msg)

    # get coords from geometry column
    inters["easting"] = inters.geometry.x
    inters["northing"] = inters.geometry.y

    # get the largest of the two distance to each lines' nearest data point to the
    # theoretical intersection
    inters["max_dist"] = inters[["line1_dist", "line2_dist"]].max(axis=1)

    # remove duplicate intersections within a block
    if block_size is not None:
        a = len(inters)
        kept = []
        # process each line pair independently
        for (_, _), group in inters.groupby(["line1", "line2"], sort=False):
            # process lowest max_dist first
            group_sorted = group.sort_values("max_dist")
            selected = []
            for idx, row in group_sorted.iterrows():
                point = row.geometry
                # keep if it is not within block_size of an already-kept intersection
                if all(
                    point.distance(group_sorted.loc[i].geometry) >= block_size
                    for i in selected
                ):
                    selected.append(idx)
            kept.extend(selected)
        inters = inters.loc[kept].sort_index()
        b = len(inters)
        if a != b:
            logger.debug(
                "Dropped %s duplicate intersections within %.1f m blocks",
                a - b,
                block_size,
            )

    # deduplicate on line names and coordinates
    # inters = (
    #     inters.sort_values(
    #         "max_dist",
    #         ascending=False,
    #     )
    #     .drop_duplicates(
    #         subset=["line1", "line2", "easting", "northing"],
    #         keep="last",
    #     )
    #     .sort_index()
    # )
    # deduplicate on bins of coordinates
    # inters["geometry"] = inters.geometry.set_precision(grid_size)
    # inters = (
    #     inters.sort_values(
    #         "max_dist",
    #         ascending=False,
    #     )
    #     .drop_duplicates(
    #         subset=[
    #             "line1",
    #             "line2",
    #             "geometry",
    #         ]
    #     )
    #     .sort_index()
    # )

    # b = len(inters)
    # if a != b:
    #     logger.debug("Dropped %s duplicate intersections", a - b)

    logger.info("found %s intersections", len(inters))

    # if intersection is not within cutoff_dist, remove rows
    if cutoff_dist is not None:
        prior_len = len(inters)
        inters = inters[inters.max_dist < cutoff_dist]
        logger.info(
            "removed %s intersection point(s) with a max distance greater than %s km",
            prior_len - len(inters),
            int(cutoff_dist / 1000),
        )

    if exclude_ints is not None:
        prior_len = len(inters)

        assert isinstance(exclude_ints, tuple | list), (
            "exclude ints must be a tuple or a list"
        )

        exclude_inds = []
        for i in exclude_ints:
            assert isinstance(i, tuple | list), (
                "elements of exclude_ints must be lists or tuples"
            )

            # if pair of lines numbers given, get those indices
            if len(i) == 2:
                ind = inters[  # type: ignore[unreachable]
                    (inters.line1 == i[0]) & (inters.line2 == i[1])
                ].index.to_numpy()
                exclude_inds.extend(ind)
                ind = inters[
                    (inters.line2 == i[0]) & (inters.line1 == i[1])
                ].index.to_numpy()
                exclude_inds.extend(ind)
            # if single line number, get all intersections of that line
            elif len(i) == 1:
                ind = inters[
                    (inters.line1 == i[0]) | (inters.line2 == i[0])
                ].index.to_numpy()
                exclude_inds.extend(ind)
        inters = inters.drop(index=exclude_inds).copy()
        logger.info(
            "manually omitted %s intersections points",
            prior_len - len(inters),
        )

    return inters.drop(columns=["line1_dist", "line2_dist"]).reset_index(drop=True)


def inspect_intersections(
    data: pd.DataFrame | gpd.GeoDataFrame,
    *,
    x: str,
    plot_variable: str | list[str],
    line_column: str,
    plot_all: bool = False,
) -> None:
    if isinstance(plot_variable, str):
        plot_variable = [plot_variable]

    for line, line_df in data.groupby(line_column):
        if len(line_df[line_df.is_intersection]) == 0:
            continue

        fig = plot_line_and_crosses(
            data,
            line=line,
            x=x,
            y=plot_variable,
            y_axes=[str(i + 1) for i in range(len(plot_variable))],
            plot_inters=True,
            line_column=line_column,
        )
        fig.show()
        if plot_all:
            continue
        input("Press key to continue...")
        clear_output(wait=True)


def add_values_to_intersections(
    df: pd.DataFrame | gpd.GeoDataFrame,
    intersections: pd.DataFrame | gpd.GeoDataFrame,
    *,
    line_column: str,
    columns: tuple[str],
) -> pd.DataFrame | gpd.GeoDataFrame:
    df = df.copy()
    inters = intersections.copy()

    for column in columns:
        # add values to intersections table
        inters[f"line1_{column}"] = np.nan
        inters[f"line2_{column}"] = np.nan
        for ind, row in inters.iterrows():
            # search data for values at intersecting lines
            line1_values = df[
                (df[line_column] == row.line1) & (df.intersecting_line == row.line2)
            ][column].to_numpy()
            line2_values = df[
                (df[line_column] == row.line2) & (df.intersecting_line == row.line1)
            ][column].to_numpy()

            # add to intersection table
            inters.loc[ind, f"line1_{column}"] = line1_values
            inters.loc[ind, f"line2_{column}"] = line2_values

    return inters


def interpolate_intersections(
    df: pd.DataFrame | gpd.GeoDataFrame,
    intersections: pd.DataFrame | gpd.GeoDataFrame,
    *,
    line_column: str,
    to_interp: str,
    interp_on: str,
    method: str = "cubic",
    extrapolate: bool = False,
    fill_value: tuple[float, float] | str | None = None,
    window_width: float | None = None,
) -> tuple[pd.DataFrame | gpd.GeoDataFrame]:
    """
    _summary_

    Parameters
    ----------
    df : pd.DataFrame | gpd.GeoDataFrame
        Dataframe containing the data to interpolate
    intersections : pd.DataFrame | gpd.GeoDataFrame
        Dataframe containing the intersection points
    line_column : str
        Column name containing the line / flight / segment names
    to_interp : str,
        specify which column to interpolate NaNs for
    interp_on : str, optional
        Decide which column interpolation is based on, such as "distance_along_line"
    method : str, optional
        Decide between interpolation methods of 'linear', 'nearest', 'nearest-up',
        'zero', 'slinear', 'quadratic','cubic', 'previous', or 'next', by default
        "cubic"
    window_width : float, optional
        window width around each NaN to use for interpolation fitting, by default None

    Returns
    -------
    pd.DataFrame | gpd.GeoDataFrame:
        the dataframe with new columns 'is_intersection' for booleans,
        'intersecting_line' containing the name of the intersection line rows which are
        intersections, '<to_interp>_interpolation_type' with string 'interpolated' or
        'extrapolated' describing what interpolation type was used for each row, and
        column specified with to_interp updated with interpolated values at the
        intersection rows.
    pd.DataFrame | gpd.GeoDataFrame:
        the intersection dataframe with new columns 'dist_along_flight_line',
        'dist_along_flight_tie', 'line1_interpolation_type', 'line2_interpolation_type'.
        Rows where the interpolation failed will be removed.
    """
    df = df.copy()
    inters = intersections.copy()

    assert isinstance(to_interp, str), "to_interp should be a single string"

    cols = [line_column, to_interp, interp_on]
    assert all(col in df.columns for col in cols), f"{cols} must be in the dataframe"

    # remove NaNs from target variable
    df = df.dropna(subset=to_interp, how="any")

    # add empty rows at each intersection to the df
    df, inters = add_intersections(
        df, inters, line_column=line_column, distance_column=interp_on
    )

    # perform interpolation
    if window_width is None:
        filled_lines = airbornegeo.interpolating.interpolate_missing_pointwise(
            df,
            to_interp=to_interp,
            interp_on=interp_on,
            method=method,
            extrapolate=extrapolate,
            fill_value=fill_value,
            groupby_column=line_column,
        )
    else:
        filled_lines = (
            airbornegeo.interpolating.interpolate_missing_pointwise_with_windows(
                df,
                to_interp=to_interp,
                window_width=window_width,
                interp_on=interp_on,
                method=method,
                extrapolate=extrapolate,
                fill_value=fill_value,
                groupby_column=line_column,
            )
        )

    # create lookup table
    interp_col = f"{to_interp}_interpolation_type"

    # keyed on the intersection's exact location so that repeat crossings between the
    # same line pair don't collapse onto a single (non-unique) index entry
    intersection_rows = (
        filled_lines[filled_lines.is_intersection]
        .set_index([line_column, "intersecting_line", "easting", "northing"])
        .sort_index()
    )

    # perform lookups
    keys1 = pd.MultiIndex.from_arrays(
        [inters.line1, inters.line2, inters.easting, inters.northing]
    )
    keys2 = pd.MultiIndex.from_arrays(
        [inters.line2, inters.line1, inters.easting, inters.northing]
    )

    inters["line1_interpolation_type"] = (
        intersection_rows[interp_col].reindex(keys1).to_numpy()
    )
    inters["line2_interpolation_type"] = (
        intersection_rows[interp_col].reindex(keys2).to_numpy()
    )

    # remove invalid intersections
    valid_mask = (
        inters.line1_interpolation_type.notna()
        & inters.line2_interpolation_type.notna()
        & (inters.line1_interpolation_type != "none")
        & (inters.line2_interpolation_type != "none")
    )
    inters_valid = inters.loc[valid_mask].copy()

    # remove invalid rows from filled_lines
    if (~valid_mask).any():
        bad = inters.loc[~valid_mask, ["line1", "line2"]]

        bad_swapped = bad.rename(
            columns={"line1": line_column, "line2": "intersecting_line"}
        )

        bad_swapped2 = bad.rename(
            columns={"line2": line_column, "line1": "intersecting_line"}
        )

        bad_all = pd.concat([bad_swapped, bad_swapped2], ignore_index=True)
        bad_all["_drop"] = True

        filled_lines = filled_lines.merge(
            bad_all,
            on=[line_column, "intersecting_line"],
            how="left",
        )

        filled_lines = filled_lines[
            ~(filled_lines.is_intersection & filled_lines["_drop"].fillna(False))
        ].drop(columns="_drop")

    # re-sort by line and distance
    filled_lines = filled_lines.sort_values([line_column, interp_on]).reset_index(
        drop=True
    )

    assert len(inters_valid) * 2 == len(filled_lines[filled_lines.is_intersection]), (
        "Number of intersection rows in the dataframe should be twice the number of rows in intersection table"
    )

    return filled_lines, inters_valid


def add_intersections(
    data: gpd.GeoDataFrame,
    intersections: gpd.GeoDataFrame,
    line_column: str,
    distance_column: str,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Add new rows to the dataframe for each intersection point and columns
    `is_intersection` and `intersection_line` to identify these intersections. All of
    the data column for these rows will have NaNs and should be filled with the
    function `interpolate_intersections()`. Add columns to the intersections table for the
    distance along each line (flight and tie) to the intersection point. During
    levelling, levelling corrections are calculated using mistie values at intersections
    and interpolated along the entire lines based on these distances. Distances are
    calculate using the geometry column, and the time column informs which end of the
    line is the start.

    Parameters
    ----------
    data : gpd.GeoDataFrame
        Flight survey dataframe containing the data points to add intersections to.
        Must contain a geometry column and columns 'line' and `time_col_name`
    intersections : gpd.GeoDataFrame
        Intersections table created by `create_intersection_table()`
    line_column : str
        Column name containing the line / flight / segment names
    distance_column : str
        Column name containing the distance along the lines

    Returns
    -------
    tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
        The updated flight survey dataframe and intersections table.
    """
    data = data.copy()
    inters = intersections.copy().reset_index(drop=True)

    cols = [line_column, distance_column, "geometry"]
    assert all(col in data.columns for col in cols), f"{cols} must be in the dataframe"

    # remove existing is_intersection column and intersection rows
    if "is_intersection" in data.columns:
        data = data.loc[~data.is_intersection].copy()

    data = data.drop(
        columns=["is_intersection", "intersecting_line"],
        errors="ignore",
    )

    prior_length = len(data)

    # initialize columns
    data["is_intersection"] = False
    data["intersecting_line"] = np.nan

    # pre-group and get numpy arrays for each line
    line_groups = {}
    for line, df in data.groupby(line_column):
        coords = np.column_stack([df.geometry.x.to_numpy(), df.geometry.y.to_numpy()])
        dist_along_line = df[distance_column].to_numpy()
        line_groups[line] = (coords, dist_along_line)

    new_rows = []

    # Track calculated distances map structured as: (intersection_idx, line_name) -> calculated_distance
    distance_lookup = {}

    pbar = tqdm(
        enumerate(inters.itertuples(index=False)),
        desc="Collecting intersections",
        total=len(inters),
    )

    for inter_idx, row in pbar:
        px, py = row.geometry.x, row.geometry.y

        for line, other in [(row.line1, row.line2), (row.line2, row.line1)]:
            # get coords and along line distances of the line
            coords, dist_along_line = line_groups[line]

            # compute distances from each data point to the intersection
            dx = coords[:, 0] - px
            dy = coords[:, 1] - py
            dist = dx * dx + dy * dy  # squared distance (no sqrt needed)

            # get closest two data points to the intersection
            idx = np.argsort(dist)[:2]
            nearest_two_dist_along_line = dist_along_line[idx]

            # choose point with smallest distance_along_line
            min_point_idx = idx[np.argmin(nearest_two_dist_along_line)]
            min_dist_along_line = dist_along_line[min_point_idx]

            # calculate distance to intersection
            dist_to_point = np.sqrt(dist[min_point_idx])

            calculated_dist = min_dist_along_line + dist_to_point
            distance_lookup[(inter_idx, line)] = calculated_dist

            new_rows.append(
                {
                    line_column: line,
                    "easting": px,
                    "northing": py,
                    "geometry": row.geometry,
                    "is_intersection": True,
                    "intersecting_line": other,
                    distance_column: calculated_dist,
                }
            )

    # add new intersection rows to dataframe
    new_df = gpd.GeoDataFrame(new_rows, crs=data.crs)
    data = pd.concat([data, new_df], ignore_index=True)
    data = data.sort_values([line_column, distance_column]).reset_index(drop=True)

    # check new dataframe is correct length
    assert len(data) == prior_length + (2 * len(inters))

    # Safely assign distances to inters based on the exact matching unique index loop
    inters["dist_along_line1"] = [
        distance_lookup[(idx, r.line1)] for idx, r in enumerate(inters.itertuples())
    ]
    inters["dist_along_line2"] = [
        distance_lookup[(idx, r.line2)] for idx, r in enumerate(inters.itertuples())
    ]

    return data, inters


def calculate_crossover_errors(
    data: gpd.GeoDataFrame,
    intersections: gpd.GeoDataFrame,
    *,
    data_col: str,
    line_column: str,
    raise_error_if_unchanged: bool = False,
) -> gpd.GeoDataFrame:
    """
    Calculate mistie values for all intersections. For each intersection, find the data
    values for the line and tie from the survey dataframe and add those values to the
    intersection table as `line_value` and `tie_value`. If they exist, overwrite them.
    Calculate the mistie value as line_value - tie_value, and save this to a column
    `crossover_error_0`. If `crossover_error_0` exists, make a new column `crossover_error_1`, and keep incrementing
    the number. If the new mistie values exactly match previous, don't make a new
    column. This allow to run the function multiple times without changing anything and
    not building up a large number of mistie columns.

    Parameters
    ----------
    intersections : gpd.GeoDataFrame
        Intersections table created by `create_intersection_table()`, then supplied to
        `add_intersections()`.
    data : gpd.GeoDataFrame
        Survey dataframe with intersection rows added by `add_intersections()` and
        interpolated with `interpolate_intersections()`.
    data_col : str
        Column name for data values to calculate misties for
    line_column : str
        Column name containing the line / flight / segment names
    raise_error_if_unchanged : bool, optional
        If true, raise a UserWarning if misties haven't changed from previous column, by
        default False.

    Returns
    -------
    gpd.GeoDataFrame
        An intersections table with new columns `line_value`, `tie_value` and `crossover_error_x`
        where x is incremented each time a new mistie is calculated.
    """

    inters = intersections.copy()
    df = data.copy()

    cols = [line_column]
    assert all(col in df.columns for col in cols), f"{cols} must be in the dataframe"

    # build lookup table, keyed on the intersection's exact location so that repeat
    # crossings between the same line pair don't collapse onto a single value
    lookup = df.set_index([line_column, "intersecting_line", "easting", "northing"])[
        data_col
    ]

    # find data values at intersection for each intersecting line
    line1_vals = [
        lookup[(r.line1, r.line2, r.easting, r.northing)] for r in inters.itertuples()
    ]
    line2_vals = [
        lookup[(r.line2, r.line1, r.easting, r.northing)] for r in inters.itertuples()
    ]

    line1_vals = np.asarray(line1_vals).flatten()
    line2_vals = np.asarray(line2_vals).flatten()

    # # sanity checks
    # if np.isnan(line1_vals).any():
    #     raise ValueError("NaN found in line1 values")
    # if np.isnan(line2_vals).any():
    #     raise ValueError("NaN found in line2 values")

    misties = line1_vals - line2_vals

    logger.debug("mistie RMSE: %s", airbornegeo.rmse(misties))

    # manage mistie column names
    cols = [c for c in inters.columns if "crossover_error_" in c]
    crossover_error_cols = []
    for c in cols:
        try:  # noqa: SIM105
            crossover_error_cols.append(int(c.split("_")[-1]))
        except ValueError:
            pass
    if len(crossover_error_cols) == 0:
        next_crossover_error_col = "crossover_error_0"
    else:
        next_crossover_error_col = f"crossover_error_{max(crossover_error_cols) + 1}"
    if len(crossover_error_cols) > 0:
        prev_col = f"crossover_error_{max(crossover_error_cols)}"
        if np.allclose(
            inters[prev_col].to_numpy(),
            misties,
            equal_nan=True,
        ):
            logger.debug("Mistie values are unchanged")
            if raise_error_if_unchanged:
                msg = "Mistie hasn't changed"
                raise UserWarning(msg)

            return inters

    # store new misties
    inters[next_crossover_error_col] = misties
    return inters


def plot_line_and_crosses(
    df: pd.DataFrame | gpd.GeoDataFrame,
    *,
    y: list[str],
    line_column: str,
    x: str,
    line: float | None = None,
    plot_inters: bool | list[bool] = False,
    use_intersection_y: bool = True,
    y_axes: list[str] | None = None,
    x_lims: tuple[float, float] | None = None,
    y_lims: tuple[float, float] | None = None,
    **kwargs: typing.Any,
) -> go.Figure:
    """
    plot lines and crosses
    """
    assert line_column in df.columns, "df must have column given by line_column"

    # turn y column name into list
    if isinstance(y, str):  # type: ignore [unreachable]
        y = [y]  # type: ignore [unreachable]

    # list of y axes to use, if none, all will be same
    if y_axes is None:
        y_axes = ["1" for _ in y]

    if len(df[line_column].unique()) <= 1:
        line = df[line_column].iloc[0]

    try:
        line_df = df[df[line_column] == line].sort_values(
            by=[line_column, "intersecting_line"]
        )
    except KeyError:
        line_df = df[df[line_column] == line].sort_values(by=[line_column])

    # if xlims is not None:
    #     line_df = line_df[line_df[x].between(*xlims)]
    # if ylims is not None:
    #     line_df = line_df[line_df[x].between(*ylims)]

    # list of which dataset to plot intersections for
    if plot_inters is True:
        inters_to_plot: list[bool] = []
        for i in y:
            if len(line_df[i].dropna()) == len(line_df[line_df.is_intersection]):
                inters_to_plot.append(False)
            else:
                inters_to_plot.append(True)

    if isinstance(plot_inters, list):
        inters_to_plot = plot_inters

    fig = airbornegeo.plotly_profiles(
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
        for i, z in enumerate(y):  # type: ignore [assignment]
            if inters_to_plot[i] is True:  # type: ignore [call-overload]
                intersections = df[df.intersecting_line == line].sort_values(
                    by=[line_column, "intersecting_line"]
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
                    text = intersections[line_column]
                if use_intersection_y is False:
                    y_val = line_df[line_df.is_intersection][z]
                fig.add_trace(
                    go.Scatter(
                        mode="markers+text",
                        x=line_df[line_df.is_intersection][x],
                        y=y_val,
                        yaxis=y_axes[i],  # type: ignore [call-overload]
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

    return fig


def lines_without_intersections(
    data: pd.DataFrame,
    intersections: pd.DataFrame,
    line_column: str,
) -> list[float]:
    return [
        i
        for i in data[line_column].unique()
        if i not in [*intersections.line1.unique(), *intersections.line2.unique()]
    ]


def update_intersections_with_eq_sources(
    data: pd.DataFrame | gpd.GeoDataFrame,
    *,
    fitted_equivalent_sources: dict,
    data_column: str,
    line_column: str,
    distance_column: str,
) -> pd.Series:
    """
    At each theoretical intersection point, replace the interpolated field value with a
    value predicted by the fitted equivalent sources for the line, at the x,y coordinate
    of the intersection point, and the higher of the two lines' elevations. This allows
    the cross-over mistie value to be comparing the field values at the same point in 3D
    space, not 2D space, due to different flight heights.

    Parameters
    ----------
    data : pd.DataFrame | gpd.GeoDataFrame
        dataframe containing the data to update
    fitted_equivalent_sources : dict
        a dictionary with keys of line names and values of fitted equivalent sources for
        each line, which can be created using the function `eq_sources_1d`
    line_column : str
        Column containing the line / flight / segment names.
    distance_column : str
        Column containing the distance along line.
    data_column : str
        name of the column containing the field values to update at the intersection
        points, this should be the same as the column that use used as 'data_column'
        when fitting the equivalent sources for each line with `eq_sources_1d`.

    Returns
    -------
    pd.Series
        the updated field values at the intersection points, which can be added to the
        dataframe as a new column or used to replace the existing values in the
        dataframe.
    """

    data = data.copy()

    # check columns are present
    cols = [line_column]
    assert all(col in data.columns for col in cols), f"{cols} must be in the dataframe"

    for segment_name, segment_data in tqdm(
        data.groupby(line_column), desc="Equivalent source segments"
    ):
        # get fitted equivalent sources for this line
        eqs = fitted_equivalent_sources[segment_name]

        # get intersection points for this line
        line_inters = segment_data[segment_data.is_intersection]

        for i, row in line_inters.iterrows():
            # get height of intersection point for the cross line
            cross_inter = data[
                (data[line_column] == row.intersecting_line)
                & (data.intersecting_line == segment_name)
            ]

            assert len(cross_inter) == 1, (
                data[data.intersecting_line == segment_name],
                row.intersecting_line,
                segment_name,
            )

            cross_height = cross_inter.height.to_numpy()[0]

            coords = (
                np.array([row[distance_column]]),
                np.array([0]),
                np.array([np.max([cross_height, row.height])]),
            )
            # predict the field value at the x,y coordinate of the intersection point,
            # and the higher of the two lines' elevations, using the supplied fitted
            # equivalent sources for each line
            up_cont_value = eqs.predict(coords)

            # add predicted value to dataframe at intersection point
            data.at[i, data_column] = up_cont_value[0]  # noqa: PD008

    return data[data_column]
