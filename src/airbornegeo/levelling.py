# pylint: disable=too-many-lines
import copy
import itertools
import typing
import warnings

import geopandas as gpd
import harmonica as hm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import seaborn as sns
import shapely
import verde as vd
from IPython.display import clear_output
from shapely.geometry import LineString, Point
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from tqdm.autonotebook import tqdm

import airbornegeo
from airbornegeo import logger

sns.set_theme()


def _end_iterations(
    rms_values: list[float],
    delta_rms_values: list[float],
    max_iterations: int,
    rms_tolerance: float | None = None,
    rms_percent_change_tolerance: float | None = None,
    rms_percent_increase_tolerance: float | None = None,
):
    end = False
    termination_reason = []

    iteration = len(rms_values)
    rms = rms_values[-1]
    delta_rms = delta_rms_values[-1]
    previous_delta_rms = delta_rms_values[-2] if iteration > 2 else np.inf

    # ignore for first iteration
    if iteration == 1:
        pass
    else:
        # end because RMS is increasing above a unreasonable amount
        if rms > np.min(rms_values) * (1 + rms_percent_increase_tolerance / 100):
            logger.info(  # pylint: disable=logging-fstring-interpolation
                f"\nEquivalent source levelling terminated after {iteration} iterations because the RMS of the levelling corrections ({round(rms, 4)}) \n"
                f"was over {rms_percent_increase_tolerance}% greater than minimum RMS ({round(np.min(rms_values), 4)}) \n"
                "Change parameter 'rms_percent_increase_tolerance' if desired.",
            )
            end = True
            termination_reason.append("RMS increasing")
        # end because RMS decrease has plateaued (defined over 2 iterations)
        if (
            (rms_percent_change_tolerance is not None)
            and (delta_rms <= rms_percent_change_tolerance)
            and (previous_delta_rms <= rms_percent_change_tolerance)
        ):
            logger.info(  # pylint: disable=logging-fstring-interpolation
                f"\nEquivalent source levelling terminated after {iteration} iterations because there was no "
                f"significant variation in the RMS (delta RMS of {round(delta_rms, 2)}%) of the levelling corrections over 2 iterations \n"
                f"Change parameter 'rms_percent_change_tolerance' ({rms_percent_change_tolerance}%) if desired.",
            )
            end = True
            termination_reason.append("RMS percent change tolerance")
        # end because RMS is below the set tolerance
        if (rms_tolerance is not None) and (rms < rms_tolerance):
            logger.info(  # pylint: disable=logging-fstring-interpolation
                f"\nEquivalent source levelling terminated after {iteration} iterations because the RMS of the levelling corrections ({rms}) was "
                f"less then set tolerance ({rms_tolerance}) \nChange parameter "
                "'rms_tolerance' if desired.",
            )
            end = True
            termination_reason.append("RMS tolerance")
    # end because max iterations reached
    if iteration >= max_iterations:
        logger.warning(  # pylint: disable=logging-fstring-interpolation
            f"\nEquivalent source levelling terminated after {iteration} iterations with RMS of levelling correction of {round(rms, 2)} because "
            f"maximum number of iterations ({max_iterations}) reached.",
        )

        end = True
        termination_reason.append("max iterations")

    return end, termination_reason


def equivalent_source_levelling(
    data: pd.DataFrame,
    data_column: str,
    max_dist: float,
    degree: int,
    lines_to_level: list[float] | None = None,
    damping: float | None = None,
    depth: str | float = "default",
    block_size: float | None = None,
    max_iterations: int = 1,
    rms_tolerance: float | None = None,
    rms_percent_change_tolerance: float = 10,
    rms_percent_increase_tolerance: float = 20,
    seed: int = 42,
    plot_iterations: bool = True,
    progressbar: bool = True,
) -> pd.Series:
    """
    _summary_

    Parameters
    ----------
    data : pd.DataFrame
        The dataframe containing columns 'easting', 'northing', 'height', 'line',
        'distance_along_line', and the data, specified by parameter `data_column`.
    data_column : str
        The name of the column containing the data values to fit equivalent sources to
        and level, typically gravity or magnetics.
    max_dist : float
        For each line to be levelled, only fit equivalent sources using data within this
        distance to the line, excluding the line data itself. This should be large
        enough to include at least 1 adjacent flight line.
    degree : int
        The degree order of the polynomial trend to fit the misfit between the
        data_column and the predicted values from the equivalent sources. 0 gives a
        DC-shift, 1 additionally allows a tilt, 2 additional allows a curve, etc.
    lines_to_level : list[float]
        Which lines to level, by default will level all lines.
    damping : float | None, optional
        The damping regularization to use when fitting the equivalent sources, by
        default None
    depth : str | float, optional
        The source depths for the equivalent sources, by default "default", which uses
        4.5 times the mean distance between first neighboring sources.
    block_size : float | None, optional
        The block size for placing the equivalent sources, by default None which place 1
        source beneath each datapoint.
    max_iterations : int, optional
        End the iterations after this value, by default 1
    rms_tolerance : float | None, optional
        End the iteration once the levelling correction RMS is less than this value, by
        default None
    rms_percent_change_tolerance : float, optional
        End the iterations if the percentage change of levelling correction RMS over 2
        consecutive iterations is less the than this percentage. This helps stop the
        iterations once improvement has plateaued, by default 10.
    rms_percent_increase_tolerance : float, optional
        End the iterations if the levelling correction RMS of the current iterations is
        more than this percent greater then the minimum RMS of past iterations. This
        helps stop run-away iterations which keep getting worse, by default 20
    seed : int, optional
        Seed supplied to the random number generator for shuffling the lines so they are
        iterated over in a random order, by default 42
    plot_iterations : bool, optional
        Plot the convergence of levelling correction RMS value, by default True
    progressbar : bool, optional
        Show progress bars for both iterations and levelling of lines, by default True

    Returns
    -------
    pd.Series
        The levelled data column, which can be assigned back to the original dataframe.
    """
    # check columns are present
    cols = ["easting", "northing", "height", "line", "distance_along_line", data_column]
    assert all(col in data.columns for col in cols), f"{cols} must be in the dataframe"

    data = data.copy()

    # save index and reset
    data = data.reset_index(names="tmp_index").reset_index(drop=True)

    if lines_to_level is not None:
        line_list = copy.deepcopy(lines_to_level)
    else:
        line_list = data.line.unique()

    if max_iterations == 1:
        progressbar = False

    if progressbar:
        pbar_iterations = tqdm(range(1, max_iterations + 1))
    else:
        pbar_iterations = range(1, max_iterations + 1)

    correction_rms_values = []
    correction_delta_rms_values = []
    iteration = 1
    for iteration in pbar_iterations:
        if progressbar:
            pbar_iterations.set_description(f"Iteration: {iteration}")

        # shuffle to order of lines to not start at the edge
        rng = np.random.default_rng(seed + iteration)
        rng.shuffle(line_list)

        pbar_lines = tqdm(line_list, leave=False) if progressbar else line_list
        for line_name in pbar_lines:
            if progressbar:
                pbar_lines.set_description(f"Levelling line: {line_name}")

            line_df = data[data.line == line_name]
            survey_df = data[data.line != line_name]

            # subset data nearby
            dist_mask = vd.distance_mask(
                (line_df.easting, line_df.northing),
                maxdist=max_dist,
                coordinates=(survey_df.easting, survey_df.northing),
            )
            survey_df = survey_df.iloc[dist_mask]

            # fit eq sources to nearby data
            coords = (
                survey_df.easting,
                survey_df.northing,
                survey_df.height,
            )
            eqs = hm.EquivalentSources(
                damping=damping, depth=depth, block_size=block_size
            )
            eqs.fit(coords, survey_df[data_column])

            # predict eq sources on the line to be levelled
            line_df["tmp_predicted_eqs"] = eqs.predict(
                (line_df.easting, line_df.northing, line_df.height)
            )

            line_df["tmp_misfit"] = line_df.tmp_predicted_eqs - line_df[data_column]

            # calculate levelling correction with a trend fit to the misfit values
            line_df = airbornegeo.levelling.skl_predict_trend(
                data_to_fit=line_df,
                cols_to_fit=["distance_along_line", "tmp_misfit"],
                data_to_predict=line_df,
                cols_to_predict=[
                    "distance_along_line",
                    f"tmp_levelling_correction_{iteration}",
                ],
                degree=degree,
            )

            # update the levelled line before moving on to the next line
            data.loc[data.line == line_name, data_column] = (
                line_df[data_column] + line_df[f"tmp_levelling_correction_{iteration}"]
            )
            data.loc[
                data.line == line_name, f"tmp_levelling_correction_{iteration}"
            ] = line_df[f"tmp_levelling_correction_{iteration}"]

        # add RMS and delta RMS of correction values for iteration to lists
        rms = airbornegeo.rmse(data[f"tmp_levelling_correction_{iteration}"])
        delta_rms = (
            (correction_rms_values[-1] / rms - 1) * 100 if iteration > 1 else np.inf
        )
        correction_rms_values.append(rms)
        correction_delta_rms_values.append(delta_rms)

        # apply levelling correction to data
        # data["tmp_levelled"] = data[data_column] + data[f"tmp_levelling_correction_{iteration}"]

        end, termination_reason = _end_iterations(
            rms_values=correction_rms_values,
            delta_rms_values=correction_delta_rms_values,
            max_iterations=max_iterations,
            rms_tolerance=rms_tolerance,
            rms_percent_change_tolerance=rms_percent_change_tolerance,
            rms_percent_increase_tolerance=rms_percent_increase_tolerance,
        )

        if end:
            if progressbar:
                pbar_iterations.set_description(
                    f"Iterations ended due to {termination_reason}"
                )
            break

    # Reset index and sort
    data = data.set_index("tmp_index").sort_values("tmp_index")

    if plot_iterations and max_iterations > 1:
        airbornegeo.plotting.plot_eqs_levelling_convergence(
            rms_values=correction_rms_values,
            delta_rms_values=correction_delta_rms_values,
            rms_tolerance=rms_tolerance,
            rms_percent_change_tolerance=rms_percent_change_tolerance,
        )

    return data[data_column]


def calculate_intersection_weights(
    gdf: gpd.GeoDataFrame,
    inters: gpd.GeoDataFrame,
    *,
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
    height_col_name: str = "height",
    plot: bool = False,
) -> gpd.GeoDataFrame:
    """
    Calculate weights for each intersection based on various criteria.
    """

    inters = inters.copy()
    gdf = gdf.copy()

    assert "line" in gdf.columns, "gdf must have column 'line'"

    # get list of lines from inters
    lines = [*inters.line.unique(), *inters.tie.unique()]

    # subset data based on lines
    gdf = gdf[gdf.line.isin(lines)]
    if weight_by in ("line", "tie", "all"):
        pass
    else:
        msg = "weight_by must be 'line', 'tie', or 'all'"
        raise ValueError(msg)

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
            inters["max_dist_weight"] = airbornegeo.normalize_values(
                inters["max_dist_weight"],
                low=1,
                high=0.001,  # reversed so large distances are bad
                # quantiles=(0.02, 0.98),
            )
        else:
            inters["max_dist_weight"] = inters.groupby(weight_by)[
                "max_dist_weight"
            ].transform(
                lambda x: airbornegeo.normalize_values(
                    x,
                    low=1,
                    high=0.001,  # reversed so large distances are bad
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
                (gdf.line == row.line) & (gdf.intersecting_line == row.tie)
            ][height_col_name].to_numpy()[0]
            tie_value = gdf[
                (gdf.line == row.tie) & (gdf.intersecting_line == row.line)
            ][height_col_name].to_numpy()[0]
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
            inters["height_difference_weight"] = airbornegeo.normalize_values(
                inters["height_difference_weight"],
                low=1,
                high=0.001,  # reversed so large differences are bad
                # quantiles=(0.02, 0.98),
            )
        else:
            inters["height_difference_weight"] = inters.groupby(weight_by)[
                "height_difference_weight"
            ].transform(
                lambda x: airbornegeo.normalize_values(
                    x,
                    low=1,
                    high=0.001,  # reversed so large differences are bad
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
            inters["interpolation_type_weight"] = airbornegeo.normalize_values(
                inters["interpolation_type_weight"],
                low=1,
                high=0.001,  # reversed so large numbers of extrapolations are bad
                # quantiles=(0.02, 0.98),
            )
        else:
            inters["interpolation_type_weight"] = inters.groupby(weight_by)[
                "interpolation_type_weight"
            ].transform(
                lambda x: airbornegeo.normalize_values(
                    x,
                    low=1,
                    high=0.001,  # reversed so large numbers of extrapolations are bad
                    # quantiles=(0.02, 0.98),
                )
            )

        weights_cols.append("interpolation_type_weight")
        weights_dict["interpolation_type_weight"] = interpolation_type_weight
        plot_cols.append("number_of_extrapolations")

    if data_1st_derive_weight is not None:
        if data_1st_derive_col_name is None:
            msg = "must provide 'data_1st_derive_col_name'"
            raise ValueError(msg)
        # find data gradient at intersection for line and tie
        for ind, row in inters.iterrows():
            # search data for values at intersecting lines
            line_value = gdf[
                (gdf.line == row.line) & (gdf.intersecting_line == row.tie)
            ][data_1st_derive_col_name].to_numpy()[0]
            tie_value = gdf[
                (gdf.line == row.tie) & (gdf.intersecting_line == row.line)
            ][data_1st_derive_col_name].to_numpy()[0]
            inters.loc[ind, "data_1st_derive"] = np.mean(
                np.abs([line_value, tie_value])
            )
        weight_vals = inters.data_1st_derive
        if data_1st_derive_floor is not None:
            weight_vals = np.where(
                weight_vals < data_1st_derive_floor,
                data_1st_derive_floor,
                weight_vals,
            )
        inters["data_1st_derive_weight"] = weight_vals

        if weight_by == "all":
            inters["data_1st_derive_weight"] = airbornegeo.normalize_values(
                inters["data_1st_derive_weight"],
                low=1,
                high=0.001,  # reversed so large gradients are bad
                # quantiles=(0.02, 0.98),
            )
        else:
            inters["data_1st_derive_weight"] = inters.groupby(weight_by)[
                "data_1st_derive_weight"
            ].transform(
                lambda x: airbornegeo.normalize_values(
                    x,
                    low=1,
                    high=0.001,  # reversed so large gradients are bad
                    # quantiles=(0.02, 0.98),
                )
            )

        weights_cols.append("data_1st_derive_weight")
        weights_dict["data_1st_derive_weight"] = data_1st_derive_weight
        plot_cols.append("data_1st_derive")

    if data_2nd_derive_weight is not None:
        if data_2nd_derive_col_name is None:
            msg = "must provide 'data_2nd_derive_col_name'"
            raise ValueError(msg)
        # find data gradient at intersection for line and tie
        for ind, row in inters.iterrows():
            # search data for values at intersecting lines
            line_value = gdf[
                (gdf.line == row.line) & (gdf.intersecting_line == row.tie)
            ][data_2nd_derive_col_name].to_numpy()[0]
            tie_value = gdf[
                (gdf.line == row.tie) & (gdf.intersecting_line == row.line)
            ][data_2nd_derive_col_name].to_numpy()[0]
            inters.loc[ind, "data_2nd_derive"] = np.mean(
                np.abs([line_value, tie_value])
            )
        weight_vals = inters.data_2nd_derive
        if data_2nd_derive_floor is not None:
            weight_vals = np.where(
                weight_vals < data_2nd_derive_floor,
                data_2nd_derive_floor,
                weight_vals,
            )
        inters["data_2nd_derive_weight"] = weight_vals

        if weight_by == "all":
            inters["data_2nd_derive_weight"] = airbornegeo.normalize_values(
                inters["data_2nd_derive_weight"],
                low=1,
                high=0.001,  # reversed so large gradients are bad
                # quantiles=(0.02, 0.98),
            )
        else:
            inters["data_2nd_derive_weight"] = inters.groupby(weight_by)[
                "data_2nd_derive_weight"
            ].transform(
                lambda x: airbornegeo.normalize_values(
                    x,
                    low=1,
                    high=0.001,  # reversed so large gradients are bad
                    # quantiles=(0.02, 0.98),
                )
            )

        weights_cols.append("data_2nd_derive_weight")
        weights_dict["data_2nd_derive_weight"] = data_2nd_derive_weight
        plot_cols.append("data_2nd_derive")

    if height_1st_derive_weight is not None:
        if height_1st_derive_col_name is None:
            msg = "must provide 'height_1st_derive_col_name'"
            raise ValueError(msg)
        # find height gradient at intersection for line and tie
        for ind, row in inters.iterrows():
            # search data for values at intersecting lines
            line_value = gdf[
                (gdf.line == row.line) & (gdf.intersecting_line == row.tie)
            ][height_1st_derive_col_name].to_numpy()[0]
            tie_value = gdf[
                (gdf.line == row.tie) & (gdf.intersecting_line == row.line)
            ][height_1st_derive_col_name].to_numpy()[0]
            inters.loc[ind, "height_1st_derive"] = np.mean(
                np.abs([line_value, tie_value])
            )
        weight_vals = inters.height_1st_derive
        if height_1st_derive_floor is not None:
            weight_vals = np.where(
                weight_vals < height_1st_derive_floor,
                height_1st_derive_floor,
                weight_vals,
            )
        inters["height_1st_derive_weight"] = weight_vals

        if weight_by == "all":
            inters["height_1st_derive_weight"] = airbornegeo.normalize_values(
                inters["height_1st_derive_weight"],
                low=1,
                high=0.001,  # reversed so large gradients are bad
                # quantiles=(0.02, 0.98),
            )
        else:
            inters["height_1st_derive_weight"] = inters.groupby(weight_by)[
                "height_1st_derive_weight"
            ].transform(
                lambda x: airbornegeo.normalize_values(
                    x,
                    low=1,
                    high=0.001,  # reversed so large gradients are bad
                    # quantiles=(0.02, 0.98),
                )
            )

        weights_cols.append("height_1st_derive_weight")
        weights_dict["height_1st_derive_weight"] = height_1st_derive_weight
        plot_cols.append("height_1st_derive")

    if height_2nd_derive_weight is not None:
        if height_2nd_derive_col_name is None:
            msg = "must provide 'height_2nd_derive_col_name'"
            raise ValueError(msg)
        # find height gradient at intersection for line and tie
        for ind, row in inters.iterrows():
            # search data for values at intersecting lines
            line_value = gdf[
                (gdf.line == row.line) & (gdf.intersecting_line == row.tie)
            ][height_2nd_derive_col_name].to_numpy()[0]
            tie_value = gdf[
                (gdf.line == row.tie) & (gdf.intersecting_line == row.line)
            ][height_2nd_derive_col_name].to_numpy()[0]
            inters.loc[ind, "height_2nd_derive"] = np.mean(
                np.abs([line_value, tie_value])
            )
        weight_vals = inters.height_2nd_derive
        if height_2nd_derive_floor is not None:
            weight_vals = np.where(
                weight_vals < height_2nd_derive_floor,
                height_2nd_derive_floor,
                weight_vals,
            )
        inters["height_2nd_derive_weight"] = weight_vals

        if weight_by == "all":
            inters["height_2nd_derive_weight"] = airbornegeo.normalize_values(
                inters["height_2nd_derive_weight"],
                low=1,
                high=0.001,  # reversed so large gradients are bad
                # quantiles=(0.02, 0.98),
            )
        else:
            inters["height_2nd_derive_weight"] = inters.groupby(weight_by)[
                "height_2nd_derive_weight"
            ].transform(
                lambda x: airbornegeo.normalize_values(
                    x,
                    low=1,
                    high=0.001,  # reversed so large gradients are bad
                    # quantiles=(0.02, 0.98),
                )
            )

        weights_cols.append("height_2nd_derive_weight")
        weights_dict["height_2nd_derive_weight"] = height_2nd_derive_weight
        plot_cols.append("height_2nd_derive")

    logger.info(
        "combining individual weight cols with following factors: %s", weights_dict
    )

    # calculated weighted mean of the weights
    def weighted_average(
        df: pd.DataFrame | gpd.GeoDataFrame, weights: dict[str, float]
    ) -> pd.Series:
        return df[list(weights)].mul(weights).sum(axis=1) / sum(weights.values())

    # inters["mistie_weight"] = weighted_average(inters, weights_dict)
    # inters["mistie_weights"] = inters[weights_cols].mean(axis=1)

    if weight_by == "all":
        inters["mistie_weight"] = weighted_average(inters, weights_dict)
        inters["mistie_weight"] = airbornegeo.normalize_values(
            inters["mistie_weight"],
            low=0.001,
            high=1,
        )
    else:
        inters["mistie_weight"] = (
            inters.groupby(weight_by)
            .apply(
                lambda x: pd.Series(weighted_average(x, weights_dict), index=x.index),
                include_groups=False,
            )
            .reset_index(drop=True)
        )
        # inters["mistie_weight"] = inters.groupby(weight_by).transform(
        #     lambda x: weighted_average(x, weights_dict),
        # )
        inters["mistie_weight"] = inters.groupby(weight_by)["mistie_weight"].transform(
            lambda x: airbornegeo.normalize_values(
                x,
                low=0.001,
                high=1,
            )
        )

    if plot:
        airbornegeo.plotly_points(
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
            size=6,
            theme=None,
        )
    return inters


def plot_levelling_convergence(
    results: gpd.GeoDataFrame | pd.DataFrame,
    *,
    logy: bool = False,
    title: str = "Levelling convergence",
    as_median: bool = False,
) -> None:
    # get mistie columns
    cols = [s for s in results.columns.to_list() if s.startswith("mistie_")]

    iters = len(cols)

    mistie_rmses = [
        airbornegeo.rmse(
            results[i],
            as_median=as_median,
        )
        for i in cols
    ]
    _fig, ax1 = plt.subplots(figsize=(5, 3.5))
    plt.title(title)
    ax1.plot(range(iters), mistie_rmses, "bo-")
    ax1.set_xlabel("Iteration")
    if logy:
        ax1.set_yscale("log")
    ax1.set_ylabel("Cross-over RMSE", color="k")
    ax1.tick_params(axis="y", colors="k", which="both")

    ax1.set_xticks(range(iters))


def create_intersection_table(
    data: gpd.GeoDataFrame,
    *,
    exclude_ints: list[list[float, float] | list[float] | float] | None = None,
    cutoff_dist: float | None = None,
    buffer_dist: float | None = None,
    grid_size: float = 1,
    plot_map: bool = True,
    plot_hist: bool = True,
    size: float = 10,
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
    data : gpd.GeoDataFrame
        Dataframe with both tie lines and flight lines.
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
    grid_size : float, optional
        The resolution to snap the intersection coordinates to.  by default 1
    plot_map : bool, optional
        Plot a map of the resulting intersection points colored by distance to the
        further of the two nearest data points, by default True
    plot_hist : bool, optional
        Plot a histogram of the max distances to the nearest points, by default True,
    size : float, optional
        Size of the points for plotting, by default 10

    Returns
    -------
    gpd.GeoDataFrame
        An intersection table containing the locations of the theoretical intersections,
        the line and tie numbers, and the distance to the further of the two nearest
        datapoints of each line, and a geometry column.
    """
    data = data.copy()

    assert "tie" in data.columns, "data must have column of booleans named 'tie'"
    assert isinstance(data, gpd.GeoDataFrame), "data must be a GeoDataFrame"
    assert data.geometry.geom_type.isin(["Point"]).all(), "geometry must be points"

    lines_df = data[~data.tie]
    ties_df = data[data.tie]

    assert "line" in lines_df.columns, "flight_lines must have column 'line'"
    assert "line" in ties_df.columns, "tie_lines must have column 'line'"

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

    # get the largest of the two distance to each lines' nearest data point to the
    # theoretical intersection
    inters["max_dist"] = inters[["line_dist", "tie_dist"]].max(axis=1)

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
        logger.debug("Dropped %s duplicate intersections", a - b)

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

    # get coords from geometry column
    inters["easting"] = inters.geometry.x
    inters["northing"] = inters.geometry.y

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
                    (inters.line == i[0]) & (inters.tie == i[1])
                ].index.to_numpy()
                exclude_inds.extend(ind)
                ind = inters[
                    (inters.tie == i[0]) & (inters.line == i[1])
                ].index.to_numpy()
                exclude_inds.extend(ind)
            # if single line number, get all intersections of that line
            elif len(i) == 1:
                ind = inters[
                    (inters.line == i[0]) | (inters.tie == i[0])
                ].index.to_numpy()
                exclude_inds.extend(ind)
        inters = inters.drop(index=exclude_inds).copy()
        logger.info(
            "manually omitted %s intersections points",
            prior_len - len(inters),
        )

    if plot_map:
        airbornegeo.plotly_points(
            inters,
            color_col="max_dist",
            hover_cols=["line", "tie", "line_dist", "tie_dist"],
            robust=True,
            size=size,
            edge_width=1,
            theme=None,
            cmap="matter",
            title="Distance from intersection to nearest data point",
        )
    if plot_hist:
        plt.hist(
            inters.max_dist,
            bins=20,
        )
        plt.xlabel("Max distance between intersection and line/tie data (m)")

    return inters.drop(columns=["line_dist", "tie_dist"]).reset_index(drop=True)


def add_intersections(
    data: gpd.GeoDataFrame,
    intersections: gpd.GeoDataFrame,
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

    Returns
    -------
    tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
        The updated flight survey dataframe and intersections table.
    """
    data = data.copy()
    inters = intersections.copy()

    assert "line" in data.columns, "data must have column 'line'"

    # if is_intersection column exists, delete it and rows where it's true
    if "is_intersection" in data.columns:
        rows_to_drop = data[data.is_intersection]
        data = data.drop(index=rows_to_drop.index)
    data = data.drop(
        columns=["is_intersection", "intersecting_line"],
        errors="ignore",
    )

    prior_length = len(data)

    # add boolean column for whether point is an intersection
    data["is_intersection"] = False
    data["intersecting_line"] = np.nan

    # collect intersections to be added
    dfs = []
    for _, row in inters.iterrows():
        for i in list(data.line.unique()):
            if i in (row.line, row.tie):
                new_row = pd.DataFrame(
                    {
                        "line": [i],
                        "easting": row.easting,
                        "northing": row.northing,
                        "is_intersection": True,
                    }
                )
                if i == row.line:
                    new_row["intersecting_line"] = row.tie
                else:
                    new_row["intersecting_line"] = row.line
                new_row["geometry"] = gpd.points_from_xy(
                    new_row.easting, new_row.northing
                )
                # print(new_row)
                # for each intersection row, find the closest 2 two points, then take
                # the one with the lower `distance_along_line` value, then calculate the
                # cumulative distance from that point to the intersection point.
                line_points = data[data.line == i]
                nearest_two_points = (
                    line_points.distance(new_row.geometry.iloc[0])
                    .sort_values(ascending=True)
                    .iloc[0:2]
                )
                nearest_two_points = line_points.loc[nearest_two_points.index]
                # print(nearest_two_points)
                min_dist_point = nearest_two_points.sort_values(
                    by="distance_along_line", ascending=True
                ).iloc[0]
                # print(min_dist_point)

                # calculate cumulative distance to the min_dist_point
                inter_and_nearest_point = pd.DataFrame(
                    {
                        "easting": [
                            *new_row.easting.to_numpy(),
                            min_dist_point.easting,
                        ],
                        "northing": [
                            *new_row.northing.to_numpy(),
                            min_dist_point.northing,
                        ],
                    }
                )

                # inter_and_nearest_point = pd.concat([new_row, min_dist_point])[["easting", "northing"]]
                # print(inter_and_nearest_point)
                dist_between = airbornegeo.relative_distance(
                    inter_and_nearest_point,
                    easting_column="easting",
                    northing_column="northing",
                )[-1]
                # print(dist_between)
                # print(min_dist_point.distance_along_line)
                new_row["distance_along_line"] = (
                    min_dist_point.distance_along_line + dist_between
                )
                # print(new_row)
                # return
                dfs.append(new_row)

    # add intersections as new rows to dataframe
    # note that they are not placed in the correct position
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="CRS not set for some of the concatenation inputs.",
        )
        data = pd.concat([data, *dfs])

    data = data.sort_values(["line", "distance_along_line"]).reset_index(drop=True)

    # check correct number of intersections were added
    assert len(data) == prior_length + (2 * len(inters))

    # recalculate distance along each line now that intersections are included
    # TODO do we really need to recalculate distance for every point? Or just for the
    # new intersection points?
    # data = data.sort_values(['easting', 'northing'])
    # data["distance_along_line"] = airbornegeo.along_track_distance(data, groupby_column="line", guess_start_position=True)

    # add dist along line to intersections dataframe
    for ind, row in inters.iterrows():
        # search data for values at intersecting lines
        line_value = data[
            (data.line == row.line) & (data.intersecting_line == row.tie)
        ].distance_along_line.to_numpy()[0]
        tie_value = data[
            (data.line == row.tie) & (data.intersecting_line == row.line)
        ].distance_along_line.to_numpy()[0]

        inters.loc[ind, "dist_along_flight_line"] = line_value
        inters.loc[ind, "dist_along_flight_tie"] = tie_value

    return data, inters


def extend_line(
    line: LineString,
    distance: float,
    plot: bool = False,
) -> LineString:
    """extend line in either direction by distance"""
    # find minimum rotated rectangle around line
    rect = line.minimum_rotated_rectangle
    angle = airbornegeo.nav.azimuth(rect)

    rect_center = shapely.centroid(rect).x, shapely.centroid(rect).y

    # get length of long edge
    x, y = rect.exterior.coords.xy
    length = max(
        (
            Point(x[0], y[0]).distance(Point(x[1], y[1])),
            Point(x[1], y[1]).distance(Point(x[2], y[2])),
        )
    )

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
    *,
    grid_size: float = 1,
    buffer_dist: float | None = None,
) -> gpd.GeoDataFrame:
    """
    adapted from https://gis.stackexchange.com/questions/137909/intersecting-lines-to-get-crossings-using-python-with-qgis

    grid_size : float, optional
        round the coordinates of the resulting intersection points to this resolution,
        by default 1
    """

    assert "line" in lines_gdf.columns, "lines_gdf must have column 'line'"
    assert "line" in ties_gdf.columns, "ties_gdf must have column 'line'"

    # group by lines/ties
    grouped_lines = lines_gdf.groupby(["line"], as_index=False)["geometry"]
    grouped_ties = ties_gdf.groupby(["line"], as_index=False)["geometry"]

    # from points to lines
    grouped_lines = grouped_lines.apply(lambda x: LineString(x.tolist()))
    grouped_ties = grouped_ties.apply(lambda x: LineString(x.tolist()))

    # extend ends of lines by buffer_dist to account for expected intersections just
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
        strict=False,
    )
    inters = []
    line_names = []
    tie_names = []
    for (line, tie), (l_name, t_name) in pbar:
        # determine intersection of line and tie
        inter = shapely.intersection(line, tie, grid_size=grid_size)
        points = [Point(i) for i in shapely.get_coordinates(inter)]

        # add each intersection and their names to lists
        inters.extend(points)
        line_names.extend([l_name] * len(points))
        tie_names.extend([t_name] * len(points))

    inters = gpd.GeoDataFrame(
        geometry=inters, data={"line": line_names, "tie": tie_names}
    )

    # get nearest 2 lines to each intersection point
    # and nearest data point on each line to the intersection point
    line_names = []
    tie_names = []
    line_dists = []
    tie_dists = []

    pbar = tqdm(
        inters.geometry,
        desc="Potential intersections",
        total=len(inters.geometry),
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
        nearest_line = grouped_lines.sort_values(by="dist")[["line"]].iloc[0]
        nearest_tie = grouped_ties.sort_values(by="dist")[["line"]].iloc[0]

        # get line/tie names
        line = nearest_line.line
        tie = nearest_tie.line

        # append names to lists
        line_names.append(line)
        tie_names.append(tie)

        # get actual datapoints for each line (not LineString representation)
        line_points = lines_gdf[lines_gdf.line == line]
        tie_points = ties_gdf[ties_gdf.line == tie]

        # get nearest data point on each line/tie to intersection point
        nearest_datapoint_line = line_points.geometry.distance(p).sort_values().iloc[0]
        nearest_datapoint_tie = tie_points.geometry.distance(p).sort_values().iloc[0]

        # add distance to nearest data point on each line to lists
        line_dists.append(nearest_datapoint_line)
        tie_dists.append(nearest_datapoint_tie)

    # add names and distances as columns
    inters["line"] = line_names
    inters["tie"] = tie_names
    inters["line_dist"] = line_dists
    inters["tie_dist"] = tie_dists

    # set CRS from supplied data
    inters.crs = lines_gdf.crs

    return inters


def get_line_intersections(
    lines: gpd.GeoSeries,
) -> list[Point]:
    """
    adapted from https://gis.stackexchange.com/questions/137909/intersecting-lines-to-get-crossings-using-python-with-qgis
    """

    inters = []
    for line1, line2 in itertools.combinations(lines, 2):
        if line1.intersects(line2):
            inter = line1.intersection(line2)

            if inter.geom_type == "Point":
                inters.append(inter)
            elif inter.geom_type == "MultiPoint":
                inters.extend(list(inter.geoms))
            elif inter.geom_type == "MultiLineString":
                multi_line = list(inter.geoms)
                first_coords = multi_line[0].coords[0]
                last_coords = multi_line[len(multi_line) - 1].coords[1]
                inters.append(Point(first_coords[0], first_coords[1]))
                inters.append(Point(last_coords[0], last_coords[1]))
            elif inter.geom_type == "GeometryCollection":
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


def interpolate_intersections(
    df: pd.DataFrame | gpd.GeoDataFrame,
    intersections: pd.DataFrame | gpd.GeoDataFrame,
    *,
    to_interp: str,
    interp_on: str = "distance_along_line",
    method: str = "cubic",
    extrapolate: bool = False,
    fill_value: tuple[float, float] | str | None = None,
    window_width: float | None = None,
) -> pd.DataFrame | gpd.GeoDataFrame:
    """
    _summary_

    Parameters
    ----------
    df : pd.DataFrame | gpd.GeoDataFrame
        Dataframe containing the data to interpolate
    intersections : pd.DataFrame | gpd.GeoDataFrame
        Dataframe containing the intersection points
    to_interp : str,
        specify which column to interpolate NaNs for
    interp_on : str, optional
        Decide which column interpolation is based on, by default "distance_along_line"
    method : str, optional
        Decide between interpolation methods of 'linear', 'nearest', 'nearest-up',
        'zero', 'slinear', 'quadratic','cubic', 'previous', or 'next', by default
        "cubic"
    window_width : float, optional
        window width around each NaN to use for interpolation fitting, by default None

    Returns
    -------
    pd.DataFrame | gpd.GeoDataFrame
        the survey dataframe with NaN's filled in the specified columns
    """
    df = df.copy()
    inters = intersections.copy()

    assert "line" in df.columns, "df must have column 'line'"

    assert isinstance(to_interp, str), "to_interp should be a single string"
    # if isinstance(to_interp, str):
    #     to_interp = [to_interp]

    # drop rows with NaNs in all rows to interp
    df = df.dropna(subset=to_interp, how="any")

    # add empty rows at each intersection to the df
    df, inters = add_intersections(df, inters)

    if window_width is None:
        filled_lines = airbornegeo.interpolating.interpolate_missing(
            df,
            to_interp=to_interp,
            interp_on=interp_on,
            method=method,
            extrapolate=extrapolate,
            fill_value=fill_value,
            groupby_column="line",
        )
    else:
        filled_lines = airbornegeo.interpolating.interpolate_missing_with_windows(
            df,
            to_interp=to_interp,
            window_width=window_width,
            interp_on=interp_on,
            method=method,
            extrapolate=extrapolate,
            fill_value=fill_value,
            groupby_column="line",
        )

    # lines = df.groupby("line")
    # filled_lines = []
    # pbar = tqdm(lines, desc="Lines")
    # for line, line_df in pbar:
    #     pbar.set_description(f"Line {line}")

    #     if window_width is None:
    #         filled = airbornegeo.interpolating.interpolate_missing(
    #             line_df,
    #             to_interp=to_interp,
    #             interp_on=interp_on,
    #             method=method,
    #             extrapolate=extrapolate,
    #             fill_value=fill_value,
    #         )
    #     else:
    #         filled = airbornegeo.interpolating.interpolate_missing_with_windows(
    #             line_df,
    #             to_interp=to_interp,
    #             window_width=window_width,
    #             interp_on=interp_on,
    #             method=method,
    #             extrapolate=extrapolate,
    #             fill_value=fill_value,
    #         )

    #     filled_lines.append(filled)

    # filled_lines = pd.concat(filled_lines)

    inters["flight_interpolation_type"] = "none"
    inters["tie_interpolation_type"] = "none"
    # inters["flight_height"] = np.nan
    # inters["tie_height"] = np.nan
    # add whether intersection was interpolated or extrapolated with respect to both lines and ties
    for ind, row in inters.iterrows():
        # search data for values at intersecting lines
        line_values = filled_lines[
            (filled_lines.line == row.line)
            & (filled_lines.intersecting_line == row.tie)
        ]
        tie_values = filled_lines[
            (filled_lines.line == row.tie)
            & (filled_lines.intersecting_line == row.line)
        ]
        # get interpolation type
        flight_interp_type = line_values[f"{to_interp}_interpolation_type"].to_numpy()[
            0
        ]
        tie_interp_type = tie_values[f"{to_interp}_interpolation_type"].to_numpy()[0]
        # get heights
        # line_height = line_values.height.to_numpy()[0]
        # tie_height = tie_values.height.to_numpy()[0]
        # add to intersection table
        inters.loc[ind, "flight_interpolation_type"] = flight_interp_type
        inters.loc[ind, "tie_interpolation_type"] = tie_interp_type
        # inters.loc[ind, "flight_height"] = line_height
        # inters.loc[ind, "tie_height"] = tie_height

    # drop inters rows if the interpolation didn't work for either line or tie
    inters_to_drop = inters[
        (inters.flight_interpolation_type == "none")
        | (inters.tie_interpolation_type == "none")
    ]
    inters = inters.drop(index=inters_to_drop.index)

    # drop the corresponding rows from filled_lines
    inters_to_drop_list = (
        inters_to_drop.line.to_numpy(),
        inters_to_drop.tie.to_numpy(),
    )
    rows_to_drop = []
    for line_name, tie_name in zip(*inters_to_drop_list, strict=True):
        rows_to_drop.append(
            filled_lines[
                (filled_lines.is_intersection)
                & (filled_lines.line == line_name)
                & (filled_lines.intersecting_line == tie_name)
            ]
        )
        rows_to_drop.append(
            filled_lines[
                (filled_lines.is_intersection)
                & (filled_lines.line == tie_name)
                & (filled_lines.intersecting_line == line_name)
            ]
        )
    if len(rows_to_drop) > 0:
        rows_to_drop = pd.concat(rows_to_drop)
        filled_lines = filled_lines.drop(index=rows_to_drop.index)

    filled_lines = filled_lines.sort_values(["line", "unixtime"]).reset_index(drop=True)

    assert len(inters) * 2 == len(filled_lines[filled_lines.is_intersection]), (
        "Number of intersection rows in the dataframe should be twice the number of rows in intersection table"
    )
    return filled_lines, inters


def inspect_intersections(
    data: pd.DataFrame | gpd.GeoDataFrame,
    *,
    plot_variable: str | list[str],
    interp_on: str = "distance_along_line",
    plot_all: bool = False,
) -> None:
    if isinstance(plot_variable, str):
        plot_variable = [plot_variable]

    for line, line_df in data.groupby("line"):
        if len(line_df[line_df.is_intersection]) == 0:
            continue

        fig = plot_line_and_crosses(
            data,
            line=line,
            x=interp_on,
            y=plot_variable,
            y_axes=[str(i + 1) for i in range(len(plot_variable))],
            plot_inters=True,
        )
        fig.show()
        if plot_all:
            continue
        input("Press key to continue...")
        clear_output(wait=True)


def calculate_crossover_errors(
    data: gpd.GeoDataFrame,
    intersections: gpd.GeoDataFrame,
    *,
    data_col: str,
    plot_map: bool = False,
    plot_hist: bool = True,
    robust: bool = True,
    warn_if_unchanged: bool = False,
) -> gpd.GeoDataFrame:
    """
    Calculate mistie values for all intersections. For each intersection, find the data
    values for the line and tie from the survey dataframe and add those values to the
    intersection table as `line_value` and `tie_value`. If they exist, overwrite them.
    Calculate the mistie value as line_value - tie_value, and save this to a column
    `mistie_0`. If `mistie_0` exists, make a new column `mistie_1`, and keep incrementing
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
    plot_map : bool, optional
        Plot the resulting mistie points on a map, by default False
    plot_hist : bool, optional
        Plot the resulting mistie histogram, by default True
    robust : bool, optional
        Use robust color limits for the map, by default True
    warn_if_unchanged : bool, optional
        If true, raise a UserWarning if misties haven't changed from previous column, by
        default False.

    Returns
    -------
    gpd.GeoDataFrame
        An intersections table with new columns `line_value`, `tie_value` and `mistie_x`
        where x is incremented each time a new mistie is calculated.
    """

    inters = intersections.copy()
    df = data.copy()

    assert "line" in df.columns, "df must have column 'line'"

    # iterate through intersections
    misties = []
    for _ind, row in inters.iterrows():
        # search data for values at intersecting lines
        line_value = df[(df.line == row.line) & (df.intersecting_line == row.tie)][
            data_col
        ].to_numpy()[0]
        tie_value = df[(df.line == row.tie) & (df.intersecting_line == row.line)][
            data_col
        ].to_numpy()[0]
        assert not np.isnan(line_value)
        assert not np.isnan(tie_value)

        # mistie is line - tie
        misties.append(line_value - tie_value)

    misties = pd.Series(misties).to_numpy()
    logger.debug("mistie RMSE: %s", airbornegeo.rmse(misties))

    cols = [c for c in inters.columns if "mistie_" in c]
    mistie_col = [int(col.split("_")[-1]) for col in cols]
    try:
        current_mistie_col = f"mistie_{max(mistie_col)}"
    except ValueError:
        current_mistie_col = "mistie_0"
    next_mistie_col = f"mistie_{len(cols)}"

    # check if new misties are identical to old misties
    if len(cols) > 0:
        try:
            pd.testing.assert_series_equal(
                inters[f"mistie_{len(cols) - 1}"],
                pd.Series(misties),
                check_names=False,
                check_index=False,
            )
            logger.debug("Mistie values are unchanged")
            if warn_if_unchanged:
                msg = "Mistie hasn't changed"
                raise UserWarning(msg)
        except AssertionError:
            inters[next_mistie_col] = misties
            current_mistie_col = next_mistie_col
    else:
        inters[next_mistie_col] = misties
        current_mistie_col = next_mistie_col
    if plot_map:
        airbornegeo.plotly_points(
            inters,
            color_col=current_mistie_col,
            hover_cols=["line", "tie"],
            robust=robust,
            absolute=True,
            cmap="balance",
            size=10,
            edge_width=1,
        )
    if plot_hist:
        plt.hist(
            misties,
            bins=20,
        )
        plt.xlabel("Mistie value")
        plt.title(f"Histogram of misties; RMSE: {round(airbornegeo.rmse(misties), 2)}")

    return inters


def skl_predict_trend(
    data_to_fit: pd.DataFrame,
    cols_to_fit: list[str],
    data_to_predict: pd.DataFrame,
    cols_to_predict: list[str],
    degree: int,
    intersection_weight_col: str | None = None,
) -> pd.DataFrame:
    """
    data_to_fit: pd.DataFrame with at least 2 columns: distance, and data
    cols_to_fit: column names representing distance and data
    data_to_predict: pd.DataFrame with at least 1 columns: distance
    cols_to_predict: column names representing distance and new column
        with predicted data
    intersection_weight_col: column name for sample weights
    """
    fit_df = data_to_fit.copy()
    predict_df = data_to_predict.copy()

    # fit a polynomial trend through the lines mistie values
    polynomial_features = PolynomialFeatures(
        degree=degree,
        include_bias=True,
    )
    linear_regression = LinearRegression()

    if intersection_weight_col is not None:
        sample_weight = fit_df[intersection_weight_col].to_numpy()
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


def level_to_grid(
    data: pd.DataFrame,
    *,
    degree: int,
    data_column: str,
    grid_column: str,
    groupby_column: str | None = None,
) -> pd.Series:
    """
    Fit a trend to the misfit between the data_column and grid_column values and
    subtract the fitted trend from the data to level it to the grid values. The grid
    values can be sampled into the dataframe with ::func`sample_grid`. If groupby_column
    is provided, the trend will be fit a line-by-line basis in 1D using the column
    distance_along_line. If groupby_column is not provided, the trend will be fit to the
    entire survey in 2D using the columns "easting" and "northing".

    Parameters
    ----------
    data : pd.DataFrame
        dataframe with the data and grid values.
    degree : int
        the degree order to fit to the misfit values.
    data_column : str
        the column name for the data to fit.
    grid_column : str
        the column name with the sample grid values.
    groupby_column : str | None
        Column name to group by, by default None

    Returns
    -------
    pd.Series
        The levelled data
    """
    data = data.copy()

    data = data.dropna(subset=grid_column)

    data["tmp_misfit"] = data[data_column] - data[grid_column]

    if groupby_column is None:
        cols = ["easting", "northing"]
        assert all(col in data.columns for col in cols), (
            f"{cols} must be in the dataframe"
        )
        # calculate correction by fitting trend to misfit values
        vdtrend = vd.Trend(degree=degree).fit(
            (data.easting, data.northing),
            data.tmp_misfit,
        )

        correction = vdtrend.predict(coordinates=((data.easting, data.northing)))

        return data[data_column] - correction

    cols = [groupby_column, "distance_along_line"]
    assert all(col in data.columns for col in cols), f"{cols} must be in the dataframe"

    # fit a trend to the misfits on line-by-line basis
    for _segment_name, segment_data in data.groupby(groupby_column):
        # calculate correction by fitting trend to misfit values
        correction = skl_predict_trend(
            data_to_fit=segment_data,
            cols_to_fit=["distance_along_line", "tmp_misfit"],
            data_to_predict=segment_data,
            cols_to_predict=["distance_along_line", "correction"],
            degree=degree,
        ).correction

        # add correction values to the main dataframe
        data.loc[data.line == _segment_name, "levelling_correction"] = correction

    return data[data_column] - data.levelling_correction


def line_levelling(
    data: gpd.GeoDataFrame | pd.DataFrame,
    inters: gpd.GeoDataFrame | pd.DataFrame,
    *,
    lines_to_level: list[float],
    data_col: str,
    levelled_col: str,
    degree: int,
    intersection_weight_col: str | None = None,
    plot: bool = False,
    warn_if_unchanged: bool = False,
) -> tuple[pd.DataFrame | gpd.GeoDataFrame, pd.DataFrame | gpd.GeoDataFrame]:
    """
    Level lines based on intersection misties values. Fit a trend of specified order to
    intersection misties, and apply the correction to the `data_col` column.
    """
    data = data.copy()
    inters = inters.copy()

    # drop lines with intersections
    lines_without_inters = lines_without_intersections(data, inters)
    lines_to_level = [x for x in lines_to_level if x not in lines_without_inters]

    assert "line" in data.columns, "data must have column 'line'"
    assert "distance_along_line" in data.columns, (
        "data must have column 'distance_along_line'"
    )

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

    if levelling_ties is True:
        cols_to_fit = ["dist_along_flight_tie"]
    elif levelling_lines is True:
        cols_to_fit = ["dist_along_flight_line"]
    else:
        msg = "need to supplied either tie lines or flight lines"
        raise ValueError(msg)

    data["levelling_correction"] = np.nan

    # get the latest mistie column
    inters2 = calculate_crossover_errors(
        data,
        inters,
        data_col=data_col,
        plot_map=False,
        plot_hist=False,
    )
    mistie_col = [
        int(col.split("_")[-1]) for col in inters2.columns if "mistie_" in col
    ]
    mistie_col = f"mistie_{max(mistie_col)}"

    logger.debug(
        "mistie before levelling: %s mGal", airbornegeo.rmse(inters2[mistie_col])
    )

    # fit a trend to the misfits on line-by-line basis
    # iterate through the chosen lines
    for line in lines_to_level:
        # subset a line
        line_df = data[data.line == line].copy()

        # get intersections of line of interest
        ints = inters2[(inters2.line == line) | (inters2.tie == line)]

        try:
            line_df = skl_predict_trend(
                data_to_fit=ints,  # data with mistie values
                cols_to_fit=cols_to_fit  # noqa: RUF005
                + [mistie_col],  # column names for distance/mistie
                data_to_predict=line_df,  # data with line data
                cols_to_predict=["distance_along_line"]  # noqa: RUF005
                + [
                    "levelling_correction"
                ],  # column names for distance/ levelling correction
                degree=degree,  # degree order for fitting line to misties
                intersection_weight_col=intersection_weight_col,
            )
        except ValueError as e:
            if "Found array with " in str(e):
                logger.error("Issue with line %s, skipping", line)
                # if issues, correction is 0
                line_df["levelling_correction"] = 0
            else:
                raise ValueError from e

        # if levelling tie lines, negate the correction
        if levelling_ties is True:
            line_df["levelling_correction"] *= -1
        else:
            pass

        # remove the trend from the gravity
        values = line_df[data_col] - line_df.levelling_correction

        # update main data
        data.loc[data.line == line, levelled_col] = values
        data.loc[data.line == line, "levelling_correction"] = (
            line_df.levelling_correction
        )

    # add unchanged values for lines not included
    for line in data.line.unique():
        if line not in lines_to_level:
            data.loc[data.line == line, levelled_col] = data.loc[
                data.line == line, data_col
            ]

    # update mistie with levelled data
    inters = calculate_crossover_errors(
        data,
        inters,
        data_col=levelled_col,
        plot_map=False,
        plot_hist=False,
        warn_if_unchanged=warn_if_unchanged,
    )
    mistie_col = [int(col.split("_")[-1]) for col in inters.columns if "mistie_" in col]
    mistie_col = f"mistie_{max(mistie_col)}"

    logger.debug(
        "mistie after levelling: %s mGal", airbornegeo.rmse(inters[mistie_col])
    )

    if plot is True:
        # plot old misties
        ints = inters2[
            inters2.line.isin(lines_to_level) | inters2.tie.isin(lines_to_level)
        ]
        airbornegeo.plotly_points(
            ints,
            color_col=mistie_col,
            hover_cols=[
                "line",
                "tie",
            ],
            cmap="balance",
            absolute=True,
            size=10,
            theme=None,
        )

        airbornegeo.plotly_points(
            data[data.line.isin(lines_to_level)],
            color_col="levelling_correction",
            hover_cols=["line", data_col, levelled_col],
            cmap="balance",
            absolute=True,
            size=5,
            theme=None,
            robust=True,
        )

    return data.drop(columns=["levelling_correction"]), inters


def iterative_line_levelling(
    data: gpd.GeoDataFrame | pd.DataFrame,
    inters: gpd.GeoDataFrame | pd.DataFrame,
    *,
    lines_to_level: list[float],
    data_col: str,
    levelled_col: str,
    degree: int,
    intersection_weight_col: str | None = None,
    iterations: int = 5,
    plot_results: bool = False,
    plot_convergence: bool = False,
    logy: bool = False,
    title: str = "Levelling convergence",
    as_median: bool = False,
) -> tuple[pd.DataFrame | gpd.GeoDataFrame, pd.DataFrame | gpd.GeoDataFrame]:
    data = data.copy()
    ints = inters.copy()

    assert "line" in data.columns, "df must have column 'line'"

    for _i in tqdm(range(1, iterations + 1), desc="Iteration"):
        try:
            data, ints = line_levelling(
                data,
                ints,
                lines_to_level=lines_to_level,
                degree=degree,
                data_col=data_col,
                levelled_col=levelled_col,
                intersection_weight_col=intersection_weight_col,
                warn_if_unchanged=True,
            )
            data_col = levelled_col
        except UserWarning:
            break
    if plot_convergence is True:
        plot_levelling_convergence(
            ints,
            logy=logy,
            title=title,
            as_median=as_median,
        )
    if plot_results is True:
        # plot flight lines
        airbornegeo.plotly_points(
            data[data.line.isin(lines_to_level)],
            color_col="levelling_correction",
            size=4,
            hover_cols=[
                "line",
            ],
        )

    return data, ints


def alternating_iterative_line_levelling(
    data: gpd.GeoDataFrame | pd.DataFrame,
    inters: gpd.GeoDataFrame | pd.DataFrame,
    *,
    data_col: str,
    levelled_col: str,
    degree: int,
    intersection_weight_col: str | None = None,
    iterations: int = 5,
    # plot_results=False,
    plot_convergence: bool = False,
    logy: bool = False,
    title: str = "Levelling convergence",
    as_median: bool = False,
) -> tuple[pd.DataFrame | gpd.GeoDataFrame, pd.DataFrame | gpd.GeoDataFrame]:
    data = data.copy()
    inters = inters.copy()

    assert "tie" in data.columns, "data must have column 'tie'"
    assert "line" in data.columns, "data must have column 'line'"
    assert "distance_along_line" in data.columns, (
        "data must have column 'distance_along_line'"
    )

    lines_to_level = data[data.tie == False].line.unique()  # noqa: E712 pylint: disable=singleton-comparison
    ties_to_level = data[data.tie == True].line.unique()  # noqa: E712 pylint: disable=singleton-comparison

    for _i in tqdm(range(1, iterations + 1), desc="Iteration"):
        # level lines to ties
        cols = [c for c in inters.columns if "mistie_" in c]
        mistie_col = [int(col.split("_")[-1]) for col in cols]
        try:
            current_mistie_col = f"mistie_{max(mistie_col)}"
            prior_mistie = airbornegeo.rmse(inters[current_mistie_col])
        except ValueError:
            prior_mistie = None

        data, inters = line_levelling(
            data,
            inters,
            lines_to_level=lines_to_level,
            degree=degree,
            data_col=data_col,
            levelled_col=levelled_col,
            intersection_weight_col=intersection_weight_col,
        )
        # level ties to lines
        data, inters = line_levelling(
            data,
            inters,
            lines_to_level=ties_to_level,
            degree=degree,
            data_col=levelled_col,
            levelled_col=levelled_col,
            intersection_weight_col=intersection_weight_col,
        )
        data_col = levelled_col

        cols = [c for c in inters.columns if "mistie_" in c]
        mistie_col = [int(col.split("_")[-1]) for col in cols]
        try:
            current_mistie_col = f"mistie_{max(mistie_col)}"
        except ValueError:
            current_mistie_col = "mistie_0"
        post_mistie = airbornegeo.rmse(inters[current_mistie_col])
        if (prior_mistie is not None) and (post_mistie > prior_mistie):
            logger.warning("Mistie increased, ending iterations")
            break

    if plot_convergence is True:
        plot_levelling_convergence(
            inters,
            logy=logy,
            title=title,
            as_median=as_median,
        )

    return data, inters


def plot_line_and_crosses(
    df: pd.DataFrame | gpd.GeoDataFrame,
    *,
    y: list[str],
    line: float | None = None,
    x: str = "distance_along_line",
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
    assert "line" in df.columns, "df must have column 'line'"

    # turn y column name into list
    if isinstance(y, str):  # type: ignore [unreachable]
        y = [y]  # type: ignore [unreachable]

    # list of y axes to use, if none, all will be same
    if y_axes is None:
        y_axes = ["1" for _ in y]

    if len(df.line.unique()) <= 1:
        line = df.line.iloc[0]

    try:
        line_df = df[df.line == line].sort_values(by=["line", "intersecting_line"])
    except KeyError:
        line_df = df[df.line == line].sort_values(by=["line"])

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
) -> list[float]:
    return [
        i
        for i in data.line.unique()
        if i not in [*intersections.line.unique(), *intersections.tie.unique()]
    ]


def update_intersections_with_eq_sources(
    data: pd.DataFrame | gpd.GeoDataFrame,
    *,
    fitted_equivalent_sources: dict,
    data_column: str,
    groupby_column: str = "line",
) -> pd.Series:
    """
    At each theoretical intersection point, replace the interpolated field value with a
    value predected by the fitted equivalent sources for the line, at the x,y coordinate
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

    assert "line" in data.columns, "data must have column 'line'"

    for segment_name, segment_data in tqdm(
        data.groupby(groupby_column), desc="Segments"
    ):
        # get fitted equivalent sources for this line
        eqs = fitted_equivalent_sources[segment_name]

        # get intersection points for this line
        line_inters = segment_data[segment_data.is_intersection]

        for i, row in line_inters.iterrows():
            # get height of intersection point for the cross line
            cross_inter = data[
                (data.line == row.intersecting_line)
                & (data.intersecting_line == segment_name)
            ]

            assert len(cross_inter) == 1, (
                data[data.intersecting_line == segment_name],
                row.intersecting_line,
                segment_name,
            )

            cross_height = cross_inter.height.to_numpy()[0]
            # assert len(cross_height)==1, f"{cross_height}"
            # assert isinstance(cross_height, (int, float)), f"{cross_inter}"

            coords = (
                np.array([row.distance_along_line]),
                np.array([0]),
                np.array([np.max([cross_height, row.height])]),
            )
            # predict the field value at the x,y coordinate of the intersection point,
            # and the higher of the two lines' elevations, using the supplied fitted
            # equivalent sources for each line
            up_cont_value = eqs.predict(coords)

            # add predicted value to dataframe at intersection point
            data.at[i, data_column] = up_cont_value  # noqa: PD008

    return data[data_column]
