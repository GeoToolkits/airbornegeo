import copy

import harmonica as hm
import numpy as np
import pandas as pd
import seaborn as sns
import verde as vd
from tqdm.autonotebook import tqdm

import airbornegeo
from airbornegeo.utils import _check_coord_columns, _init_iteration_progressbar

sns.set_theme()


def equivalent_source_levelling(
    data: pd.DataFrame,
    *,
    data_column: str,
    distance_column: str,
    max_dist: float,
    degree: int,
    lines_to_level: list[float] | None = None,
    damping: float | None = None,
    depth: str | float = "default",
    data_block_size: float | None = None,
    source_block_size: float | None = None,
    max_iterations: int = 1,
    rms_tolerance: float | None = None,
    rms_percent_change_tolerance: float = 10,
    rms_percent_increase_tolerance: float = 20,
    seed: int = 42,
    plot_levelling_convergence: bool = True,
    progressbar: bool = True,
) -> pd.Series:
    """
    Iteratively levelling lines by comparing the line to the forward calculated effect
    of equivalent sources which have been fitted to nearby data from other lines. The
    difference between the prediction and the observed data gives a misfit, which a
    trend is then fit to, creating the levelling correction, which is subsequently added
    to the line. Randomly iterate through all the lines, creating new equivalent source
    models and calculating levelling corrections for each line. Then repeat this entire
    process several times, until one of several stopping criteria are met.

    Parameters
    ----------
    data : pd.DataFrame
        The dataframe containing columns 'easting', 'northing', 'height', 'line',
        'distance_along_line', and the data, specified by parameter `data_column`.
    data_column : str
        The name of the column containing the data values to fit equivalent sources to
        and level, typically gravity or magnetics.
    distance_column : str
        The column containing the distance along the line in meters.
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
        4.5 times the median distance between first neighboring sources.
    data_block_size : float | None, optional
        The block size for reducing the number of data points used in fitting the
        equivalent sources, by default None. If given, data will be block reduced along
        individual lines.
    source_block_size : float | None, optional
        The block size for placing the equivalent sources, by default None which places
        1 source beneath each (blocked) datapoint. If given, will instead place 1 source
        in each window of line data with a width of the block size.
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
    plot_levelling_convergence : bool, optional
        Plot the convergence of levelling correction RMS value, by default True
    progressbar : bool, optional
        Show progress bars for both iterations and levelling of lines, by default True

    Returns
    -------
    pd.Series
        The levelled data column, which can be assigned back to the original dataframe.
    """
    # check columns are present
    _check_coord_columns(data)
    cols = ["height", "line", distance_column, data_column]

    assert all(col in data.columns for col in cols), f"{cols} must be in the dataframe"

    data = data.copy()

    # save index and reset
    data = data.reset_index(names="tmp_index").reset_index(drop=True)

    if lines_to_level is not None:
        line_list = copy.deepcopy(lines_to_level)
    else:
        line_list = data.line.unique()

    progressbar, pbar_iterations = _init_iteration_progressbar(
        max_iterations, progressbar
    )

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
            nearby_survey_df = survey_df.iloc[dist_mask]

            if len(nearby_survey_df) == 0:
                continue

            # block reduce data used for fitting eq sources
            if data_block_size is not None:
                nearby_survey_df = airbornegeo.block_reduce(
                    nearby_survey_df,
                    np.median,
                    reduce_by=distance_column,
                    spacing=data_block_size,
                    groupby_column="line",
                    progressbar=False,
                )

            # fit eq sources to nearby data
            coords = (
                nearby_survey_df.easting,
                nearby_survey_df.northing,
                nearby_survey_df.height,
            )
            # # if block-reducing for sources, do it manually so sources are along lines
            # if source_block_size is not None:
            #     blocked_nearby_survey = airbornegeo.block_reduce(
            #         nearby_survey_df,
            #         np.median,
            #         spacing=source_block_size,
            #         reduce_by=distance_column,
            #         groupby_column='line',
            #         progressbar=False,
            #     )
            #     blocked_nearby_survey = airbornegeo.block_reduce(
            #         blocked_nearby_survey,
            #         np.median,
            #         spacing=source_block_size,
            #         reduce_by=('easting','northing'),
            #         progressbar=False,
            #     )
            #     if depth == "default":
            #         source_depth = 4.5 * np.mean(
            #         bd.neighbor_distance_statistics(
            #             (blocked_nearby_survey.easting, blocked_nearby_survey.northing),
            #             "median",
            #             k=1,
            #         )
            #     )
            #     else:
            #         source_depth = depth
            #     points = (
            #         blocked_nearby_survey.easting,
            #         blocked_nearby_survey.northing,
            #         blocked_nearby_survey.height - source_depth,
            #     )
            # else:
            #     points = None

            eqs = hm.EquivalentSources(
                damping=damping,
                depth=depth,
                # points=points,
                block_size=source_block_size,
            )
            eqs.fit(coords, nearby_survey_df[data_column])

            # predict eq sources on the line to be levelled
            line_df["tmp_predicted_eqs"] = eqs.predict(
                (line_df.easting, line_df.northing, line_df.height)
            )

            line_df["tmp_misfit"] = line_df.tmp_predicted_eqs - line_df[data_column]

            # calculate levelling correction with a trend fit to the misfit values
            line_df = airbornegeo.trend(
                data_to_fit=line_df,
                cols_to_fit=["distance_along_line", "tmp_misfit"],
                data_to_predict=line_df,
                cols_to_predict=[
                    "distance_along_line",
                    f"tmp_levelling_correction_{iteration}",
                ],
                degree=degree,
            )

            # TODO: make levelling correction 0 if more than max_dist from other line data

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

        end, termination_reason = airbornegeo.crossover_levelling._end_iterations(  # pylint: disable=protected-access
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

    if plot_levelling_convergence and iteration > 2:
        airbornegeo.plotting.plot_levelling_convergence(
            rms_values=correction_rms_values,
            delta_rms_values=correction_delta_rms_values,
            rms_tolerance=rms_tolerance,
            rms_percent_change_tolerance=rms_percent_change_tolerance,
        )

    return data[data_column]
