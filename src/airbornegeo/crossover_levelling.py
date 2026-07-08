import geopandas as gpd  # pylint: disable=too-many-lines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import airbornegeo
from airbornegeo import logger
from airbornegeo.utils import _init_iteration_progressbar

sns.set_theme()


def crossover_network_levelling(
    data: gpd.GeoDataFrame | pd.DataFrame,
    inters: gpd.GeoDataFrame | pd.DataFrame,
    *,
    data_col: str,
    levelled_col: str,
    line_column: str,
    distance_column: str,
    degree: int | None = None,
    filter_type: str | None = None,
    lines_to_level: list[float] | None = None,
    intersection_weight_col: str | None = None,
    mistie_interp_method: str = "linear",
    relaxation_factor: float = 0.5,
    warn_if_unchanged: bool = True,
    max_iterations: int = 5,
    rms_tolerance: float | None = None,
    rms_percent_change_tolerance: float = 10,
    rms_percent_increase_tolerance: float = 20,
    plot_convergence: bool = True,
    plot_dynamic_convergence: bool = False,
    progressbar: bool = True,
) -> tuple[pd.DataFrame | gpd.GeoDataFrame, pd.DataFrame | gpd.GeoDataFrame]:
    """
    Level a network of lines by fitting a trend (or low-pass filter) to the cross-over
    errors at every intersection each line participates in, whether it appears as
    `line1` or `line2` in the intersection table (i.e. `method='network'` from
    `create_intersection_table`).

    This differs from `crossover_pair_levelling`, which levels one group of lines onto a
    second, fixed, reference group (`method='pairs'`). In a network, every line can
    intersect many other lines, all of which are simultaneously being adjusted, so a
    line's correction is derived from *all* of its misties (signed so that a positive
    mistie always means "this line is higher than the intersecting line"), and only a
    fraction of the correction (`relaxation_factor`) is removed per iteration —
    splitting the mistie between both intersecting lines rather than pushing it fully
    onto one of them. Repeated iterations (controlled by `max_iterations`, as with
    `crossover_pair_levelling`) let corrections propagate through the network until the
    cross-over misfits converge.

    Parameters
    ----------
    data : gpd.GeoDataFrame | pd.DataFrame
        Survey dataframe with intersection rows added by `add_intersections()` and
        interpolated with `interpolate_intersections()`.
    inters : gpd.GeoDataFrame | pd.DataFrame
        Intersection table created with `create_intersection_table(..., method='network')`.
    data_col : str
        Column containing the values to level.
    levelled_col : str
        Column name to store the levelled values in.
    line_column : str
        Column containing the line / flight / segment names.
    distance_column : str
        Column containing the distance along each line / segment.
    degree : int | None, optional
        Polynomial degree used to fit a trend to the misties along each line.
    filter_type : str | None, optional
        Alternative to `degree`; low-pass filter type applied to the misties along
        each line.
    lines_to_level : list[float] | None
        All lines in the network to be levelled together. By default is all lines.
    intersection_weight_col : str | None, optional
        Column in `inters` with per-intersection weights.
    mistie_interp_method : str, optional
        Method used to fill gaps between misties along a line before filtering (only
        used when `filter_type` is given), by default "linear".
    relaxation_factor : float, optional
        Fraction of each line's fitted mistie trend to remove per iteration, by
        default 0.5 (i.e. split the mistie evenly between the two lines at each
        crossover). Values close to 1 correct faster but are more likely to
        overshoot/oscillate; values closer to 0 converge more slowly but more
        stably.
    warn_if_unchanged : bool, optional
        Raise a UserWarning if misties are unchanged from the previous iteration, by
        default True.
    max_iterations : int, optional
        Maximum number of iterations, by default 5. Network levelling generally needs
        more iterations than pairs levelling since corrections must propagate through
        the network.
    rms_tolerance, rms_percent_change_tolerance, rms_percent_increase_tolerance : float, optional
        Convergence criteria, as in `crossover_pair_levelling`.
    plot_convergence, plot_dynamic_convergence : bool, optional
        Plot convergence of the levelling corrections.
    progressbar : bool, optional
        Show a progress bar over iterations.

    Returns
    -------
    tuple[pd.DataFrame | gpd.GeoDataFrame, pd.DataFrame | gpd.GeoDataFrame]
        The levelled dataframe and updated intersections table.
    """
    data = data.copy()
    inters = inters.copy()

    progressbar, pbar_iterations = _init_iteration_progressbar(
        max_iterations, progressbar
    )

    if plot_dynamic_convergence:
        monitor = airbornegeo.plotting.LevellingConvergenceMonitor(
            rms_tolerance=rms_tolerance,
            rms_percent_change_tolerance=rms_percent_change_tolerance,
        )

    if lines_to_level is None:
        lines_to_level = data[line_column].unique()

    correction_rms_values = []
    correction_delta_rms_values = []
    mistie_values = []
    iteration = 1
    for iteration in pbar_iterations:
        if progressbar:
            pbar_iterations.set_description(f"Iteration: {iteration}")

        try:
            original_values = data[data_col]
            data, inters = _crossover_network_levelling(
                data,
                inters,
                degree=degree,
                filter_type=filter_type,
                data_col=data_col,
                levelled_col=levelled_col,
                line_column=line_column,
                distance_column=distance_column,
                lines_to_level=lines_to_level,
                intersection_weight_col=intersection_weight_col,
                mistie_interp_method=mistie_interp_method,
                relaxation_factor=relaxation_factor,
                warn_if_unchanged=warn_if_unchanged,
            )
            final_values = data[levelled_col]
        except UserWarning:
            break

        cols = [
            c for c in inters.columns if "mistie_" in c and c != intersection_weight_col
        ]
        mistie_col = [int(col.split("_")[-1]) for col in cols]
        try:
            current_mistie_col = f"mistie_{max(mistie_col)}"
        except ValueError:
            current_mistie_col = "mistie_0"
        mistie_values.append(airbornegeo.rmse(inters[current_mistie_col]))

        levelling_correction = original_values - final_values
        rms = airbornegeo.rmse(levelling_correction)
        delta_rms = (
            (correction_rms_values[-1] / rms - 1) * 100 if iteration > 1 else np.inf
        )
        correction_rms_values.append(rms)
        correction_delta_rms_values.append(delta_rms)

        end, termination_reason = _end_iterations(
            rms_values=correction_rms_values,
            delta_rms_values=correction_delta_rms_values,
            max_iterations=max_iterations,
            mistie_values=mistie_values,
            rms_tolerance=rms_tolerance,
            rms_percent_change_tolerance=rms_percent_change_tolerance,
            rms_percent_increase_tolerance=rms_percent_increase_tolerance,
        )

        if plot_dynamic_convergence:
            monitor.update(correction_rms_values, correction_delta_rms_values)

        if end:
            if progressbar:
                pbar_iterations.set_description(
                    f"Iterations ended due to {termination_reason}"
                )
            break

        # use levelled column as input for next iteration
        data_col = levelled_col

    if plot_convergence and iteration > 2 and not plot_dynamic_convergence:
        airbornegeo.plotting.plot_levelling_convergence(
            rms_values=correction_rms_values,
            delta_rms_values=correction_delta_rms_values,
            rms_tolerance=rms_tolerance,
            rms_percent_change_tolerance=rms_percent_change_tolerance,
        )

    return data, inters


def _line_network_misties(
    inters: pd.DataFrame | gpd.GeoDataFrame,
    line: float,
    mistie_col: str,
    intersection_weight_col: str | None = None,
) -> pd.DataFrame:
    """
    Build a table of (distance_along_line, signed mistie[, weight]) for a single line,
    combining the intersections where it appears as `line1` and as `line2`. Misties
    are flipped in sign for the `line2` case so that every value in the returned
    'network_mistie' column consistently represents (this_line's value - the
    intersecting line's value), matching the convention used by
    `calculate_crossover_errors` (mistie = line1_value - line2_value).
    """
    cols_out = ["dist_along_line", "network_mistie"]
    if intersection_weight_col is not None:
        cols_out.append(intersection_weight_col)

    as_line1 = inters[inters.line1 == line].copy()
    as_line1["dist_along_line"] = as_line1["dist_along_line1"]
    as_line1["network_mistie"] = as_line1[mistie_col]

    as_line2 = inters[inters.line2 == line].copy()
    as_line2["dist_along_line"] = as_line2["dist_along_line2"]
    as_line2["network_mistie"] = -as_line2[mistie_col]

    return pd.concat(
        [as_line1[cols_out], as_line2[cols_out]],
        ignore_index=True,
    )


def _crossover_network_levelling(
    data: gpd.GeoDataFrame | pd.DataFrame,
    inters: gpd.GeoDataFrame | pd.DataFrame,
    *,
    data_col: str,
    levelled_col: str,
    line_column: str,
    distance_column: str,
    degree: int | None = None,
    filter_type: str | None = None,
    lines_to_level: list[float] | None = None,
    intersection_weight_col: str | None = None,
    mistie_interp_method: str = "linear",
    relaxation_factor: float = 0.5,
    warn_if_unchanged: bool = False,
) -> tuple[pd.DataFrame | gpd.GeoDataFrame, pd.DataFrame | gpd.GeoDataFrame]:
    """
    Single iteration of network levelling: fit a trend / filter to the (signed)
    cross-over misties of every line against all lines it intersects, and remove a
    damped fraction (`relaxation_factor`) of that trend from `data_col`.
    """
    data = data.copy()
    inters = inters.copy()

    if lines_to_level is None:
        lines_to_level = data[line_column].unique()

    if (filter_type is not None) and (degree is not None):
        msg = "only provide either `filter_type` or `degree`, not both"
        raise UserWarning(msg)
    if (filter_type is None) and (degree is None):
        msg = "must provide either `filter_type` or `degree`"
        raise UserWarning(msg)

    assert line_column in data.columns, (
        "data must have column specified by `line_column`"
    )
    assert distance_column in data.columns, (
        "data must have column denote by distance_column"
    )

    # drop lines without intersections
    lines_without_inters = airbornegeo.lines_without_intersections(
        data,
        inters,
        line_column=line_column,
    )
    lines_to_level = [x for x in lines_to_level if x not in lines_without_inters]

    data["levelling_correction"] = np.nan

    # get the latest mistie column, calculated from current data_col
    inters2 = airbornegeo.calculate_crossover_errors(
        data,
        inters,
        data_col=data_col,
        line_column=line_column,
    )

    mistie_col = [
        c for c in inters2.columns if "mistie_" in c and c != intersection_weight_col
    ]
    mistie_col = [int(col.split("_")[-1]) for col in mistie_col]
    mistie_col = f"mistie_{max(mistie_col)}"

    logger.debug(
        "network mistie before levelling: %s", airbornegeo.rmse(inters2[mistie_col])
    )

    # fit a trend/filter to each line's (signed) misties, independently, using the
    # mistie table computed above (i.e. Jacobi-style: all lines corrected relative to
    # the same starting state, not sequentially against each other's updated values)
    for line in lines_to_level:
        line_df = data[data[line_column] == line].copy()

        line_ints = _line_network_misties(
            inters2, line, mistie_col, intersection_weight_col
        )

        if degree is not None:
            try:
                line_df = airbornegeo.trend(
                    data_to_fit=line_ints,
                    cols_to_fit=["dist_along_line", "network_mistie"],
                    data_to_predict=line_df,
                    cols_to_predict=[distance_column, "levelling_correction"],
                    degree=degree,
                    intersection_weight_col=intersection_weight_col,
                )
            except (ValueError, UserWarning) as e:
                if isinstance(e, UserWarning) or "Found array with " in str(e):
                    logger.error("Issue with line %s, skipping", line)
                    line_df["levelling_correction"] = 0
                else:
                    raise ValueError from e

        if filter_type is not None:
            # add signed misties to line dataframe at each intersection row
            for ind, row in line_df[line_df.is_intersection].iterrows():
                match = line_ints[
                    np.isclose(line_ints.dist_along_line, row[distance_column])
                ]
                if len(match) == 0:
                    continue
                line_df.loc[ind, mistie_col] = match["network_mistie"].to_numpy()[0]

            n_valid_misties = line_df[mistie_col].notna().sum()
            if n_valid_misties == 0:
                # line was flagged as having intersections, but none of them could be
                # matched to a mistie value (e.g. distance mismatch between `data` and
                # `inters`) -- nothing to interpolate or filter, so leave uncorrected
                logger.warning(
                    "Line %s has intersections but no mistie values could be matched; "
                    "skipping levelling correction for this line (set to 0)",
                    line,
                )
                line_df["levelling_correction"] = 0
            else:
                # `interpolate_missing` falls back internally (mistie_interp_method ->
                # linear -> nearest) based on how many misties this line has
                line_df = airbornegeo.interpolate_missing(
                    line_df,
                    to_interp=mistie_col,
                    interp_on=distance_column,
                    method=mistie_interp_method,
                    extrapolate=False,
                    groupby_column=None,
                )
                line_df = airbornegeo.interpolate_missing(
                    line_df,
                    to_interp=mistie_col,
                    interp_on=distance_column,
                    method="nearest",
                    extrapolate=True,
                    groupby_column=None,
                )

                line_df["levelling_correction"] = airbornegeo.filter_line(
                    line_df,
                    filter_type=filter_type,
                    data_column=mistie_col,
                    filter_by_column=distance_column,
                    groupby_column=None,
                    progressbar=False,
                )

        # damp the correction so both lines at each crossover move only part-way
        # towards each other, rather than one fully absorbing the other's mistie
        line_df["levelling_correction"] *= relaxation_factor

        values = line_df[data_col] - line_df.levelling_correction

        data.loc[data[line_column] == line, levelled_col] = values
        data.loc[data[line_column] == line, "levelling_correction"] = (
            line_df.levelling_correction
        )

    # unchanged values for lines not included
    for line in data[line_column].unique():
        if line not in lines_to_level:
            data.loc[data[line_column] == line, levelled_col] = data.loc[
                data[line_column] == line, data_col
            ]

    inters = airbornegeo.calculate_crossover_errors(
        data,
        inters,
        data_col=levelled_col,
        line_column=line_column,
        warn_if_unchanged=warn_if_unchanged,
    )
    mistie_col = [
        int(col.split("_")[-1])
        for col in inters.columns
        if "mistie_" in col and col != intersection_weight_col
    ]
    mistie_col = f"mistie_{max(mistie_col)}"

    logger.debug(
        "network mistie after levelling: %s", airbornegeo.rmse(inters[mistie_col])
    )

    return data.drop(columns=["levelling_correction"]), inters


def _end_iterations(
    rms_values: list[float],
    delta_rms_values: list[float],
    max_iterations: int,
    mistie_values: list[float] | None = None,
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

    if mistie_values is not None:
        mistie = mistie_values[-1]
        previous_mistie = mistie_values[-2] if iteration > 2 else np.inf

    # ignore for first iteration
    if iteration == 1:
        pass
    else:
        # end because RMS is increasing above a unreasonable amount
        if rms > np.min(rms_values) * (1 + rms_percent_increase_tolerance / 100):
            logger.info(  # pylint: disable=logging-fstring-interpolation
                f"\nLevelling terminated after {iteration} iterations because the RMS of the levelling corrections ({round(rms, 4)}) \n"
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
                f"\nLevelling terminated after {iteration} iterations because there was no "
                f"significant variation in the RMS (delta RMS of {round(delta_rms, 2)}%) of the levelling corrections over 2 iterations \n"
                f"Change parameter 'rms_percent_change_tolerance' ({rms_percent_change_tolerance}%) if desired.",
            )
            end = True
            termination_reason.append("RMS percent change tolerance")
        # end because RMS is below the set tolerance
        if (rms_tolerance is not None) and (rms < rms_tolerance):
            logger.info(  # pylint: disable=logging-fstring-interpolation
                f"\nLevelling terminated after {iteration} iterations because the RMS of the levelling corrections ({rms}) was "
                f"less then set tolerance ({rms_tolerance}) \nChange parameter "
                "'rms_tolerance' if desired.",
            )
            end = True
            termination_reason.append("RMS tolerance")
        # end because RMS of cross-overs is increasing
        if (mistie_values is not None) and (mistie > previous_mistie):  # pylint: disable=possibly-used-before-assignment
            logger.info(  # pylint: disable=logging-fstring-interpolation
                f"\nLevelling terminated after {iteration} iterations because the RMS of the cross-over errors ({rms}) "
                f"began to increase.",
            )
            end = True
            termination_reason.append("cross-over RMS increasing")

    # end because max iterations reached
    if iteration >= max_iterations:
        if max_iterations > 1:
            logger.warning(  # pylint: disable=logging-fstring-interpolation
                f"\nLevelling terminated after {iteration} iterations with RMS of levelling correction of {round(rms, 2)} because "
                f"maximum number of iterations ({max_iterations}) reached.",
            )

        end = True
        termination_reason.append("max iterations")

    return end, termination_reason


def plot_levelling_convergence(
    results: gpd.GeoDataFrame | pd.DataFrame,
    *,
    logy: bool = False,
    title: str = "Levelling convergence",
    as_median: bool = False,
) -> None:
    # get mistie columns
    cols = [c for c in results.columns if "mistie_" in c]
    cols = [col.split("_")[-1] for col in cols]
    mistie_cols = []
    for c in cols:
        try:  # noqa: SIM105
            mistie_cols.append(int(c))
        except ValueError:
            pass
    cols = [f"mistie_{c}" for c in mistie_cols]
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


def crossover_pair_levelling(
    data: gpd.GeoDataFrame | pd.DataFrame,
    inters: gpd.GeoDataFrame | pd.DataFrame,
    *,
    lines_to_level: list[float],
    data_col: str,
    levelled_col: str,
    line_column: str,
    distance_column: str,
    degree: int | None = None,
    filter_type: str | None = None,
    intersection_weight_col: str | None = None,
    mistie_interp_method: str = "linear",
    warn_if_unchanged: bool = True,
    max_iterations: int = 1,
    rms_tolerance: float | None = None,
    rms_percent_change_tolerance: float = 10,
    rms_percent_increase_tolerance: float = 20,
    plot_convergence: bool = True,
    plot_dynamic_convergence: bool = False,
    progressbar: bool = True,
) -> tuple[pd.DataFrame | gpd.GeoDataFrame, pd.DataFrame | gpd.GeoDataFrame]:
    """
    Level lines by fitting a trend of specified order to cross-over errors and apply
    the correction to the `data_col` column.
    """
    data = data.copy()
    inters = inters.copy()

    progressbar, pbar_iterations = _init_iteration_progressbar(
        max_iterations, progressbar
    )

    if plot_dynamic_convergence:
        # initialize figure
        monitor = airbornegeo.plotting.LevellingConvergenceMonitor(
            rms_tolerance=rms_tolerance,
            rms_percent_change_tolerance=rms_percent_change_tolerance,
        )

    correction_rms_values = []
    correction_delta_rms_values = []
    mistie_values = []
    iteration = 1
    for iteration in pbar_iterations:
        if progressbar:
            pbar_iterations.set_description(f"Iteration: {iteration}")

        try:
            original_values = data[data_col]
            data, inters = _crossover_pair_levelling(
                data,
                inters,
                lines_to_level=lines_to_level,
                degree=degree,
                filter_type=filter_type,
                data_col=data_col,
                levelled_col=levelled_col,
                line_column=line_column,
                distance_column=distance_column,
                intersection_weight_col=intersection_weight_col,
                mistie_interp_method=mistie_interp_method,
                warn_if_unchanged=warn_if_unchanged,
            )
            final_values = data[levelled_col]
        except UserWarning:
            break

        cols = [
            c for c in inters.columns if "mistie_" in c and c != intersection_weight_col
        ]
        mistie_col = [int(col.split("_")[-1]) for col in cols]
        try:
            current_mistie_col = f"mistie_{max(mistie_col)}"
        except ValueError:
            current_mistie_col = "mistie_0"
        mistie_values.append(airbornegeo.rmse(inters[current_mistie_col]))

        # add RMS and delta RMS of correction values for iteration to lists
        levelling_correction = original_values - final_values
        rms = airbornegeo.rmse(levelling_correction)
        delta_rms = (
            (correction_rms_values[-1] / rms - 1) * 100 if iteration > 1 else np.inf
        )
        correction_rms_values.append(rms)
        correction_delta_rms_values.append(delta_rms)

        end, termination_reason = _end_iterations(
            rms_values=correction_rms_values,
            delta_rms_values=correction_delta_rms_values,
            max_iterations=max_iterations,
            mistie_values=mistie_values,
            rms_tolerance=rms_tolerance,
            rms_percent_change_tolerance=rms_percent_change_tolerance,
            rms_percent_increase_tolerance=rms_percent_increase_tolerance,
        )

        if plot_dynamic_convergence:
            monitor.update(correction_rms_values, correction_delta_rms_values)

        if end:
            if progressbar:
                pbar_iterations.set_description(
                    f"Iterations ended due to {termination_reason}"
                )
            break

        # use levelled column as input for next iteration
        data_col = levelled_col

    if plot_convergence and iteration > 2 and not plot_dynamic_convergence:
        airbornegeo.plotting.plot_levelling_convergence(
            rms_values=correction_rms_values,
            delta_rms_values=correction_delta_rms_values,
            rms_tolerance=rms_tolerance,
            rms_percent_change_tolerance=rms_percent_change_tolerance,
        )

    return data, inters


def _crossover_pair_levelling(
    data: gpd.GeoDataFrame | pd.DataFrame,
    inters: gpd.GeoDataFrame | pd.DataFrame,
    *,
    lines_to_level: list[float],
    data_col: str,
    levelled_col: str,
    line_column: str,
    distance_column: str,
    degree: int | None = None,
    filter_type: str | None = None,
    intersection_weight_col: str | None = None,
    mistie_interp_method: str = "linear",
    warn_if_unchanged: bool = False,
) -> tuple[pd.DataFrame | gpd.GeoDataFrame, pd.DataFrame | gpd.GeoDataFrame]:
    """
    Level lines by fitting a trend of specified order to cross-over errors and apply
    the correction to the `data_col` column.
    """
    data = data.copy()
    inters = inters.copy()

    if (filter_type is not None) and (degree is not None):
        msg = "only provide either `filter_type` or `degree`, not both"
        raise UserWarning(msg)
    if (filter_type is None) and (degree is None):
        msg = "must provide either `filter_type` or `degree`"
        raise UserWarning(msg)

    # drop lines without intersections
    lines_without_inters = airbornegeo.lines_without_intersections(
        data,
        inters,
        line_column=line_column,
    )
    lines_to_level = [x for x in lines_to_level if x not in lines_without_inters]

    assert line_column in data.columns, (
        "data must have column specified by `line_column`"
    )
    assert distance_column in data.columns, (
        "data must have column denote by distance_column"
    )
    # check if levelling lines to ties or vice versa
    levelling_lines2 = False
    levelling_lines1 = False
    for j in lines_to_level:
        if j in inters.line2.unique():
            levelling_lines2 = True
        if j in inters.line1.unique():
            levelling_lines1 = True
    if (levelling_lines2 is True) & (levelling_lines1 is True):
        msg = "Supplied two types of lines to be levelled!"
        raise ValueError(msg)
    if levelling_lines1 is True:
        logger.debug("Levelling line type 0 to line type 1")
    elif levelling_lines2 is True:
        logger.debug("Levelling line type 1 to line type 0")

    if levelling_lines2 is True:
        cols_to_fit = ["dist_along_line2"]
    elif levelling_lines1 is True:
        cols_to_fit = ["dist_along_line1"]
    else:
        msg = "need to supplied either lines of type 0 or 1"
        raise ValueError(msg)

    data["levelling_correction"] = np.nan

    # get the latest mistie column
    inters2 = airbornegeo.calculate_crossover_errors(
        data,
        inters,
        data_col=data_col,
        line_column=line_column,
    )

    mistie_col = [
        c for c in inters2.columns if "mistie_" in c and c != intersection_weight_col
    ]
    mistie_col = [int(col.split("_")[-1]) for col in mistie_col]
    mistie_col = f"mistie_{max(mistie_col)}"

    logger.debug(
        "mistie before levelling: %s mGal", airbornegeo.rmse(inters2[mistie_col])
    )

    # fit a trend to the misfits on line-by-line basis
    # iterate through the chosen lines
    for line in lines_to_level:
        # subset a line
        line_df = data[data[line_column] == line].copy()

        # get intersections of line of interest
        ints = inters2[(inters2.line1 == line) | (inters2.line2 == line)]

        if degree is not None:
            try:
                line_df = airbornegeo.trend(
                    data_to_fit=ints,  # data with mistie values
                    cols_to_fit=cols_to_fit  # noqa: RUF005
                    + [mistie_col],  # column names for distance/mistie
                    data_to_predict=line_df,  # data with line data
                    cols_to_predict=[distance_column]  # noqa: RUF005
                    + [
                        "levelling_correction"
                    ],  # column names for distance/ levelling correction
                    degree=degree,  # degree order for fitting line to misties
                    intersection_weight_col=intersection_weight_col,
                )
            except (ValueError, UserWarning) as e:
                if isinstance(e, UserWarning) or "Found array with " in str(e):
                    logger.error("Issue with line %s, skipping", line)
                    line_df["levelling_correction"] = 0
                else:
                    raise ValueError from e

        if filter_type is not None:
            # add misties to line dataframe from intersections dataframe
            for ind, row in line_df[line_df.is_intersection].iterrows():
                # search intersections for mistie values
                mistie_row = ints[
                    ((ints.line1 == line) & (ints.line2 == row.intersecting_line))
                    | ((ints.line1 == row.intersecting_line) & (ints.line2 == line))
                ]
                assert len(mistie_row) >= 1

                # add misties to line dataframe
                line_df.loc[ind, mistie_col] = mistie_row[mistie_col].to_numpy()

            n_valid_misties = line_df[mistie_col].notna().sum()
            if n_valid_misties == 0:
                # line was flagged as having intersections, but none of them could be
                # matched to a mistie value (e.g. distance mismatch between `data` and
                # `inters`) -- nothing to interpolate or filter, so leave uncorrected
                logger.warning(
                    "Line %s has intersections but no mistie values could be matched; "
                    "skipping levelling correction for this line (set to 0)",
                    line,
                )
                line_df["levelling_correction"] = 0
            else:
                # interpolate mistie NaNs along the line. `interpolate_missing` falls
                # back internally (mistie_interp_method -> linear -> nearest) based on how
                # many mistie values this line actually has
                line_df = airbornegeo.interpolate_missing(
                    line_df,
                    to_interp=mistie_col,
                    interp_on=distance_column,
                    method=mistie_interp_method,
                    extrapolate=False,
                    groupby_column=None,
                )
                line_df = airbornegeo.interpolate_missing(
                    line_df,
                    to_interp=mistie_col,
                    interp_on=distance_column,
                    method="nearest",
                    extrapolate=True,
                    groupby_column=None,
                )

                # calculate levelling correction by low pass filtering the misfits values
                line_df["levelling_correction"] = airbornegeo.filter_line(
                    line_df,
                    filter_type=filter_type,
                    data_column=mistie_col,
                    filter_by_column=distance_column,
                    groupby_column=None,  # already giving a single group
                    progressbar=False,
                )

        # if levelling tie lines, negate the correction
        if levelling_lines2 is True:
            line_df["levelling_correction"] *= -1
        else:
            pass

        # remove the levelling correction from the gravity
        values = line_df[data_col] - line_df.levelling_correction

        # update main data
        data.loc[data[line_column] == line, levelled_col] = values
        data.loc[data[line_column] == line, "levelling_correction"] = (
            line_df.levelling_correction
        )

    # add unchanged values for lines not included
    for line in data[line_column].unique():
        if line not in lines_to_level:
            data.loc[data[line_column] == line, levelled_col] = data.loc[
                data[line_column] == line, data_col
            ]

    # update mistie with levelled data
    inters = airbornegeo.calculate_crossover_errors(
        data,
        inters,
        data_col=levelled_col,
        line_column=line_column,
        warn_if_unchanged=warn_if_unchanged,
    )
    mistie_col = [
        int(col.split("_")[-1])
        for col in inters.columns
        if "mistie_" in col and col != intersection_weight_col
    ]
    mistie_col = f"mistie_{max(mistie_col)}"

    logger.debug(
        "mistie after levelling: %s mGal", airbornegeo.rmse(inters[mistie_col])
    )

    return data.drop(columns=["levelling_correction"]), inters


def alternating_iterative_line_levelling(
    data: gpd.GeoDataFrame | pd.DataFrame,
    inters: gpd.GeoDataFrame | pd.DataFrame,
    *,
    data_col: str,
    levelled_col: str,
    line_column: str,
    distance_column: str,
    lines_to_level: list[str] | None = None,
    degree: int | None = None,
    filter_type: str | None = None,
    intersection_weight_col: str | None = None,
    mistie_interp_method: str = "linear",
    max_iterations: int = 5,
    rms_tolerance: float | None = None,
    rms_percent_change_tolerance: float = 10,
    rms_percent_increase_tolerance: float = 20,
    plot_convergence: bool = True,
    plot_dynamic_convergence: bool = False,
    progressbar: bool = True,
) -> tuple[pd.DataFrame | gpd.GeoDataFrame, pd.DataFrame | gpd.GeoDataFrame]:
    data = data.copy()
    inters = inters.copy()

    if (filter_type is not None) and (degree is not None):
        msg = "only provide either `filter_type` or `degree`, not both"
        raise UserWarning(msg)
    if (filter_type is None) and (degree is None):
        msg = "must provide either `filter_type` or `degree`"
        raise UserWarning(msg)

    # check columns are present
    cols = [line_column, "line_type", distance_column, data_col]
    assert all(col in data.columns for col in cols), f"{cols} must be in the dataframe"

    lines1_to_level = data[data.line_type == 0][line_column].unique()
    lines2_to_level = data[data.line_type == 1][line_column].unique()

    if lines_to_level is not None:
        lines1_to_level = [x for x in lines1_to_level if x in lines_to_level]
        lines2_to_level = [x for x in lines2_to_level if x in lines_to_level]

    progressbar, pbar_iterations = _init_iteration_progressbar(
        max_iterations, progressbar
    )

    if plot_dynamic_convergence:
        # initialize figure
        monitor = airbornegeo.plotting.LevellingConvergenceMonitor(
            rms_tolerance=rms_tolerance,
            rms_percent_change_tolerance=rms_percent_change_tolerance,
        )

    correction_rms_values = []
    correction_delta_rms_values = []
    mistie_values = []
    iteration = 1
    for iteration in pbar_iterations:
        if progressbar:
            pbar_iterations.set_description(f"Iteration: {iteration}")

        # level lines1 to lines2
        original_values = data[data_col]
        data, inters = crossover_pair_levelling(
            data,
            inters,
            lines_to_level=lines1_to_level,
            degree=degree,
            filter_type=filter_type,
            data_col=data_col,
            line_column=line_column,
            levelled_col=levelled_col,
            distance_column=distance_column,
            intersection_weight_col=intersection_weight_col,
            mistie_interp_method=mistie_interp_method,
            max_iterations=1,
        )
        # level lines2 to lines1
        data, inters = crossover_pair_levelling(
            data,
            inters,
            lines_to_level=lines2_to_level,
            degree=degree,
            filter_type=filter_type,
            data_col=levelled_col,
            line_column=line_column,
            levelled_col=levelled_col,
            distance_column=distance_column,
            intersection_weight_col=intersection_weight_col,
            mistie_interp_method=mistie_interp_method,
            max_iterations=1,
        )
        final_values = data[levelled_col]

        cols = [
            c for c in inters.columns if "mistie_" in c and c != intersection_weight_col
        ]
        mistie_col = [int(col.split("_")[-1]) for col in cols]
        try:
            current_mistie_col = f"mistie_{max(mistie_col)}"
        except ValueError:
            current_mistie_col = "mistie_0"
        mistie_values.append(airbornegeo.rmse(inters[current_mistie_col]))

        # add RMS and delta RMS of correction values for iteration to lists
        levelling_correction = original_values - final_values
        rms = airbornegeo.rmse(levelling_correction)
        delta_rms = (
            (correction_rms_values[-1] / rms - 1) * 100 if iteration > 1 else np.inf
        )
        correction_rms_values.append(rms)
        correction_delta_rms_values.append(delta_rms)
        # print(f"\t{rms=}")
        # print(f"\t{delta_rms=}")
        end, termination_reason = _end_iterations(
            rms_values=correction_rms_values,
            delta_rms_values=correction_delta_rms_values,
            max_iterations=max_iterations,
            mistie_values=mistie_values,
            rms_tolerance=rms_tolerance,
            rms_percent_change_tolerance=rms_percent_change_tolerance,
            rms_percent_increase_tolerance=rms_percent_increase_tolerance,
        )

        if plot_dynamic_convergence:
            monitor.update(correction_rms_values, correction_delta_rms_values)

        if end:
            if progressbar:
                pbar_iterations.set_description(
                    f"Iterations ended due to {termination_reason}"
                )
            break

        # use levelled column as input for next iteration
        data_col = levelled_col

    if plot_convergence and iteration > 2 and not plot_dynamic_convergence:
        airbornegeo.plotting.plot_levelling_convergence(
            rms_values=correction_rms_values,
            delta_rms_values=correction_delta_rms_values,
            rms_tolerance=rms_tolerance,
            rms_percent_change_tolerance=rms_percent_change_tolerance,
        )

    return data, inters


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
    height_1st_derive_weight: float | None = None,
    height_1st_derive_floor: float | None = None,
    height_1st_derive_col_name: str | None = None,
    height_col_name: str | None = None,
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
        if height_col_name is None:
            msg = "must provide 'height_col_name'"
            raise ValueError(msg)
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
            inters.loc[ind, "data_1st_derive"] = np.max(np.abs([line_value, tie_value]))
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
            inters.loc[ind, "height_1st_derive"] = np.max(
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
