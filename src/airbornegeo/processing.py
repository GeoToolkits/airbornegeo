import typing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from airbornegeo import logger
from airbornegeo.nav import turn_rate

sns.set_theme()


def split_into_segments(
    data: pd.DataFrame,
    threshold: float,
    column_name: str,
    min_points_per_segment: int = 0,
    angular_difference: bool = False,
    segment_column: str | None = None,
    smoothing_window: int | None = None,
    absolute: bool = False,
) -> pd.Series:
    """
    Split dataframe into segments where there is a gap in the supplied values greater
    than the threshold. Data are sorted by the column given by column_name which can be
    quantities such as time in seconds, cumulative distances, or aircraft heading or
    track. By default, a series of sequential segments names as integers starting at 1
    will be returned. Alternatively, existing segments can be divided into new segments,
    via the column given by segment_column. The first sub-segment keeps the existing
    name and later ones have a suffix added (i.e., 210 -> 210 and 210_1) or the
    suffix will be incremented (210_1 -> 210_1 and 210_2). Names are returned as
    strings.

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
    angular_difference : bool
        if True, will interpret the data as degrees between 0 and 360, so the calculated
        differences are the shorted, for example the difference between 0 and 350
        degrees will be -10 instead of 340.
    segment_column: str or None
        Column of existing segment or line names to split within, by default None
    smoothing_window : int, optional
        Window size in number of data points for smoothing each difference after it's
        been calculated, by default None. Note this averages the SIGNED difference, so
        it suppresses noise on an isolated break but cancels a change spread over many
        points. It is not a way to detect a sustained change such as a turn: use
        :func:`airbornegeo.split_at_turns` for that.
    absolute : bool
        if True, compare the magnitude of each difference against the threshold, so
        that decreases split the data as well as increases. Needed when the values can
        step either way, such as a track angle turning left or right, by default False

    Returns
    -------
    pd.Series
        A series with new segments identified with sequential integers starting with 1,
        or if segment_column is provided, string names extending the existing
        segment names (i.e. 210 -> 210 and 210_1).
    """
    df = data.copy()

    if segment_column is None:
        col_list = [column_name]

        assert all(x in df.columns for x in col_list), (
            f"dataframe must contain columns {col_list} "
        )

        # Calculate difference between each point
        df["dif"] = df[column_name].diff()

        # If data are angles between 0-360, differences around 0 / 360 should be small
        if angular_difference:
            df["dif"] = df.dif % 360
            df["dif"] = np.where(df.dif > 180, df.dif - 360, df.dif)

        if smoothing_window is not None:
            df["dif"] = df.dif.rolling(window=smoothing_window, min_periods=1).mean()

        # Create new segment when gap > threshold
        difference = df.dif.abs() if absolute else df.dif
        df["segment"] = (difference > threshold).cumsum()

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

        return df.segment

    assert segment_column in df.columns, (
        f"dataframe must contain column '{segment_column}'"
    )
    names = pd.Series(np.nan, index=df.index, dtype=object)
    taken = set(df[segment_column].dropna().astype(str))
    for name, group in df.groupby(segment_column, sort=False):
        pieces = split_into_segments(
            group,
            threshold,
            column_name,
            min_points_per_segment=min_points_per_segment,
            angular_difference=angular_difference,
            smoothing_window=smoothing_window,
            absolute=absolute,
        )
        names.loc[group.index] = _subline_names(name, pieces, taken)
    return names


def _subline_names(
    name: typing.Any,
    pieces: pd.Series,
    taken: set[str],
) -> list[str | float]:
    """
    Name the pieces a line was split into. The first piece keeps the line's
    name, and later pieces add a suffix (``12`` -> ``12_1``, ``12_2``) or, if
    the name already has one, continue counting from it (``12_1`` -> ``12_2``).
    Names already in ``taken`` are skipped, and new names are added to it.
    NaN pieces stay NaN.
    """
    name = str(name)
    base, _, suffix = name.rpartition("_")
    if base and suffix.isdigit():
        count = int(suffix)
    else:
        base, count = name, 0

    lookup: dict[typing.Any, str] = {}
    for piece in pd.unique(pieces.dropna()):
        if not lookup:
            lookup[piece] = name
            continue
        count += 1
        while f"{base}_{count}" in taken:
            count += 1
        lookup[piece] = f"{base}_{count}"
        taken.add(lookup[piece])
    return [np.nan if pd.isna(p) else lookup[p] for p in pieces]


def split_at_turns(
    data: pd.DataFrame,
    *,
    track_column: str,
    distance_column: str,
    groupby_column: str | None = None,
    max_turn_rate: float = 0.5,
    smoothing_distance: float = 10e3,
    min_length: float | None = None,
    keep_turns: bool = True,
) -> pd.Series:
    """
    Split flights into lines wherever the aircraft turns.

    A released survey often gives a whole flight one line number, so an
    out-and-back flight is a single line that doubles back on itself. Anything
    defined along the line -- a filter, a trend, a levelling correction --
    then treats the two legs as far apart when on the ground they are side by
    side. This cuts each flight into the straight runs between its turns.

    Unlike :func:`split_into_segments`, which cuts at an isolated jump in one
    column, turning is a condition sustained over many points, so it is found
    from the smoothed turn rate rather than from a single difference. See
    :func:`airbornegeo.turn_rate`.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing the data points, which must have the columns
        given by ``track_column`` and ``distance_column``.
    track_column : str
        Name of the column holding the track angle in degrees clockwise from
        north, for example as calculated by :func:`airbornegeo.track`.
    distance_column : str
        Name of the column holding the cumulative along-track distance in
        metres.
    groupby_column : str | None, optional
        The existing flight or line column. The first line of each flight
        keeps its name and later ones extend it, so flight ``12`` becomes
        ``12``, ``12_1``, ``12_2`` and so on, by default None
    max_turn_rate : float, optional
        Degrees per kilometre above which the aircraft counts as turning.
        Straight survey flight holds well below 0.5, by default 0.5
    smoothing_distance : float, optional
        Passed to :func:`airbornegeo.turn_rate`, by default 10e3
    min_length : float | None, optional
        Runs shorter than this, in metres, are given a line name of NaN, by
        default None which keeps them all
    keep_turns : bool, optional
        If True each turn becomes a line of its own, so an out-and-back
        flight gives an outward leg, a turn, and a return leg. If False the
        turns are given NaN and only the straight runs are named, by default
        True

    Returns
    -------
    pd.Series
        The new line name for each row, NaN where a row belongs to no line.

    See Also
    --------
    airbornegeo.turn_rate : the turn rate this thresholds.
    airbornegeo.split_into_segments : split at a gap in time, distance or angle.
    """
    if groupby_column is not None:
        names = pd.Series(np.nan, index=data.index, dtype=object)
        taken = set(data[groupby_column].dropna().astype(str))
        for name, group in data.groupby(groupby_column, sort=False):
            part = split_at_turns(
                group,
                track_column=track_column,
                distance_column=distance_column,
                max_turn_rate=max_turn_rate,
                smoothing_distance=smoothing_distance,
                min_length=min_length,
                keep_turns=keep_turns,
            )
            names.loc[group.index] = _subline_names(name, part, taken)
        logger.info(
            "split %s flights into %s lines at their turns.",
            data[groupby_column].nunique(),
            names.nunique(),
        )
        return names

    rate = turn_rate(
        data,
        track_column=track_column,
        distance_column=distance_column,
        smoothing_distance=smoothing_distance,
    )
    turning = rate > max_turn_rate
    # a new line begins wherever the aircraft starts, or stops, turning
    segment = np.cumsum(np.r_[False, turning[1:] != turning[:-1]])

    names = pd.Series(segment, index=data.index, dtype=float)
    if not keep_turns:
        names[turning] = np.nan
    if min_length is not None:
        distance = data[distance_column].to_numpy(dtype=float)
        named = names.notna().to_numpy()
        for value in np.unique(segment):
            mask = (segment == value) & named
            if mask.any() and np.ptp(distance[mask]) < min_length:
                names[mask] = np.nan
    return names


def unique_line_id(
    df: pd.DataFrame,
    line_col_name: str,
) -> pd.Series:
    """
    Convert supplied lines names into integers.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the data points and the line labels.
    line_col_name : str
        Column name specifying the line number

    Returns
    -------
    pd.Series
        The line names for each point in the dataframe
    """
    df1 = df.copy()

    line_names = list(df1[line_col_name].unique())

    line_series = df1[line_col_name]
    df1[line_col_name] = np.nan

    for i, n in enumerate(line_names):
        df1.loc[line_series == n, line_col_name] = int(i + 1)

    return df1[line_col_name].astype(int)


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
