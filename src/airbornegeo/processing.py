import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from airbornegeo import logger

sns.set_theme()


def split_into_segments(
    data: pd.DataFrame,
    threshold: float,
    column_name: str,
    min_points_per_segment: int = 0,
    angular_difference: bool = False,
    segment_column: str | None = None,
    smoothing_window: int | None = None,
) -> pd.Series:
    """
    Split dataframe into segments where there is a gap in the supplied values greater
    than the threshold. Data are sorted by the column given by column_name which can be
    quantities such as time in seconds, cumulative distances, or aircraft heading or
    track. By default, a series of sequential segments names as integers starting at 1
    will be returned. Alternatively, existing segments can be divided into new segments.
    These existing segment ID's should be provided as floats, via the column given by
    segment_column. New sub-segments will have a decimal added (i.e., 210 -> 210.0 and
    210.1) or the decimal will be incremented (210.1 -> 210.1 and 210.2).

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
        The segment ID's, as floats.
    smoothing_window : int, optional
        Window size in number of data points for smoothing each difference after it's
        been calculated, by default None

    Returns
    -------
    pd.Series
        A series with new segments identified with sequential integers starting with 1,
        or if segment_column is provided, existing segment ID numbers will be
        incremented by 0.1 if the segment is split (i.e. 210 -> 210.0 and 210.1).
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
        df["segment"] = (df.dif > threshold).cumsum()

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

    raise NotImplementedError

    # col_list = [column_name, segment_column]

    # assert all(x in df.columns for x in col_list), (
    #     f"dataframe must contain columns {col_list} "
    # )

    # # check segment_column is right dtype
    # assert isinstance(df[segment_column], float)

    # # save index, sort by time and reset index
    # segments = []
    # for _segment_name, segment_data in df.groupby(segment_column):
    #     segment_data = segment_data.reset_index(names="tmp_index")
    #     segment_data = segment_data.sort_values(by=column_name).reset_index(drop=True)

    #     # Calculate difference between each point
    #     segment_data["diff"] = segment_data[column_name].diff()

    #     # Create new segment when difference >threshold
    #     segment_data["segment"] = (segment_data["diff"] > threshold).cumsum()

    #     # remove segments which are less than specified number of points
    #     if min_points_per_segment > 0:
    #         groups = segment_data.groupby("segment")
    #         prior_len = len(segment_data.segment.unique())
    #         # make segment ID nan for small segments
    #         small_segments = groups.filter(lambda x: len(x) < min_points_per_segment)
    #         segment_data.loc[small_segments.index, "segment"] = np.nan
    #         post_len = len(segment_data.segment.unique())
    #         logger.info(
    #             "dropped %s segments which contained less than %s points.",
    #             prior_len - post_len,
    #             min_points_per_segment,
    #         )

    #     # Reset index and sort
    #     segment_data = segment_data.set_index("tmp_index").sort_values("tmp_index")

    #     # if segment has been split, increment the decimal of the provided ID
    #     if len(segment_data.segment.unique() > 1):
    #         increment = 0.1
    #         for _subsegment_name, subsegment_data in segment_data.groupby("segment"):
    #             segment_data["segment"] = subsegment_data[segment_column] + increment
    #             increment += 0.1
    #     else:
    #         segment_data["segment"] = segment_data[segment_column]

    # df = pd.concat(segments)

    # return df.segment


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
