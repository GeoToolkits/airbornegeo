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
