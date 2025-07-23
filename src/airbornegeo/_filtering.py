from __future__ import annotations
from polartoolkit import utils as polar_utils
import numpy as np
import pandas as pd

def block_reduce_by_line(
    df: gpd.GeoDataFrame | pd.DataFrame,
    line_column: str = "line",
    block_size: int = 10,
) -> pd.DataFrame:
    """
    Reduce data by line using block size.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to be reduced.
    data_column : str
        The column containing the data to be reduced.
    line_column : str, optional
        The column containing the line identifiers, by default "line".
    block_size : int, optional
        The size of the blocks to reduce the data by, by default 10.

    Returns
    -------
    pd.DataFrame
        DataFrame with reduced data.
    """

    df = df.copy()

    df = df.groupby(line_column)

    blocked_dfs = []
    for name, line in df:
        blocked = polar_utils.block_reduce(
            line,
            np.median,
            spacing=100,
            center_coordinates=False,
            input_coord_names=["easting", "northing"],
            input_data_names=line.columns.drop([line_column])
        )
        blocked[line_column] = name
        blocked_dfs.append(blocked)

    return pd.concat(blocked_dfs).reset_index(drop=True).sort_values(
        by=[line_column]
    )




def filter_by_distance(
    df: gpd.GeoDataFrame | pd.DataFrame,
    filt_type: str,
    data_column: str,
    distance_column: str = "dist_along_line",
    line_column: str = "line",
    pad_width_percentage: float = 10,
) -> gpd.GeoDataFrame | pd.DataFrame:
    """
    _summary_

    Parameters
    ----------
    df : gpd.GeoDataFrame | pd.DataFrame
        _description_
    filt_type : str
        a string with format "<type><width>+h" where type is GMT filter type, width is
        the filter width in same units as distance column, and optional +h switches from
        low-pass to high-pass filter; e.g. "g10+h" is a 10m high-pass Gaussian filter.
    data_column : str
        _description_
    distance_column : str, optional
        _description_, by default "dist_along_line"
    line_column : str, optional
        _description_, by default "line"
    pad_width_percentage : float, optional
        _description_, by default 10

    Returns
    -------
    gpd.GeoDataFrame | pd.DataFrame
        _description_
    """

    df = df.copy()

    for i in df[line_column].unique():
        # subset data from 1 line
        line = df[df[line_column] == i]
        line = line[[distance_column, data_column]]

        # get data spacing
        distance = line[distance_column].values
        data_spacing = np.median(np.diff(distance))

        # pad distance of 10% of line distance
        pad_dist = (distance.max() - distance.min()) * (pad_width_percentage / 100)
        pad_dist = round(pad_dist / data_spacing) * data_spacing

        # get the number of points to pad
        # n_pad = int(pad_dist / data_spacing)

        # add pad points to distance values
        lower_pad = np.arange(
            distance.min() - pad_dist,
            distance.min(),
            data_spacing,
        )
        upper_pad = np.arange(
            distance.max(),
            distance.max() + pad_dist,
            data_spacing,
        )
        vals = np.concatenate((lower_pad, upper_pad))
        new_dist = pd.DataFrame({distance_column: vals})

        # pad the line, fill padded values in data with nearest value
        padded = (
            pd.concat(
                [line.reset_index(), new_dist],
            )
            .sort_values(by=distance_column)
            .set_index("index")
        )
        padded = padded.fillna(method="ffill").fillna(method="bfill").reset_index()
        padded = padded.rename(columns={"index": "true_index"})

        # filter the padded data
        filtered = pygmt.filter1d(
            padded[[distance_column, data_column]],
            end=True,
            time_col=0,
            filter_type=filt_type,
        ).rename(columns={0: distance_column, 1: data_column})

        # un-pad the data
        filtered["index"] = padded.true_index
        filtered = filtered.set_index("index")
        filtered = filtered[filtered.index.isin(line.index)]

        # replace original data with filtered data
        df.loc[df[line_column] == i, data_column] = filtered[data_column]

    return df[data_column]