import bordado as bd
import numpy as np
import pandas as pd
import verde as vd
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from tqdm.autonotebook import tqdm


def resample(
    data: pd.DataFrame,
    *,
    spacing: float,
    resample_by: str,
    maxdist: float | None,
    groupby_column: str | None = None,
) -> pd.DataFrame:
    """
    Resample all numeric columns in a dataframe at a supplied spacing of the supplied
    resample_by column. For example, if resample_by is `time` in seconds, and spacing is
    0.5, this will return a new dataframe with all numeric columns resampled at a
    resolution of 0.5 seconds.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the data to be reduced.
    spacing : float,
        The new resolution to resample the data onto
    resample_by : str or tuple of str
        Column name to use for the resampling, for example time or distance.
    maxdist : float | None, optional
        Only return resampled points which are within this distance of the nearest data
        point provided in resample_by, by default None,
    groupby_column : str | None, optional
        Column name to group by before resampling, by default None.

    Returns
    -------
    pd.DataFrame
        DataFrame with reduced data.
    """
    data = data.copy()

    # get only numeric columns
    data = data.select_dtypes(include="number")

    # drop any rows with nans
    data = data.dropna(how="any")

    # get column names
    original_col_names = data.columns

    cols = [resample_by]
    assert all(col in data.columns for col in cols), (
        f"{resample_by} must be in the dataframe"
    )

    # get list of data columns to reduce
    input_data_names = tuple(
        original_col_names.drop(
            [resample_by, groupby_column, "geometry"], errors="ignore"
        )
    )
    input_data_dtypes = data[list(input_data_names)].dtypes

    if groupby_column is None:
        # get tuples of pd.Series
        original_values = data[resample_by].to_numpy()
        input_data = tuple([data[col].to_numpy() for col in input_data_names])  # pylint: disable=consider-using-generator

        # define verde reducer function
        new_values = bd.line_coordinates(
            start=data[resample_by].min(),
            stop=data[resample_by].max(),
            spacing=spacing,
            adjust="region",
        )

        # drop values if further than max dist from nearest point to avoid interpolating
        # over data gaps
        dist_mask = vd.distance_mask(
            data_coordinates=(original_values, np.zeros_like(original_values)),
            maxdist=maxdist,
            coordinates=(new_values, np.zeros_like(new_values)),
        )
        new_values = new_values[dist_mask]

        # create interpolator
        f = interp1d(
            original_values,
            input_data,
            axis=1,
            kind="cubic",
        )

        # interpolate and make new dataframe
        df_new = pd.DataFrame(
            f(new_values).T,
            columns=input_data_names,
        )
        df_new = df_new.astype(input_data_dtypes.to_dict())
        df_new[resample_by] = new_values

        # reorder column to match original
        df_new = df_new[original_col_names]

        return df_new.reset_index(drop=True)

    assert groupby_column in data.columns, "groupby_column must be in the dataframe"

    resampled_dfs = []
    for segment_name, segment_data in tqdm(
        data.groupby(groupby_column), desc="Segments"
    ):
        # get tuples of pd.Series
        original_values = segment_data[resample_by].to_numpy()
        input_data = tuple([segment_data[col].to_numpy() for col in input_data_names])  # pylint: disable=consider-using-generator

        # define verde reducer function
        new_values = bd.line_coordinates(
            start=segment_data[resample_by].min(),
            stop=segment_data[resample_by].max(),
            spacing=spacing,
            adjust="region",
        )

        # drop values if further than max dist from nearest point to avoid interpolating
        # over data gaps
        dist_mask = vd.distance_mask(
            data_coordinates=(original_values, np.zeros_like(original_values)),
            maxdist=maxdist,
            coordinates=(new_values, np.zeros_like(new_values)),
        )
        new_values = new_values[dist_mask]

        # create interpolator
        f = interp1d(
            original_values,
            input_data,
            axis=1,
            kind="cubic",
        )

        # interpolate and make new dataframe
        df_new = pd.DataFrame(
            f(new_values).T,
            columns=input_data_names,
        )
        df_new = df_new.astype(input_data_dtypes.to_dict())
        df_new[resample_by] = new_values
        df_new[groupby_column] = segment_name

        # reorder column to match original
        df_new = df_new[original_col_names]

        resampled_dfs.append(df_new)

    return pd.concat(resampled_dfs).reset_index(drop=True)


def resample_as(
    data: pd.DataFrame,
    *,
    resample_by: str,
    resample_values: NDArray,
    groupby_column: str | None = None,
) -> pd.DataFrame:
    """

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the data to be reduced.
    resample_by : str or tuple of str
        Column name to use for the resampling, for example time or distance.
    groupby_column : str | None, optional
        Column name to group by before resampling, by default None.

    Returns
    -------
    pd.DataFrame
        DataFrame with reduced data.
    """
    data = data.copy()

    # get only numeric columns
    data = data.select_dtypes(include="number")

    # drop any rows with nans
    data = data.dropna(how="any")

    # get column names
    original_col_names = data.columns

    cols = [resample_by]
    assert all(col in data.columns for col in cols), (
        f"{resample_by} must be in the dataframe"
    )

    # get list of data columns to reduce
    input_data_names = tuple(
        original_col_names.drop(
            [resample_by, groupby_column, "geometry"], errors="ignore"
        )
    )
    input_data_dtypes = data[list(input_data_names)].dtypes

    if groupby_column is None:
        # get tuples of pd.Series
        original_values = data[resample_by].to_numpy()
        input_data = tuple([data[col].to_numpy() for col in input_data_names])  # pylint: disable=consider-using-generator

        # create interpolator
        f = interp1d(
            original_values,
            input_data,
            axis=1,
            kind="cubic",
        )

        resample_values = resample_values[
            (resample_values >= min(original_values))
            & (resample_values <= max(original_values))
        ]
        # interpolate and make new dataframe
        df_new = pd.DataFrame(
            f(resample_values).T,
            columns=input_data_names,
        )
        df_new = df_new.astype(input_data_dtypes.to_dict())
        df_new[resample_by] = resample_values

        # reorder column to match original
        df_new = df_new[original_col_names]

        return df_new.reset_index(drop=True)

    assert groupby_column in data.columns, "groupby_column must be in the dataframe"

    resampled_dfs = []
    for segment_name, segment_data in tqdm(
        data.groupby(groupby_column), desc="Segments"
    ):
        # get tuples of pd.Series
        original_values = segment_data[resample_by].to_numpy()
        input_data = tuple([segment_data[col].to_numpy() for col in input_data_names])  # pylint: disable=consider-using-generator

        # create interpolator
        f = interp1d(
            original_values,
            input_data,
            axis=1,
            kind="cubic",
        )

        resample_values = resample_values[
            (resample_values >= min(original_values))
            & (resample_values <= max(original_values))
        ]
        # interpolate and make new dataframe
        df_new = pd.DataFrame(
            f(resample_values).T,
            columns=input_data_names,
        )
        df_new = df_new.astype(input_data_dtypes.to_dict())
        df_new[resample_by] = resample_values
        df_new[groupby_column] = segment_name

        # reorder column to match original
        df_new = df_new[original_col_names]

        resampled_dfs.append(df_new)

    return pd.concat(resampled_dfs).reset_index(drop=True)
