import typing

import pandas as pd
import verde as vd
from tqdm.autonotebook import tqdm


def block_reduce(
    data: pd.DataFrame,
    reduction: typing.Callable[..., float | int],
    *,
    spacing: float,
    reduce_by: str | tuple[str, str],
    groupby_column: str | None = None,
    progressbar: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Reduce data by line based on the column(s) provided by reduce_by and the reduction
    function. For example, if 'reduce_by' is 'distance_along_line', 'spacing' is 1000,
    and 'reduction' is np.mean', then the data will be reduced by taking the mean value
    along every 1 km. If 'reduce_by' is 'unixtime', this would take the mean of every
    1000 seconds. If 'reduce_by' is ('easting', 'northing'), this would take the mean of
    every point in a 1x1 km block. The reduction function can be any function that takes
    an array of values and  returns a single value, such as np.mean, np.median, np.max,
    etc. The function will return a new dataframe with the reduced data.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the data to be reduced.
    reduction : typing.Callable
        function to use in reduction, e.g. np.mean
    spacing : float,
        The spacing to reduce the data by, in the same units as whatever
        column is specified by reduce_by
    reduce_by : str or tuple of str
        Column name(s) to reduce by. If a single column name is provided, the data will
        be reduced by only column. If a tuple of two column names are provided, the data
        will be reduced by both columns, e.g. in 1x1 km blocks if the columns are
        'easting' and 'northing'.
    groupby_column : str | None, optional
        Column name to group by before block reducing, by default None.
    progressbar : bool, optional
        Show progress bar for each group, by default True
    kwargs : typing.Any
        Any additional keyword arguments to pass to verde.BlockReduce.

    Returns
    -------
    pd.DataFrame
        DataFrame with reduced data.
    """
    data = data.copy()

    if isinstance(reduce_by, str):
        reduce_by = (reduce_by,)

    if len(reduce_by) == 1:
        reduce_by = (reduce_by[0], "tmp")  # add dummy column for second coordinate
        data["tmp"] = 0.0

    assert all(col in data.columns for col in reduce_by), (
        f"{reduce_by} must be in the dataframe"
    )

    # get only numeric columns
    numeric_cols = data.select_dtypes(include="number").columns
    input_data_names = [c for c in numeric_cols if c not in [*reduce_by, "geometry"]]

    # define verde reducer function
    reducer = vd.BlockReduce(
        reduction,
        spacing=spacing,
        **kwargs,
    )

    def run_reduce(df):
        input_coords = tuple(df[c].to_numpy() for c in reduce_by)
        input_data = tuple(df[c].to_numpy() for c in input_data_names)

        # apply reduction
        coordinates, blocked_data = reducer.filter(
            coordinates=input_coords,
            data=input_data,
        )

        out = {c: coordinates[i] for i, c in enumerate(reduce_by)}

        if len(input_data_names) == 1:
            out[input_data_names[0]] = blocked_data
        else:
            out.update({c: blocked_data[i] for i, c in enumerate(input_data_names)})

        return pd.DataFrame(out)

    if groupby_column is None:
        return run_reduce(data).reset_index(drop=True)

    assert groupby_column in data.columns

    groups = data.groupby(groupby_column, sort=False)

    pbar = tqdm(groups, desc="Block-reducing segments") if progressbar else groups

    blocked_dfs = []
    for segment_name, segment_data in pbar:
        blocked = run_reduce(segment_data)
        blocked[groupby_column] = segment_name
        blocked_dfs.append(blocked)

    return pd.concat(blocked_dfs, ignore_index=True)
