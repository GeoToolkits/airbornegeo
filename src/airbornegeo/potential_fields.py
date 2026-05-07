import harmonica as hm
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.autonotebook import tqdm

sns.set_theme()


def eq_sources_1d(
    data: pd.DataFrame,
    *,
    data_column: str,
    damping: float,
    depth: float | str = "default",
    block_size: float | None = None,
    groupby_column: str | None = None,
) -> dict | hm.EquivalentSources:
    """
    Fit a set of equivalent sources along 1 dimension. These fitted sources
    can then be used to predict the  data at the intersection points, on a regular line
    spacing, or to upward continue  the line. If groupby_column is provided, the source
    will be fit individually for each group,

    Parameters
    ----------
    data : pd.DataFrame
        the dataframe containing the columns distance_along_line, grouby_column,
        data_column, and height.
    data_column : str
        the column name for the data to fit.
    damping : float
        the damping value to use in fitting the equivalent sources
    depth : float | str, optional
        the source depths, by default "default"
    block_size : float | None, optional
        Block reduce the number of sources. This doesn't block reduce the data, that
        should be done before with func::`block_reduce`, by default None
    groupby_column : str | None, optional
        Column name to group by before fitting sources, by default None

    Returns
    -------
    dict | hm.EquivalentSources
        a dictionary with a keys of each group name and a values of fitted equivalent
        sources, or if groupby_column is not provided, just a single fitted set of
        equivalent sources.
    """

    data = data.copy()

    data["tmp"] = 0

    if groupby_column is None:
        coords = (
            data.distance_along_line,
            data.tmp,
            data.height,
        )

        # define equivalent source parameters
        eqs_line = hm.EquivalentSources(
            damping=damping,
            depth=depth,
            block_size=block_size,
        )

        eqs_line.fit(coords, data[data_column])

        return eqs_line

    assert groupby_column in data.columns, "groupby_column must be in dataframe"

    fitted_eqs = {}
    for segment_name, segment_data in tqdm(data.groupby(groupby_column), desc="Groups"):
        coords = (
            segment_data.distance_along_line,
            segment_data.tmp,
            segment_data.height,
        )

        # define equivalent source parameters
        eqs_line = hm.EquivalentSources(
            damping=damping,
            depth=depth,
            block_size=block_size,
        )

        eqs_line.fit(coords, segment_data[data_column])

        fitted_eqs[segment_name] = eqs_line

    return fitted_eqs


def upward_continue_by_line(
    data: pd.DataFrame,
    fitted_equivalent_sources: dict,
    height: float,
    groupby_column: str = "line",
    no_downward_continuation: bool = True,
) -> pd.Series:
    """
    For each light line in a dataframe, fit a set of equivalent sources and then upward
    continue to data to a specified height and return the upward continued data.
    """

    data = data.copy()

    assert "line" in data.columns, "line column must be in dataframe"
    assert "height" in data.columns, "height column must be in dataframe"

    data["tmp"] = 0

    for segment_name, segment_data in tqdm(data.groupby(groupby_column), desc="Groups"):
        eqs = fitted_equivalent_sources[segment_name]

        upward = np.full_like(segment_data.tmp, height)

        if no_downward_continuation is True:
            upward = np.where(
                upward > segment_data.height.to_numpy(),
                upward,
                segment_data.height.to_numpy(),
            )

        upward_continued = eqs.predict(
            (
                segment_data.distance_along_line,
                segment_data.tmp,
                upward,
            )
        )

        data.loc[data[groupby_column] == segment_name, "upward_continued"] = (
            upward_continued
        )

    return data.upward_continued
