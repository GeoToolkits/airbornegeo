import boule as bl
import harmonica as hm
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import NDArray

from airbornegeo.utils import _apply_grouped, _iter_groups

sns.set_theme()


def eq_sources_1d(
    data: pd.DataFrame,
    *,
    data_column: str,
    damping: float,
    depth: float | str = "default",
    block_size: float | None = None,
    groupby_column: str | None = None,
    progressbar: bool = True,
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
    progressbar : bool, optional
        Show progress bar for each group, by default True

    Returns
    -------
    dict | hm.EquivalentSources
        a dictionary with a keys of each group name and a values of fitted equivalent
        sources, or if groupby_column is not provided, just a single fitted set of
        equivalent sources.
    """

    data = data.copy()

    data["tmp"] = 0

    def _fit(segment: pd.DataFrame) -> hm.EquivalentSources:
        coords = (
            segment.distance_along_line,
            segment.tmp,
            segment.height,
        )
        eqs_line = hm.EquivalentSources(
            damping=damping,
            depth=depth,
            block_size=block_size,
        )
        eqs_line.fit(coords, segment[data_column])
        return eqs_line

    if groupby_column is None:
        return _fit(data)

    assert groupby_column in data.columns, "groupby_column must be in dataframe"

    return {
        segment_name: _fit(segment_data)
        for segment_name, segment_data in _iter_groups(
            data, groupby_column, progressbar
        )
    }


def upward_continue_by_line(
    data: pd.DataFrame,
    fitted_equivalent_sources: dict,
    height: float,
    groupby_column: str = "line",
    progressbar: bool = True,
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

    for segment_name, segment_data in _iter_groups(data, groupby_column, progressbar):
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


def igrf(
    data: pd.DataFrame,
    *,
    datetime_column: str,
    latitude_column: str,
    longitude_column: str,
    height_column: str,
    groupby_column: str | None = None,
    progressbar: bool = True,
    ellipsoid=bl.WGS84,
    min_degree=1,
    max_degree=13,
) -> tuple[NDArray, NDArray, NDArray]:
    """
    Calculate IGRF intensity, inclication and declination using the time from the first
    row of the supplied datetime_column. If groupby_column is given, then will use the
    first row from each group.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing the data
    datetime_column : str
        name of the column containing the datetime values, of which the first (or first
        within a group), will be used to calculate the IGRF.
    latitude_column : str
        name of the column containing the geodetic latitudes in decimal degrees
    longitude_column : str
        name of the column containing the geodetic longitudes in decimal degrees
    height_column : str
        name of the column containing the ellipsoidal heights in meters
    groupby_column : str | None, optional
        Column name to group by before calculation, by default None
    progressbar : bool, optional
        Show progress bar for each group, by default True
    ellipsoid:
        The ellipsoid used to convert geodetic to geocentric spherical coordinates and
        convert the magnetic field vector from a geocentric spherical to a geodetic
        system. Default is `boule.WGS84`
    min_degree:
        The minimum degree used in the expansion. Default is 1 (magnetic fields don't
        have the 0 degree term).
    max_degree:
        The maximum degree used in the expansion. Default is 13.

    Returns
    -------
    tuple[NDArray, NDArray, NDArray]
        The intensity (nT), inclination (degrees) and declination (degrees) of the IGRF.
    """
    data = data.copy()

    col_list = [datetime_column, latitude_column, longitude_column, height_column]
    if groupby_column is not None:
        col_list.append(groupby_column)
    assert all(x in data.columns for x in col_list), (
        f"dataframe must contain columns {col_list} "
    )

    def _predict(segment: pd.DataFrame) -> tuple[NDArray, NDArray, NDArray]:
        # select the first row's datetime (or first within the group)
        datetime = segment[datetime_column].iloc[0]

        # initialize a IGRF class with the date
        igrf_model = hm.IGRF14(
            datetime,
            ellipsoid=ellipsoid,
            min_degree=min_degree,
            max_degree=max_degree,
        )

        # predict the IGRF field vectors at all survey locations
        field = igrf_model.predict(
            (
                segment[longitude_column],
                segment[latitude_column],
                segment[height_column],
            )
        )

        # convert vector to intensity, inclination and declination
        return hm.magnetic_vec_to_angles(*field)

    return _apply_grouped(
        data,
        groupby_column=groupby_column,
        progressbar=progressbar,
        func=_predict,
    )
