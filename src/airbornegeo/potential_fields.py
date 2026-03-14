import geopandas as gpd
import harmonica as hm
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import NDArray
from tqdm.autonotebook import tqdm

import airbornegeo

sns.set_theme()


def eq_sources_by_line(
    data: pd.DataFrame,
    *,
    data_column: str,
    damping: float,
    depth: float | str = "default",
    block_size: float | None = None,
    groupby_column: str = "line",
) -> dict:
    """
    Fit a set of equivalent sources to each group of a dataframe. These fitted sources
    can then be used to predict the  data at the intersection points, on a regular line
    spacing, or to upward continue  the line.

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
        Column name to group by before sorting by time, by default None

    Returns
    -------
    dict
        a dictionary with a keys of each group name and a values of fitted equivalent
        sources
    """

    data = data.copy()

    data["tmp"] = 0

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


def eotvos_correction(lat_deg, lon_deg, time):
    """
    Compute simple Eötvös correction (mGal) from latitude, longitude, and time.

    Parameters
    ----------
    lat_deg : array-like
        Latitude in degrees
    lon_deg : array-like
        Longitude in degrees
    time : array-like
        Time in seconds

    Returns
    -------
    E : ndarray
        Eötvös correction in mGal
    """
    v_east = airbornegeo.processing.eastward_velocity(lat_deg, lon_deg, time)
    lat_rad = np.radians(lat_deg)

    omega = 7.292115e-5  # Earth's rotation rate, rad/s
    r = 6371000  # Earth radius in meters

    # Classical Eötvös formula
    e = 2 * omega * v_east * np.cos(lat_rad) + (v_east**2) / r

    # Convert to mGal
    return e * 1e5


def eotvos_correction_full(
    coordinates: tuple[NDArray, NDArray, NDArray],
    time: NDArray,
) -> pd.Series:
    # ell = bl.WGS84
    # angular_velocity = ell.angular_velocity # radians per second
    # semi_major_axis = ell.semimajor_axis
    # flattening = ell.flattening
    a = 6378137.0
    b = 6356752.3142

    lon, lat, ht = coordinates

    omega = 0.00007292115  # siderial rotation rate, radians/sec
    ecc = (a - b) / a

    latr = np.deg2rad(lat)
    lonr = np.deg2rad(lon)
    lonr = np.unwrap(lonr)

    # get time derivatives of position
    dt = np.diff(time, prepend=np.nan)
    dlat = np.diff(latr, prepend=np.nan) / dt
    dlon = np.diff(lonr, prepend=np.nan) / dt
    dht = np.diff(ht, prepend=np.nan) / dt
    ddlat = np.diff(dlat, prepend=np.nan) / dt
    ddlon = np.diff(dlon, prepend=np.nan) / dt
    ddht = np.diff(dht, prepend=np.nan) / dt

    # sines and cosines etc
    slat = np.sin(latr)
    clat = np.cos(latr)
    s2lat = np.sin(2 * latr)
    c2lat = np.cos(2 * latr)

    # calculate r' and its derivatives
    rp = a * (1 - ecc * slat * slat)
    drp = -a * dlat * ecc * s2lat
    ddrp = -a * ddlat * ecc * s2lat - 2 * a * dlat * dlat * ecc * c2lat

    # calculate deviation from normal and derivatives
    d = np.arctan(ecc * s2lat)
    dd = 2 * dlat * ecc * c2lat
    ddd = 2 * ddlat * ecc * c2lat - 4 * dlat * dlat * ecc * s2lat

    # define r and its derivatives
    r = np.vstack((-rp * np.sin(d), np.zeros(len(rp)), -rp * np.cos(d) - ht)).T
    rdot = np.vstack(
        (
            -drp * np.sin(d) - rp * dd * np.cos(d),
            np.zeros(len(rp)),
            -drp * np.cos(d) + rp * dd * np.sin(d) - dht,
        )
    ).T
    ci = (
        -ddrp * np.sin(d)
        - 2.0 * drp * dd * np.cos(d)
        - rp * (ddd * np.cos(d) - dd * dd * np.sin(d))
    )
    ck = (
        -ddrp * np.cos(d)
        + 2.0 * drp * dd * np.sin(d)
        + rp * (ddd * np.sin(d) + dd * dd * np.cos(d) - ddht)
    )
    rdotdot = np.vstack((ci, np.zeros(len(ci)), ck)).T

    # define w and derivative
    w = np.vstack(((dlon + omega) * clat, -dlat, -(dlon + omega) * slat)).T
    wdot = np.vstack(
        (
            dlon * clat - (dlon + omega) * dlat * slat,
            -ddlat,
            -ddlon * slat - (dlon + omega) * dlat * clat,
        )
    ).T

    w2xrdot = np.cross(2 * w, rdot)
    wdotxr = np.cross(wdot, r)
    wxr = np.cross(w, r)
    wxwxr = np.cross(w, wxr)

    # calculate wexwexre, centrifugal acceleration due to the Earth
    re = np.vstack((-rp * np.sin(d), np.zeros(len(rp)), -rp * np.cos(d))).T
    we = np.vstack((omega * clat, np.zeros(len(slat)), -omega * slat)).T
    wexre = np.cross(we, re)
    _wexwexre = np.cross(we, wexre)
    wexr = np.cross(we, r)
    wexwexr = np.cross(we, wexr)

    # calculate total acceleration for the aircraft
    accel = rdotdot + w2xrdot + wdotxr + wxwxr

    # Eotvos correction is the vertical component of the total acceleration of
    # the aircraft minus the centrifugal acceleration of the Earth, convert to mGal
    return (accel[:, 2] - wexwexr[:, 2]) * 100e3


def update_intersections_with_eq_sources(
    data: pd.DataFrame | gpd.GeoDataFrame,
    *,
    fitted_equivalent_sources: dict,
    data_column: str,
    groupby_column: str = "line",
) -> pd.Series:
    """
    At each theoretical intersection point, replace the interpolated field value with a
    value predected by the fitted equivalent sources for the line, at the x,y coordinate
    of the intersection point, and the higher of the two lines' elevations. This allows
    the cross-over mistie value to be comparing the field values at the same point in 3D
    space, not 2D space, due to different flight heights.

    Parameters
    ----------
    data : pd.DataFrame | gpd.GeoDataFrame
        dataframe containing the data to update
    fitted_equivalent_sources : dict
        a dictionary with keys of line names and values of fitted equivalent sources for
        each line, which can be created using the function `eq_sources_by_line`
    data_column : str
        name of the column containing the field values to update at the intersection
        points, this should be the same as the column that use used as 'data_column'
        when fitting the equivalent sources for each line with `eq_sources_by_line`.

    Returns
    -------
    pd.Series
        the updated field values at the intersection points, which can be added to the
        dataframe as a new column or used to replace the existing values in the
        dataframe.
    """

    data = data.copy()

    assert "line" in data.columns, "data must have column 'line'"

    for segment_name, segment_data in tqdm(
        data.groupby(groupby_column), desc="Segments"
    ):
        # get fitted equivalent sources for this line
        eqs = fitted_equivalent_sources[segment_name]

        # get intersection points for this line
        line_inters = segment_data[segment_data.is_intersection]

        for i, row in line_inters.iterrows():
            # get height of intersection point for the cross line
            cross_inter = data[
                (data.line == row.intersecting_line)
                & (data.intersecting_line == segment_name)
            ]

            assert len(cross_inter) == 1, (
                data[data.intersecting_line == segment_name],
                row.intersecting_line,
                segment_name,
            )

            cross_height = cross_inter.height.to_numpy()[0]
            # assert len(cross_height)==1, f"{cross_height}"
            # assert isinstance(cross_height, (int, float)), f"{cross_inter}"

            coords = (
                np.array([row.distance_along_line]),
                np.array([0]),
                np.array([np.max([cross_height, row.height])]),
            )
            # predict the field value at the x,y coordinate of the intersection point,
            # and the higher of the two lines' elevations, using the supplied fitted
            # equivalent sources for each line
            up_cont_value = eqs.predict(coords)

            # add predicted value to dataframe at intersection point
            data.at[i, data_column] = up_cont_value  # noqa: PD008

    return data[data_column]
