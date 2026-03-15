import harmonica as hm
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import NDArray
from tqdm.autonotebook import tqdm

import airbornegeo

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
        Column name to group by before sorting by time, by default None

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
