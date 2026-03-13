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
    df: pd.DataFrame,
    data_column: str,
    damping: float,
    depth: float | str = "default",
    block_size: float | None = None,
) -> pd.Series:
    """
    For each light line in a dataframe, fit a set of equivalent sources to them. These
    can then be used to predict the data at the intersection points, on a regular line
    spacing, or to upward continue the line.
    """

    df = df.copy()

    assert "line" in df.columns, "line column must be in dataframe"

    df["tmp"] = 0

    fitted_eqs = {}
    for name, line_df in tqdm(df.groupby("line"), desc="Lines"):
        coords = (
            line_df.distance_along_line,
            line_df.tmp,
            line_df.height,
        )

        # define equivalent source parameters
        eqs_line = hm.EquivalentSources(
            damping=damping,
            depth=depth,
            block_size=block_size,
        )

        eqs_line.fit(coords, line_df[data_column])

        fitted_eqs[name] = eqs_line

    return fitted_eqs


def upward_continue_by_line(
    df: pd.DataFrame,
    fitted_equivalent_sources: dict,
    height: float,
    no_downward_continuation: bool = True,
) -> pd.Series:
    """
    For each light line in a dataframe, fit a set of equivalent sources and then upward
    continue to data to a specified height and return the upward continued data.
    """

    df = df.copy()

    assert "line" in df.columns, "line column must be in dataframe"
    assert "height" in df.columns, "height column must be in dataframe"

    df["tmp"] = 0

    for name, line_df in tqdm(df.groupby("line"), desc="Lines"):
        eqs = fitted_equivalent_sources[name]

        upward = np.full_like(line_df.tmp, height)

        if no_downward_continuation is True:
            upward = np.where(
                upward > line_df.height.to_numpy(), upward, line_df.height.to_numpy()
            )

        upward_continued = eqs.predict(
            (
                line_df.distance_along_line,
                line_df.tmp,
                upward,
            )
        )

        df.loc[df.line == name, "upward_continued"] = upward_continued

    return df.upward_continued


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
    fitted_equivalent_sources: dict,
    data_column: str,
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

    pbar = tqdm(data.groupby("line"), desc="Lines")
    for line, line_df in pbar:
        pbar.set_description(f"Line {line}")

        # get fitted equivalent sources for this line
        eqs = fitted_equivalent_sources[line]

        # get intersection points for this line
        line_inters = line_df[line_df.is_intersection]

        for i, row in line_inters.iterrows():
            # get height of intersection point for the cross line
            cross_inter = data[
                (data.line == row.intersecting_line) & (data.intersecting_line == line)
            ]

            assert len(cross_inter) == 1, (
                data[data.intersecting_line == line],
                row.intersecting_line,
                line,
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
