import boule
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


def eotvos_correction_glicken(latitude, track, ground_speed):
    """
    Compute the simple Eötvös correction (mGal) from latitude, aircraft track and ground
    speed using the Glicken 1962 simplified equation.

    Parameters
    ----------
    latitude : array-like
        Latitude in degrees
    track : array-like
        Aircraft track in degrees from geographic north, positive clock-wise
    ground_speed : array-like
        Ground speed in meters per seconds

    Returns
    -------
    E : ndarray
        Eötvös correction in mGal
    """
    # mps to knots
    ground_speed = ground_speed * 1.94384

    # degrees to radian
    lat_rad = np.radians(latitude)
    track_rad = np.radians(track)

    return 7.503 * ground_speed * np.cos(lat_rad) * np.sin(track_rad) + (
        0.004154 * ground_speed**2
    )


def eotvos_correction_harlan(latitude, track, ground_speed, height):
    """
    Compute the Eötvös correction (mGal) from latitude, track, ground speed, and height
    using the Harlan 1968 equation 16, which uses ground speed instead of air speed.

    Ground speed and velocity at altitude differ due to the Coriolis term
    Parameters
    ----------
    latitude : array-like
        Latitude in degrees
    track : array-like
        Aircraft track in degrees from geographic north, positive clock-wise
    ground_speed : array-like
        Ground speed in meters per seconds

    Returns
    -------
    E : ndarray
        Eötvös correction in mGal
    """
    # define an ellipsoid
    ell = boule.WGS84

    # mps to knots
    ground_speed = ground_speed * 1.94384

    # degrees to radian
    lat_rad = np.radians(latitude)
    track_rad = np.radians(track)

    return ((ground_speed**2) / ell.semimajor_axis) * (
        1
        + height / ell.semimajor_axis
        - ell.flattening * (1 - np.cos(lat_rad) ** 2 * (3 - 2 * np.sin(track_rad) ** 2))
    ) + 2 * ground_speed * ell.angular_velocity * np.cos(lat_rad) * np.sin(
        track_rad
    ) * (1 + height / ell.semimajor_axis) * 1e5

    # # use airspeed instead of ground speed
    # velocity_easting = np.sin(track_rad) * ground_speed
    # velocity_northing = np.cos(track_rad) * ground_speed

    # Eotvos correction in mGal
    # return ((velocity_northing**2) / ell.semimajor_axis) * (1 - height / ell.semimajor_axis + ell.flattening * (2 - 3 * np.sin(lat_rad)**2)) \
    #       + ((velocity_easting**2) / ell.semimajor_axis) * (1 - height / ell.semimajor_axis - ell.flattening * (np.sin(lat_rad)**2)) \
    #       + 2 * ell.angular_velocity * velocity_easting * np.cos(lat_rad) * 1e5


def eotvos_correction(latitude, track, ground_speed, height):
    """
    Compute Eötvös correction (mGal) from latitude, longitude, and time.

    Parameters
    ----------
    latitude : array-like
        Latitude in degrees
    track : array-like
        Aircraft track in degrees from geographic north, positive clock-wise
    ground_speed : array-like
        Ground speed in meters per seconds
    height : array-like
        Height above ...

    Returns
    -------
    E : ndarray
        Eötvös correction in mGal
    """
    # define an ellipsoid
    ell = boule.WGS84

    # mps to knots
    ground_speed = ground_speed * 1.94384

    # degrees to radian
    lat_rad = np.radians(latitude)
    track_rad = np.radians(track)

    velocity_easting = np.sin(track_rad) * ground_speed
    velocity_northing = np.cos(track_rad) * ground_speed

    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)

    # Eotvos correction in mGal
    return (
        ((velocity_northing**2) / ell.semimajor_axis)
        * (1 - height / ell.semimajor_axis + ell.flattening * (2 - 3 * sin_lat**2))
        + ((velocity_easting**2) / ell.semimajor_axis)
        * (1 - height / ell.semimajor_axis - ell.flattening * (sin_lat**2))
        + 2 * ell.angular_velocity * velocity_easting * cos_lat * 1e5
    )


# def eotvos_correction(data_in, differentiator=central_difference):
#     """
#     Eotvos correction from https://github.com/DynamicGravitySystems/DGP/blob/5c0b566b846eb25f1e5ede64b2caaaa6a3352a29/dgp/lib/transform/gravity.py#L52

#     Parameters
#     ----------
#     data_in: DataFrame
#         trajectory frame containing latitude, longitude, and
#         height above the ellipsoid
#     dt: float
#         sample period

#     Returns
#     -------
#         Series
#             index taken from the input

#     Notes
#     -----
#     Added to observed gravity when the positive direction of the sensitive axis is
#     down, otherwise, subtracted.

#     References
#     ---------
#     Harlan 1968, "Eotvos Corrections for Airborne Gravimetry" JGR 73,n14
#     """

#     dt = 0.1

#     lat = np.deg2rad(data_in['lat'].values)
#     lon = np.deg2rad(data_in['long'].values)
#     ht = data_in['ell_ht'].values

#     dlat = differentiator(lat, n=1, dt=dt)
#     ddlat = differentiator(lat, n=2, dt=dt)
#     dlon = differentiator(lon, n=1, dt=dt)
#     ddlon = differentiator(lon, n=2, dt=dt)
#     dht = differentiator(ht, n=1, dt=dt)
#     ddht = differentiator(ht, n=2, dt=dt)

#     sin_lat = np.sin(lat)
#     cos_lat = np.cos(lat)
#     sin_2lat = np.sin(2.0 * lat)
#     cos_2lat = np.cos(2.0 * lat)

#     # Calculate the r' and its derivatives
#     r_prime = a * (1.0 - ecc * sin_lat ** 2)
#     dr_prime = -a * dlat * ecc * sin_2lat
#     ddr_prime = (-a * ddlat * ecc * sin_2lat - 2.0 * a * (dlat ** 2) *
#                  ecc * cos_2lat)

#     # Calculate the deviation from the normal and its derivatives
#     D = np.arctan(ecc * sin_2lat)
#     dD = 2.0 * dlat * ecc * cos_2lat
#     ddD = (2.0 * ddlat * ecc * cos_2lat - 4.0 * dlat * dlat *
#            ecc * sin_2lat)

#     # Calculate this value once (used many times)
#     sinD = np.sin(D)
#     cosD = np.cos(D)

#     # Calculate r and its derivatives
#     r = array([
#         -r_prime * sinD,
#         np.zeros(r_prime.size),
#         -r_prime * cosD - ht
#     ])

#     rdot = array([
#         (-dr_prime * sinD - r_prime * dD * cosD),
#         np.zeros(r_prime.size),
#         (-dr_prime * cosD + r_prime * dD * sinD - dht)
#     ])

#     ci = (-ddr_prime * sinD - 2.0 * dr_prime * dD * cosD - r_prime *
#           (ddD * cosD - dD * dD * sinD))
#     ck = (-ddr_prime * cosD + 2.0 * dr_prime * dD * sinD + r_prime *
#           (ddD * sinD + dD * dD * cosD) - ddht)
#     r2dot = array([
#         ci,
#         np.zeros(ci.size),
#         ck
#     ])

#     # Define w and its derivative
#     w = array([
#         (dlon + We) * cos_lat,
#         -dlat,
#         (-(dlon + We)) * sin_lat
#     ])

#     wdot = array([
#         dlon * cos_lat - (dlon + We) * dlat * sin_lat,
#         -ddlat,
#         (-ddlon * sin_lat - (dlon + We) * dlat * cos_lat)
#     ])

#     w2_x_rdot = np.cross(2.0 * w, rdot, axis=0)
#     wdot_x_r = np.cross(wdot, r, axis=0)
#     w_x_r = np.cross(w, r, axis=0)
#     wxwxr = np.cross(w, w_x_r, axis=0)

#     we = array([
#         We * cos_lat,
#         np.zeros(sin_lat.shape),
#         -We * sin_lat
#     ])

#     wexr = np.cross(we, r, axis=0)
#     wexwexr = np.cross(we, wexr, axis=0)

#     kin_accel = r2dot * mps2mgal
#     eotvos = (w2_x_rdot + wdot_x_r + wxwxr - wexwexr) * mps2mgal

#     # acc = r2dot + w2_x_rdot + wdot_x_r + wxwxr

#     eotvos = pd.Series(eotvos[2], index=data_in.index, name='eotvos')
#     kin_accel = pd.Series(kin_accel[2], index=data_in.index, name='kin_accel')

#     return pd.concat([eotvos, kin_accel], axis=1, join='outer')


# def eotvos_correction_full(
#     coordinates: tuple[NDArray, NDArray, NDArray],
#     time: NDArray,
# ) -> pd.Series:
#     # ell = bl.WGS84
#     # angular_velocity = ell.angular_velocity # radians per second
#     # semi_major_axis = ell.semimajor_axis
#     # flattening = ell.flattening
#     a = 6378137.0
#     b = 6356752.3142

#     lon, lat, ht = coordinates

#     omega = 0.00007292115  # siderial rotation rate, radians/sec
#     ecc = (a - b) / a

#     latr = np.deg2rad(lat)
#     lonr = np.deg2rad(lon)
#     lonr = np.unwrap(lonr)

#     # get time derivatives of position
#     dt = np.diff(time, prepend=np.nan)
#     dlat = np.diff(latr, prepend=np.nan) / dt
#     dlon = np.diff(lonr, prepend=np.nan) / dt
#     dht = np.diff(ht, prepend=np.nan) / dt
#     ddlat = np.diff(dlat, prepend=np.nan) / dt
#     ddlon = np.diff(dlon, prepend=np.nan) / dt
#     ddht = np.diff(dht, prepend=np.nan) / dt

#     # sines and cosines etc
#     slat = np.sin(latr)
#     clat = np.cos(latr)
#     s2lat = np.sin(2 * latr)
#     c2lat = np.cos(2 * latr)

#     # calculate r' and its derivatives
#     rp = a * (1 - ecc * slat * slat)
#     drp = -a * dlat * ecc * s2lat
#     ddrp = -a * ddlat * ecc * s2lat - 2 * a * dlat * dlat * ecc * c2lat

#     # calculate deviation from normal and derivatives
#     d = np.arctan(ecc * s2lat)
#     dd = 2 * dlat * ecc * c2lat
#     ddd = 2 * ddlat * ecc * c2lat - 4 * dlat * dlat * ecc * s2lat

#     # define r and its derivatives
#     r = np.vstack((-rp * np.sin(d), np.zeros(len(rp)), -rp * np.cos(d) - ht)).T
#     rdot = np.vstack(
#         (
#             -drp * np.sin(d) - rp * dd * np.cos(d),
#             np.zeros(len(rp)),
#             -drp * np.cos(d) + rp * dd * np.sin(d) - dht,
#         )
#     ).T
#     ci = (
#         -ddrp * np.sin(d)
#         - 2.0 * drp * dd * np.cos(d)
#         - rp * (ddd * np.cos(d) - dd * dd * np.sin(d))
#     )
#     ck = (
#         -ddrp * np.cos(d)
#         + 2.0 * drp * dd * np.sin(d)
#         + rp * (ddd * np.sin(d) + dd * dd * np.cos(d) - ddht)
#     )
#     rdotdot = np.vstack((ci, np.zeros(len(ci)), ck)).T

#     # define w and derivative
#     w = np.vstack(((dlon + omega) * clat, -dlat, -(dlon + omega) * slat)).T
#     wdot = np.vstack(
#         (
#             dlon * clat - (dlon + omega) * dlat * slat,
#             -ddlat,
#             -ddlon * slat - (dlon + omega) * dlat * clat,
#         )
#     ).T

#     w2xrdot = np.cross(2 * w, rdot)
#     wdotxr = np.cross(wdot, r)
#     wxr = np.cross(w, r)
#     wxwxr = np.cross(w, wxr)

#     # calculate wexwexre, centrifugal acceleration due to the Earth
#     re = np.vstack((-rp * np.sin(d), np.zeros(len(rp)), -rp * np.cos(d))).T
#     we = np.vstack((omega * clat, np.zeros(len(slat)), -omega * slat)).T
#     wexre = np.cross(we, re)
#     _wexwexre = np.cross(we, wexre)
#     wexr = np.cross(we, r)
#     wexwexr = np.cross(we, wexr)

#     # calculate total acceleration for the aircraft
#     accel = rdotdot + w2xrdot + wdotxr + wxwxr

#     # Eotvos correction is the vertical component of the total acceleration of
#     # the aircraft minus the centrifugal acceleration of the Earth, convert to mGal
#     return (accel[:, 2] - wexwexr[:, 2]) * 100e3


# def eotvos_correction_full(lon, lat, ht, samp, a=6378137.0, b=6356752.3142):
#     """ Full Eotvos correction in mGals.
#     From https://github.com/PFPE/shipgrav/blob/e84b33cc6fcfddf0e18736cd1205620d82e16e91/shipgrav/grav.py#L194

#     The Eotvos correction is the effect on measured gravity due to horizontal
#     motion over the Earth's surface.

#     This formulation is from Harlan (1968), "Eotvos Corrections for Airborne Gravimetry" in
#     *Journal of Geophysical Research*, 73(14), DOI: 10.1029/JB073i014p04675

#     Implementation modified from matlab script written by Sandra Preaux, NGS, NOAA August 24 2009

#     Components of the correction:

#     * rdoubledot
#     * angular acceleration of the reference frame
#     * corliolis
#     * centrifugal
#     * centrifugal acceleration of Earth

#     :param lon: longitudes in degrees
#     :type lon: array_like
#     :param lat: latitudes in degrees
#     :type lat: array_like
#     :param ht: elevation (referenced to sea level)
#     :type ht: array_like
#     :param samp: sampling rate
#     :type samp: float
#     :param a: major axis of ellipsoid (default is WGS84)
#     :type a: float, optional
#     :param b: minor axis of ellipsoid (default is WGS84)
#     :type b: float, optional

#     :return: **E** (*ndarray*), Eotvos correction in mGal
#     """
#     ell = boule.WGS84
#     ecc = (a-b)/a

#     latr = np.deg2rad(lat)
#     lonr = np.deg2rad(lon)

#     # get time derivatives of position
#     dlat = center_diff(latr, 1, samp)
#     ddlat = center_diff(latr, 2, samp)
#     dlon = center_diff(lonr, 1, samp)
#     ddlon = center_diff(lonr, 2, samp)
#     dht = center_diff(ht, 1, samp)
#     ddht = center_diff(ht, 2, samp)

#     # sines and cosines etc
#     slat = np.sin(latr[1:-1])
#     clat = np.cos(latr[1:-1])
#     s2lat = np.sin(2*latr[1:-1])
#     c2lat = np.cos(2*latr[1:-1])

#     # calculate r' and its derivatives
#     rp = a*(1 - ecc*slat*slat)
#     drp = -a*dlat*ecc*s2lat
#     ddrp = -a*ddlat*ecc*s2lat - 2*a*dlat*dlat*ecc*c2lat

#     # calculate deviation from normal and derivatives
#     D = np.arctan(ecc*s2lat)
#     dD = 2*dlat*ecc*c2lat
#     ddD = 2*ddlat*ecc*c2lat - 4*dlat*dlat*ecc*s2lat

#     # define r and its derivatives
#     r = np.vstack((-rp*np.sin(D), np.zeros(len(rp)), -
#                   rp*np.cos(D) - ht[1:-1])).T
#     rdot = np.vstack((-drp*np.sin(D) - rp*dD*np.cos(D),
#                      np.zeros(len(rp)), -drp*np.cos(D) + rp*dD*np.sin(D) - dht)).T
#     ci = -ddrp*np.sin(D) - 2.*drp*dD*np.cos(D) - rp * \
#         (ddD*np.cos(D) - dD*dD*np.sin(D))
#     ck = -ddrp*np.cos(D) + 2.*drp*dD*np.sin(D) + rp * \
#         (ddD*np.sin(D) + dD*dD*np.cos(D) - ddht)
#     rdotdot = np.vstack((ci, np.zeros(len(ci)), ck)).T

#     # define w and derivative
#     w = np.vstack(((dlon + ell.angular_velocity)*clat, -dlat, -(dlon + ell.angular_velocity)*slat)).T
#     wdot = np.vstack((dlon*clat - (dlon + ell.angular_velocity)*dlat*slat, -
#                      ddlat, -ddlon*slat - (dlon + ell.angular_velocity)*dlat*clat)).T

#     w2xrdot = np.cross(2*w, rdot)
#     wdotxr = np.cross(wdot, r)
#     wxr = np.cross(w, r)
#     wxwxr = np.cross(w, wxr)

#     # calculate wexwexre, centrifugal acceleration due to the Earth
#     re = np.vstack((-rp*np.sin(D), np.zeros(len(rp)), -rp*np.cos(D))).T
#     we = np.vstack((ell.angular_velocity*clat, np.zeros(len(slat)), -ell.angular_velocity*slat)).T
#     wexre = np.cross(we, re)
#     wexr = np.cross(we, r)
#     wexwexr = np.cross(we, wexr)

#     # calculate total acceleration for the aircraft
#     a = rdotdot + w2xrdot + wdotxr + wxwxr

#     # Eotvos correction is the vertical component of the total acceleration of
#     # the aircraft minus the centrifugal acceleration of the Earth, convert to mGal
#     E = (a[:, 2] - wexwexr[:, 2]) * 1e5
#     E = np.hstack((E[0], E, E[-1]))

#     return E
