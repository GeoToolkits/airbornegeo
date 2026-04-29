import boule  # pylint: disable=too-many-lines
import harmonica as hm
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import NDArray
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


def eotvos_correction_full(
    latitude: NDArray,
    longitude: NDArray,
    height: NDArray,
    time: NDArray,
):
    """
    Parameters
    ----------
    TODO: for deviation, should height be 0 or observation height?

    Returns
    -------
    ndarray
        Eötvös correction in mGal
    """

    # define an ellipsoid
    ell = boule.WGS84

    # convert degrees to radians, and unwrap
    lat_rad = np.deg2rad(latitude)
    lon_rad = np.deg2rad(longitude)
    lon_rad = np.unwrap(lon_rad)

    # compute 1st and 2nd time derivatives of lat, lon, and heights
    # dot means time derivative
    # dt = np.diff(time, prepend=np.nan)
    # lat_dot = np.diff(lat_rad, prepend=np.nan) / dt
    # lat_dotdot = np.diff(lat_dot, prepend=np.nan) / dt
    # lon_dot = np.diff(lon_rad, prepend=np.nan) / dt
    # lon_dotdot = np.diff(lon_dot, prepend=np.nan) / dt
    # height_dot = np.diff(height, prepend=np.nan) / dt
    # height_dotdot = np.diff(height_dot, prepend=np.nan) / dt
    lat_dot = np.gradient(lat_rad, time)
    lat_dotdot = np.gradient(lat_dot, time)
    lon_dot = np.gradient(lon_rad, time)
    lon_dotdot = np.gradient(lon_dot, time)
    height_dot = np.gradient(height, time)
    height_dotdot = np.gradient(height_dot, time)

    # pre-compute trig functions
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_2lat = np.sin(2 * lat_rad)
    cos_2lat = np.cos(2 * lat_rad)

    # calculate r' and its derivatives (Eqs. 6,7,8)
    # r' is the vector from center of ellipsoid to point on ellipsoid directly (normal)
    # below the aircraft
    asquared_bsquared = ell.semimajor_axis**2 * ell.semiminor_axis**2
    bsquared_minus_asquared = ell.semiminor_axis**2 - ell.semimajor_axis**2
    k = asquared_bsquared * bsquared_minus_asquared
    g = ell.semimajor_axis**2 * cos_lat**2 + ell.semiminor_axis**2 * sin_lat**2
    r_prime = ell.geocentric_radius(latitude, coordinate_system="geodetic")
    r_prime_dot = (1 / (2 * r_prime)) * lat_dot * ((k * sin_2lat) / g**2)
    drdlat = (k * sin_2lat) / (2 * r_prime * g**2)
    d2rdlat2 = (k / (2 * r_prime * g**2)) * (
        (2 * cos_2lat)
        - (
            sin_2lat**2
            / (r_prime * g**2)
            * ((k / (2 * r_prime)) + 2 * r_prime * g * bsquared_minus_asquared)
        )
    )
    r_prime_dotdot = lat_dot**2 * d2rdlat2 + lat_dotdot * drdlat

    # below are approximations
    # r_prime = ell.semimajor_axis * (1 - (ell.flattening * sin_lat * sin_lat))
    # r_prime_dot = -ell.semimajor_axis * lat_dot * ell.flattening * sin_2lat
    # r_prime_dotdot = (-ell.semimajor_axis * lat_dotdot * ell.flattening * sin_2lat) - (2 * ell.semimajor_axis * lat_dot * lat_dot * ell.flattening * cos_2lat)

    # calculate deviation between geodetic and geocentric latitudes and derivatives (Eqs. 10,11,12)
    two_e_squared_minus_e_to_forth = 2 * ell.eccentricity**2 - ell.eccentricity**4
    dev = (
        latitude
        - ell.geodetic_to_spherical((None, latitude, np.zeros_like(latitude)))[1]
    )  # Fig 1 from Harlan 1968 shows dev as the angle with a height of 0 ...
    dev = np.deg2rad(dev)
    devdot = lat_dot * (
        (ell.eccentricity**2)
        - two_e_squared_minus_e_to_forth
        * sin_lat**2
        / (1 - two_e_squared_minus_e_to_forth * sin_lat**2)
    )
    term1 = (ell.eccentricity**2 - two_e_squared_minus_e_to_forth * sin_lat**2) / (
        1 - two_e_squared_minus_e_to_forth * sin_lat**2
    )
    term2 = ((1 - ell.eccentricity**2) * two_e_squared_minus_e_to_forth * sin_2lat) / (
        (1 - (two_e_squared_minus_e_to_forth * sin_lat**2)) ** 2
    )
    devdotdot = term1 * lat_dotdot - term2 * lat_dot**2

    # below are approximations
    # dev = np.arctan(ell.flattening * sin_2lat) # eq. 9 is D = arctan(flattening * sin2lat)
    # devdot = 2 * lat_dot * ell.flattening * cos_2lat
    # devdotdot = 2 * lat_dotdot * ell.flattening * cos_2lat - 4 * lat_dot * lat_dot * ell.flattening * sin_2lat

    # pre-compute trig functions
    sin_dev = np.sin(dev)
    cos_dev = np.cos(dev)

    # define r and its derivatives (Eq. 2)
    r = np.vstack(
        (
            -r_prime * sin_dev,
            np.zeros(len(r_prime)),
            -r_prime * cos_dev - height,
        )
    ).T

    rdot = np.vstack(
        (
            -r_prime_dot * sin_dev - r_prime * devdot * cos_dev,
            np.zeros(len(r_prime)),
            -r_prime_dot * cos_dev + r_prime * devdot * sin_dev - height_dot,
        )
    ).T

    rdotdot_x = (
        -r_prime_dotdot * sin_dev
        - 2.0 * r_prime_dot * devdot * cos_dev
        - r_prime * (devdotdot * cos_dev - devdot**2 * sin_dev)
    )
    rdotdot_z = (
        -r_prime_dotdot * cos_dev
        + 2.0 * r_prime_dot * devdot * sin_dev
        + r_prime * (devdotdot * sin_dev + devdot**2 * cos_dev)
        - height_dotdot
    )
    rdotdot = np.vstack(
        (
            rdotdot_x,
            np.zeros(len(rdotdot_x)),
            rdotdot_z,
        )
    ).T

    # define w and derivative (Eq. 3)
    w = np.vstack(
        (
            (lon_dot + ell.angular_velocity) * cos_lat,
            -lat_dot,
            -(lon_dot + ell.angular_velocity) * sin_lat,
        )
    ).T
    wdot = np.vstack(
        (
            lon_dotdot * cos_lat - (lon_dot + ell.angular_velocity) * lat_dot * sin_lat,
            -lat_dotdot,
            -lon_dotdot * sin_lat
            - (lon_dot + ell.angular_velocity) * lat_dot * cos_lat,
        )
    ).T

    # total acceleration of the aircraft Eq.1 from Harlan 1968
    w2xrdot = np.cross(2 * w, rdot)
    wdotxr = np.cross(wdot, r)
    wxr = np.cross(w, r)
    wxwxr = np.cross(w, wxr)
    accel = rdotdot + w2xrdot + wdotxr + wxwxr

    # calculate wexwexre, centrifugal acceleration due to the Earth
    we = np.vstack(
        (
            ell.angular_velocity * cos_lat,
            np.zeros(len(sin_lat)),
            -ell.angular_velocity * sin_lat,
        )
    ).T
    wexr = np.cross(we, r)
    wexwexr = np.cross(we, wexr)

    # Eotvos correction is the vertical component of the total acceleration of
    # the aircraft minus the centrifugal acceleration of the Earth
    eotvos_corr = accel[:, 2] - wexwexr[:, 2]

    # eotvos correction in mGal
    return 1e5 * eotvos_corr


def eotvos_correction_approx(
    latitude: NDArray,
    longitude: NDArray,
    height: NDArray,
    time: NDArray,
):
    """
    Parameters
    ----------

    Returns
    -------
    ndarray
        Eötvös correction in mGal
    """

    # define an ellipsoid
    ell = boule.WGS84

    # convert degrees to radians, and unwrap
    lat_rad = np.deg2rad(latitude)
    lon_rad = np.deg2rad(longitude)
    lon_rad = np.unwrap(lon_rad)

    # compute 1st and 2nd time derivatives of lat, lon, and heights
    # dot means time derivative
    # dt = np.diff(time, prepend=np.nan)
    # lat_dot = np.diff(lat_rad, prepend=np.nan) / dt
    # lon_dot = np.diff(lon_rad, prepend=np.nan) / dt
    # height_dot = np.diff(height, prepend=np.nan) / dt
    # lat_dotdot = np.diff(lat_dot, prepend=np.nan) / dt
    # lon_dotdot = np.diff(lon_dot, prepend=np.nan) / dt
    # height_dotdot = np.diff(height_dot, prepend=np.nan) / dt
    lat_dot = np.gradient(lat_rad, time)
    lat_dotdot = np.gradient(lat_dot, time)
    lon_dot = np.gradient(lon_rad, time)
    lon_dotdot = np.gradient(lon_dot, time)
    height_dot = np.gradient(height, time)
    height_dotdot = np.gradient(height_dot, time)

    # pre-compute trig functions
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_2lat = np.sin(2 * lat_rad)
    cos_2lat = np.cos(2 * lat_rad)

    # calculate r' and its derivatives (Eqs. 6,7,8)
    # r' is the vector from center of ellipsoid to point on ellipsoid directly (normal)
    # below the aircraft
    r_prime = ell.semimajor_axis * (
        1 - ell.flattening * sin_lat**2
    )  # Eq 6, simplification
    r_prime_dot = (
        -ell.semimajor_axis * lat_dot * ell.flattening * sin_2lat
    )  # Eq 7, simplification
    r_prime_dotdot = (
        -ell.semimajor_axis * lat_dotdot * ell.flattening * sin_2lat
        - 2 * ell.semimajor_axis * lat_dot**2 * ell.flattening * cos_2lat
    )  # Eq 8, simplification

    # calculate deviation from normal and derivatives (Eqs. 10,11,12)
    d = np.arctan(ell.flattening * sin_2lat)  # Eq 10, simplification
    ddot = 2 * lat_dot * ell.flattening * cos_2lat  # Eq 11, simplification
    ddotdot = (
        2 * lat_dotdot * ell.flattening * cos_2lat
        - 4 * lat_dot**2 * ell.flattening * sin_2lat
    )  # Eq 12, simplification

    # pre-compute trig functions
    sin_d = np.sin(d)
    cos_d = np.cos(d)

    # define r and its derivatives
    # r is vector
    r = np.vstack(
        (  # Eq 2
            -r_prime * sin_d,
            np.zeros(len(r_prime)),
            -r_prime * cos_d - height,
        )
    ).T
    rdot = np.vstack(
        (  # this is just time derivative of above components
            -r_prime_dot * sin_d - r_prime * ddot * cos_d,
            np.zeros(len(r_prime)),
            -r_prime_dot * cos_d + r_prime * ddot * sin_d - height_dot,
        )
    ).T
    ci = (
        -r_prime_dotdot * sin_d
        - 2.0 * r_prime_dot * ddot * cos_d
        - r_prime * (ddotdot * cos_d - ddot * ddot * sin_d)
    )
    ck = (
        -r_prime_dotdot * cos_d
        + 2.0 * r_prime_dot * ddot * sin_d
        + r_prime * (ddotdot * sin_d + ddot * ddot * cos_d)
        - height_dotdot
    )
    rdotdot = np.vstack(
        (ci, np.zeros(len(ci)), ck)
    ).T  # this is just 2nd time derivative of above components

    # define w and derivative (Eq. 3)
    w = np.vstack(
        (
            (lon_dot + ell.angular_velocity) * cos_lat,
            -lat_dot,
            -(lon_dot + ell.angular_velocity) * sin_lat,
        )
    ).T
    wdot = np.vstack(
        (
            lon_dotdot * cos_lat  # I think the first lon_dot should be lon_dotdot!
            - (lon_dot + ell.angular_velocity) * lat_dot * sin_lat,
            -lat_dotdot,
            -lon_dotdot * sin_lat
            - (lon_dot + ell.angular_velocity) * lat_dot * cos_lat,
        )
    ).T

    # total acceleration of the aircraft Eq.1 from Harlan 1968
    w2xrdot = np.cross(2 * w, rdot)
    wdotxr = np.cross(wdot, r)
    wxr = np.cross(w, r)
    wxwxr = np.cross(w, wxr)
    accel = rdotdot + w2xrdot + wdotxr + wxwxr

    # calculate wexwexre, centrifugal acceleration due to the Earth
    we = np.vstack(
        (
            ell.angular_velocity * cos_lat,
            np.zeros(len(sin_lat)),
            -ell.angular_velocity * sin_lat,
        )
    ).T
    wexr = np.cross(we, r)
    wexwexr = np.cross(we, wexr)

    # Eotvos correction is the vertical component of the total acceleration of
    # the aircraft minus the centrifugal acceleration of the Earth
    eotvos_corr = accel[:, 2] - wexwexr[:, 2]

    # eotvos correction in mGal
    return 1e5 * eotvos_corr
    # return eotvos_corr


def eotvos_correction_glicken(
    latitude: NDArray,
    track: NDArray,
    ground_speed: NDArray,
) -> NDArray:
    """
    Compute the simple Eötvös correction from latitude, aircraft track and ground
    speed using the Glicken 1962 simplified equation.

    Parameters
    ----------
    latitude : ndarray
        Latitude in decimal degrees
    track : ndarray
        Aircraft track in decimal degrees from geographic north, positive clock-wise
    ground_speed : array-like
        Ground speed in meters per second

    Returns
    -------
    ndarray
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


def eotvos_correction_harlan_track(
    latitude: NDArray,
    track: NDArray,
    ground_speed: NDArray,
    height: NDArray,
):
    """
    Compute the Eötvös correction from latitude, track, ground speed, and height
    using equation 15 from Harlan 1968 .

    Parameters
    ----------
    latitude : ndarray
        Latitude in degrees
    track : ndarray
        Aircraft track in degrees from geographic north, positive clock-wise
    ground_speed : ndarray
        Ground speed in meters per seconds
    height : ndarray
        Aircraft height in meters

    Returns
    -------
    E : ndarray
        Eötvös correction in mGal
    """

    # define an ellipsoid
    ell = boule.WGS84

    # degrees to radian
    lat_rad = np.radians(latitude)
    track_rad = np.radians(track)

    # # Equation 16 from Harlan
    # return 1e5 * (((ground_speed**2) / ell.semimajor_axis) * (
    #     1
    #     + height / ell.semimajor_axis
    #     - ell.flattening * (1 - np.cos(lat_rad) ** 2 * (3 - 2 * np.sin(track_rad) ** 2))
    # ) + 2 * ground_speed * ell.angular_velocity * np.cos(lat_rad) * np.sin(
    #     track_rad
    # ) * (1 + height / ell.semimajor_axis))

    # # Equation 18 from Harlan
    # return 1e5 * (((ground_speed**2) / ell.semimajor_axis) * (
    #     1
    #     + height / ell.semimajor_axis
    #     - ell.flattening * (1 - np.cos(lat_rad) ** 2 * (3 - 2 * np.sin(track_rad) ** 2))
    # ) + 2 * ground_speed * ell.angular_velocity * np.cos(lat_rad) * np.sin(
    #     track_rad
    # ))

    # Equation 15 from Harlan
    velocity_easting = np.sin(track_rad) * ground_speed
    velocity_northing = np.cos(track_rad) * ground_speed
    return (
        1e5
        * (
            ((velocity_northing**2) / ell.semimajor_axis)
            * (
                1
                + (height / ell.semimajor_axis)
                + ell.flattening * (2 - 3 * np.sin(lat_rad) ** 2)
            )
            + ((velocity_easting**2) / ell.semimajor_axis)
            * (
                1
                + (height / ell.semimajor_axis)
                - ell.flattening * (np.sin(lat_rad) ** 2)
            )
            + 2 * ell.angular_velocity * velocity_easting * np.cos(lat_rad)
        )
        * (1 + (height / ell.semimajor_axis))
    )


def eotvos_correction_harlan_velocity(
    latitude: NDArray,
    velocity_latitudinal: NDArray,
    velocity_longitudinal: NDArray,
    height: NDArray,
):
    """
    Compute the complete Eötvös correction from latitude, latitude and longitude
    components of aircraft velocity, and height. This follows equation 17 of Harlan 1968
    which assumes the velocities are given as aircraft velocities at altitude, not
    ground velocities.

    Parameters
    ----------
    latitude : ndarray
        Latitude in decimal degrees
    velocity_latitudinal : ndarray
        Latitudinal component of aircraft ground velocity in decimal degrees per second
    velocity_latitudinal : ndarray
        Longitudinal component of aircraft ground velocity in decimal degrees per second
    height : ndarray
        Aircraft height in meters

    Returns
    -------
    ndarray
        Eötvös correction in mGal
    """

    # define an ellipsoid
    ell = boule.WGS84

    # degrees to radian
    lat_rad = np.radians(latitude)
    velocity_lat_rad = np.radians(velocity_latitudinal)
    velocity_lon_rad = np.radians(velocity_longitudinal)

    sin_lat2 = np.sin(lat_rad) ** 2
    ecc2 = ell.first_eccentricity**2
    r = ell.semimajor_axis

    # local radii of curvature
    denom = 1 - ecc2 * sin_lat2
    curvature_meridional = r * (1 - ecc2) / (denom**1.5)
    curvature_normal = r / np.sqrt(denom)

    # velocity components
    velocity_northing = (curvature_meridional + height) * velocity_lat_rad
    velocity_easting_height = (
        (curvature_normal + height) * np.cos(lat_rad) * velocity_lon_rad
    )

    # northerly centrifugal term
    term_n = (velocity_northing**2 / r) * (
        1 - (height / r) + ell.flattening * (2 - 3 * sin_lat2)
    )

    # easterly centrifugal term
    term_e = (velocity_easting_height**2 / r) * (
        1 - (height / r) - ell.flattening * sin_lat2
    )

    # coriolis term
    term_coriolis = 2 * ell.angular_velocity * velocity_easting_height * np.cos(lat_rad)

    # eotvos correction in mGal
    return 1e5 * (term_n + term_e + term_coriolis)


# def central_difference(data_in, n=1, order=2, dt=0.1):
#     """central difference differentiator"""
#     if order == 2:
#         # first derivative
#         if n == 1:
#             dy = (data_in[2:] - data_in[0:-2]) / (2 * dt)
#         # second derivative
#         elif n == 2:
#             dy = (data_in[0:-2] - 2 * data_in[1:-1] + data_in[2:]) / np.power(dt, 2)
#         else:
#             raise ValueError("Invalid value for parameter n {1 or 2}")
#     else:
#         raise NotImplementedError

#     return np.pad(dy, (1, 1), "edge")


# def eotvos_correction_dgs(
#     latitude: NDArray,
#     longitude: NDArray,
#     height: NDArray,
#     dt,
#     differentiator=central_difference,
# ):
#     """
#     from https://github.com/DynamicGravitySystems/DGP/blob/5c0b566b846eb25f1e5ede64b2caaaa6a3352a29/dgp/lib/transform/gravity.py#L73
#     I changed sampling rate to match my data
#     final eotvos correction is column eotvos plus column kin_accel
#     I changed variables names to match with the shipgrav implementation to compare
#     I changed to output eotvos correction, which is columns eotvos + kin_accel
#     I added sampling rate (dt) as parameter
#     I changed inputs to be same as other functions

#     Eotvos correction

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

#     # dt = 0.1

#     # constants # added these to the function
#     a = 6378137.0  # Default semi-major axis
#     b = 6356752.3142  # Default semi-minor axis
#     ecc = (a - b) / a  # Eccentricity
#     omega = 0.00007292115  # sidereal rotation rate, radians/sec
#     mps2mgal = 100000  # m/s/s to mgal

#     latr = np.deg2rad(latitude)
#     lonr = np.deg2rad(longitude)
#     # lonr = np.unwrap(lonr)

#     dlat = differentiator(latr, n=1, dt=dt)
#     ddlat = differentiator(latr, n=2, dt=dt)
#     dlon = differentiator(lonr, n=1, dt=dt)
#     ddlon = differentiator(lonr, n=2, dt=dt)
#     dht = differentiator(height, n=1, dt=dt)
#     ddht = differentiator(height, n=2, dt=dt)

#     sin_lat = np.sin(latr)
#     cos_lat = np.cos(latr)
#     sin_2lat = np.sin(2.0 * latr)
#     cos_2lat = np.cos(2.0 * latr)

#     # Calculate the r' and its derivatives
#     r_prime = a * (1.0 - ecc * sin_lat**2)
#     dr_prime = -a * dlat * ecc * sin_2lat
#     ddr_prime = -a * ddlat * ecc * sin_2lat - 2.0 * a * (dlat**2) * ecc * cos_2lat

#     # Calculate the deviation from the normal and its derivatives
#     d = np.arctan(ecc * sin_2lat)
#     dd = 2.0 * dlat * ecc * cos_2lat
#     ddd = 2.0 * ddlat * ecc * cos_2lat - 4.0 * dlat * dlat * ecc * sin_2lat

#     # Calculate this value once (used many times)
#     sinD = np.sin(d)
#     cosD = np.cos(d)

#     # Calculate r and its derivatives
#     r = np.array([-r_prime * sinD, np.zeros(r_prime.size), -r_prime * cosD - height])

#     rdot = np.array(
#         [
#             (-dr_prime * sinD - r_prime * dd * cosD),
#             np.zeros(r_prime.size),
#             (-dr_prime * cosD + r_prime * dd * sinD - dht),
#         ]
#     )

#     ci = (
#         -ddr_prime * sinD
#         - 2.0 * dr_prime * dd * cosD
#         - r_prime * (ddd * cosD - dd * dd * sinD)
#     )
#     ck = (
#         -ddr_prime * cosD
#         + 2.0 * dr_prime * dd * sinD
#         + r_prime * (ddd * sinD + dd * dd * cosD)
#         - ddht  # here ddht is correctly outsite parenthesis, in shipgrap its inside
#     )
#     rdotdot = np.array([ci, np.zeros(ci.size), ck])

#     # Define w and its derivative
#     w = np.array([(dlon + omega) * cos_lat, -dlat, (-(dlon + omega)) * sin_lat])

#     wdot = np.array(
#         [
#             ddlon * cos_lat  # first term, dlon, should be ddlon
#             - (dlon + omega) * dlat * sin_lat,
#             -ddlat,
#             (-ddlon * sin_lat - (dlon + omega) * dlat * cos_lat),
#         ]
#     )

#     w2xrdot = np.cross(2.0 * w, rdot, axis=0)
#     wdotxr = np.cross(wdot, r, axis=0)
#     wxr = np.cross(w, r, axis=0)
#     wxwxr = np.cross(w, wxr, axis=0)

#     we = np.array([omega * cos_lat, np.zeros(sin_lat.shape), -omega * sin_lat])
#     wexr = np.cross(we, r, axis=0)
#     wexwexr = np.cross(we, wexr, axis=0)

#     kin_accel = rdotdot * mps2mgal
#     eotvos = (w2xrdot + wdotxr + wxwxr - wexwexr) * mps2mgal

#     # acc = rdotdot + w2xrdot + wdotxr + wxwxr

#     # eotvos = pd.Series(eotvos[2], index=data_in.index, name='eotvos')
#     # kin_accel = pd.Series(kin_accel[2], index=data_in.index, name='kin_accel')

#     # df = pd.concat([eotvos, kin_accel], axis=1, join='outer')

#     return eotvos[2] + kin_accel[2]


# def eotvos_correction_shipgrav(
#     latitude: NDArray,
#     longitude: NDArray,
#     height: NDArray,
#     time: NDArray,
# ) -> pd.Series:
#     """
#     This is my re-writting of the below `eotvos_correction_shipgrav` function, which is
#     directly from the shipgrav python package. They assume a constant sampling rate
#     between points, but here I've re-written to allow varying sampling rates.
#     I changed variable names to match with DGS package implementation
#     ck and wdot both had mistakes I fixed
#     I changed inputs to match format of other functions
#     """
#     a = 6378137.0
#     b = 6356752.3142

#     omega = 0.00007292115  # siderial rotation rate, radians/sec
#     ecc = (a - b) / a  # this is referred to as flattening in Boule

#     latr = np.deg2rad(latitude)
#     lonr = np.deg2rad(longitude)
#     lonr = np.unwrap(lonr)

#     # get time derivatives of position
#     # dt = np.diff(time, prepend=np.nan)
#     # dlat = np.diff(latr, prepend=np.nan) / dt
#     # dlon = np.diff(lonr, prepend=np.nan) / dt
#     # dht = np.diff(height, prepend=np.nan) / dt
#     # ddlat = np.diff(dlat, prepend=np.nan) / dt
#     # ddlon = np.diff(dlon, prepend=np.nan) / dt
#     # ddht = np.diff(dht, prepend=np.nan) / dt

#     dlat = np.gradient(latr, time)
#     ddlat = np.gradient(dlat, time)
#     dlon = np.gradient(lonr, time)
#     ddlon = np.gradient(dlon, time)
#     dht = np.gradient(height, time)
#     ddht = np.gradient(dht, time)

#     # sines and cosines etc
#     sin_lat = np.sin(latr)
#     cos_lat = np.cos(latr)
#     sin_2lat = np.sin(2 * latr)
#     cos_2lat = np.cos(2 * latr)

#     # calculate r' and its derivatives
#     r_prime = a * (1 - ecc * sin_lat * sin_lat)
#     dr_prime = -a * dlat * ecc * sin_2lat
#     ddr_prime = -a * ddlat * ecc * sin_2lat - 2 * a * dlat * dlat * ecc * cos_2lat

#     # calculate deviation from normal and derivatives
#     d = np.arctan(ecc * sin_2lat)  # eq. 9 is D = arctan(flattening * sin2lat)
#     dd = 2 * dlat * ecc * cos_2lat
#     ddd = 2 * ddlat * ecc * cos_2lat - 4 * dlat * dlat * ecc * sin_2lat

#     # define r and its derivatives
#     # r is the vector from the earth's center to the aircraft
#     r = np.vstack(
#         (
#             -r_prime * np.sin(d),
#             np.zeros(len(r_prime)),
#             -r_prime * np.cos(d) - height,
#         )
#     ).T
#     rdot = np.vstack(
#         (
#             -dr_prime * np.sin(d) - r_prime * dd * np.cos(d),
#             np.zeros(len(r_prime)),
#             -dr_prime * np.cos(d) + r_prime * dd * np.sin(d) - dht,
#         )
#     ).T
#     ci = (
#         -ddr_prime * np.sin(d)
#         - 2.0 * dr_prime * dd * np.cos(d)
#         - r_prime * (ddd * np.cos(d) - dd * dd * np.sin(d))
#     )
#     ck = (
#         -ddr_prime * np.cos(d)
#         + 2.0 * dr_prime * dd * np.sin(d)
#         # + r_prime * (ddd * np.sin(d) + dd * dd * np.cos(d) - ddht) # this is wrong, ddht should be outside parenthesis
#         + r_prime * (ddd * np.sin(d) + dd * dd * np.cos(d))
#         - ddht  # this is correct
#     )
#     rdotdot = np.vstack((ci, np.zeros(len(ci)), ck)).T

#     # define w and derivative
#     w = np.vstack(
#         (
#             (dlon + omega) * cos_lat,
#             -dlat,
#             -(dlon + omega) * sin_lat,
#         )
#     ).T
#     wdot = np.vstack(
#         (
#             ddlon * cos_lat
#             - (dlon + omega)
#             * dlat
#             * sin_lat,  # changed first term dlong to ddlon, no dicernable difference
#             -ddlat,
#             -ddlon * sin_lat - (dlon + omega) * dlat * cos_lat,
#         )
#     ).T

#     w2xrdot = np.cross(2 * w, rdot)
#     wdotxr = np.cross(wdot, r)
#     wxr = np.cross(w, r)
#     wxwxr = np.cross(w, wxr)

#     # calculate wexwexre, centrifugal acceleration due to the Earth
#     we = np.vstack(
#         (
#             omega * cos_lat,
#             np.zeros(len(sin_lat)),
#             -omega * sin_lat,
#         )
#     ).T
#     wexr = np.cross(we, r)
#     wexwexr = np.cross(we, wexr)

#     # calculate total acceleration for the aircraft
#     accel = rdotdot + w2xrdot + wdotxr + wxwxr

#     # Eotvos correction is the vertical component of the total acceleration of
#     # the aircraft minus the centrifugal acceleration of the Earth, convert to mGal
#     return (accel[:, 2] - wexwexr[:, 2]) * 100e3


# def center_diff(y, n, samp):
#     """ Numerical derivatives, central difference of nth order

#     :param y: data to differentiate
#     :type y: array_like
#     :param n: order, either 1 or 2
#     :type n: int
#     :param samp: sampling rate
#     :type samp: float

#     :return: 1st or 2nd order derivative of **y**
#     """
#     if n == 1:
#         return (y[2:] - y[:-2])*(samp/2)
#     elif n == 2:
#         return (y[:-2] - 2*y[1:-1] + y[2:])*(samp**2)
#     else:
#         print('bad order for derivative')
#         return -999


# def eotvos_correction_shipgrav(lon, lat, ht, samp, a=6378137.0, b=6356752.3142):
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
#     We = 0.00007292115    # siderial rotation rate, radians/sec
#     mps2mgal = 100000     # m/s/s to mgal
#     ecc = (a-b)/a # this is referred to as flattening in Boule

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
#     w = np.vstack(((dlon + We)*clat, -dlat, -(dlon + We)*slat)).T
#     wdot = np.vstack((dlon*clat - (dlon + We)*dlat*slat, - # should 1st term be ddlon??
#                      ddlat, -ddlon*slat - (dlon + We)*dlat*clat)).T

#     w2xrdot = np.cross(2*w, rdot)
#     wdotxr = np.cross(wdot, r)
#     wxr = np.cross(w, r)
#     wxwxr = np.cross(w, wxr)

#     # calculate wexwexre, centrifugal acceleration due to the Earth
#     # re = np.vstack((-rp*np.sin(D), np.zeros(len(rp)), -rp*np.cos(D))).T # this is actually used, comment out?
#     we = np.vstack((We*clat, np.zeros(len(slat)), -We*slat)).T
#     # wexre = np.cross(we, re) # this is actually used, comment out?
#     # wexwexre = np.cross(we, wexre) # this is actually used, comment out?
#     wexr = np.cross(we, r)
#     wexwexr = np.cross(we, wexr)

#     # calculate total acceleration for the aircraft
#     a = rdotdot + w2xrdot + wdotxr + wxwxr

#     # Eotvos correction is the vertical component of the total acceleration of
#     # the aircraft minus the centrifugal acceleration of the Earth, convert to mGal
#     E = (a[:, 2] - wexwexr[:, 2])*mps2mgal
#     E = np.hstack((E[0], E, E[-1]))

#     return E
