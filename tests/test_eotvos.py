import boule
import numpy as np
import pandas as pd
import pytest

from airbornegeo.eotvos import (
    _eotvos_correction_approx,
    _eotvos_correction_full,
    _eotvos_correction_glicken,
    _eotvos_correction_harlan,
    eotvos_correction,
)


def test_glicken_zero_ground_speed_is_zero():
    """Zero ground speed should give exactly zero Eotvos correction."""
    lat = np.array([45.0, 45.0, 45.0])
    track = np.array([90.0, 90.0, 90.0])
    result = _eotvos_correction_glicken(lat, track, np.zeros(3))
    assert result == pytest.approx([0.0, 0.0, 0.0])


def test_glicken_matches_hand_calculation():
    """The Glicken correction should match a direct hand calculation of its formula."""
    lat = np.array([0.0])
    track = np.array([90.0])
    gs_mps = np.array([100.0])
    result = _eotvos_correction_glicken(lat, track, gs_mps)
    gs_knots = 100.0 * 1.94384
    expected = 7.503 * gs_knots * np.cos(0.0) * np.sin(np.radians(90.0)) + (
        0.004154 * gs_knots**2
    )
    assert result == pytest.approx([expected])


def test_glicken_single_point():
    """The Glicken correction should work for a single-point array input."""
    result = _eotvos_correction_glicken(
        np.array([45.0]), np.array([90.0]), np.array([50.0])
    )
    assert result == pytest.approx([554.88445977], rel=1e-6)


def _stationary(n=20):
    time = np.linspace(0, 100, n)
    lat = np.full(n, 45.0)
    lon = np.full(n, 10.0)
    height = np.full(n, 500.0)
    return lat, lon, time, height


@pytest.mark.parametrize("ground_speed", [True, False])
def test_harlan_stationary_aircraft_gives_zero(ground_speed):
    """A stationary aircraft (constant lat/lon/height) should give zero correction, for both Harlan equations."""
    lat, lon, time, height = _stationary()
    result = _eotvos_correction_harlan(
        lat, lon, time, height, ground_speed=ground_speed
    )
    assert result == pytest.approx(np.zeros_like(result), abs=1e-8)


def test_full_stationary_aircraft_gives_zero():
    """A stationary aircraft should give zero correction from the full Harlan implementation."""
    lat, lon, time, height = _stationary()
    result = _eotvos_correction_full(lat, lon, time, height)
    assert result == pytest.approx(np.zeros_like(result), abs=1e-8)


def test_approx_stationary_aircraft_gives_zero():
    """A stationary aircraft should give zero correction from the approximate implementation."""
    lat, lon, time, height = _stationary()
    result = _eotvos_correction_approx(lat, lon, time, height)
    assert result == pytest.approx(np.zeros_like(result), abs=1e-8)


def _straight_line_east_flight(n=50, speed_mps=70.0, lat_deg=45.0):
    time = np.linspace(0, 500, n)
    lat = np.full(n, lat_deg)
    ell = boule.WGS84
    r = ell.geocentric_radius(lat_deg, coordinate_system="geodetic")
    dlon_dt = speed_mps / (r * np.cos(np.radians(lat_deg)))
    lon = 10.0 + np.degrees(dlon_dt) * time
    height = np.full(n, 500.0)
    return lat, lon, time, height


def test_harlan_full_approx_agree_for_constant_velocity_flight():
    """The four independent correction implementations should agree closely for a realistic constant-velocity flight."""
    lat, lon, time, height = _straight_line_east_flight()
    harlan_true = _eotvos_correction_harlan(lat, lon, time, height, ground_speed=True)
    harlan_false = _eotvos_correction_harlan(lat, lon, time, height, ground_speed=False)
    full = _eotvos_correction_full(lat, lon, time, height)
    approx = _eotvos_correction_approx(lat, lon, time, height)

    interior = slice(5, -5)
    assert harlan_true[interior] == pytest.approx(full[interior], rel=1e-4)
    assert harlan_false[interior] == pytest.approx(full[interior], rel=1e-4)
    assert approx[interior] == pytest.approx(full[interior], rel=1e-2)
    # physically sane magnitude for a ~70 m/s survey flight (verified regression value)
    assert np.mean(full[interior]) == pytest.approx(801.577, rel=1e-3)


def test_glicken_matches_harlan_order_of_magnitude_for_constant_velocity():
    """The simplified Glicken correction should agree with the full correction to within a few mGal."""
    lat, lon, time, height = _straight_line_east_flight()
    full = _eotvos_correction_full(lat, lon, time, height)
    track = np.full(len(lat), 90.0)
    gs = np.full(len(lat), 70.0)
    glicken = _eotvos_correction_glicken(lat, track, gs)
    assert glicken[0] == pytest.approx(np.mean(full[5:-5]), abs=5)


def test_harlan_single_point_raises_indexerror():
    """A single-point input should raise IndexError rather than a clean validation error."""
    with pytest.raises(IndexError):
        _eotvos_correction_harlan(
            np.array([45.0]), np.array([10.0]), np.array([0.0]), np.array([500.0])
        )


def _flight_frame(n=50):
    lat, lon, time, height = _straight_line_east_flight(n=n)
    return pd.DataFrame(
        {"lat": lat, "lon": lon, "unixtime": time, "height": height, "line": "L1"}
    )


_TRAJECTORY_COLUMNS = {
    "latitude_column": "lat",
    "longitude_column": "lon",
    "time_column": "unixtime",
    "height_column": "height",
}


@pytest.mark.parametrize(
    ("method", "implementation"),
    [
        ("full", _eotvos_correction_full),
        ("harlan", _eotvos_correction_harlan),
        ("approx", _eotvos_correction_approx),
    ],
)
def test_eotvos_correction_dispatches_to_its_implementation(method, implementation):
    """Each method name should give exactly what its implementation gives."""
    data = _flight_frame()
    result = eotvos_correction(
        data, method=method, progressbar=False, **_TRAJECTORY_COLUMNS
    )
    expected = implementation(
        data.lat.to_numpy(),
        data.lon.to_numpy(),
        data.unixtime.to_numpy(),
        data.height.to_numpy(),
    )
    assert result == pytest.approx(expected)


def test_eotvos_correction_passes_kwargs_to_implementation():
    """Extra kwargs should reach the implementation, e.g. Harlan's ground_speed."""
    data = _flight_frame()
    result = eotvos_correction(
        data,
        method="harlan",
        ground_speed=False,
        progressbar=False,
        **_TRAJECTORY_COLUMNS,
    )
    expected = _eotvos_correction_harlan(
        data.lat.to_numpy(),
        data.lon.to_numpy(),
        data.unixtime.to_numpy(),
        data.height.to_numpy(),
        ground_speed=False,
    )
    assert result == pytest.approx(expected)


def test_eotvos_correction_glicken_uses_track_and_ground_speed_columns():
    """Method 'glicken' should read the track and ground speed columns."""
    data = _flight_frame()
    data["track"] = 90.0
    data["ground_speed"] = 70.0
    result = eotvos_correction(
        data,
        method="glicken",
        latitude_column="lat",
        track_column="track",
        ground_speed_column="ground_speed",
        progressbar=False,
    )
    expected = _eotvos_correction_glicken(
        data.lat.to_numpy(), data.track.to_numpy(), data.ground_speed.to_numpy()
    )
    assert result == pytest.approx(expected)


def test_eotvos_correction_groupby_matches_per_line_calls():
    """Grouping should give each line the value it gets when computed alone."""
    line1 = _flight_frame()
    line2 = _flight_frame()
    line2["line"] = "L2"
    line2["lat"] += 1.0
    data = pd.concat([line1, line2], ignore_index=True)

    grouped = eotvos_correction(
        data,
        method="full",
        groupby_column="line",
        progressbar=False,
        **_TRAJECTORY_COLUMNS,
    )

    for name, group in data.groupby("line"):
        alone = _eotvos_correction_full(
            group.lat.to_numpy(),
            group.lon.to_numpy(),
            group.unixtime.to_numpy(),
            group.height.to_numpy(),
        )
        assert grouped[data.line == name] == pytest.approx(alone), name


def test_eotvos_correction_without_groupby_differs_at_the_line_join():
    """Not grouping should corrupt the time derivatives where two lines meet."""
    line1 = _flight_frame()
    line2 = _flight_frame()
    line2["line"] = "L2"
    line2["lat"] += 1.0
    data = pd.concat([line1, line2], ignore_index=True)

    grouped = eotvos_correction(
        data,
        method="full",
        groupby_column="line",
        progressbar=False,
        **_TRAJECTORY_COLUMNS,
    )
    ungrouped = eotvos_correction(
        data, method="full", progressbar=False, **_TRAJECTORY_COLUMNS
    )

    join = len(line1)
    assert ungrouped[join] != pytest.approx(grouped[join])


def test_eotvos_correction_unknown_method_raises():
    with pytest.raises(ValueError, match="method must be one of"):
        eotvos_correction(_flight_frame(), method="bogus")


def test_eotvos_correction_missing_columns_names_them_and_usable_methods():
    """Choosing a method the dataframe can't support should say what's missing."""
    data = _flight_frame()
    with pytest.raises(ValueError, match="track_column") as error:
        eotvos_correction(data, method="glicken", latitude_column="lat")
    # the trajectory methods are still usable if their columns are named
    assert "ground_speed_column" in str(error.value)


def test_eotvos_correction_reports_alternative_usable_methods():
    data = _flight_frame()
    data["ground_speed"] = 70.0
    with pytest.raises(ValueError, match="usable with the columns given") as error:
        eotvos_correction(
            data,
            method="glicken",
            ground_speed_column="ground_speed",
            **_TRAJECTORY_COLUMNS,
        )
    assert sorted(("harlan", "full", "approx")) == sorted(
        method for method in ("harlan", "full", "approx") if method in str(error.value)
    )


def test_eotvos_correction_unknown_groupby_column_raises():
    with pytest.raises(ValueError, match="groupby_column 'nope'"):
        eotvos_correction(
            _flight_frame(),
            method="full",
            groupby_column="nope",
            **_TRAJECTORY_COLUMNS,
        )
