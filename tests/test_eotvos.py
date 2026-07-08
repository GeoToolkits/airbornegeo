import boule
import numpy as np
import pytest

from airbornegeo.eotvos import (
    eotvos_correction_approx,
    eotvos_correction_full,
    eotvos_correction_glicken,
    eotvos_correction_harlan,
)


def test_glicken_zero_ground_speed_is_zero():
    """Zero ground speed should give exactly zero Eotvos correction."""
    lat = np.array([45.0, 45.0, 45.0])
    track = np.array([90.0, 90.0, 90.0])
    result = eotvos_correction_glicken(lat, track, np.zeros(3))
    assert result == pytest.approx([0.0, 0.0, 0.0])


def test_glicken_matches_hand_calculation():
    """The Glicken correction should match a direct hand calculation of its formula."""
    lat = np.array([0.0])
    track = np.array([90.0])
    gs_mps = np.array([100.0])
    result = eotvos_correction_glicken(lat, track, gs_mps)
    gs_knots = 100.0 * 1.94384
    expected = 7.503 * gs_knots * np.cos(0.0) * np.sin(np.radians(90.0)) + (
        0.004154 * gs_knots**2
    )
    assert result == pytest.approx([expected])


def test_glicken_single_point():
    """The Glicken correction should work for a single-point array input."""
    result = eotvos_correction_glicken(
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
    result = eotvos_correction_harlan(lat, lon, time, height, ground_speed=ground_speed)
    assert result == pytest.approx(np.zeros_like(result), abs=1e-8)


def test_full_stationary_aircraft_gives_zero():
    """A stationary aircraft should give zero correction from the full Harlan implementation."""
    lat, lon, time, height = _stationary()
    result = eotvos_correction_full(lat, lon, time, height)
    assert result == pytest.approx(np.zeros_like(result), abs=1e-8)


def test_approx_stationary_aircraft_gives_zero():
    """A stationary aircraft should give zero correction from the approximate implementation."""
    lat, lon, time, height = _stationary()
    result = eotvos_correction_approx(lat, lon, time, height)
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
    harlan_true = eotvos_correction_harlan(lat, lon, time, height, ground_speed=True)
    harlan_false = eotvos_correction_harlan(lat, lon, time, height, ground_speed=False)
    full = eotvos_correction_full(lat, lon, time, height)
    approx = eotvos_correction_approx(lat, lon, time, height)

    interior = slice(5, -5)
    assert harlan_true[interior] == pytest.approx(full[interior], rel=1e-4)
    assert harlan_false[interior] == pytest.approx(full[interior], rel=1e-4)
    assert approx[interior] == pytest.approx(full[interior], rel=1e-2)
    # physically sane magnitude for a ~70 m/s survey flight (verified regression value)
    assert np.mean(full[interior]) == pytest.approx(801.577, rel=1e-3)


def test_glicken_matches_harlan_order_of_magnitude_for_constant_velocity():
    """The simplified Glicken correction should agree with the full correction to within a few mGal."""
    lat, lon, time, height = _straight_line_east_flight()
    full = eotvos_correction_full(lat, lon, time, height)
    track = np.full(len(lat), 90.0)
    gs = np.full(len(lat), 70.0)
    glicken = eotvos_correction_glicken(lat, track, gs)
    assert glicken[0] == pytest.approx(np.mean(full[5:-5]), abs=5)


def test_harlan_single_point_raises_indexerror():
    """A single-point input should raise IndexError rather than a clean validation error."""
    with pytest.raises(IndexError):
        eotvos_correction_harlan(
            np.array([45.0]), np.array([10.0]), np.array([0.0]), np.array([500.0])
        )
