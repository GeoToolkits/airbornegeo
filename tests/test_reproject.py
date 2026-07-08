import numpy as np
import pandas as pd
import pyproj
import pytest

from airbornegeo.reproject import reproject


def test_reproject_wgs84_to_utm():
    """Reproject WGS84 lon/lat coordinates to a UTM projection and check the exact values."""
    lon = np.array([10.0, 10.5, 11.0])
    lat = np.array([50.0, 50.1, 50.2])
    x, y = reproject(lon, lat, input_crs="EPSG:4326", output_crs="EPSG:32632")
    assert x == pytest.approx([571666.44750344, 607275.30981045, 642733.45012827])
    assert y == pytest.approx([5539109.8152988, 5550826.64580076, 5562782.28153253])


def test_reproject_round_trip():
    """Reprojecting to UTM and back to WGS84 should recover the original coordinates."""
    lon = np.array([10.0, 10.5, 11.0])
    lat = np.array([50.0, 50.1, 50.2])
    x, y = reproject(lon, lat, input_crs="EPSG:4326", output_crs="EPSG:32632")
    lon2, lat2 = reproject(x, y, input_crs="EPSG:32632", output_crs="EPSG:4326")
    assert lon2 == pytest.approx(lon)
    assert lat2 == pytest.approx(lat)


def test_reproject_crs_string_is_case_insensitive():
    """Lowercase and uppercase EPSG CRS strings should produce identical results."""
    lon = np.array([10.0, 10.5])
    lat = np.array([50.0, 50.1])
    x1, y1 = reproject(lon, lat, input_crs="EPSG:4326", output_crs="EPSG:32632")
    x2, y2 = reproject(lon, lat, input_crs="epsg:4326", output_crs="epsg:32632")
    assert x1 == pytest.approx(x2)
    assert y1 == pytest.approx(y2)


def test_reproject_accepts_pandas_series():
    """Passing pandas Series instead of numpy arrays should give the same result."""
    lon = np.array([10.0, 10.5, 11.0])
    lat = np.array([50.0, 50.1, 50.2])
    x_array, y_array = reproject(
        lon, lat, input_crs="EPSG:4326", output_crs="EPSG:32632"
    )
    x_series, y_series = reproject(
        pd.Series(lon), pd.Series(lat), input_crs="EPSG:4326", output_crs="EPSG:32632"
    )
    assert np.asarray(x_series) == pytest.approx(x_array)
    assert np.asarray(y_series) == pytest.approx(y_array)


def test_reproject_invalid_crs_raises():
    """An unrecognized CRS string should raise a pyproj CRSError."""
    lon = np.array([10.0])
    lat = np.array([50.0])
    with pytest.raises(pyproj.exceptions.CRSError):
        reproject(lon, lat, input_crs="not_a_crs", output_crs="EPSG:4326")
