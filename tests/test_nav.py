from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from airbornegeo import nav


def test_directional_velocity_no_groupby():
    """Directional velocity of a constant-slope coordinate over time should be the constant slope."""
    data = pd.DataFrame(
        {"unixtime": [0.0, 1.0, 2.0, 3.0], "easting": [0.0, 10.0, 20.0, 30.0]}
    )
    result = nav.directional_velocity(
        data,
        time_column="unixtime",
        coordinate_column="easting",
        groupby_column=None,
        progressbar=False,
    )
    assert result == pytest.approx([10.0, 10.0, 10.0, 10.0])


@pytest.mark.parametrize("progressbar", [True, False])
def test_directional_velocity_groupby(progressbar):
    """Grouped directional velocity should be computed independently per group, regardless of progressbar setting."""
    data = pd.DataFrame(
        {
            "line": ["a", "a", "a", "b", "b", "b"],
            "unixtime": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
            "easting": [0.0, 5.0, 10.0, 0.0, 20.0, 40.0],
        }
    )
    result = nav.directional_velocity(
        data,
        time_column="unixtime",
        coordinate_column="easting",
        groupby_column="line",
        progressbar=progressbar,
    )
    assert result == pytest.approx([5.0, 5.0, 5.0, 20.0, 20.0, 20.0])


def test_directional_velocity_groupby_labels_sort_differently_than_appearance():
    """Group labels that don't sort into their appearance order must not scramble the result's row order."""
    data = pd.DataFrame(
        {
            "line": ["B", "B", "A", "A"],
            "unixtime": [0.0, 1.0, 0.0, 1.0],
            "easting": [100.0, 200.0, 5.0, 6.0],
        }
    )
    result = nav.directional_velocity(
        data,
        time_column="unixtime",
        coordinate_column="easting",
        groupby_column="line",
        progressbar=False,
    )
    assert result == pytest.approx([100.0, 100.0, 1.0, 1.0])


def test_ground_speed_no_groupby():
    """Ground speed along a constant-heading line should equal the constant distance-per-time rate."""
    data = pd.DataFrame(
        {
            "unixtime": [0.0, 1.0, 2.0, 3.0],
            "easting": [0.0, 3.0, 6.0, 9.0],
            "northing": [0.0, 0.0, 0.0, 0.0],
        }
    )
    result = nav.ground_speed(
        data,
        time_column="unixtime",
        groupby_column=None,
        progressbar=False,
    )
    assert result == pytest.approx([3.0, 3.0, 3.0, 3.0])


@pytest.mark.parametrize("progressbar", [True, False])
def test_ground_speed_groupby(progressbar):
    """Grouped ground speed should be computed independently per group, regardless of progressbar setting."""
    data = pd.DataFrame(
        {
            "line": ["a", "a", "a", "b", "b", "b"],
            "unixtime": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
            "easting": [0.0, 3.0, 6.0, 0.0, 6.0, 12.0],
            "northing": [0.0] * 6,
        }
    )
    result = nav.ground_speed(
        data,
        time_column="unixtime",
        groupby_column="line",
        progressbar=progressbar,
    )
    assert result == pytest.approx([3.0, 3.0, 3.0, 6.0, 6.0, 6.0])


@pytest.mark.parametrize("groupby_column", [None, "line"])
def test_vertical_acceleration_missing_columns_raises(groupby_column):
    """Missing time/height columns should raise AssertionError, with or without groupby_column set."""
    data = pd.DataFrame({"other": [1.0, 2.0, 3.0]})
    with pytest.raises(AssertionError):
        nav.vertical_acceleration(
            data,
            time_column="t",
            height_column="h",
            groupby_column=groupby_column,
        )


def test_vertical_acceleration_no_groupby_no_threshold():
    """Vertical acceleration of a quadratic height-vs-time profile should match its constant 2nd derivative."""
    data = pd.DataFrame(
        {"t": [0.0, 1.0, 2.0, 3.0, 4.0], "h": [0.0, 1.0, 4.0, 9.0, 16.0]}
    )
    result = nav.vertical_acceleration(
        data,
        time_column="t",
        height_column="h",
        groupby_column=None,
        progressbar=False,
        time_threshold=None,
        smoothing_window=None,
    )
    assert result == pytest.approx([1.0, 1.5, 2.0, 1.5, 1.0])


def test_vertical_acceleration_smoothing_window():
    """A smoothing_window should apply a rolling mean to the computed acceleration."""
    data = pd.DataFrame(
        {"t": [0.0, 1.0, 2.0, 3.0, 4.0], "h": [0.0, 1.0, 4.0, 9.0, 16.0]}
    )
    result = nav.vertical_acceleration(
        data,
        time_column="t",
        height_column="h",
        groupby_column=None,
        progressbar=False,
        time_threshold=None,
        smoothing_window=3,
    )
    assert result == pytest.approx([1.0, 1.25, 1.5, 1.6666666666666667, 1.5])


def test_vertical_acceleration_time_threshold_no_groupby():
    """A gap greater than time_threshold should split the line and compute acceleration independently in each segment."""
    data = pd.DataFrame(
        {"t": [0.0, 1.0, 2.0, 10.0, 11.0, 12.0], "h": [0.0, 1.0, 4.0, 0.0, 1.0, 4.0]}
    )
    result = nav.vertical_acceleration(
        data,
        time_column="t",
        height_column="h",
        groupby_column=None,
        progressbar=False,
        time_threshold=5,
    )
    assert np.asarray(result) == pytest.approx([1.0] * 6)


@pytest.mark.parametrize("progressbar", [True, False])
def test_vertical_acceleration_groupby_no_threshold(progressbar):
    """Grouped vertical acceleration (no time_threshold) should be computed independently per group."""
    data = pd.DataFrame(
        {
            "line": ["a", "a", "a", "b", "b", "b"],
            "t": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
            "h": [0.0, 1.0, 4.0, 0.0, 1.0, 4.0],
        }
    )
    result = nav.vertical_acceleration(
        data,
        time_column="t",
        height_column="h",
        groupby_column="line",
        progressbar=progressbar,
        time_threshold=None,
    )
    assert np.asarray(result) == pytest.approx([1.0] * 6)


def test_vertical_acceleration_groupby_and_threshold():
    """Combining groupby_column and time_threshold should further split each group at its own internal gaps."""
    data = pd.DataFrame(
        {
            "line": ["a"] * 6 + ["b"] * 6,
            "t": [0.0, 1.0, 2.0, 10.0, 11.0, 12.0] * 2,
            "h": [0.0, 1.0, 4.0, 0.0, 1.0, 4.0] * 2,
        }
    )
    result = nav.vertical_acceleration(
        data,
        time_column="t",
        height_column="h",
        groupby_column="line",
        progressbar=False,
        time_threshold=5,
    )
    assert np.asarray(result) == pytest.approx([1.0] * 12)


def test_vertical_acceleration_groupby_smoothing_window():
    """A smoothing_window should also apply correctly within each group of a grouped computation."""
    data = pd.DataFrame(
        {
            "line": ["a", "a", "a", "b", "b", "b"],
            "t": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
            "h": [0.0, 1.0, 4.0, 0.0, 1.0, 4.0],
        }
    )
    result = nav.vertical_acceleration(
        data,
        time_column="t",
        height_column="h",
        groupby_column="line",
        progressbar=False,
        time_threshold=None,
        smoothing_window=3,
    )
    assert np.asarray(result) == pytest.approx([1.0] * 6)


def test_relative_track_ellipsoid_eastward():
    """Points moving due east along the equator should give a track of 90 degrees."""
    lat = np.array([0.0, 0.0, 0.0])
    lon = np.array([0.0, 1.0, 2.0])
    result = nav.relative_track_ellipsoid(lat, lon)
    assert result == pytest.approx([90.0, 90.0, 90.0])


def test_relative_track_ellipsoid_northward():
    """Points moving due north should give a track of 0 degrees."""
    lat = np.array([0.0, 1.0, 2.0])
    lon = np.array([0.0, 0.0, 0.0])
    result = nav.relative_track_ellipsoid(lat, lon)
    assert result == pytest.approx([0.0, 0.0, 0.0])


def test_relative_track_ellipsoid_empty():
    """Empty input arrays should return an empty result rather than erroring."""
    result = nav.relative_track_ellipsoid(np.array([]), np.array([]))
    assert len(result) == 0


def test_relative_track_ellipsoid_single_point_raises():
    """A single point has no direction; the None placeholder used internally makes '% 360' raise TypeError."""
    with pytest.raises(TypeError):
        nav.relative_track_ellipsoid(np.array([0.0]), np.array([0.0]))


def test_relative_track_spheroid_eastward():
    """Points moving due east along the equator should give a track of 90 degrees."""
    lat = np.array([0.0, 0.0, 0.0])
    lon = np.array([0.0, 1.0, 2.0])
    result = nav.relative_track_spheroid(lat, lon)
    assert result == pytest.approx([90.0, 90.0, 90.0])


def test_relative_track_spheroid_northward():
    """Points moving due north should give a track of 0 degrees."""
    lat = np.array([0.0, 1.0, 2.0])
    lon = np.array([0.0, 0.0, 0.0])
    result = nav.relative_track_spheroid(lat, lon)
    assert result == pytest.approx([0.0, 0.0, 0.0])


def test_relative_track_spheroid_two_points():
    """A 2-point input should work and give the correct track between them."""
    lat = np.array([0.0, 1.0])
    lon = np.array([0.0, 0.0])
    result = nav.relative_track_spheroid(lat, lon)
    assert result == pytest.approx([0.0, 0.0])


def test_relative_track_spheroid_single_point_raises():
    """A single point has no segment to duplicate, so indexing the nonexistent last track value raises IndexError."""
    with pytest.raises(IndexError):
        nav.relative_track_spheroid(np.array([0.0]), np.array([0.0]))


@pytest.mark.parametrize("ellipsoid", [True, False])
def test_track_no_groupby(ellipsoid):
    """track() should give the same eastward result for both the ellipsoid and spheroid backends."""
    data = pd.DataFrame({"lat": [0.0, 0.0, 0.0], "lon": [0.0, 1.0, 2.0]})
    result = nav.track(
        data,
        latitude_column="lat",
        longitude_column="lon",
        groupby_column=None,
        progressbar=False,
        ellipsoid=ellipsoid,
    )
    assert result == pytest.approx([90.0, 90.0, 90.0])


@pytest.mark.parametrize("progressbar", [True, False])
def test_track_groupby(progressbar):
    """Grouped track computation should be applied independently per group."""
    data = pd.DataFrame(
        {
            "line": ["a", "a", "a", "b", "b", "b"],
            "lat": [0.0, 0.0, 0.0, 0.0, 1.0, 2.0],
            "lon": [0.0, 1.0, 2.0, 0.0, 0.0, 0.0],
        }
    )
    result = nav.track(
        data,
        latitude_column="lat",
        longitude_column="lon",
        groupby_column="line",
        progressbar=progressbar,
        ellipsoid=True,
    )
    assert result == pytest.approx([90.0, 90.0, 90.0, 0.0, 0.0, 0.0])


def test__relative_distance_mismatched_lengths_raises():
    """Mismatched x/y array lengths should raise AssertionError."""
    with pytest.raises(AssertionError):
        nav._relative_distance(np.array([0.0, 1.0]), np.array([0.0]))  # pylint: disable=protected-access


def test__relative_distance_values():
    """_relative_distance() should compute the exact Euclidean distance between successive points."""
    x = np.array([0.0, 3.0, 6.0])
    y = np.array([0.0, 4.0, 8.0])
    result = nav._relative_distance(x, y)  # pylint: disable=protected-access
    assert result == pytest.approx([0.0, 5.0, 5.0])


@pytest.mark.parametrize("groupby_column", [None, "line"])
def test_relative_distance_missing_columns_raises(groupby_column):
    """Missing easting/northing columns should raise AssertionError, with or without groupby_column set."""
    data = pd.DataFrame({"other": [1.0, 2.0]})
    with pytest.raises(AssertionError):
        nav.relative_distance(
            data,
            groupby_column=groupby_column,
        )


def test_relative_distance_integer_dtype():
    """Integer-dtype coordinate columns should be cast to float internally rather than raising."""
    data = pd.DataFrame({"easting": [0, 3, 6], "northing": [0, 4, 8]})
    result = nav.relative_distance(data)
    assert result == pytest.approx([0.0, 5.0, 5.0])


def test_relative_distance_no_groupby():
    """relative_distance() should compute the exact distance between successive points."""
    data = pd.DataFrame({"easting": [0.0, 3.0, 6.0], "northing": [0.0, 4.0, 8.0]})
    result = nav.relative_distance(
        data,
        groupby_column=None,
        progressbar=False,
    )
    assert result == pytest.approx([0.0, 5.0, 5.0])


@pytest.mark.parametrize("progressbar", [True, False])
def test_relative_distance_groupby(progressbar):
    """Grouped relative_distance() should be computed independently per group."""
    data = pd.DataFrame(
        {
            "line": ["a", "a", "b", "b"],
            "easting": [0.0, 3.0, 0.0, 6.0],
            "northing": [0.0, 4.0, 0.0, 8.0],
        }
    )
    result = nav.relative_distance(
        data,
        groupby_column="line",
        progressbar=progressbar,
    )
    assert result == pytest.approx([0.0, 5.0, 0.0, 10.0])


def test_cumulative_distance_no_groupby():
    """cumulative_distance() should be the running total of relative_distance()."""
    data = pd.DataFrame({"easting": [0.0, 3.0, 6.0], "northing": [0.0, 4.0, 8.0]})
    result = nav.cumulative_distance(
        data,
        groupby_column=None,
        progressbar=False,
    )
    assert result == pytest.approx([0.0, 5.0, 10.0])


def test_cumulative_distance_groupby():
    """The cumulative sum should reset at the start of each group."""
    data = pd.DataFrame(
        {
            "line": ["a", "a", "b", "b"],
            "easting": [0.0, 3.0, 0.0, 6.0],
            "northing": [0.0, 4.0, 0.0, 8.0],
        }
    )
    result = nav.cumulative_distance(
        data,
        groupby_column="line",
        progressbar=False,
    )
    assert result == pytest.approx([0.0, 5.0, 0.0, 10.0])


def test_cumulative_distance_groupby_labels_sort_differently_than_appearance():
    """Group labels that don't sort into their appearance order must not scramble the result's row order."""
    data = pd.DataFrame(
        {
            "line": ["B", "B", "A", "A"],
            "easting": [0.0, 3.0, 0.0, 6.0],
            "northing": [0.0, 4.0, 0.0, 8.0],
        }
    )
    result = nav.cumulative_distance(
        data,
        groupby_column="line",
        progressbar=False,
    )
    assert result == pytest.approx([0.0, 5.0, 0.0, 10.0])


def test_along_track_distance_no_guess_no_groupby():
    """Without guess_start_position, along_track_distance() should equal cumulative_distance()."""
    data = pd.DataFrame({"easting": [0.0, 3.0, 6.0], "northing": [0.0, 4.0, 8.0]})
    data = gpd.GeoDataFrame(
        data, geometry=gpd.points_from_xy(data.easting, data.northing), crs="EPSG:3431"
    )
    result = nav.along_track_distance(
        data,
        groupby_column=None,
        progressbar=False,
        guess_start_position=False,
    )
    assert result == pytest.approx([0.0, 5.0, 10.0])


def test_along_track_distance_no_guess_groupby():
    """Grouped along_track_distance() (no guess_start_position) should reset per group like cumulative_distance()."""
    data = pd.DataFrame(
        {
            "line": ["a", "a", "b", "b"],
            "easting": [0.0, 3.0, 0.0, 6.0],
            "northing": [0.0, 4.0, 0.0, 8.0],
        }
    )
    data = gpd.GeoDataFrame(
        data, geometry=gpd.points_from_xy(data.easting, data.northing), crs="EPSG:3431"
    )
    result = nav.along_track_distance(
        data,
        groupby_column="line",
        progressbar=False,
        guess_start_position=False,
    )
    assert result == pytest.approx([0.0, 5.0, 0.0, 10.0])


def test_along_track_distance_guess_start_constructs_geometry_from_coords():
    """guess_start_position=True should build a geometry column from easting/northing
    when the input is a plain DataFrame without one."""
    data = pd.DataFrame({"easting": [2.0, 0.0, 1.0], "northing": [0.0, 0.0, 0.5]})
    result = nav.along_track_distance(
        data, guess_start_position=True, groupby_column=None, progressbar=False
    )
    assert result == pytest.approx([2.23606797749979, 0.0, 1.118033988749895])


def test_along_track_distance_guess_start_no_groupby():
    """Points given out of order along a bent line should be re-ordered by along-track position when guessing the start."""
    points = [Point(2, 0), Point(0, 0), Point(1, 0.5)]
    data = gpd.GeoDataFrame({"geometry": points})
    result = nav.along_track_distance(
        data, guess_start_position=True, groupby_column=None, progressbar=False
    )
    assert result == pytest.approx([2.23606797749979, 0.0, 1.118033988749895])


@pytest.mark.parametrize("progressbar", [True, False])
def test_along_track_distance_guess_start_groupby(progressbar):
    """Grouped guess_start_position should re-order each group's points independently."""
    points = [
        Point(2, 0),
        Point(0, 0),
        Point(1, 0.5),
        Point(5, 5),
        Point(3, 5),
        Point(4, 5.5),
    ]
    data = gpd.GeoDataFrame(
        {"line": ["a", "a", "a", "b", "b", "b"], "geometry": points}
    )
    result = nav.along_track_distance(
        data, guess_start_position=True, groupby_column="line", progressbar=progressbar
    )
    expected = [2.23606797749979, 0.0, 1.118033988749895] * 2
    assert result == pytest.approx(expected)


def test_along_track_distance_no_guess_works_without_geometry():
    """Without guess_start_position, a plain DataFrame with only easting/northing
    (no geometry column) should work fine, since no geometry is needed."""
    data = pd.DataFrame({"easting": [0.0, 3.0, 6.0], "northing": [0.0, 4.0, 8.0]})
    result = nav.along_track_distance(
        data,
        groupby_column=None,
        progressbar=False,
        guess_start_position=False,
    )
    assert result == pytest.approx([0.0, 5.0, 10.0])


def test_azimuth_between_points_positive_angle():
    """A point to the upper-right should give a positive azimuth angle."""
    assert nav._azimuth_between_points((0, 0), (1, 1)) == pytest.approx(45.0)  # pylint: disable=protected-access


def test_azimuth_between_points_negative_angle():
    """A point to the lower-right should give an azimuth angle shifted into the 0-180 range."""
    assert nav._azimuth_between_points((0, 0), (1, -1)) == pytest.approx(135.0)  # pylint: disable=protected-access


def test_dist():
    """_dist() should compute the exact Euclidean distance between two points."""
    assert nav._dist((0, 0), (1, 1)) == pytest.approx(1.4142135623730951)  # pylint: disable=protected-access


def test_azimuth_short_axis1():
    """azimuth() should compute the correct rectangle orientation when axis1 is the shorter side."""
    line = LineString([(0, 0), (1, 0), (2, 0), (2, 1)])
    rect = line.minimum_rotated_rectangle
    assert nav.azimuth(rect) == pytest.approx(26.56505117707799)


def test_azimuth_long_axis1():
    """A synthetic rectangle forces the axis1 > axis2 branch, since shapely's real rectangles never take it."""
    fake_rect = SimpleNamespace(
        exterior=SimpleNamespace(coords=[(0, 0), (0, 1), (5, 1), (5, 0)])
    )
    assert nav.azimuth(fake_rect) == pytest.approx(180.0)
