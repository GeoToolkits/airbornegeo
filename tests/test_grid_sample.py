import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.interpolate import RectBivariateSpline

from airbornegeo.grid_sample import (
    _crop_to_bounding_box,
    _nearest_index,
    _sample_grid_nearest,
    sample_grid,
)


def _linear_ramp_grid():
    easting = np.linspace(0, 100, 21)
    northing = np.linspace(0, 100, 21)
    e, n = np.meshgrid(easting, northing)
    return xr.DataArray(
        e + n,
        coords={"northing": northing, "easting": easting},
        dims=("northing", "easting"),
        name="z",
    )


def test_sample_grid_basic():
    """Sampling a linear-ramp grid at known points should return the exact grid values."""
    grid = _linear_ramp_grid()
    result = sample_grid(
        grid, x=np.array([0.0, 50.0, 100.0]), y=np.array([0.0, 50.0, 100.0])
    )
    assert result.tolist() == pytest.approx([0.0, 100.0, 200.0])


def test_sample_grid_mismatched_shapes_raises_valueerror():
    """x and y of different shapes should raise ValueError."""
    grid = _linear_ramp_grid()
    with pytest.raises(ValueError, match="must have the same shape"):
        sample_grid(grid, x=np.array([0.0, 1.0]), y=np.array([0.0]))


def test_sample_grid_default_interpolation_is_cubic_between_cells():
    """The default interpolation='cubic' should differ from the nearest-cell value between cell centers, for a non-linear surface."""
    easting = np.linspace(0, 100, 21)
    northing = np.linspace(0, 100, 21)
    e, n = np.meshgrid(easting, northing)
    grid = xr.DataArray(
        np.sin(e / 20) + np.cos(n / 15),
        coords={"northing": northing, "easting": easting},
        dims=("northing", "easting"),
        name="z",
    )
    x = np.array([52.0])
    y = np.array([37.0])

    cubic = sample_grid(grid, x, y)
    nearest = sample_grid(grid, x, y, interpolation="nearest")
    assert cubic[0] != pytest.approx(nearest[0])


def test_sample_grid_cubic_matches_exact_values_at_cell_centers():
    """Sampling exactly at cell centers should return the exact grid value even with cubic interpolation."""
    grid = _linear_ramp_grid()
    result = sample_grid(
        grid, x=np.array([0.0, 50.0, 100.0]), y=np.array([0.0, 50.0, 100.0])
    )
    assert result.tolist() == pytest.approx([0.0, 100.0, 200.0])


def test_sample_grid_cubic_handles_nan_coordinates():
    """NaN x or y coordinates should give a NaN sample for interpolation='cubic', without raising."""
    grid = _linear_ramp_grid()
    result = sample_grid(
        grid, x=np.array([0.0, np.nan, 50.0]), y=np.array([0.0, 50.0, np.nan])
    )
    assert result[0] == pytest.approx(0.0)
    assert np.isnan(result[1])
    assert np.isnan(result[2])


def test_sample_grid_cubic_handles_nan_in_grid():
    """A point near a NaN patch in the grid should sample as NaN under interpolation='cubic'."""
    grid = _linear_ramp_grid()
    grid = grid.copy()
    grid.loc[{"northing": 50.0, "easting": 50.0}] = np.nan
    result = sample_grid(grid, x=np.array([50.0, 0.0]), y=np.array([50.0, 0.0]))
    assert np.isnan(result[0])
    assert result[1] == pytest.approx(0.0)


def test_sample_grid_invalid_interpolation_raises_valueerror():
    """An unrecognized interpolation value should raise ValueError."""
    grid = _linear_ramp_grid()
    with pytest.raises(ValueError, match="interpolation must be 'cubic' or 'nearest'"):
        sample_grid(grid, x=np.array([0.0]), y=np.array([0.0]), interpolation="bogus")


def test_crop_to_bounding_box_covers_query_points_with_buffer():
    """The crop should cover all query points plus the requested buffer of grid cells on each side."""
    x_coord = np.arange(0.0, 101.0, 1.0)  # 0..100, spacing 1
    y_coord = np.arange(0.0, 101.0, 1.0)
    e, n = np.meshgrid(x_coord, y_coord)
    values = e + n

    x = np.array([40.0, 60.0])
    y = np.array([30.0, 70.0])
    x_crop, y_crop, values_crop = _crop_to_bounding_box(
        x_coord, y_coord, values, x, y, buffer_cells=6
    )

    assert x_crop.min() <= 34.0
    assert x_crop.max() >= 66.0
    assert y_crop.min() <= 24.0
    assert y_crop.max() >= 76.0
    # cropped values should be an exact sub-block of the original grid
    assert values_crop.shape == (len(y_crop), len(x_crop))
    expected = x_crop[np.newaxis, :] + y_crop[:, np.newaxis]
    assert np.array_equal(values_crop, expected)


def test_crop_to_bounding_box_clips_at_grid_edges():
    """A buffer that would extend past the grid edge should clip, not raise or wrap."""
    x_coord = np.arange(0.0, 21.0, 1.0)
    y_coord = np.arange(0.0, 21.0, 1.0)
    e, n = np.meshgrid(x_coord, y_coord)
    values = e + n

    x = np.array([1.0, 19.0])
    y = np.array([1.0, 19.0])
    x_crop, y_crop, _ = _crop_to_bounding_box(
        x_coord, y_coord, values, x, y, buffer_cells=6
    )

    assert x_crop.min() == 0.0
    assert x_crop.max() == 20.0
    assert y_crop.min() == 0.0
    assert y_crop.max() == 20.0


def test_sample_grid_cubic_localized_query_matches_full_grid_result():
    """Cropping to a bounding box before fitting should give the same result as fitting the full grid would."""
    easting = np.linspace(0, 100_000, 200)
    northing = np.linspace(0, 100_000, 200)
    e, n = np.meshgrid(easting, northing)
    grid = xr.DataArray(
        np.sin(e / 10_000) + np.cos(n / 8_000),
        coords={"northing": northing, "easting": easting},
        dims=("northing", "easting"),
        name="z",
    )
    # query points clustered in a small sub-region, the scenario cropping targets
    rng = np.random.default_rng(0)
    x = rng.uniform(40_000, 42_000, 50)
    y = rng.uniform(40_000, 42_000, 50)

    full_spline = RectBivariateSpline(northing, easting, grid.to_numpy())
    expected = full_spline.ev(y, x)

    result = sample_grid(grid, x, y)
    assert result == pytest.approx(expected, abs=1e-8)


def test_sample_grid_cubic_matches_pygmt_grdtrack():
    """sample_grid's default cubic interpolation should closely match pygmt.grdtrack's default (interpolation='c')."""
    pygmt = pytest.importorskip("pygmt")

    rng = np.random.default_rng(0)
    easting = np.linspace(0, 100_000, 60)
    northing = np.linspace(0, 100_000, 60)
    e, n = np.meshgrid(easting, northing)
    values = np.sin(e / 10_000) + np.cos(n / 8_000)
    grid = xr.DataArray(
        values,
        coords={"northing": northing, "easting": easting},
        dims=("northing", "easting"),
        name="z",
    )

    x = rng.uniform(2000, 98_000, 300)
    y = rng.uniform(2000, 98_000, 300)
    data = pd.DataFrame({"easting": x, "northing": y})

    ours = sample_grid(grid, x, y)
    gmt = pygmt.grdtrack(
        points=data,
        grid=grid,
        newcolname="z",
        no_skip=True,
        verbose="warning",
        interpolation="c",
    ).z

    assert np.max(np.abs(ours - gmt.to_numpy())) < 0.02


def test_sample_grid_cubic_nan_mask_matches_pygmt_grdtrack():
    """sample_grid's cubic interpolation should mark the same points NaN as pygmt.grdtrack near a data gap."""
    pygmt = pytest.importorskip("pygmt")

    rng = np.random.default_rng(1)
    easting = np.linspace(0, 100_000, 60)
    northing = np.linspace(0, 100_000, 60)
    e, n = np.meshgrid(easting, northing)
    values = np.sin(e / 10_000) + np.cos(n / 8_000)
    values[25:30, 25:30] = np.nan
    grid = xr.DataArray(
        values,
        coords={"northing": northing, "easting": easting},
        dims=("northing", "easting"),
        name="z",
    )

    x = rng.uniform(1000, 99_000, 1000)
    y = rng.uniform(1000, 99_000, 1000)
    data = pd.DataFrame({"easting": x, "northing": y})

    ours = sample_grid(grid, x, y)
    gmt = pygmt.grdtrack(
        points=data,
        grid=grid,
        newcolname="z",
        no_skip=True,
        verbose="warning",
        interpolation="c",
    ).z

    assert np.array_equal(np.isnan(ours), np.isnan(gmt.to_numpy()))


def test_sample_grid_nearest_exact_cell_centers():
    """Sampling exactly at cell centers should return the exact grid values."""
    grid = _linear_ramp_grid()
    result = _sample_grid_nearest(
        grid, x=np.array([0.0, 50.0, 100.0]), y=np.array([0.0, 50.0, 100.0])
    )
    assert result == pytest.approx([0.0, 100.0, 200.0])


def test_sample_grid_nearest_between_cells_uses_closest():
    """A point between two cell centers should return the value of whichever cell center is closer."""
    grid = _linear_ramp_grid()  # 21 cells from 0 to 100, spacing 5
    # x=52 is closer to the cell at x=50 than x=55
    result = _sample_grid_nearest(grid, x=np.array([52.0]), y=np.array([0.0]))
    assert result == pytest.approx([50.0])


def test_sample_grid_nearest_matches_sample_grid_at_cell_centers():
    """At exact cell centers, _sample_grid_nearest should agree with sample_grid(interpolation='nearest')."""
    grid = _linear_ramp_grid()
    x = np.array([0.0, 25.0, 50.0, 75.0, 100.0])
    y = np.array([0.0, 25.0, 50.0, 75.0, 100.0])
    nearest = _sample_grid_nearest(grid, x, y)
    via_sample_grid = sample_grid(grid, x, y, interpolation="nearest")
    assert nearest == pytest.approx(via_sample_grid)


def test_sample_grid_nearest_handles_nan_coordinates():
    """NaN x or y coordinates should give a NaN sample, without raising."""
    grid = _linear_ramp_grid()
    result = _sample_grid_nearest(
        grid, x=np.array([0.0, np.nan, 50.0]), y=np.array([0.0, 50.0, np.nan])
    )
    assert result[0] == pytest.approx(0.0)
    assert np.isnan(result[1])
    assert np.isnan(result[2])


def test_sample_grid_nearest_handles_nan_in_grid():
    """A point whose nearest cell is NaN in the grid should sample as NaN, without raising."""
    grid = _linear_ramp_grid()
    grid = grid.copy()
    grid.loc[{"northing": 50.0, "easting": 50.0}] = np.nan
    result = _sample_grid_nearest(
        grid, x=np.array([50.0, 0.0]), y=np.array([50.0, 0.0])
    )
    assert np.isnan(result[0])
    assert result[1] == pytest.approx(0.0)


def test_sample_grid_nearest_tie_is_safe_and_deterministic():
    """A point exactly equidistant between two grid cells should not raise, and should pick consistently."""
    grid = (
        _linear_ramp_grid()
    )  # spacing 5, so x=52.5 is exactly between cells at 50 and 55
    result_1 = _sample_grid_nearest(grid, x=np.array([52.5]), y=np.array([0.0]))
    result_2 = _sample_grid_nearest(grid, x=np.array([52.5]), y=np.array([0.0]))
    assert not np.isnan(result_1[0])
    assert result_1 == pytest.approx(result_2)


def test_sample_grid_nearest_descending_coordinates():
    """A grid with descending (e.g. north-up) coordinate order should be sampled correctly."""
    easting = np.linspace(0, 100, 21)
    northing = np.linspace(100, 0, 21)  # descending
    e, n = np.meshgrid(easting, northing)
    grid = xr.DataArray(
        e + n,
        coords={"northing": northing, "easting": easting},
        dims=("northing", "easting"),
        name="z",
    )
    result = _sample_grid_nearest(
        grid, x=np.array([0.0, 100.0]), y=np.array([100.0, 0.0])
    )
    assert result == pytest.approx([100.0, 100.0])


def test_sample_grid_nearest_mismatched_shapes_raises_valueerror():
    """x and y of different shapes should raise ValueError."""
    grid = _linear_ramp_grid()
    with pytest.raises(ValueError, match="must have the same shape"):
        _sample_grid_nearest(grid, x=np.array([0.0, 1.0]), y=np.array([0.0]))


def test_sample_grid_nearest_invalid_method_raises_valueerror():
    """An unrecognized method should raise ValueError."""
    grid = _linear_ramp_grid()
    with pytest.raises(ValueError, match="method must be 'grid' or 'geodesic'"):
        _sample_grid_nearest(grid, x=np.array([0.0]), y=np.array([0.0]), method="bogus")


def _lonlat_grid(lon_coord, lat_coord, nan_fraction=0.0, seed=0):
    rng = np.random.default_rng(seed)
    ll, la = np.meshgrid(lon_coord, lat_coord)
    values = ll + la
    if nan_fraction:
        mask = rng.random(values.shape) < nan_fraction
        values[mask] = np.nan
    return xr.DataArray(
        values,
        coords={"lat": lat_coord, "lon": lon_coord},
        dims=("lat", "lon"),
        name="z",
    )


def test_sample_grid_nearest_geodesic_matches_grid_at_low_latitude():
    """At low/mid latitude with fine spacing, 'geodesic' and 'grid' should agree, since longitude compression is negligible there."""
    grid = _lonlat_grid(np.arange(-30, 30.01, 0.5), np.arange(-30, 30.01, 0.5))
    rng = np.random.default_rng(1)
    lon = rng.uniform(-25, 25, 5000)
    lat = rng.uniform(-25, 25, 5000)

    grid_method = _sample_grid_nearest(grid, lon, lat, method="grid")
    geodesic_method = _sample_grid_nearest(grid, lon, lat, method="geodesic")
    assert grid_method == pytest.approx(geodesic_method)


def test_sample_grid_nearest_geodesic_diverges_from_grid_near_pole():
    """With coarse latitude spacing near the pole, 'geodesic' should meaningfully diverge from 'grid', which ignores longitude compression."""
    grid = _lonlat_grid(np.arange(-30, 30.01, 0.5), np.arange(85, 90.01, 5.0))
    rng = np.random.default_rng(2)
    n = 20_000
    lon = rng.uniform(-25, 25, n)
    lat = rng.uniform(85, 90, n)

    grid_method = _sample_grid_nearest(grid, lon, lat, method="grid")
    geodesic_method = _sample_grid_nearest(grid, lon, lat, method="geodesic")
    mismatch_fraction = np.mean(grid_method != geodesic_method)
    assert mismatch_fraction > 0.2  # ~48% observed for this configuration


def test_sample_grid_nearest_geodesic_handles_nan_coordinates():
    """NaN x or y coordinates should give a NaN sample under method='geodesic', without raising."""
    grid = _lonlat_grid(np.arange(-10, 10.01, 1.0), np.arange(-10, 10.01, 1.0))
    result = _sample_grid_nearest(
        grid,
        x=np.array([0.0, np.nan, 5.0]),
        y=np.array([0.0, 5.0, np.nan]),
        method="geodesic",
    )
    assert result[0] == pytest.approx(0.0)
    assert np.isnan(result[1])
    assert np.isnan(result[2])


def test_sample_grid_nearest_geodesic_handles_nan_in_grid():
    """A point whose true-nearest cell is NaN in the grid should sample as NaN under method='geodesic', without raising."""
    grid = _lonlat_grid(np.arange(-10, 10.01, 1.0), np.arange(-10, 10.01, 1.0))
    grid = grid.copy()
    grid.loc[{"lat": 0.0, "lon": 0.0}] = np.nan
    result = _sample_grid_nearest(
        grid, x=np.array([0.0, 5.0]), y=np.array([0.0, 5.0]), method="geodesic"
    )
    assert np.isnan(result[0])
    assert result[1] == pytest.approx(10.0)


@pytest.mark.parametrize("ascending", [True, False])
def test_nearest_index_tie_picks_lower_original_index(ascending):
    """On an exact tie, _nearest_index should pick the lower original array index, regardless of coordinate direction."""
    coord = (
        np.array([0.0, 10.0, 20.0, 30.0])
        if ascending
        else np.array([30.0, 20.0, 10.0, 0.0])
    )
    query = np.array(
        [15.0]
    )  # exactly between index 1 (10 or 20) and index 2 (20 or 10)
    result = _nearest_index(coord, query)
    assert result[0] == 1


def test_nearest_index_nan_query_gives_negative_one():
    """A NaN query value should give the placeholder index -1."""
    coord = np.array([0.0, 10.0, 20.0])
    result = _nearest_index(coord, np.array([5.0, np.nan]))
    assert result[0] == 0
    assert result[1] == -1
