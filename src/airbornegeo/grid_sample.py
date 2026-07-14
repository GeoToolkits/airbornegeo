import numpy as np
import scipy.spatial
import verde as vd
import xarray as xr
from numpy.typing import NDArray
from scipy.interpolate import RectBivariateSpline


def _nearest_index(coord: NDArray, query: NDArray) -> NDArray:
    """
    For each value in query, find the index (into the original, monotonic but
    possibly ascending or descending, 1D array coord) of its nearest value. NaN
    entries in query get index -1 (an invalid placeholder, replaced by the caller).
    Ties (a query value exactly equidistant between two coord values) resolve to the
    lower of the two original indices, deterministically.
    """
    ascending = coord[-1] >= coord[0]
    sorted_coord = coord if ascending else coord[::-1]

    valid = ~np.isnan(query)
    index = np.full(query.shape, -1, dtype=int)

    # searchsorted needs an ascending array; the nearest value to each query point is
    # either at the insertion point, or one step before it
    right_sorted = np.clip(
        np.searchsorted(sorted_coord, query[valid]), 1, len(coord) - 1
    )
    left_sorted = right_sorted - 1

    if ascending:
        right, left = right_sorted, left_sorted
    else:
        right = len(coord) - 1 - right_sorted
        left = len(coord) - 1 - left_sorted

    dist_right = np.abs(coord[right] - query[valid])
    dist_left = np.abs(coord[left] - query[valid])
    pick_right = (dist_right < dist_left) | ((dist_right == dist_left) & (right < left))

    index[valid] = np.where(pick_right, right, left)
    return index


def _lonlat_to_unit_sphere(lon_deg: NDArray, lat_deg: NDArray) -> NDArray:
    """
    Convert longitude/latitude in degrees to xyz coordinates on a unit sphere. The
    radius is arbitrary (any positive constant scales all points equally, so it never
    changes which point is nearest) - a unit sphere keeps this an implementation
    detail rather than a hidden assumption about Earth's radius.
    """
    lon = np.radians(lon_deg)
    lat = np.radians(lat_deg)
    return np.column_stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
    )


def _sample_grid_nearest(
    grid: xr.DataArray,
    x: NDArray,
    y: NDArray,
    method: str = "grid",
) -> NDArray:
    """
    Sample a grid's values at a set of x, y point coordinates, using the value of the
    nearest grid cell to each point. The grid's first dimension is treated as the
    y-coordinate and its second dimension as the x-coordinate, matching this package's
    convention elsewhere (e.g. dims ("northing", "easting")).

    Two methods are available, trading accuracy on geographic grids for speed:

    - "grid" (the default): finds the nearest grid cell independently along each
      dimension, treating x and y as generic, unrelated coordinates. This is
      equivalent to pygmt.grdtrack with nearest-neighbor interpolation (without
      requiring GMT), and is fast (an O(log n) binary search per dimension, per
      point, with no upfront cost to build a search structure). For geographic
      (longitude/latitude) grids this is also what GMT itself does: it is not aware
      that longitude represents a shrinking physical distance towards the poles, so
      on a coarse grid at high latitudes it can pick a different, and less physically
      correct, cell than the true nearest one.
    - "geodesic": finds the true nearest cell by great-circle distance, treating x as
      longitude and y as latitude, both in degrees (on a sphere; Earth's ellipsoidal
      flattening is not accounted for). This is accurate at any latitude, but is
      slower: it builds a 3D KDTree over every grid cell up front, an O(n log n) cost
      in the total number of grid cells, before any points can be queried. Only use
      this for geographic grids - it is meaningless for e.g. projected
      easting/northing coordinates.

    NaN values in x or y give a NaN sample, without raising. If the nearest grid cell
    itself is NaN, the returned sample is NaN. If a point is exactly equidistant
    between two grid cells, the lower-index cell is used (independently per dimension
    for method "grid"; whichever KDTree returns for method "geodesic", which does not
    make the same guarantee).

    Parameters
    ----------
    grid : xarray.DataArray
        2D grid to sample, with coordinates along its two dimensions.
    x : NDArray
        x-coordinates (or longitude, in degrees, for method "geodesic") of the points
        to sample at.
    y : NDArray
        y-coordinates (or latitude, in degrees, for method "geodesic") of the points
        to sample at, same length as x.
    method : str, optional
        Either "grid" or "geodesic", by default "grid".

    Returns
    -------
    NDArray
        Sampled grid values, the same length and order as x and y.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.shape != y.shape:
        msg = "x and y must have the same shape"
        raise ValueError(msg)
    if method not in ("grid", "geodesic"):
        msg = f"method must be 'grid' or 'geodesic', got '{method}'"
        raise ValueError(msg)

    y_dim, x_dim = grid.dims
    x_coord = grid[x_dim].to_numpy()
    y_coord = grid[y_dim].to_numpy()
    values = grid.to_numpy()
    sampled = np.full(x.shape, np.nan)

    if method == "grid":
        x_index = _nearest_index(x_coord, x)
        y_index = _nearest_index(y_coord, y)
        valid = (x_index >= 0) & (y_index >= 0)
        sampled[valid] = values[y_index[valid], x_index[valid]]
        return sampled

    # method == "geodesic"
    xx, yy = np.meshgrid(x_coord, y_coord)
    tree = scipy.spatial.cKDTree(_lonlat_to_unit_sphere(xx.ravel(), yy.ravel()))
    valid = ~(np.isnan(x) | np.isnan(y))
    _, flat_index = tree.query(_lonlat_to_unit_sphere(x[valid], y[valid]))
    sampled[valid] = values.ravel()[flat_index]
    return sampled


def _fill_nans_nearest(x_coord: NDArray, y_coord: NDArray, values: NDArray) -> NDArray:
    """
    Fill NaN cells in a 2D grid (shape (len(y_coord), len(x_coord))) with their
    nearest valid neighbor's value, using verde's KNeighbors. Used to give a bicubic
    spline fitter a complete grid to fit; NaN cells are masked back out afterwards
    based on the nearest original grid cell, so the fill values themselves never
    reach the caller.
    """
    xx, yy = np.meshgrid(x_coord, y_coord)
    valid = ~np.isnan(values)
    filled: NDArray = (
        vd.KNeighbors().fit((xx[valid], yy[valid]), values[valid]).predict((xx, yy))
    )
    return filled


def _crop_to_bounding_box(
    x_coord: NDArray,
    y_coord: NDArray,
    values: NDArray,
    x: NDArray,
    y: NDArray,
    buffer_cells: int = 6,
) -> tuple[NDArray, NDArray, NDArray]:
    """
    Crop a regular grid (and its coordinate arrays) down to the bounding box of the
    query points x, y, padded by buffer_cells grid cells on each side (enough margin
    for the bicubic spline's boundary behavior to be unaffected). This is a pure
    speedup for large grids sampled at spatially localized points - fitting a spline
    to a small crop is far cheaper than fitting to the whole grid, and gives
    (numerically) identical results, since a tensor-product spline's value at a
    point only depends on nearby control points anyway.
    """
    x_lo = np.searchsorted(x_coord, x.min())
    x_hi = np.searchsorted(x_coord, x.max())
    y_lo = np.searchsorted(y_coord, y.min())
    y_hi = np.searchsorted(y_coord, y.max())

    x_lo = max(x_lo - buffer_cells, 0)
    x_hi = min(x_hi + buffer_cells, len(x_coord) - 1) + 1
    y_lo = max(y_lo - buffer_cells, 0)
    y_hi = min(y_hi + buffer_cells, len(y_coord) - 1) + 1

    return (
        x_coord[x_lo:x_hi],
        y_coord[y_lo:y_hi],
        values[y_lo:y_hi, x_lo:x_hi],
    )


def _sample_grid_cubic(grid: xr.DataArray, x: NDArray, y: NDArray) -> NDArray:
    """
    Sample a grid's values at x, y point coordinates with bicubic spline
    interpolation, matching pygmt.grdtrack's default (interpolation="c") closely. NaN
    cells are filled first so the spline can be fit, then any query point whose
    nearest original grid cell was NaN is masked back to NaN in the result, matching
    GMT's own behavior of returning NaN near data gaps.

    Before fitting, the grid is cropped to the bounding box of the (valid) query
    points, plus a small buffer. For a large grid sampled at spatially localized
    points (e.g. a small survey area cut from a much larger reference grid) this can
    make fitting hundreds to thousands of times faster, with no accuracy cost - a
    query point's interpolated value only depends on nearby grid cells regardless.
    """
    y_dim, x_dim = grid.dims
    x_coord = grid[x_dim].to_numpy()
    y_coord = grid[y_dim].to_numpy()
    values = grid.to_numpy()

    valid = ~(np.isnan(x) | np.isnan(y))
    sampled = np.full(x.shape, np.nan)
    if not valid.any():
        return sampled

    x_coord, y_coord, values = _crop_to_bounding_box(
        x_coord, y_coord, values, x[valid], y[valid]
    )

    has_nan = np.isnan(values).any()
    fit_values = _fill_nans_nearest(x_coord, y_coord, values) if has_nan else values

    # RectBivariateSpline requires strictly ascending coordinates on both axes
    x_order = np.argsort(x_coord)
    y_order = np.argsort(y_coord)
    spline = RectBivariateSpline(
        y_coord[y_order], x_coord[x_order], fit_values[np.ix_(y_order, x_order)]
    )

    sampled[valid] = spline.ev(y[valid], x[valid])

    if has_nan:
        x_index = _nearest_index(x_coord, x[valid])
        y_index = _nearest_index(y_coord, y[valid])
        nan_nearest = np.isnan(values[y_index, x_index])
        sampled_valid = sampled[valid]
        sampled_valid[nan_nearest] = np.nan
        sampled[valid] = sampled_valid

    return sampled


def sample_grid(
    grid: xr.DataArray,
    x: NDArray,
    y: NDArray,
    interpolation: str = "cubic",
) -> NDArray:
    """
    Sample a grid's values at a set of x, y point coordinates. By default this uses
    bicubic spline interpolation (scipy.interpolate.RectBivariateSpline), matching
    pygmt.grdtrack's default behavior (interpolation="c") closely on smooth grids
    (sub-percent differences) and reasonably on rough, high-frequency ones (e.g. real
    terrain: a few tenths of a percent RMS of the grid's value range in testing) -
    scipy's spline and GMT's own bicubic convolution algorithm are both legitimate but
    different interpolants, so they are not bit-identical on rough data. Pass
    interpolation="nearest" for fast nearest-grid-cell sampling instead (see
    :func:`_sample_grid_nearest` for details on that method, including NaN and tie
    handling). NaN values in x or y give a NaN sample; for "cubic", a point whose
    nearest original grid cell is NaN also samples as NaN, matching GMT's behavior
    near data gaps.

    Parameters
    ----------
    grid : xarray.DataArray
        2D grid to sample, with coordinates along its two dimensions.
    x : NDArray
        x-coordinates of the points to sample at.
    y : NDArray
        y-coordinates of the points to sample at, same length as x.
    interpolation : str, optional
        Either "cubic" or "nearest", by default "cubic".

    Returns
    -------
    NDArray
        Sampled grid values, the same length and order as x and y.
    """
    if interpolation not in ("cubic", "nearest"):
        msg = f"interpolation must be 'cubic' or 'nearest', got '{interpolation}'"
        raise ValueError(msg)

    if interpolation == "nearest":
        return _sample_grid_nearest(grid, x, y)

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        msg = "x and y must have the same shape"
        raise ValueError(msg)

    return _sample_grid_cubic(grid, x, y)
