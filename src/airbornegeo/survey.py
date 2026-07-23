"""
Object-oriented wrapper around the functional airbornegeo API.

The :class:`Survey` class stores a survey dataframe, its intersection table, a
coordinate reference system, free-form metadata, and the names of the survey's
structural columns (line, distance, time, height, latitude, longitude, line
type). Most package functions are exposed as methods which inject those stored
column names, so they don't need to be re-passed on every call. Summary
statistics (bounding region, line counts, lengths, spacings, orientations) are
computed lazily and cached.
"""

import functools  # pylint: disable=too-many-lines
import html
import json
import pathlib
import typing

import geopandas as gpd
import harmonica as hm
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyproj
import shapely
import verde as vd
import xarray as xr
from numpy.typing import NDArray

from airbornegeo.block_reduce import block_reduce
from airbornegeo.crossover_levelling import (
    alternating_iterative_line_levelling,
    crossover_network_levelling,
    crossover_pair_levelling,
)
from airbornegeo.crossovers import (
    add_intersections,
    add_values_to_intersections,
    calculate_crossover_errors,
    create_intersection_table,
    inspect_intersections,
    interpolate_intersections,
    lines_without_intersections,
    plot_line_and_crosses,
    update_intersections_with_eq_sources,
)
from airbornegeo.eotvos import eotvos_correction
from airbornegeo.eq_source_levelling import equivalent_source_levelling
from airbornegeo.filtering import filter_line
from airbornegeo.grid_levelling import level_to_grid
from airbornegeo.grid_sample import sample_grid
from airbornegeo.interpolating import (
    interpolate_missing,
    interpolate_missing_pointwise,
    interpolate_missing_pointwise_with_windows,
)
from airbornegeo.nav import (
    _azimuth_between_points,
    along_track_distance,
    azimuth,
    directional_velocity,
    ground_speed,
    relative_distance,
    track,
    vertical_acceleration,
)
from airbornegeo.plotting import (
    add_scalebar,
    choose_colormap,
    inspect_lines,
    nice_scalebar_width,
    plot_profiles,
    plotly_points,
    plotly_profiles,
)
from airbornegeo.potential_fields import (
    eq_sources_1d,
    igrf,
    upward_continue_by_line,
)
from airbornegeo.processing import split_into_segments, unique_line_id
from airbornegeo.reproject import reproject
from airbornegeo.resample import resample, resample_as
from airbornegeo.utils import (
    _check_coord_columns,
    largest_line_dimensions,
    median_line_spacing,
)

_TYPE_NAMES = {0: "flight", 1: "tie", 2: "excluded"}


def _axial_mean(azimuths: pd.Series) -> float:
    """
    Mean of axial (0-180 degree) angles via angle doubling, robust to values
    straddling the 0/180 wraparound (e.g. N-S lines at 179 and 1 degrees).
    """
    doubled = np.deg2rad(2 * azimuths.to_numpy())
    mean = np.arctan2(np.sin(doubled).mean(), np.cos(doubled).mean())
    return float(np.rad2deg(mean) / 2 % 180)


def _median(values: pd.Series) -> float:
    return float(values.median())


class Survey:
    """
    Container for an airborne survey dataframe with its structural column
    names, coordinate reference system, metadata, intersection table, and
    lazily computed summary statistics.

    Methods which add or alter a few columns operate in place on
    ``survey.data`` and return the survey itself, so calls can be chained,
    e.g. ``survey.along_track_distance().filter_line(...)``. Methods which
    replace the whole dataframe (:meth:`block_reduce`, :meth:`resample`,
    :meth:`resample_as`) instead return a new :class:`Survey`, leaving the
    original untouched.

    Summary statistics (:attr:`region`, :attr:`line_counts`,
    :attr:`line_lengths`, :attr:`total_length`, :attr:`median_line_lengths`,
    :attr:`line_azimuths`, :attr:`mean_line_azimuths`,
    :attr:`median_line_spacings`, :attr:`line_point_spacings`,
    :attr:`median_point_spacings`) are computed on first access and cached.
    The cache is cleared automatically when methods replace or restructure the
    dataframe, but not when ``survey.data`` is mutated directly from outside —
    call :meth:`invalidate_cache` after such external edits.

    Parameters
    ----------
    data : pd.DataFrame | gpd.GeoDataFrame
        the survey dataframe. Must contain projected coordinate columns; if
        they are not named 'easting'/'northing', give their names with
        easting_column/northing_column and they will be renamed on ingest
        (the rest of the package requires those literal names).

        Apart from the coordinates, no column names are assumed. Name the
        columns a workflow needs when building the Survey; methods which need
        an unset binding raise a ValueError naming the missing one.
    line_column : str | None, optional
        name of the column containing line names/numbers, by default None
    line_type_column : str | None, optional
        name of the column containing line type codes (0 = flight line,
        1 = tie line, 2 = excluded), by default None
    distance_column : str | None, optional
        name of the column containing distance along line, by default None
    time_column : str | None, optional
        name of the column containing time values, by default None
    height_column : str | None, optional
        name of the column containing flight heights, by default None
    latitude_column, longitude_column : str | None, optional
        names of the geodetic latitude/longitude columns, by default None
    easting_column, northing_column : str, optional
        names of the projected coordinate columns in the supplied dataframe,
        renamed to 'easting'/'northing' on ingest, by default already
        'easting'/'northing'
    crs : typing.Any | None, optional
        coordinate reference system, anything accepted by
        ``pyproj.CRS.from_user_input``. Taken from the GeoDataFrame if not
        given, by default None
    metadata : dict[str, typing.Any] | None, optional
        free-form survey metadata (name, year, instrument, units, ...),
        by default None
    copy : bool, optional
        copy the dataframe on ingest so in-place methods never alter the
        caller's frame. Set to False to avoid the copy for very large
        surveys, at the cost of methods mutating the supplied frame,
        by default True
    """

    _STAT_NAMES: typing.ClassVar[tuple[str, ...]]  # filled in after class body

    def __init__(
        self,
        data: pd.DataFrame | gpd.GeoDataFrame,
        *,
        line_column: str | None = None,
        line_type_column: str | None = None,
        distance_column: str | None = None,
        time_column: str | None = None,
        height_column: str | None = None,
        latitude_column: str | None = None,
        longitude_column: str | None = None,
        easting_column: str = "easting",
        northing_column: str = "northing",
        crs: typing.Any | None = None,
        metadata: dict[str, typing.Any] | None = None,
        copy: bool = True,
    ) -> None:
        if copy:
            data = data.copy()
        renames = {
            old: new
            for old, new in (
                (easting_column, "easting"),
                (northing_column, "northing"),
            )
            if old != new
        }
        if renames:
            data = data.rename(columns=renames)
        _check_coord_columns(data)

        gdf_crs = getattr(data, "crs", None)
        if crs is None:
            crs = gdf_crs
        elif gdf_crs is not None and pyproj.CRS.from_user_input(
            crs
        ) != pyproj.CRS.from_user_input(gdf_crs):
            msg = f"crs argument ({crs}) conflicts with GeoDataFrame crs ({gdf_crs})"
            raise ValueError(msg)
        self.crs: pyproj.CRS | None = (
            None if crs is None else pyproj.CRS.from_user_input(crs)
        )

        self._line_column = line_column
        self._line_type_column = line_type_column
        self._distance_column = distance_column
        self._time_column = time_column
        self._height_column = height_column
        self.latitude_column = latitude_column
        self.longitude_column = longitude_column
        self.metadata: dict[str, typing.Any] = dict(metadata) if metadata else {}
        self.intersections: gpd.GeoDataFrame | None = None
        self._data: pd.DataFrame = data

    ####################
    # data access and cache invalidation
    ####################

    @property
    def data(self) -> pd.DataFrame:
        """The survey dataframe."""
        return self._data

    @data.setter
    def data(self, value: pd.DataFrame) -> None:
        _check_coord_columns(value)
        self._data = value
        self._invalidate_stats()

    def invalidate_cache(self) -> None:
        """
        Drop all cached statistics so they recompute on next access. Call this
        after mutating ``survey.data`` directly from outside the class.
        """
        self._invalidate_stats()

    ####################
    # structural column bindings
    ####################

    def _get_column(self, attribute: str) -> str:
        """
        Value of a structural column binding, raising if it was never set.
        Only 'easting'/'northing' have defaults; everything else must be named
        explicitly when the Survey is built, since column naming conventions
        vary too much between surveys to guess.
        """
        value = getattr(self, f"_{attribute}")
        if value is None:
            msg = (
                f"this method needs {attribute} set on the Survey; pass "
                f"{attribute}='...' to Survey() or assign survey.{attribute}"
            )
            raise ValueError(msg)
        return str(value)

    @property
    def line_column(self) -> str:
        """Name of the line name/number column."""
        return self._get_column("line_column")

    @line_column.setter
    def line_column(self, value: str | None) -> None:
        self._line_column = value
        self._invalidate_stats()

    @property
    def line_type_column(self) -> str:
        """Name of the line type code column (0 = flight, 1 = tie, 2 = excluded)."""
        return self._get_column("line_type_column")

    @line_type_column.setter
    def line_type_column(self, value: str | None) -> None:
        self._line_type_column = value
        self._invalidate_stats()

    @property
    def distance_column(self) -> str:
        """Name of the distance-along-line column."""
        return self._get_column("distance_column")

    @distance_column.setter
    def distance_column(self, value: str | None) -> None:
        self._distance_column = value

    @property
    def time_column(self) -> str:
        """Name of the time column."""
        return self._get_column("time_column")

    @time_column.setter
    def time_column(self, value: str | None) -> None:
        self._time_column = value

    @property
    def height_column(self) -> str:
        """Name of the flight height column."""
        return self._get_column("height_column")

    @height_column.setter
    def height_column(self, value: str | None) -> None:
        self._height_column = value

    def _invalidate_stats(self) -> None:
        for name in self._STAT_NAMES:
            self.__dict__.pop(name, None)

    ####################
    # validation and column-name helpers
    ####################

    def _require(self, *columns: str) -> None:
        missing = [col for col in columns if col not in self._data.columns]
        if missing:
            msg = (
                f"column(s) {missing} not found in survey.data (check the "
                "*_column attributes set on this Survey)"
            )
            raise ValueError(msg)

    def _require_geographic(self) -> tuple[str, str]:
        if self.latitude_column is None or self.longitude_column is None:
            msg = (
                "this method needs latitude_column and longitude_column set "
                "on the Survey"
            )
            raise ValueError(msg)
        self._require(self.latitude_column, self.longitude_column)
        return self.latitude_column, self.longitude_column

    def _require_intersections(self) -> pd.DataFrame:
        if self.intersections is None:
            msg = (
                "this survey has no intersections table; call "
                "create_intersection_table() first"
            )
            raise ValueError(msg)
        return self.intersections

    def _canonical_pairs(self) -> tuple[tuple[str, str], ...]:
        """Bound (current name, literal name) pairs; unset bindings are skipped."""
        return tuple(
            (current, literal)
            for current, literal in (
                (self._line_column, "line"),
                (self._height_column, "height"),
                (self._line_type_column, "line_type"),
                (self._distance_column, "distance_along_line"),
            )
            if current is not None
        )

    def _canonical(self) -> pd.DataFrame:
        """
        View of the survey dataframe with structural columns renamed to the
        literal names ('line', 'height', 'line_type', 'distance_along_line')
        which some package functions require.
        """
        renames = {
            current: literal
            for current, literal in self._canonical_pairs()
            if current != literal
        }
        for current, literal in renames.items():
            if literal in self._data.columns:
                msg = (
                    f"cannot rename column '{current}' to '{literal}': "
                    f"survey.data already has an unrelated '{literal}' column"
                )
                raise ValueError(msg)
        return self._data.rename(columns=renames) if renames else self._data

    def _canonical_inverse(self, data: pd.DataFrame) -> pd.DataFrame:
        renames = {
            literal: current
            for current, literal in self._canonical_pairs()
            if current != literal
        }
        return data.rename(columns=renames) if renames else data

    def _spawn(self, data: pd.DataFrame) -> "Survey":
        """New Survey with this survey's column bindings, crs, and metadata."""
        return Survey(
            data,
            line_column=self._line_column,
            line_type_column=self._line_type_column,
            distance_column=self._distance_column,
            time_column=self._time_column,
            height_column=self._height_column,
            latitude_column=self.latitude_column,
            longitude_column=self.longitude_column,
            crs=self.crs,
            metadata=dict(self.metadata),
            copy=False,
        )

    ####################
    # saving / reloading
    ####################

    def _sidecar(self, has_intersections: bool) -> dict[str, typing.Any]:
        """Column bindings, CRS, and metadata needed to reconstruct this Survey."""
        return {
            "line_column": self._line_column,
            "line_type_column": self._line_type_column,
            "distance_column": self._distance_column,
            "time_column": self._time_column,
            "height_column": self._height_column,
            "latitude_column": self.latitude_column,
            "longitude_column": self.longitude_column,
            "crs": self.crs.to_wkt() if self.crs else None,
            "metadata": self.metadata,
            "has_intersections": has_intersections,
        }

    @classmethod
    def _from_sidecar(
        cls, data: pd.DataFrame, sidecar: dict[str, typing.Any]
    ) -> "Survey":
        return cls(
            data,
            line_column=sidecar["line_column"],
            line_type_column=sidecar["line_type_column"],
            distance_column=sidecar["distance_column"],
            time_column=sidecar["time_column"],
            height_column=sidecar["height_column"],
            latitude_column=sidecar["latitude_column"],
            longitude_column=sidecar["longitude_column"],
            crs=sidecar["crs"],
            metadata=sidecar["metadata"],
            copy=False,
        )

    def to_parquet(self, path: str | pathlib.Path, **kwargs: typing.Any) -> None:
        """
        Save this survey to Parquet: ``<path>.parquet`` (the data, a plain
        Parquet file readable without airbornegeo), ``<path>_intersections.
        parquet`` (only written if :attr:`intersections` is set), and
        ``<path>.json`` (a sidecar with the CRS, metadata, and column
        bindings needed to reconstruct the ``Survey``). See
        :meth:`from_parquet` to reload. Extra keyword arguments are passed
        to ``DataFrame.to_parquet``.
        """
        prefix = pathlib.Path(path)
        if prefix.suffix:
            msg = f"path must not include a file extension, got {path!r}"
            raise ValueError(msg)
        self._data.to_parquet(prefix.with_suffix(".parquet"), **kwargs)

        has_intersections = self.intersections is not None
        if has_intersections:
            self.intersections.to_parquet(
                prefix.with_name(f"{prefix.name}_intersections.parquet"), **kwargs
            )
        prefix.with_suffix(".json").write_text(
            json.dumps(self._sidecar(has_intersections), indent=2), encoding="utf-8"
        )

    @classmethod
    def from_parquet(cls, path: str | pathlib.Path, **kwargs: typing.Any) -> "Survey":
        """
        Reload a survey saved with :meth:`to_parquet`. Extra keyword
        arguments are passed to ``pandas.read_parquet``.
        """
        prefix = pathlib.Path(path).with_suffix("")
        sidecar = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))

        data = pd.read_parquet(prefix.with_suffix(".parquet"), **kwargs)
        survey = cls._from_sidecar(data, sidecar)
        if sidecar["has_intersections"]:
            survey.intersections = gpd.read_parquet(
                prefix.with_name(f"{prefix.name}_intersections.parquet"), **kwargs
            )
        return survey

    def to_csv(self, path: str | pathlib.Path, **kwargs: typing.Any) -> None:
        """
        Save this survey to CSV: ``<path>.csv`` (the data, a plain CSV file
        readable without airbornegeo), ``<path>_intersections.csv`` (only
        written if :attr:`intersections` is set, with its geometry column
        as WKT text), and ``<path>.json`` (a sidecar with the CRS,
        metadata, and column bindings needed to reconstruct the
        ``Survey``). See :meth:`from_csv` to reload. CSV round-trips numeric
        data exactly but not dtypes (e.g. integer columns become floats) —
        prefer :meth:`to_parquet` unless a plain-text format is needed.
        Extra keyword arguments are passed to ``DataFrame.to_csv``.
        """
        prefix = pathlib.Path(path)
        if prefix.suffix:
            msg = f"path must not include a file extension, got {path!r}"
            raise ValueError(msg)
        self._data.to_csv(prefix.with_suffix(".csv"), index=False, **kwargs)

        has_intersections = self.intersections is not None
        if has_intersections:
            self.intersections.to_csv(
                prefix.with_name(f"{prefix.name}_intersections.csv"),
                index=False,
                **kwargs,
            )
        prefix.with_suffix(".json").write_text(
            json.dumps(self._sidecar(has_intersections), indent=2), encoding="utf-8"
        )

    @classmethod
    def from_csv(cls, path: str | pathlib.Path, **kwargs: typing.Any) -> "Survey":
        """
        Reload a survey saved with :meth:`to_csv`. Extra keyword arguments
        are passed to ``pandas.read_csv``.
        """
        prefix = pathlib.Path(path).with_suffix("")
        sidecar = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))

        data = pd.read_csv(prefix.with_suffix(".csv"), **kwargs)
        survey = cls._from_sidecar(data, sidecar)
        if sidecar["has_intersections"]:
            inters = pd.read_csv(
                prefix.with_name(f"{prefix.name}_intersections.csv"), **kwargs
            )
            inters["geometry"] = gpd.GeoSeries.from_wkt(inters["geometry"])
            survey.intersections = gpd.GeoDataFrame(
                inters, geometry="geometry", crs=survey.crs
            )
        return survey

    ####################
    # cached summary statistics
    ####################

    def _line_types(self) -> pd.Series | None:
        """Per-line line-type code, or None if the columns are unset or missing."""
        line, line_type = self._line_column, self._line_type_column
        if (
            line is None
            or line_type is None
            or line not in self._data.columns
            or line_type not in self._data.columns
        ):
            return None
        return self._data.groupby(line)[line_type].first()

    def _per_type(
        self,
        per_line: pd.Series,
        agg: typing.Callable[[pd.Series], float] = _median,
    ) -> dict[str, float]:
        """Aggregate a per-line series into a value per line-type name."""
        per_line = per_line.dropna()
        types = self._line_types()
        if types is None:
            return {"all": agg(per_line)} if len(per_line) > 0 else {}
        out = {}
        for code, name in _TYPE_NAMES.items():
            lines = types.index[types == code]
            values = per_line.loc[per_line.index.intersection(lines)]
            if len(values) > 0:
                out[name] = agg(values)
        return out

    @functools.cached_property
    def region(self) -> tuple[float, float, float, float]:
        """Bounding region (west, east, south, north) of the survey."""
        data = self._data.dropna(subset=["easting", "northing"])
        return (
            float(data.easting.min()),
            float(data.easting.max()),
            float(data.northing.min()),
            float(data.northing.max()),
        )

    @functools.cached_property
    def line_counts(self) -> dict[str, int]:
        """Number of lines, in total and per line type."""
        self._require(self.line_column)
        counts = {"total": int(self._data[self.line_column].nunique())}
        types = self._line_types()
        if types is not None:
            for code, name in _TYPE_NAMES.items():
                number = int((types == code).sum())
                if number > 0:
                    counts[name] = number
        return counts

    @functools.cached_property
    def line_lengths(self) -> pd.Series:
        """
        Largest bounding dimension of each line, indexed by line. See
        :func:`airbornegeo.largest_line_dimensions`.
        """
        self._require(self.line_column)
        return largest_line_dimensions(self._data, line_column=self.line_column)

    @functools.cached_property
    def total_length(self) -> float:
        """Total survey length; the sum of all line lengths."""
        return float(self.line_lengths.sum())

    @functools.cached_property
    def median_line_lengths(self) -> dict[str, float]:
        """Median line length per line type."""
        return self._per_type(self.line_lengths)

    @functools.cached_property
    def line_azimuths(self) -> pd.Series:
        """
        Compass azimuth (degrees clockwise from north, axial 0-180 range) of
        each line's minimum rotated bounding box long axis, indexed by line.
        """
        self._require(self.line_column)
        data = self._data.dropna(subset=["easting", "northing"])
        azimuths = {}
        for line, line_data in data.groupby(self.line_column):
            points = shapely.MultiPoint(
                np.column_stack((line_data["easting"], line_data["northing"]))
            )
            rect = shapely.oriented_envelope(points)
            if isinstance(rect, shapely.Polygon):
                math_azimuth = azimuth(rect)
            elif isinstance(rect, shapely.LineString):
                # degenerate: perfectly straight line
                coords = list(rect.coords)
                math_azimuth = _azimuth_between_points(coords[0], coords[-1])
            else:  # single point
                math_azimuth = np.nan
            azimuths[line] = (90 - math_azimuth) % 180
        return pd.Series(azimuths, name="azimuth").rename_axis(self.line_column)

    @functools.cached_property
    def mean_line_azimuths(self) -> dict[str, float]:
        """
        Mean compass azimuth per line type, computed as an axial circular
        mean so lines straddling the 0/180 wraparound average correctly.
        """
        return self._per_type(self.line_azimuths, _axial_mean)

    @functools.cached_property
    def median_line_spacings(self) -> dict[str, float]:
        """
        Median line spacing per line type (line types with fewer than 2 lines
        are skipped). See :func:`airbornegeo.median_line_spacing`.
        """
        self._require(self.line_column)
        if (
            self._line_type_column is None
            or self._line_type_column not in self._data.columns
        ):
            if self._data[self.line_column].nunique() < 2:
                return {}
            return {
                "all": float(
                    median_line_spacing(self._data, line_column=self.line_column)
                )
            }
        spacings = {}
        for code, name in _TYPE_NAMES.items():
            subset = self._data[self._data[self.line_type_column] == code]
            if subset[self.line_column].nunique() < 2:
                continue
            spacings[name] = float(
                median_line_spacing(subset, line_column=self.line_column)
            )
        return spacings

    @functools.cached_property
    def line_point_spacings(self) -> pd.Series:
        """
        Median along-line distance between successive data points of each
        line, indexed by line. See :func:`airbornegeo.relative_distance`.
        """
        self._require(self.line_column)
        distances = relative_distance(
            self._data,
            groupby_column=self.line_column,
            progressbar=False,
        )
        return (
            pd.Series(distances, index=self._data.index)
            .groupby(self._data[self.line_column])
            .median()
            .rename("point_spacing")
        )

    @functools.cached_property
    def median_point_spacings(self) -> dict[str, float]:
        """Median along-line data point spacing per line type."""
        return self._per_type(self.line_point_spacings)

    ####################
    # repr and summary
    ####################

    def __repr__(self) -> str:
        computed = [name for name in self._STAT_NAMES if name in self.__dict__]
        n_inters = (
            "None" if self.intersections is None else f"{len(self.intersections)} rows"
        )
        return (
            f"airbornegeo.Survey ({len(self._data):,} rows x "
            f"{len(self._data.columns)} cols)\n"
            f"  crs: {self.crs.to_string() if self.crs else None}\n"
            f"  line_column: {self._line_column!r} | "
            f"line_type_column: {self._line_type_column!r}\n"
            f"  distance_column: {self._distance_column!r} | "
            f"time_column: {self._time_column!r} | "
            f"height_column: {self._height_column!r}\n"
            f"  latitude_column: {self.latitude_column!r} | "
            f"longitude_column: {self.longitude_column!r}\n"
            f"  metadata keys: {sorted(self.metadata) or 'none'}\n"
            f"  intersections: {n_inters}\n"
            f"  computed stats: {', '.join(computed) or 'none'} "
            "(use .describe() to compute all)"
        )

    def describe(self) -> str:
        """
        Compute all summary statistics and return them as a formatted string.
        """

        def _fmt(values: dict[str, float]) -> str:
            return (
                ", ".join(f"{key}: {value:,.1f}" for key, value in values.items())
                or "n/a"
            )

        lines = [
            f"airbornegeo.Survey ({len(self._data):,} rows x "
            f"{len(self._data.columns)} cols)",
            f"  crs: {self.crs.to_string() if self.crs else None}",
            f"  metadata: {self.metadata or 'none'}",
            "  region (W, E, S, N): "
            + ", ".join(f"{value:,.1f}" for value in self.region),
            "  line counts: "
            + ", ".join(f"{key}: {value}" for key, value in self.line_counts.items()),
            f"  total length: {self.total_length:,.1f}",
            f"  median line lengths: {_fmt(self.median_line_lengths)}",
            f"  mean line azimuths: {_fmt(self.mean_line_azimuths)}",
            f"  median line spacings: {_fmt(self.median_line_spacings)}",
            f"  median point spacings: {_fmt(self.median_point_spacings)}",
        ]
        return "\n".join(lines)

    def _format_stat(self, name: str) -> str:
        """Compact one-line rendering of an already-computed statistic."""
        value = self.__dict__[name]
        if isinstance(value, pd.Series):
            return (
                f"{len(value)} lines | min {value.min():,.1f} | "
                f"median {value.median():,.1f} | max {value.max():,.1f}"
            )
        if isinstance(value, dict):
            return (
                ", ".join(
                    f"{key}: {val:,.1f}" if isinstance(val, float) else f"{key}: {val}"
                    for key, val in value.items()
                )
                or "n/a"
            )
        if isinstance(value, tuple):
            return ", ".join(f"{val:,.1f}" for val in value)
        return f"{value:,.1f}" if isinstance(value, float) else str(value)

    def _repr_html_(self) -> str:
        """
        Rich HTML representation with collapsible sections, shown in Jupyter.
        Only reports statistics which have already been computed — it never
        triggers computation itself.
        """

        def esc(value: typing.Any) -> str:
            return html.escape(str(value))

        def rows(pairs: list[tuple[str, str]]) -> str:
            return "".join(
                f"<tr><td>{name}</td><td>{value}</td></tr>" for name, value in pairs
            )

        def section(title: str, body: str, note: str = "", open_: bool = True) -> str:
            note_html = f'<span class="ag-note">{note}</span>' if note else ""
            return (
                f"<details{' open' if open_ else ''}>"
                f"<summary>{title}{note_html}</summary>{body}</details>"
            )

        def binding(name: str) -> str:
            # structural bindings are stored privately behind raising properties,
            # so read the raw value here — an unset binding must render, not raise
            attribute = f"{name}_column"
            private = f"_{attribute}"
            value = (
                getattr(self, private)
                if hasattr(self, private)
                else getattr(self, attribute)
            )
            return (
                f"<code>{esc(value)}</code>"
                if value is not None
                else '<span class="ag-note">not set</span>'
            )

        column_rows = rows(
            [
                (name, binding(name))
                for name in (
                    "line",
                    "line_type",
                    "distance",
                    "time",
                    "height",
                    "latitude",
                    "longitude",
                )
            ]
        )

        meta_pairs = [("crs", esc(self.crs.to_string()) if self.crs else "None")]
        meta_pairs += [(esc(key), esc(val)) for key, val in self.metadata.items()]

        computed = [name for name in self._STAT_NAMES if name in self.__dict__]
        pending = [name for name in self._STAT_NAMES if name not in self.__dict__]
        stat_rows = rows(
            [(name, esc(self._format_stat(name))) for name in computed]
            + [(name, '<span class="ag-note">not computed</span>') for name in pending]
        )
        stats_note = (
            "all computed"
            if not pending
            else f"{len(computed)}/{len(self._STAT_NAMES)} computed "
            "&mdash; access an attribute or call .describe()"
        )

        if self.intersections is None:
            inters_html = '<p class="ag-note">None &mdash; call '
            inters_html += "<code>create_intersection_table()</code></p>"
            inters_note = "None"
        else:
            inters_html = (
                f"<p>{len(self.intersections):,} intersections with columns: "
                f"<code>{esc(list(self.intersections.columns))}</code></p>"
            )
            inters_note = f"{len(self.intersections):,} rows"

        return (
            '<div class="ag-survey"><style>'
            ".ag-survey{font-family:var(--jp-code-font-family,monospace);"
            "font-size:0.9em;line-height:1.4;max-width:60em;}"
            ".ag-survey .ag-header{font-weight:bold;padding:4px 0;"
            "border-bottom:1px solid rgba(128,128,128,0.4);margin-bottom:4px;}"
            ".ag-survey .ag-shape{font-weight:normal;opacity:0.65;}"
            ".ag-survey details{padding:1px 0 1px 12px;"
            "border-left:2px solid rgba(128,128,128,0.25);margin:3px 0;}"
            ".ag-survey summary{cursor:pointer;font-weight:bold;"
            "list-style-position:outside;margin-left:-6px;}"
            ".ag-survey .ag-note{font-weight:normal;font-style:italic;"
            "opacity:0.6;margin-left:8px;}"
            ".ag-survey table{border-collapse:collapse;margin:2px 0;}"
            ".ag-survey td{padding:1px 12px 1px 0;border:none;"
            "text-align:left;vertical-align:top;}"
            ".ag-survey td:first-child{opacity:0.7;}"
            ".ag-survey .ag-data{overflow-x:auto;}"
            "</style>"
            '<div class="ag-header">airbornegeo.Survey '
            f'<span class="ag-shape">{len(self._data):,} rows &times; '
            f"{len(self._data.columns)} columns</span></div>"
            + section("Column attributes", f"<table>{column_rows}</table>")
            + section(
                "CRS &amp; metadata",
                f"<table>{rows(meta_pairs)}</table>",
                note=f"{len(self.metadata)} metadata entries",
            )
            + section(
                "Statistics",
                f"<table>{stat_rows}</table>",
                note=stats_note,
                open_=bool(computed),
            )
            + section("Intersections", inters_html, note=inters_note, open_=False)
            + section(
                "Data",
                f'<div class="ag-data">{self._data.head()._repr_html_()}</div>',  # pylint: disable=protected-access
                note="first 5 rows",
                open_=False,
            )
            + "</div>"
        )

    ####################
    # 2D plotting
    ####################

    def plot(
        self,
        *,
        ax: mpl.axes.Axes | None = None,
        color_by: str | None = None,
        categorical: bool = False,
        coarsen: int | None = None,
        cmap: str | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        absolute: bool = False,
        robust: bool = True,
        label_categories: bool = False,
        markerscale: float = 4,
        s: float = 0.5,
        scalebar: bool = True,
        scalebar_length: float | None = None,
        title: str | None = None,
        colorbar_label: str | None = None,
        **kwargs: typing.Any,
    ) -> mpl.axes.Axes:
        """
        Plot the survey points in map view, optionally colored by a column.

        Parameters
        ----------
        ax : mpl.axes.Axes | None, optional
            Axes to plot into. A new figure and axes are created if not given.
        color_by : str | None, optional
            Column to color the points by. If None, all points are drawn in a
            single color.
        categorical : bool, optional
            Treat `color_by` as categorical (e.g. the line or line type
            column), drawing each category in its own color, by default False.
        coarsen : int | None, optional
            Plot only every nth point, to speed up plotting of large surveys,
            by default None.
        cmap : str | None, optional
            Colormap name. If not given (and the plot isn't categorical), a
            suitable colormap is chosen with
            :func:`airbornegeo.choose_colormap`.
        vmin, vmax : float | None, optional
            Color limits. If not given they are determined from the data.
        absolute : bool, optional
            Force symmetric color limits centered on zero, by default False.
        robust : bool, optional
            Use percentiles instead of the full data range for the color
            limits, so outliers don't dominate, by default True.
        label_categories : bool, optional
            Add a legend naming each category, by default False. Only used
            when `categorical` is True.
        markerscale : float, optional
            Scale factor for the marker size in the legend, by default 4.
        s : float, optional
            Marker size, by default 0.5.
        scalebar : bool, optional
            Add a scalebar, by default True.
        scalebar_length : float | None, optional
            Length of the scalebar in the coordinates' units. If not given, a
            nicely rounded fraction of the map width is used.
        title : str | None, optional
            Plot title. Defaults to the survey's 'survey' metadata entry, if
            there is one.
        colorbar_label : str | None, optional
            Label for the colorbar, by default the `color_by` column name.
        kwargs : typing.Any
            Additional keyword arguments passed to the matplotlib scatter call.

        Returns
        -------
        mpl.axes.Axes
            The axes the survey was plotted into.
        """
        if color_by is not None:
            self._require(color_by)
        if categorical and color_by is None:
            msg = "categorical=True requires a color_by column"
            raise ValueError(msg)

        data = self._data if coarsen is None else self._data[::coarsen]

        if ax is None:
            _fig, ax = plt.subplots()

        if categorical:
            for category, category_data in data.groupby(color_by):
                ax.scatter(
                    category_data.easting,
                    category_data.northing,
                    s=s,
                    label=category if label_categories else None,
                    lw=0,
                    **kwargs,
                )
            if label_categories:
                ax.legend(title=color_by, markerscale=markerscale)
        elif color_by is None:
            ax.scatter(data.easting, data.northing, s=s, lw=0, **kwargs)
        else:
            values = data[color_by]
            if absolute:
                maxabs = float(vd.maxabs(values, percentile=98 if robust else 100))
                default_cmap, default_min, default_max = "RdBu", -maxabs, maxabs
            else:
                default_cmap, default_min, default_max = choose_colormap(
                    values,
                    robust=robust,
                )
            points = ax.scatter(
                data.easting,
                data.northing,
                c=values,
                s=s,
                lw=0,
                cmap=cmap if cmap is not None else default_cmap,
                vmin=default_min if vmin is None else vmin,
                vmax=default_max if vmax is None else vmax,
                **kwargs,
            )
            colorbar = ax.figure.colorbar(points, ax=ax)
            colorbar.set_label(color_by if colorbar_label is None else colorbar_label)

        ax.set_xlabel("easting")
        ax.set_ylabel("northing")
        ax.set_aspect("equal")

        if title is None:
            survey_name = self.metadata.get("survey")
            title = None if survey_name is None else f"Survey: {survey_name}"
        if title is not None:
            ax.set_title(title)

        if scalebar:
            if scalebar_length is None:
                x_width = abs(ax.get_xlim()[1] - ax.get_xlim()[0])
                scalebar_length = nice_scalebar_width(x_width)
            add_scalebar(ax, scalebar_length)

        return ax

    def plotly_points(self, **kwargs: typing.Any) -> go.Figure:
        """
        Interactive map-view scatter of the survey points. See
        :func:`airbornegeo.plotly_points` for all parameters.
        """
        color_col = kwargs.get("color_col")
        if color_col is not None:
            self._require(color_col)
        kwargs.setdefault("title", self.metadata.get("survey"))
        return plotly_points(self._data, **kwargs)

    def plot_profiles(
        self,
        *,
        y: list[str] | str,
        x: str | None = None,
        **kwargs: typing.Any,
    ) -> None:
        """
        Plot data profiles with matplotlib, against ``x`` (default the
        survey's distance column). See :func:`airbornegeo.plot_profiles` for
        all parameters.
        """
        x = x or self.distance_column
        self._require(x, *([y] if isinstance(y, str) else y))
        return plot_profiles(self._data, y=y, x=x, **kwargs)

    def plotly_profiles(
        self,
        *,
        y: list[str] | str,
        x: str | None = None,
        **kwargs: typing.Any,
    ) -> go.Figure:
        """
        Plot data profiles with plotly, against ``x`` (default the survey's
        distance column). See :func:`airbornegeo.plotly_profiles` for all
        parameters.
        """
        x = x or self.distance_column
        self._require(x, *([y] if isinstance(y, str) else y))
        return plotly_profiles(self._data, y=y, x=x, **kwargs)

    def inspect_lines(
        self,
        *,
        plot_variable: str | list[str],
        interp_on: str | None = None,
        **kwargs: typing.Any,
    ) -> None:
        """
        Step through a plotly profile of each line in turn, plotted against
        ``interp_on`` (default the survey's distance column). Blocks for a
        keypress between lines. See :func:`airbornegeo.inspect_lines` for all
        parameters.
        """
        interp_on = interp_on or self.distance_column
        self._require(
            interp_on,
            self.line_column,
            *([plot_variable] if isinstance(plot_variable, str) else plot_variable),
        )
        return inspect_lines(
            self._canonical(),
            plot_variable=plot_variable,
            interp_on=interp_on,
            **kwargs,
        )

    def plot_line_and_crosses(
        self,
        *,
        y: list[str] | str,
        x: str | None = None,
        **kwargs: typing.Any,
    ) -> go.Figure:
        """
        Plot one line's profiles against ``x`` (default the survey's distance
        column), marking its intersections. Needs the intersection rows in the
        dataframe, added by :meth:`add_intersections`. See
        :func:`airbornegeo.plot_line_and_crosses` for all parameters.
        """
        x = x or self.distance_column
        self._require(x, self.line_column, *([y] if isinstance(y, str) else y))
        return plot_line_and_crosses(
            self._data,
            y=y,
            line_column=self.line_column,
            x=x,
            **kwargs,
        )

    def inspect_intersections(
        self,
        *,
        plot_variable: str | list[str],
        x: str | None = None,
        **kwargs: typing.Any,
    ) -> None:
        """
        Step through a plotly profile of each line with intersections in turn,
        plotted against ``x`` (default the survey's distance column). Blocks
        for a keypress between lines unless ``plot_all=True``. Needs the
        intersection rows in the dataframe, added by :meth:`add_intersections`.
        See :func:`airbornegeo.inspect_intersections` for all parameters.
        """
        x = x or self.distance_column
        self._require(
            x,
            self.line_column,
            *([plot_variable] if isinstance(plot_variable, str) else plot_variable),
        )
        return inspect_intersections(
            self._data,
            x=x,
            plot_variable=plot_variable,
            line_column=self.line_column,
            **kwargs,
        )

    ####################
    # coordinate methods (in place, chainable)
    ####################

    def reproject(
        self,
        output_crs: typing.Any,
        *,
        input_crs: typing.Any | None = None,
    ) -> typing.Self:
        """
        Reproject the survey's easting/northing columns into ``output_crs``
        and set it as the survey's crs. ``input_crs`` defaults to the
        survey's current crs, which must then be set. See
        :func:`airbornegeo.reproject`.
        """
        source = (
            self.crs if input_crs is None else pyproj.CRS.from_user_input(input_crs)
        )
        if source is None:
            msg = (
                "this survey has no crs set; give the coordinates' crs with "
                "input_crs, or set survey.crs"
            )
            raise ValueError(msg)
        target = pyproj.CRS.from_user_input(output_crs)

        easting, northing = reproject(
            self._data["easting"],
            self._data["northing"],
            source.to_string(),
            target.to_string(),
        )
        # a GeoDataFrame's geometry would otherwise be left in the old crs
        if isinstance(self._data, gpd.GeoDataFrame) and self._data.crs is not None:
            self._data = self._data.to_crs(target)
        self._data["easting"] = easting
        self._data["northing"] = northing
        self.crs = target
        self._invalidate_stats()
        return self

    ####################
    # navigation / trajectory methods (in place, chainable)
    ####################

    def along_track_distance(self, **kwargs: typing.Any) -> typing.Self:
        """
        Compute distance along each line, written to the survey's distance
        column. See :func:`airbornegeo.along_track_distance` for all
        parameters.
        """
        self._require(self.line_column)
        self._data[self.distance_column] = along_track_distance(
            self._data,
            groupby_column=self.line_column,
            **kwargs,
        )
        return self

    def relative_distance(
        self,
        *,
        result_column: str = "relative_distance",
        **kwargs: typing.Any,
    ) -> typing.Self:
        """
        Compute the distance between consecutive points of each line, written
        to ``result_column``. See :func:`airbornegeo.relative_distance` for
        all parameters.
        """
        self._require(self.line_column)
        self._data[result_column] = relative_distance(
            self._data,
            groupby_column=self.line_column,
            **kwargs,
        )
        return self

    def ground_speed(
        self,
        *,
        result_column: str = "ground_speed",
        **kwargs: typing.Any,
    ) -> typing.Self:
        """
        Compute ground speed along each line, written to ``result_column``.
        See :func:`airbornegeo.ground_speed` for all parameters.
        """
        self._require(self.line_column, self.time_column)
        self._data[result_column] = ground_speed(
            self._data,
            time_column=self.time_column,
            groupby_column=self.line_column,
            **kwargs,
        )
        return self

    def track(
        self,
        *,
        result_column: str = "track",
        **kwargs: typing.Any,
    ) -> typing.Self:
        """
        Compute the travel azimuth (track) along each line, written to
        ``result_column``. See :func:`airbornegeo.track` for all parameters.
        """
        latitude_column, longitude_column = self._require_geographic()
        self._require(self.line_column)
        self._data[result_column] = track(
            self._data,
            latitude_column=latitude_column,
            longitude_column=longitude_column,
            groupby_column=self.line_column,
            **kwargs,
        )
        return self

    def directional_velocity(
        self,
        *,
        coordinate_column: str = "easting",
        result_column: str | None = None,
        **kwargs: typing.Any,
    ) -> typing.Self:
        """
        Compute velocity along one coordinate direction, written to
        ``result_column`` (default ``'<coordinate_column>_velocity'``). See
        :func:`airbornegeo.directional_velocity` for all parameters.
        """
        self._require(self.line_column, self.time_column, coordinate_column)
        self._data[result_column or f"{coordinate_column}_velocity"] = (
            directional_velocity(
                self._data,
                time_column=self.time_column,
                coordinate_column=coordinate_column,
                groupby_column=self.line_column,
                **kwargs,
            )
        )
        return self

    def vertical_acceleration(
        self,
        *,
        result_column: str = "vertical_acceleration",
        **kwargs: typing.Any,
    ) -> typing.Self:
        """
        Compute vertical acceleration along each line, written to
        ``result_column``. See :func:`airbornegeo.vertical_acceleration` for
        all parameters.
        """
        self._require(self.line_column, self.time_column, self.height_column)
        self._data[result_column] = vertical_acceleration(
            self._data,
            time_column=self.time_column,
            height_column=self.height_column,
            groupby_column=self.line_column,
            **kwargs,
        )
        return self

    ####################
    # line creation methods (in place, chainable)
    ####################

    def split_into_segments(
        self,
        threshold: float,
        *,
        column_name: str | None = None,
        result_column: str | None = None,
        **kwargs: typing.Any,
    ) -> typing.Self:
        """
        Split the survey into segments where ``column_name`` (default the
        survey's time column) jumps by more than ``threshold``, writing the
        segment names to ``result_column`` (default the survey's line
        column). See :func:`airbornegeo.split_into_segments` for all
        parameters.
        """
        column_name = column_name or self.time_column
        self._require(column_name)
        self._data[result_column or self.line_column] = split_into_segments(
            self._data,
            threshold,
            column_name,
            **kwargs,
        )
        self._invalidate_stats()
        return self

    def unique_line_id(self) -> typing.Self:
        """
        Replace the line column's values with unique sequential integer ids.
        See :func:`airbornegeo.unique_line_id`.
        """
        self._require(self.line_column)
        self._data[self.line_column] = unique_line_id(
            self._data,
            line_col_name=self.line_column,
        )
        self._invalidate_stats()
        return self

    ####################
    # filtering / interpolation / sampling methods (in place, chainable)
    ####################

    def filter_line(
        self,
        *,
        data_column: str,
        filter_width: float,
        result_column: str | None = None,
        filter_by_column: str | None = None,
        **kwargs: typing.Any,
    ) -> typing.Self:
        """
        Filter a data column along each line, written to ``result_column``
        (default ``'<data_column>_filtered'``). See
        :func:`airbornegeo.filter_line` for all parameters.
        """
        filter_by = filter_by_column or self.distance_column
        self._require(data_column, filter_by, self.line_column)
        self._data[result_column or f"{data_column}_filtered"] = filter_line(
            self._data,
            filter_width=filter_width,
            data_column=data_column,
            filter_by_column=filter_by,
            groupby_column=self.line_column,
            **kwargs,
        )
        return self

    def sample_grid(
        self,
        grid: xr.DataArray,
        *,
        result_column: str,
        interpolation: str = "cubic",
    ) -> typing.Self:
        """
        Sample a grid at the survey's coordinates, written to
        ``result_column``. See :func:`airbornegeo.sample_grid`.
        """
        self._data[result_column] = sample_grid(
            grid,
            self._data["easting"].to_numpy(),
            self._data["northing"].to_numpy(),
            interpolation=interpolation,
        )
        return self

    def interpolate_missing(
        self,
        *,
        to_interp: str,
        interp_on: str | None = None,
        **kwargs: typing.Any,
    ) -> typing.Self:
        """
        Fill NaN values of ``to_interp`` by interpolating along ``interp_on``
        (default the survey's distance column) for each line. See
        :func:`airbornegeo.interpolate_missing` for all parameters.
        """
        interp_on = interp_on or self.distance_column
        self._require(to_interp, interp_on, self.line_column)
        self.data = interpolate_missing(
            self._data,
            to_interp=to_interp,
            interp_on=interp_on,
            groupby_column=self.line_column,
            **kwargs,
        )
        return self

    def interpolate_missing_pointwise(
        self,
        *,
        to_interp: str,
        interp_on: str | None = None,
        **kwargs: typing.Any,
    ) -> typing.Self:
        """
        Fill NaN values of ``to_interp`` one point at a time by interpolating
        along ``interp_on`` (default the survey's distance column) for each
        line. See :func:`airbornegeo.interpolate_missing_pointwise` for all
        parameters.
        """
        interp_on = interp_on or self.distance_column
        self._require(to_interp, interp_on, self.line_column)
        self.data = interpolate_missing_pointwise(
            self._data,
            to_interp=to_interp,
            interp_on=interp_on,
            groupby_column=self.line_column,
            **kwargs,
        )
        return self

    def interpolate_missing_pointwise_with_windows(
        self,
        *,
        window_width: float,
        to_interp: str,
        interp_on: str | None = None,
        **kwargs: typing.Any,
    ) -> typing.Self:
        """
        Fill NaN values of ``to_interp`` one point at a time using a moving
        window along ``interp_on`` (default the survey's distance column) for
        each line. See
        :func:`airbornegeo.interpolate_missing_pointwise_with_windows` for
        all parameters.
        """
        interp_on = interp_on or self.distance_column
        self._require(to_interp, interp_on, self.line_column)
        self.data = interpolate_missing_pointwise_with_windows(
            self._data,
            window_width=window_width,
            to_interp=to_interp,
            interp_on=interp_on,
            groupby_column=self.line_column,
            **kwargs,
        )
        return self

    ####################
    # levelling and potential-field methods (in place, chainable)
    ####################

    def level_to_grid(
        self,
        *,
        data_column: str,
        grid_column: str,
        result_column: str | None = None,
        fit_by_column: str | tuple[str] | None = None,
        **kwargs: typing.Any,
    ) -> typing.Self:
        """
        Level a data column to values sampled from a grid, written to
        ``result_column`` (default ``'<data_column>_levelled'``). See
        :func:`airbornegeo.level_to_grid` for all parameters.
        """
        fit_by = fit_by_column or self.distance_column
        self._require(data_column, grid_column, self.line_column)
        self._data[result_column or f"{data_column}_levelled"] = level_to_grid(
            self._data,
            data_column=data_column,
            grid_column=grid_column,
            fit_by_column=fit_by,
            groupby_column=self.line_column,
            **kwargs,
        )
        return self

    def equivalent_source_levelling(
        self,
        *,
        data_column: str,
        max_dist: float,
        degree: int,
        result_column: str | None = None,
        **kwargs: typing.Any,
    ) -> typing.Self:
        """
        Level lines with iteratively fitted equivalent sources, written to
        ``result_column`` (default ``'<data_column>_levelled'``). See
        :func:`airbornegeo.equivalent_source_levelling` for all parameters.
        """
        self._require(
            data_column,
            self.line_column,
            self.height_column,
            self.distance_column,
        )
        self._data[result_column or f"{data_column}_levelled"] = (
            equivalent_source_levelling(
                self._canonical(),
                data_column=data_column,
                distance_column="distance_along_line",
                max_dist=max_dist,
                degree=degree,
                **kwargs,
            )
        )
        return self

    def eq_sources_1d(
        self,
        *,
        data_column: str,
        damping: float,
        **kwargs: typing.Any,
    ) -> dict | hm.EquivalentSources:
        """
        Fit equivalent sources along each line and return them (a dict keyed
        by line name), for use with :meth:`upward_continue_by_line` or
        :meth:`update_intersections_with_eq_sources`. See
        :func:`airbornegeo.eq_sources_1d` for all parameters.
        """
        self._require(
            data_column,
            self.line_column,
            self.height_column,
            self.distance_column,
        )
        return eq_sources_1d(
            self._canonical(),
            data_column=data_column,
            damping=damping,
            groupby_column="line",
            **kwargs,
        )

    def upward_continue_by_line(
        self,
        fitted_equivalent_sources: dict,
        height: float,
        *,
        result_column: str = "upward_continued",
        **kwargs: typing.Any,
    ) -> typing.Self:
        """
        Upward continue each line to a constant height using fitted
        equivalent sources, written to ``result_column``. See
        :func:`airbornegeo.upward_continue_by_line` for all parameters.
        """
        self._require(self.line_column, self.height_column, self.distance_column)
        self._data[result_column] = upward_continue_by_line(
            self._canonical(),
            fitted_equivalent_sources,
            height,
            groupby_column="line",
            **kwargs,
        )
        return self

    def igrf(
        self,
        *,
        datetime_column: str,
        result_columns: tuple[str, str, str] = (
            "igrf_intensity",
            "igrf_inclination",
            "igrf_declination",
        ),
        **kwargs: typing.Any,
    ) -> typing.Self:
        """
        Compute the IGRF intensity, inclination and declination, written to
        the three ``result_columns``. See :func:`airbornegeo.igrf` for all
        parameters.
        """
        latitude_column, longitude_column = self._require_geographic()
        self._require(datetime_column, self.height_column, self.line_column)
        intensity, inclination, declination = igrf(
            self._data,
            datetime_column=datetime_column,
            latitude_column=latitude_column,
            longitude_column=longitude_column,
            height_column=self.height_column,
            groupby_column=self.line_column,
            **kwargs,
        )
        for column, values in zip(
            result_columns,
            (intensity, inclination, declination),
            strict=True,
        ):
            self._data[column] = values
        return self

    def eotvos_correction(
        self,
        *,
        method: str = "full",
        result_column: str = "eotvos_correction",
        track_column: str = "track",
        ground_speed_column: str = "ground_speed",
        by_line: bool = True,
        **kwargs: typing.Any,
    ) -> typing.Self:
        """
        Compute the Eötvös correction, written to ``result_column``. Computed
        line by line unless ``by_line=False``, since most methods take time
        derivatives which are meaningless across the join between two lines.
        Method 'glicken' needs ``track_column`` and ``ground_speed_column``,
        which :meth:`track` and :meth:`ground_speed` create. See
        :func:`airbornegeo.eotvos_correction` for all parameters.
        """
        latitude_column, longitude_column = self._require_geographic()
        if by_line:
            self._require(self.line_column)
        self._data[result_column] = eotvos_correction(
            self._data,
            method=method,
            latitude_column=latitude_column,
            longitude_column=longitude_column,
            time_column=self._time_column,
            height_column=self._height_column,
            track_column=track_column,
            ground_speed_column=ground_speed_column,
            groupby_column=self.line_column if by_line else None,
            **kwargs,
        )
        return self

    ####################
    # frame-replacing methods (return a NEW Survey)
    ####################

    def block_reduce(
        self,
        reduction: typing.Callable[..., float | int],
        *,
        spacing: float,
        reduce_by: str | tuple[str, str] | None = None,
        **kwargs: typing.Any,
    ) -> "Survey":
        """
        Return a new Survey with block-reduced data; this survey is
        unchanged and its intersection table is not carried over. Only the
        numeric columns, the ``reduce_by`` column(s) and the line column
        survive the reduction. See :func:`airbornegeo.block_reduce` for all
        parameters.
        """
        reduce_by = reduce_by or self.distance_column
        self._require(self.line_column)
        reduced = block_reduce(
            self._data,
            reduction,
            spacing=spacing,
            reduce_by=reduce_by,
            groupby_column=self.line_column,
            **kwargs,
        )
        return self._spawn(reduced)

    def resample(
        self,
        *,
        spacing: float,
        maxdist: float | None,
        resample_by: str | None = None,
        **kwargs: typing.Any,
    ) -> "Survey":
        """
        Return a new Survey resampled onto a regular spacing of
        ``resample_by`` (default the survey's distance column); this survey
        is unchanged and its intersection table is not carried over. Only
        numeric columns survive resampling — a survey with string line names
        must call :meth:`unique_line_id` first. See
        :func:`airbornegeo.resample` for all parameters.
        """
        resample_by = resample_by or self.distance_column
        self._require(resample_by, self.line_column)
        resampled = resample(
            self._data,
            spacing=spacing,
            resample_by=resample_by,
            maxdist=maxdist,
            groupby_column=self.line_column,
            **kwargs,
        )
        return self._spawn(resampled)

    def resample_as(
        self,
        *,
        resample_values: NDArray,
        resample_by: str | None = None,
        **kwargs: typing.Any,
    ) -> "Survey":
        """
        Return a new Survey resampled onto the supplied values of
        ``resample_by`` (default the survey's distance column); this survey
        is unchanged and its intersection table is not carried over. Only
        numeric columns survive resampling — a survey with string line names
        must call :meth:`unique_line_id` first. See
        :func:`airbornegeo.resample_as` for all parameters.
        """
        resample_by = resample_by or self.distance_column
        self._require(resample_by, self.line_column)
        resampled = resample_as(
            self._data,
            resample_by=resample_by,
            resample_values=resample_values,
            groupby_column=self.line_column,
            **kwargs,
        )
        return self._spawn(resampled)

    ####################
    # intersection / crossover methods (in place, chainable)
    ####################

    def create_intersection_table(
        self,
        *,
        method: str,
        **kwargs: typing.Any,
    ) -> typing.Self:
        """
        Create the intersection table of the survey's lines, stored as
        ``survey.intersections``. ``method='groups'`` intersects flight lines
        (line type 0) with tie lines (line type 1); ``method='network'``
        intersects all line pairs. See
        :func:`airbornegeo.create_intersection_table` for all parameters.
        """
        if method == "groups":
            data = self._canonical()
            line_column = "line"
        else:
            self._require(self.line_column)
            data = self._data
            line_column = self.line_column
        self.intersections = create_intersection_table(
            data,
            line_column=line_column,
            method=method,
            **kwargs,
        )
        return self

    def add_intersections(self, **kwargs: typing.Any) -> typing.Self:
        """
        Insert the intersection points as rows into the survey dataframe and
        add their along-line distances to the intersection table. See
        :func:`airbornegeo.add_intersections` for all parameters.
        """
        intersections = self._require_intersections()
        self._require(self.line_column, self.distance_column)
        new_data, new_intersections = add_intersections(
            self._data,
            intersections,
            line_column=self.line_column,
            distance_column=self.distance_column,
            **kwargs,
        )
        self.data = new_data
        self.intersections = new_intersections
        return self

    def interpolate_intersections(
        self,
        *,
        to_interp: str,
        interp_on: str | None = None,
        **kwargs: typing.Any,
    ) -> typing.Self:
        """
        Interpolate ``to_interp`` onto the intersection rows along
        ``interp_on`` (default the survey's distance column), updating both
        the survey dataframe and the intersection table. See
        :func:`airbornegeo.interpolate_intersections` for all parameters.
        """
        intersections = self._require_intersections()
        interp_on = interp_on or self.distance_column
        self._require(to_interp, interp_on, self.line_column)
        new_data, new_intersections = interpolate_intersections(
            self._data,
            intersections,
            line_column=self.line_column,
            to_interp=to_interp,
            interp_on=interp_on,
            **kwargs,
        )
        self.data = new_data
        self.intersections = new_intersections
        return self

    def add_values_to_intersections(
        self,
        columns: tuple[str],
    ) -> typing.Self:
        """
        Add each line's values of the given columns to the intersection
        table. See :func:`airbornegeo.add_values_to_intersections`.
        """
        intersections = self._require_intersections()
        self._require(self.line_column, *columns)
        self.intersections = add_values_to_intersections(
            self._data,
            intersections,
            line_column=self.line_column,
            columns=columns,
        )
        return self

    def calculate_crossover_errors(
        self,
        *,
        data_col: str,
        **kwargs: typing.Any,
    ) -> typing.Self:
        """
        Calculate crossover errors of ``data_col``, added to the intersection
        table as a ``crossover_error_N`` column. See
        :func:`airbornegeo.calculate_crossover_errors` for all parameters.
        """
        intersections = self._require_intersections()
        self._require(data_col, self.line_column)
        self.intersections = calculate_crossover_errors(
            self._data,
            intersections,
            data_col=data_col,
            line_column=self.line_column,
            **kwargs,
        )
        return self

    def lines_without_intersections(self) -> list[float]:
        """
        Return the lines which have no intersections in the intersection
        table. See :func:`airbornegeo.lines_without_intersections`.
        """
        intersections = self._require_intersections()
        self._require(self.line_column)
        return lines_without_intersections(
            self._data,
            intersections,
            self.line_column,
        )

    def update_intersections_with_eq_sources(
        self,
        *,
        fitted_equivalent_sources: dict,
        data_column: str,
        result_column: str | None = None,
        **kwargs: typing.Any,
    ) -> typing.Self:
        """
        Replace the interpolated values at the intersection rows with values
        predicted by fitted equivalent sources, written to ``result_column``
        (default overwriting ``data_column``). See
        :func:`airbornegeo.update_intersections_with_eq_sources` for all
        parameters.
        """
        self._require(data_column, self.line_column, self.distance_column)
        self._data[result_column or data_column] = update_intersections_with_eq_sources(
            self._canonical(),
            fitted_equivalent_sources=fitted_equivalent_sources,
            data_column=data_column,
            line_column="line",
            distance_column="distance_along_line",
            **kwargs,
        )
        return self

    ####################
    # crossover levelling methods (in place, chainable)
    ####################

    def crossover_pair_levelling(
        self,
        *,
        data_col: str,
        lines_to_level: list[float],
        levelled_col: str | None = None,
        **kwargs: typing.Any,
    ) -> typing.Self:
        """
        Level the given lines to the rest of the survey using crossover
        errors, written to ``levelled_col`` (default
        ``'<data_col>_levelled'``); updates the survey dataframe and the
        intersection table in place. See
        :func:`airbornegeo.crossover_pair_levelling` for all parameters.
        """
        intersections = self._require_intersections()
        self._require(data_col, self.line_column, self.distance_column)
        new_data, new_intersections = crossover_pair_levelling(
            self._data,
            intersections,
            lines_to_level=lines_to_level,
            data_col=data_col,
            levelled_col=levelled_col or f"{data_col}_levelled",
            line_column=self.line_column,
            distance_column=self.distance_column,
            **kwargs,
        )
        self.data = new_data
        self.intersections = new_intersections
        return self

    def crossover_network_levelling(
        self,
        *,
        data_col: str,
        levelled_col: str | None = None,
        **kwargs: typing.Any,
    ) -> typing.Self:
        """
        Network-level all lines using crossover errors, written to
        ``levelled_col`` (default ``'<data_col>_levelled'``); updates the
        survey dataframe and the intersection table in place. See
        :func:`airbornegeo.crossover_network_levelling` for all parameters.
        """
        intersections = self._require_intersections()
        self._require(data_col, self.line_column, self.distance_column)
        new_data, new_intersections = crossover_network_levelling(
            self._data,
            intersections,
            data_col=data_col,
            levelled_col=levelled_col or f"{data_col}_levelled",
            line_column=self.line_column,
            distance_column=self.distance_column,
            **kwargs,
        )
        self.data = new_data
        self.intersections = new_intersections
        return self

    def alternating_iterative_line_levelling(
        self,
        *,
        data_col: str,
        levelled_col: str | None = None,
        **kwargs: typing.Any,
    ) -> typing.Self:
        """
        Alternately level flight lines (line type 0) and tie lines (line type
        1) using crossover errors, written to ``levelled_col`` (default
        ``'<data_col>_levelled'``); updates the survey dataframe and the
        intersection table in place. See
        :func:`airbornegeo.alternating_iterative_line_levelling` for all
        parameters.
        """
        intersections = self._require_intersections()
        self._require(
            data_col,
            self.line_column,
            self.line_type_column,
            self.distance_column,
        )
        new_data, new_intersections = alternating_iterative_line_levelling(
            self._canonical(),
            intersections,
            data_col=data_col,
            levelled_col=levelled_col or f"{data_col}_levelled",
            line_column="line",
            distance_column="distance_along_line",
            **kwargs,
        )
        self.data = self._canonical_inverse(new_data)
        self.intersections = new_intersections
        return self


Survey._STAT_NAMES = tuple(  # pylint: disable=protected-access
    name
    for name, attribute in vars(Survey).items()
    if isinstance(attribute, functools.cached_property)
)
