import functools

import matplotlib as mpl

mpl.use("Agg")

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import pytest

from airbornegeo.eotvos import eotvos_correction
from airbornegeo.reproject import reproject as ag_reproject
from airbornegeo.survey import Survey

CRS = "EPSG:3431"


def _synthetic_survey(
    line_column="line",
    line_type_column="line_type",
    biases=(5.0, -3.0, 0.0),
    **extra_columns,
):
    """
    Two E-W flight lines (northing 0 and 100, line_type 0) and one N-S tie
    line (easting 50, line_type 1), points every 10 m over 0-100. The 'value'
    column holds a constant per-line bias.
    """
    along = np.arange(0.0, 101.0, 10.0)
    frames = []
    lines = [
        ("L1", along, np.zeros_like(along), 0, biases[0]),
        ("L2", along, np.full_like(along, 100.0), 0, biases[1]),
        ("T1", np.full_like(along, 50.0), along, 1, biases[2]),
    ]
    for name, easting, northing, line_type, bias in lines:
        frame = pd.DataFrame(
            {
                "easting": easting,
                "northing": northing,
                "unixtime": np.arange(len(along), dtype=float),
                "height": 1000.0,
                "value": bias,
            }
        )
        frame[line_column] = name
        frame[line_type_column] = line_type
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    for column, values in extra_columns.items():
        data[column] = values
    return data


# the Survey only defaults easting/northing, so tests name the rest explicitly
_BINDINGS = {
    "line_column": "line",
    "line_type_column": "line_type",
    "distance_column": "distance_along_line",
    "time_column": "unixtime",
    "height_column": "height",
}


def _survey(**kwargs):
    return Survey(_synthetic_survey(), **{**_BINDINGS, **kwargs})


# ---------------------------------------------------------------------------
# init and validation
# ---------------------------------------------------------------------------


def test_init_missing_coordinates_raises():
    data = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
    with pytest.raises(AssertionError, match="easting"):
        Survey(data)


def test_init_renames_coordinate_columns():
    data = _synthetic_survey().rename(columns={"easting": "x", "northing": "y"})
    survey = Survey(data, easting_column="x", northing_column="y")
    assert "easting" in survey.data.columns
    assert "northing" in survey.data.columns
    assert "x" not in survey.data.columns


def test_init_copy_isolates_caller_frame():
    data = _synthetic_survey()
    survey = Survey(data)
    survey.data["new_column"] = 1.0
    assert "new_column" not in data.columns


def test_init_no_copy_shares_caller_frame():
    data = _synthetic_survey()
    survey = Survey(data, copy=False)
    survey.data["new_column"] = 1.0
    assert "new_column" in data.columns


def test_init_metadata_defaults_to_fresh_dict():
    survey1 = _survey()
    survey2 = _survey()
    survey1.metadata["name"] = "a"
    assert survey2.metadata == {}


# ---------------------------------------------------------------------------
# crs handling
# ---------------------------------------------------------------------------


def test_crs_taken_from_geodataframe():
    data = _synthetic_survey()
    gdf = gpd.GeoDataFrame(
        data, geometry=gpd.points_from_xy(data.easting, data.northing), crs=CRS
    )
    survey = Survey(gdf)
    assert survey.crs is not None
    assert survey.crs.to_string() == CRS


def test_crs_explicit_on_plain_dataframe():
    survey = _survey(crs=CRS)
    assert survey.crs.to_string() == CRS


def test_crs_conflict_with_geodataframe_raises():
    data = _synthetic_survey()
    gdf = gpd.GeoDataFrame(
        data, geometry=gpd.points_from_xy(data.easting, data.northing), crs=CRS
    )
    with pytest.raises(ValueError, match="conflicts"):
        Survey(gdf, crs="EPSG:3031")


def test_crs_matching_geodataframe_passes():
    data = _synthetic_survey()
    gdf = gpd.GeoDataFrame(
        data, geometry=gpd.points_from_xy(data.easting, data.northing), crs=CRS
    )
    survey = Survey(gdf, crs=CRS)
    assert survey.crs.to_string() == CRS


# ---------------------------------------------------------------------------
# column injection and chaining
# ---------------------------------------------------------------------------


def test_along_track_distance_writes_distance_column():
    survey = _survey().along_track_distance(progressbar=False)
    assert "distance_along_line" in survey.data.columns
    grouped = survey.data.groupby("line")["distance_along_line"]
    assert (grouped.min() == 0.0).all()
    assert (grouped.max() == 100.0).all()


def test_missing_column_raises_with_helpful_message():
    survey = Survey(_synthetic_survey(line_column="flight"), **_BINDINGS)
    with pytest.raises(ValueError, match=r"not found in survey\.data"):
        survey.along_track_distance(progressbar=False)


def test_unset_column_binding_raises_with_helpful_message():
    survey = Survey(_synthetic_survey())
    with pytest.raises(ValueError, match="needs line_column set on the Survey"):
        survey.along_track_distance(progressbar=False)


def test_custom_line_column_is_injected():
    survey = Survey(
        _synthetic_survey(line_column="flight"),
        **{**_BINDINGS, "line_column": "flight"},
    ).along_track_distance(progressbar=False)
    assert "distance_along_line" in survey.data.columns


def test_unique_line_id_overwrites_line_column():
    survey = _survey().unique_line_id()
    assert survey.data["line"].nunique() == 3
    assert pd.api.types.is_integer_dtype(survey.data["line"])


def test_split_into_segments_writes_line_column():
    data = _synthetic_survey().drop(columns=["line", "line_type"])
    data["unixtime"] = np.arange(len(data), dtype=float)
    data.loc[11:, "unixtime"] += 100
    data.loc[22:, "unixtime"] += 100
    survey = Survey(data, **_BINDINGS).split_into_segments(50.0)
    assert survey.data["line"].nunique() == 3


def test_track_requires_geographic_columns():
    with pytest.raises(ValueError, match="latitude_column"):
        _survey().track(progressbar=False)


def test_chaining_returns_same_object():
    survey = _survey()
    result = survey.along_track_distance(progressbar=False).filter_line(
        data_column="value", filter_width=30.0, progressbar=False
    )
    assert result is survey
    assert "value_filtered" in survey.data.columns


# ---------------------------------------------------------------------------
# frame-replacing methods return a new Survey
# ---------------------------------------------------------------------------


def test_block_reduce_returns_new_survey():
    survey = Survey(
        _synthetic_survey(),
        crs=CRS,
        metadata={"name": "demo"},
        **_BINDINGS,
    ).along_track_distance(progressbar=False)
    survey.create_intersection_table(method="network", progressbar=False)
    original_length = len(survey.data)

    reduced = survey.block_reduce(np.mean, spacing=20.0, progressbar=False)

    assert isinstance(reduced, Survey)
    assert reduced is not survey
    assert len(survey.data) == original_length
    assert len(reduced.data) < original_length
    assert reduced.crs == survey.crs
    assert reduced.metadata == survey.metadata
    assert reduced.metadata is not survey.metadata
    assert reduced.line_column == survey.line_column
    assert reduced.intersections is None
    # string line names survive block_reduce (re-attached per group)
    assert set(reduced.data["line"]) == {"L1", "L2", "T1"}


def test_resample_returns_new_survey():
    survey = _survey().along_track_distance(progressbar=False).unique_line_id()
    resampled = survey.resample(spacing=20.0, maxdist=50.0, progressbar=False)
    assert isinstance(resampled, Survey)
    assert resampled is not survey
    assert resampled.data["line"].nunique() == 3


# ---------------------------------------------------------------------------
# cached statistics
# ---------------------------------------------------------------------------


def test_region():
    assert _survey().region == (0.0, 100.0, 0.0, 100.0)


def test_line_counts():
    assert _survey().line_counts == {"total": 3, "flight": 2, "tie": 1}


def test_line_lengths_and_total_length():
    survey = _survey()
    assert survey.line_lengths.to_numpy() == pytest.approx([100.0] * 3)
    assert survey.total_length == pytest.approx(300.0)


def test_median_line_lengths():
    assert _survey().median_line_lengths == {
        "flight": pytest.approx(100.0),
        "tie": pytest.approx(100.0),
    }


def test_line_azimuths_are_compass_azimuths():
    azimuths = _survey().line_azimuths
    # E-W flight lines -> 90 degrees, N-S tie line -> 0 degrees
    assert azimuths["L1"] == pytest.approx(90.0)
    assert azimuths["L2"] == pytest.approx(90.0)
    assert azimuths["T1"] == pytest.approx(0.0)


def test_mean_line_azimuths():
    means = _survey().mean_line_azimuths
    assert means["flight"] == pytest.approx(90.0)
    assert means["tie"] == pytest.approx(0.0, abs=1e-6)


def test_mean_line_azimuths_wraparound():
    # two nearly N-S lines at ~1 and ~179 degrees compass azimuth: their
    # axial mean is ~0/180, not the naive median of 90
    along = np.linspace(0.0, 100.0, 11)
    frames = []
    for name, azimuth_deg in (("A", 1.0), ("B", 179.0)):
        angle = np.deg2rad(azimuth_deg)
        frames.append(
            pd.DataFrame(
                {
                    "easting": along * np.sin(angle),
                    "northing": along * np.cos(angle),
                    "line": name,
                }
            )
        )
    survey = Survey(pd.concat(frames, ignore_index=True), line_column="line")
    mean = survey.mean_line_azimuths["all"]
    assert min(mean, 180.0 - mean) == pytest.approx(0.0, abs=1.0)


def test_median_line_spacings():
    spacings = _survey().median_line_spacings
    assert spacings["flight"] == pytest.approx(100.0)
    # only one tie line, so no tie spacing can be computed
    assert "tie" not in spacings


def test_median_line_spacings_without_line_type_column():
    data = _synthetic_survey().drop(columns="line_type")
    assert Survey(data, line_column="line").median_line_spacings["all"] > 0


# ---------------------------------------------------------------------------
# cache invalidation
# ---------------------------------------------------------------------------


def test_stat_names_covers_every_cached_property():
    cached = {
        name
        for name, attribute in vars(Survey).items()
        if isinstance(attribute, functools.cached_property)
    }
    assert set(Survey._STAT_NAMES) == cached
    assert len(cached) > 0


def test_data_setter_invalidates_cache():
    survey = _survey()
    assert survey.region == (0.0, 100.0, 0.0, 100.0)
    shifted = survey.data.copy()
    shifted["easting"] += 50.0
    survey.data = shifted
    assert survey.region == (50.0, 150.0, 0.0, 100.0)


def test_line_creating_methods_invalidate_cache():
    survey = _survey()
    _ = survey.line_counts
    assert "line_counts" in survey.__dict__
    survey.unique_line_id()
    assert "line_counts" not in survey.__dict__


def test_data_column_methods_do_not_invalidate_cache():
    survey = _survey().along_track_distance(progressbar=False)
    _ = survey.region
    survey.filter_line(data_column="value", filter_width=30.0, progressbar=False)
    assert "region" in survey.__dict__


def test_invalidate_cache():
    survey = _survey()
    _ = survey.region
    survey.invalidate_cache()
    assert "region" not in survey.__dict__


# ---------------------------------------------------------------------------
# repr and describe
# ---------------------------------------------------------------------------


def test_repr_is_cheap_and_reports_computed_stats():
    survey = _survey()
    text = repr(survey)
    assert "computed stats: none" in text
    assert "region" not in survey.__dict__
    _ = survey.region
    assert "region" in repr(survey)


def test_repr_html_is_cheap_and_has_sections():
    survey = Survey(_synthetic_survey(), crs=CRS, metadata={"name": "demo"})
    text = survey._repr_html_()
    assert "airbornegeo.Survey" in text
    assert "<details" in text
    assert "name" in text
    assert "not computed" in text
    assert "region" not in survey.__dict__  # repr never triggers computation
    _ = survey.region
    assert "100.0" in survey._repr_html_()


def test_describe_computes_and_formats_all_stats():
    survey = _survey()
    text = survey.describe()
    assert "region" in text
    assert "100.0" in text
    assert all(name in survey.__dict__ for name in Survey._STAT_NAMES)


# ---------------------------------------------------------------------------
# intersections pipeline
# ---------------------------------------------------------------------------


def test_pipeline_method_before_create_raises():
    with pytest.raises(ValueError, match="create_intersection_table"):
        _survey().add_intersections(progressbar=False)


def test_intersections_roundtrip():
    survey = Survey(_synthetic_survey(), crs=CRS, **_BINDINGS).along_track_distance(
        progressbar=False
    )

    survey.create_intersection_table(method="network", progressbar=False)
    assert survey.intersections is not None
    assert len(survey.intersections) == 2  # L1 x T1 and L2 x T1

    _ = survey.region
    rows_before = len(survey.data)
    survey.add_intersections(progressbar=False)
    assert len(survey.data) > rows_before
    assert "is_intersection" in survey.data.columns
    assert "region" not in survey.__dict__  # cache invalidated

    survey.interpolate_intersections(to_interp="value", progressbar=False)
    survey.calculate_crossover_errors(data_col="value")
    assert "crossover_error_0" in survey.intersections.columns
    errors = np.sort(np.abs(survey.intersections["crossover_error_0"]))
    # per-line biases are 5, -3 (flights) and 0 (tie)
    assert errors == pytest.approx([3.0, 5.0])


def test_crossover_network_levelling_reduces_errors():
    survey = Survey(_synthetic_survey(), **_BINDINGS).along_track_distance(
        progressbar=False
    )
    survey.create_intersection_table(method="network", progressbar=False)
    survey.add_intersections(progressbar=False)
    survey.interpolate_intersections(to_interp="value", progressbar=False)
    survey.calculate_crossover_errors(data_col="value")

    survey.crossover_network_levelling(
        data_col="value",
        degree=0,
        max_iterations=2,
        plot_convergence=False,
        progressbar=False,
        raise_error_if_unchanged=False,
    )

    assert "value_levelled" in survey.data.columns
    error_columns = [
        column
        for column in survey.intersections.columns
        if column.startswith("crossover_error_")
    ]
    first = np.abs(survey.intersections[error_columns[0]]).max()
    last = np.abs(survey.intersections[error_columns[-1]]).max()
    assert last < first


# ---------------------------------------------------------------------------
# canonical column-name shim
# ---------------------------------------------------------------------------


def test_canonical_collision_raises():
    data = _synthetic_survey(line_column="flight", unrelated=1.0).rename(
        columns={"unrelated": "line"}
    )
    survey = Survey(data, **{**_BINDINGS, "line_column": "flight"})
    with pytest.raises(ValueError, match="unrelated 'line' column"):
        survey.create_intersection_table(method="groups", progressbar=False)


def test_canonical_renames_for_groups_method():
    data = _synthetic_survey(line_column="flight", line_type_column="lt")
    survey = Survey(
        data, **{**_BINDINGS, "line_column": "flight", "line_type_column": "lt"}
    )
    survey.create_intersection_table(method="groups", progressbar=False)
    assert len(survey.intersections) == 2
    # the rename is never persisted
    assert "flight" in survey.data.columns
    assert "line" not in survey.data.columns


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _plot_survey():
    data = _synthetic_survey()
    data["mag"] = np.linspace(-20.0, 20.0, len(data))
    return Survey(data, crs=CRS, metadata={"survey": "demo"})


def test_plot_returns_axes_with_scalebar():
    """plot() must return the axes, not None, when the default scalebar is drawn."""
    ax = _plot_survey().plot(color_by="mag")
    assert isinstance(ax, mpl.axes.Axes)


def test_plot_returns_axes_without_scalebar():
    ax = _plot_survey().plot(color_by="mag", scalebar=False)
    assert isinstance(ax, mpl.axes.Axes)


def test_plot_draws_into_supplied_axes():
    """A supplied ax must be used rather than a new figure being created."""
    _fig, ax = plt.subplots()
    n_figures = len(plt.get_fignums())
    result = _plot_survey().plot(color_by="mag", ax=ax)
    assert result is ax
    assert len(plt.get_fignums()) == n_figures


def test_plot_without_color_by():
    """Omitting color_by should plot plain points rather than raising a KeyError."""
    ax = _plot_survey().plot(scalebar=False)
    assert len(ax.collections) == 1


def test_plot_missing_column_raises_helpful_error():
    with pytest.raises(ValueError, match=r"not found in survey\.data"):
        _plot_survey().plot(color_by="nonexistent")


def test_plot_categorical_without_color_by_raises():
    with pytest.raises(ValueError, match="categorical=True requires"):
        _plot_survey().plot(categorical=True)


def test_plot_categorical_draws_one_collection_per_category():
    ax = _plot_survey().plot(color_by="line", categorical=True, scalebar=False)
    assert len(ax.collections) == 3
    assert ax.get_legend() is None


def test_plot_categorical_with_labels_adds_legend():
    ax = _plot_survey().plot(
        color_by="line",
        categorical=True,
        label_categories=True,
        scalebar=False,
    )
    legend = ax.get_legend()
    assert legend is not None
    assert legend.get_title().get_text() == "line"


def test_plot_categorical_legend_goes_to_supplied_axes():
    """The legend must land on the supplied axes, not whatever pyplot's current axes is."""
    _fig, other_ax = plt.subplots()
    _fig2, ax = plt.subplots()
    plt.sca(other_ax)
    _plot_survey().plot(
        color_by="line",
        categorical=True,
        label_categories=True,
        ax=ax,
        scalebar=False,
    )
    assert ax.get_legend() is not None
    assert other_ax.get_legend() is None


def test_plot_adds_colorbar_labelled_with_column():
    ax = _plot_survey().plot(color_by="mag", scalebar=False)
    labels = [a.get_ylabel() for a in ax.figure.axes if a is not ax]
    assert "mag" in labels


def test_plot_title_defaults_to_survey_metadata():
    ax = _plot_survey().plot(color_by="mag", scalebar=False)
    assert ax.get_title() == "Survey: demo"


def test_plot_explicit_vmin_vmax_are_used():
    ax = _plot_survey().plot(color_by="mag", vmin=-5.0, vmax=5.0, scalebar=False)
    assert ax.collections[0].get_clim() == pytest.approx((-5.0, 5.0))


def test_plot_absolute_gives_symmetric_limits():
    data = _synthetic_survey()
    data["mag"] = np.linspace(-5.0, 40.0, len(data))
    ax = Survey(data).plot(color_by="mag", absolute=True, scalebar=False)
    vmin, vmax = ax.collections[0].get_clim()
    assert vmin == pytest.approx(-vmax)


def test_plot_coarsen_reduces_plotted_points():
    survey = _plot_survey()
    ax = survey.plot(color_by="mag", coarsen=3, scalebar=False)
    assert len(ax.collections[0].get_offsets()) == len(survey.data[::3])


# ---------------------------------------------------------------------------
# to_parquet / from_parquet
# ---------------------------------------------------------------------------


def test_parquet_roundtrip_preserves_data_crs_bindings_and_metadata(tmp_path):
    survey = Survey(
        _synthetic_survey(),
        crs=CRS,
        metadata={"name": "demo"},
        **_BINDINGS,
    )

    survey.to_parquet(tmp_path / "survey")
    reloaded = Survey.from_parquet(tmp_path / "survey")

    pd.testing.assert_frame_equal(reloaded.data, survey.data)
    assert reloaded.crs == survey.crs
    assert reloaded.metadata == survey.metadata
    assert reloaded.line_column == survey.line_column
    assert reloaded.line_type_column == survey.line_type_column
    assert reloaded.distance_column == survey.distance_column
    assert reloaded.time_column == survey.time_column
    assert reloaded.height_column == survey.height_column
    assert reloaded.intersections is None


def test_parquet_roundtrip_preserves_intersections(tmp_path):
    survey = _survey()
    survey.create_intersection_table(method="network", progressbar=False)

    survey.to_parquet(tmp_path / "survey")
    reloaded = Survey.from_parquet(tmp_path / "survey")

    assert isinstance(reloaded.intersections, gpd.GeoDataFrame)
    assert len(reloaded.intersections) == len(survey.intersections)
    assert list(reloaded.intersections.columns) == list(survey.intersections.columns)


def test_parquet_data_file_is_plain_parquet_without_airbornegeo(tmp_path):
    """The data file must be readable with bare pandas, no airbornegeo import."""
    survey = _survey()
    survey.to_parquet(tmp_path / "survey")

    plain = pd.read_parquet(tmp_path / "survey.parquet")

    pd.testing.assert_frame_equal(plain, survey.data)


def test_parquet_accepts_string_path(tmp_path):
    survey = _survey()
    path = str(tmp_path / "survey")

    survey.to_parquet(path)
    reloaded = Survey.from_parquet(path)

    pd.testing.assert_frame_equal(reloaded.data, survey.data)


def test_parquet_rejects_path_with_extension(tmp_path):
    survey = _survey()

    with pytest.raises(ValueError, match="must not include a file extension"):
        survey.to_parquet(tmp_path / "survey.parquet")


# ---------------------------------------------------------------------------
# to_csv / from_csv
# ---------------------------------------------------------------------------


def test_csv_roundtrip_preserves_data_crs_bindings_and_metadata(tmp_path):
    survey = Survey(
        _synthetic_survey(),
        crs=CRS,
        metadata={"name": "demo"},
        **_BINDINGS,
    )

    survey.to_csv(tmp_path / "survey")
    reloaded = Survey.from_csv(tmp_path / "survey")

    pd.testing.assert_frame_equal(reloaded.data, survey.data, check_dtype=False)
    assert reloaded.crs == survey.crs
    assert reloaded.metadata == survey.metadata
    assert reloaded.line_column == survey.line_column
    assert reloaded.intersections is None


def test_csv_roundtrip_preserves_intersections_geometry(tmp_path):
    survey = _survey()
    survey.create_intersection_table(method="network", progressbar=False)

    survey.to_csv(tmp_path / "survey")
    reloaded = Survey.from_csv(tmp_path / "survey")

    assert isinstance(reloaded.intersections, gpd.GeoDataFrame)
    assert len(reloaded.intersections) == len(survey.intersections)
    assert list(reloaded.intersections.geometry) == list(survey.intersections.geometry)


def test_csv_data_file_is_plain_csv_without_airbornegeo(tmp_path):
    """The data file must be readable with bare pandas, no airbornegeo import."""
    survey = _survey()
    survey.to_csv(tmp_path / "survey")

    plain = pd.read_csv(tmp_path / "survey.csv")

    pd.testing.assert_frame_equal(plain, survey.data, check_dtype=False)


def test_csv_rejects_path_with_extension(tmp_path):
    survey = _survey()

    with pytest.raises(ValueError, match="must not include a file extension"):
        survey.to_csv(tmp_path / "survey.csv")


# ---------------------------------------------------------------------------
# reproject
# ---------------------------------------------------------------------------


def test_reproject_moves_coordinates_and_sets_crs():
    survey = _survey(crs=CRS)
    before = survey.data[["easting", "northing"]].to_numpy().copy()

    survey.reproject("EPSG:3413")

    assert survey.crs == pyproj.CRS.from_user_input("EPSG:3413")
    assert not np.allclose(survey.data[["easting", "northing"]].to_numpy(), before)


def test_reproject_roundtrip_returns_original_coordinates():
    survey = _survey(crs=CRS)
    before = survey.data[["easting", "northing"]].to_numpy().copy()

    survey.reproject("EPSG:4326").reproject(CRS)

    assert survey.data[["easting", "northing"]].to_numpy() == pytest.approx(
        before, abs=1e-6
    )
    assert survey.crs == pyproj.CRS.from_user_input(CRS)


def test_reproject_uses_explicit_input_crs_when_survey_has_none():
    survey = _survey()
    survey.reproject("EPSG:4326", input_crs=CRS)
    assert survey.crs == pyproj.CRS.from_user_input("EPSG:4326")


def test_reproject_without_any_crs_raises():
    survey = _survey()
    with pytest.raises(ValueError, match="no crs set"):
        survey.reproject("EPSG:4326")


def test_reproject_invalidates_cached_stats():
    survey = _survey(crs=CRS)
    region = survey.region
    survey.reproject("EPSG:4326")
    assert survey.region != region


# ---------------------------------------------------------------------------
# eotvos correction
# ---------------------------------------------------------------------------


def _geographic_survey():
    """Synthetic survey with the latitude/longitude/time columns eotvos needs."""
    data = _synthetic_survey()
    longitude, latitude = ag_reproject(
        data.easting.to_numpy(), data.northing.to_numpy(), CRS, "EPSG:4326"
    )
    data["lat"] = latitude
    data["lon"] = longitude
    return Survey(
        data,
        **_BINDINGS,
        latitude_column="lat",
        longitude_column="lon",
        crs=CRS,
    )


def test_eotvos_correction_writes_result_column():
    survey = _geographic_survey()
    survey.eotvos_correction(progressbar=False)
    assert "eotvos_correction" in survey.data.columns
    assert survey.data.eotvos_correction.notna().all()


def test_eotvos_correction_groups_by_line_by_default():
    """Each line's values must match computing that line on its own."""
    survey = _geographic_survey()
    survey.eotvos_correction(result_column="eotvos", progressbar=False)

    for line, group in survey.data.groupby("line"):
        alone = eotvos_correction(
            group,
            method="full",
            latitude_column="lat",
            longitude_column="lon",
            time_column="unixtime",
            height_column="height",
            progressbar=False,
        )
        assert group.eotvos.to_numpy() == pytest.approx(alone), line


def test_eotvos_correction_by_line_false_matches_ungrouped_call():
    survey = _geographic_survey()
    survey.eotvos_correction(by_line=False, result_column="eotvos", progressbar=False)

    expected = eotvos_correction(
        survey.data,
        method="full",
        latitude_column="lat",
        longitude_column="lon",
        time_column="unixtime",
        height_column="height",
        progressbar=False,
    )
    assert survey.data.eotvos.to_numpy() == pytest.approx(expected)


def test_eotvos_correction_glicken_uses_computed_track_and_ground_speed():
    survey = _geographic_survey()
    survey.along_track_distance(progressbar=False)
    survey.track(progressbar=False)
    survey.ground_speed(progressbar=False)

    survey.eotvos_correction(method="glicken", progressbar=False)

    assert survey.data.eotvos_correction.notna().any()


def test_eotvos_correction_without_geographic_columns_raises():
    survey = _survey()
    with pytest.raises(ValueError, match="latitude_column and longitude_column"):
        survey.eotvos_correction()


# ---------------------------------------------------------------------------
# plotting wrappers
# ---------------------------------------------------------------------------


def test_plotly_points_returns_figure():
    survey = _survey()
    assert survey.plotly_points(color_col="value") is not None


def test_plotly_points_unknown_column_raises():
    survey = _survey()
    with pytest.raises(ValueError, match=r"not found in survey\.data"):
        survey.plotly_points(color_col="nope")


def test_profile_wrappers_default_x_to_the_distance_column():
    survey = _survey().along_track_distance(progressbar=False)
    figure = survey.plotly_profiles(y="value")
    assert figure.data[0].x == pytest.approx(survey.data.distance_along_line.to_numpy())


def test_plot_profiles_runs_with_default_x():
    survey = _survey().along_track_distance(progressbar=False)
    survey.plot_profiles(y="value")
    plt.close("all")


def test_plot_line_and_crosses_returns_figure_for_one_line():
    survey = _survey().along_track_distance(progressbar=False)
    figure = survey.plot_line_and_crosses(y=["value"], line="L1")
    assert figure.layout.title.text == "Line: L1"


def test_profile_wrappers_unknown_column_raises():
    survey = _survey().along_track_distance(progressbar=False)
    with pytest.raises(ValueError, match=r"not found in survey\.data"):
        survey.plotly_profiles(y="nope")
