import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from airbornegeo import crossovers as cx_module
from airbornegeo.crossovers import (
    add_intersections,
    add_values_to_intersections,
    calculate_crossover_errors,
    create_intersection_table,
    extend_line,
    get_line_intersections,
    interpolate_intersections,
    lines_without_intersections,
    update_intersections_with_eq_sources,
)

CRS = "EPSG:3431"


def _line_points(name, eastings, northings, line_column="line", **extra_columns):
    frame = pd.DataFrame({"easting": eastings, "northing": northings, **extra_columns})
    frame[line_column] = name
    return gpd.GeoDataFrame(
        frame, geometry=gpd.points_from_xy(frame.easting, frame.northing), crs=CRS
    )


def _crossing_lines_gdf():
    """Two perpendicular lines, 'A' horizontal and 'B' vertical, crossing once at (50, 50)."""
    line_a = _line_points("A", [0, 20, 40, 60, 80, 100], [50] * 6)
    line_b = _line_points("B", [50] * 6, [0, 20, 40, 60, 80, 100])
    return pd.concat([line_a, line_b], ignore_index=True)


# ---------------------------------------------------------------------------
# extend_line
# ---------------------------------------------------------------------------


def test_extend_line_extends_both_ends_by_distance():
    """A straight line should be extended by exactly `distance` at each end, along its direction."""
    line = LineString([(0, 0), (10, 0)])
    result = extend_line(line, distance=5)
    coords = list(result.coords)
    assert coords[0] == pytest.approx((-5, 0))
    assert coords[-1] == pytest.approx((15, 0))
    assert coords[1:-1] == [(0.0, 0.0), (10.0, 0.0)]


def test_extend_line_returns_unchanged_for_degenerate_line():
    """A line collapsing to fewer than 2 unique points after removing consecutive duplicates should be returned unchanged."""
    line = LineString([(5, 5), (5, 5)])
    result = extend_line(line, distance=10)
    assert result is line


def test_extend_line_removes_consecutive_duplicate_points():
    """Consecutive duplicate coordinates should be collapsed before computing the extension direction."""
    line = LineString([(0, 0), (0, 0), (10, 0), (10, 0), (10, 0), (20, 0)])
    result = extend_line(line, distance=5)
    coords = list(result.coords)
    assert len(coords) == 5
    assert coords[0] == pytest.approx((-5, 0))
    assert coords[-1] == pytest.approx((25, 0))


# ---------------------------------------------------------------------------
# get_line_intersections
# ---------------------------------------------------------------------------


def test_get_line_intersections_network_finds_single_crossing():
    """Two perpendicular lines crossing once should yield exactly one intersection near their true crossing."""
    data = _crossing_lines_gdf()
    result = get_line_intersections(data, line_column="line")
    assert len(result) == 1
    row = result.iloc[0]
    assert {row.line1, row.line2} == {"A", "B"}
    assert row.geometry.x == pytest.approx(50, abs=1)
    assert row.geometry.y == pytest.approx(50, abs=1)
    assert not row.is_buffered


def test_get_line_intersections_no_crossing_raises_valueerror():
    """Two lines that never cross should raise ValueError."""
    line_a = _line_points("A", [0, 20, 40], [50, 50, 50])
    line_b = _line_points("B", [200, 220, 240], [0, 20, 40])
    data = pd.concat([line_a, line_b], ignore_index=True)
    with pytest.raises(ValueError, match="No intersections found"):
        get_line_intersections(data, line_column="line")


def test_get_line_intersections_buffer_dist_finds_extended_crossing():
    """A line that falls short of crossing another should intersect once extended via buffer_dist, and be flagged as buffered."""
    line_a = _line_points("A", [0, 20, 40], [50, 50, 50])
    line_b = _line_points("B", [50, 50, 50, 50, 50], [0, 25, 50, 75, 100])
    data = pd.concat([line_a, line_b], ignore_index=True)

    result = get_line_intersections(data, line_column="line", buffer_dist=20)
    assert len(result) == 1
    assert result.iloc[0].is_buffered


def test_get_line_intersections_builds_geometry_from_plain_dataframes_and_buffers_second_group():
    """Plain DataFrames (no geometry column) supplied for both lines1_gdf and lines2_gdf
    should have geometry built from their easting/northing columns, and buffer_dist
    should extend the ends of both groups' lines (not just the first)."""
    line_a = pd.DataFrame(
        {"easting": [0, 20, 40], "northing": [50, 50, 50], "line": ["A"] * 3}
    )
    line_b = pd.DataFrame(
        {"easting": [50, 50, 50], "northing": [0, 20, 40], "line": ["B"] * 3}
    )
    result = get_line_intersections(
        line_a, line_b, line_column="line", buffer_dist=20, progressbar=False
    )
    assert len(result) == 1
    assert result.iloc[0].is_buffered


# ---------------------------------------------------------------------------
# create_intersection_table
# ---------------------------------------------------------------------------


def test_create_intersection_table_invalid_method_raises():
    """An unrecognized method string should raise ValueError."""
    data = _crossing_lines_gdf()
    with pytest.raises(ValueError, match="method must be either 'groups' or 'network'"):
        create_intersection_table(data, line_column="line", method="bogus")


def test_create_intersection_table_network_mode():
    """Network mode should find the crossing between two lines without needing a line_type column."""
    data = _crossing_lines_gdf()
    result = create_intersection_table(data, line_column="line", method="network")
    assert len(result) == 1
    assert {result.iloc[0].line1, result.iloc[0].line2} == {"A", "B"}


def test_create_intersection_table_groups_mode_uses_line_type():
    """Groups mode should only find intersections between line_type 0 and 1, ignoring type 2."""
    line_a = _line_points("A", [0, 20, 40, 60, 80, 100], [50] * 6, line_type=0)
    line_b = _line_points("B", [50] * 6, [0, 20, 40, 60, 80, 100], line_type=1)
    line_c = _line_points("C", [90] * 6, [0, 20, 40, 60, 80, 100], line_type=2)
    data = pd.concat([line_a, line_b, line_c], ignore_index=True)

    result = create_intersection_table(data, line_column="line", method="groups")
    assert len(result) == 1
    assert result.iloc[0].line1 == "A"
    assert result.iloc[0].line2 == "B"


def test_create_intersection_table_cutoff_dist_filters_far_intersections():
    """An intersection whose nearest data point is farther than cutoff_dist should be dropped."""
    line_a = _line_points("A", [0, 20, 90, 100], [50, 50, 50, 50])
    line_b = _line_points("B", [50] * 6, [0, 20, 40, 60, 80, 100])
    data = pd.concat([line_a, line_b], ignore_index=True)

    kept = create_intersection_table(
        data, line_column="line", method="network", cutoff_dist=35
    )
    assert len(kept) == 1

    dropped = create_intersection_table(
        data, line_column="line", method="network", cutoff_dist=15
    )
    assert len(dropped) == 0


def test_create_intersection_table_exclude_ints_removes_specified_pairs():
    """exclude_ints should drop intersections either by exact line pair or by a single line number."""
    line_a = _line_points("A", [0, 20, 40, 60, 80, 100], [50] * 6)
    line_b = _line_points("B", [30] * 6, [0, 20, 40, 60, 80, 100])
    line_c = _line_points("C", [70] * 6, [0, 20, 40, 60, 80, 100])
    data = pd.concat([line_a, line_b, line_c], ignore_index=True)

    baseline = create_intersection_table(data, line_column="line", method="network")
    assert len(baseline) == 2

    pair_excluded = create_intersection_table(
        data, line_column="line", method="network", exclude_ints=[["A", "B"]]
    )
    assert len(pair_excluded) == 1
    assert "B" not in pair_excluded[["line1", "line2"]].to_numpy()

    line_excluded = create_intersection_table(
        data, line_column="line", method="network", exclude_ints=["A"]
    )
    assert len(line_excluded) == 0


def test_create_intersection_table_drops_preexisting_is_intersection_rows():
    """Rows already flagged is_intersection (e.g. left over from a prior
    add_intersections call) should be dropped before recomputing intersections, so
    running the function again on already-processed data doesn't accumulate
    duplicates or feed synthetic rows back into the calculation."""
    data = _crossing_lines_gdf()
    data["is_intersection"] = False
    result1 = create_intersection_table(data, line_column="line", method="network")
    assert len(result1) == 1

    # simulate data that has already had intersection rows added
    fake_intersection_row = data.iloc[[0]].copy()
    fake_intersection_row["is_intersection"] = True
    data_with_inter_rows = pd.concat([data, fake_intersection_row], ignore_index=True)

    result2 = create_intersection_table(
        data_with_inter_rows, line_column="line", method="network"
    )
    assert len(result2) == 1


def test_create_intersection_table_block_size_drops_near_duplicates(monkeypatch):
    """Intersections between the same line pair within block_size of each other should
    be deduplicated (keeping the lowest max_dist one), and a debug log should report
    how many were dropped."""
    fake_inters = gpd.GeoDataFrame(
        {
            "line1": ["A", "A", "A"],
            "line2": ["B", "B", "B"],
            "line1_data_dist": [1.0, 2.0, 1.0],
            "line2_data_dist": [1.0, 2.0, 1.0],
            "is_buffered": [False, False, False],
            "geometry": [Point(0, 0), Point(0.5, 0), Point(100, 100)],
        },
        geometry="geometry",
        crs=CRS,
    )
    monkeypatch.setattr(
        cx_module, "get_line_intersections", lambda **_kwargs: fake_inters
    )

    data = _crossing_lines_gdf()
    result = create_intersection_table(
        data, line_column="line", method="network", block_size=5
    )
    # the two near-duplicate points (0,0) and (0.5,0) collapse into one, the far
    # point at (100, 100) is kept separately
    assert len(result) == 2


# ---------------------------------------------------------------------------
# add_intersections
# ---------------------------------------------------------------------------


def test_add_intersections_adds_two_rows_with_correct_distance():
    """add_intersections should append one row per line for each intersection, with distance-along-line computed from the nearest data point."""
    line_a = _line_points(
        "A",
        [0, 20, 40, 60, 80, 100],
        [50] * 6,
        dist_along_line=[0, 20, 40, 60, 80, 100],
    )
    line_b = _line_points(
        "B",
        [50] * 6,
        [0, 20, 40, 60, 80, 100],
        dist_along_line=[0, 20, 40, 60, 80, 100],
    )
    data = pd.concat([line_a, line_b], ignore_index=True)

    inters = create_intersection_table(data, line_column="line", method="network")
    assert len(inters) == 1

    new_data, new_inters = add_intersections(
        data, inters, line_column="line", distance_column="dist_along_line"
    )

    assert len(new_data) == len(data) + 2
    added = new_data[new_data.is_intersection]
    assert len(added) == 2
    assert set(added.line) == {"A", "B"}
    assert set(added.intersecting_line) == {"A", "B"}
    assert added.dist_along_line.to_numpy() == pytest.approx([50, 50], abs=1)
    assert new_inters.loc[0, "dist_along_line1"] == pytest.approx(50, abs=1)
    assert new_inters.loc[0, "dist_along_line2"] == pytest.approx(50, abs=1)


def test_add_intersections_called_twice_does_not_duplicate():
    """Calling add_intersections again on data that already has intersection rows
    should drop the existing intersection rows first and replace them, rather than
    accumulating duplicates."""
    line_a = _line_points(
        "A",
        [0, 20, 40, 60, 80, 100],
        [50] * 6,
        dist_along_line=[0, 20, 40, 60, 80, 100],
    )
    line_b = _line_points(
        "B",
        [50] * 6,
        [0, 20, 40, 60, 80, 100],
        dist_along_line=[0, 20, 40, 60, 80, 100],
    )
    data = pd.concat([line_a, line_b], ignore_index=True)
    inters = create_intersection_table(data, line_column="line", method="network")

    data1, inters1 = add_intersections(
        data, inters, line_column="line", distance_column="dist_along_line"
    )
    assert len(data1) == len(data) + 2

    data2, _inters2 = add_intersections(
        data1, inters1, line_column="line", distance_column="dist_along_line"
    )
    assert len(data2) == len(data) + 2
    assert data2.is_intersection.sum() == 2


# ---------------------------------------------------------------------------
# add_values_to_intersections
# ---------------------------------------------------------------------------


def test_add_values_to_intersections_looks_up_values_from_each_side():
    """add_values_to_intersections should look up each line's value at the
    intersection from the survey dataframe using the intersecting_line match, and
    write both sides' values back onto the intersection table."""
    df = pd.DataFrame(
        {
            "line": ["A", "B"],
            "intersecting_line": ["B", "A"],
            "mag": [12.0, 10.0],
        }
    )
    inters = pd.DataFrame({"line1": ["A"], "line2": ["B"]})

    result = add_values_to_intersections(
        df, inters, line_column="line", columns=("mag",)
    )
    assert result.loc[0, "line1_mag"] == 12.0
    assert result.loc[0, "line2_mag"] == 10.0


# ---------------------------------------------------------------------------
# interpolate_intersections
# ---------------------------------------------------------------------------


def test_interpolate_intersections_fills_value_at_single_crossing():
    """A single crossing should get its value filled by linear interpolation along each line."""
    data = _crossing_lines_gdf()
    data["dist_along_line"] = data.groupby("line")["easting"].transform(
        lambda s: s - s.min()
    ) + data.groupby("line")["northing"].transform(lambda s: s - s.min())
    data["value"] = data["easting"] + data["northing"]

    inters = create_intersection_table(data, line_column="line", method="network")
    filled, inters_valid = interpolate_intersections(
        data,
        inters,
        line_column="line",
        to_interp="value",
        interp_on="dist_along_line",
        method="linear",
    )
    assert len(inters_valid) == 1
    intersection_rows = filled[filled.is_intersection]
    assert len(intersection_rows) == 2
    assert not intersection_rows.value.isna().any()


def test_interpolate_intersections_handles_repeated_line_pair_crossings():
    """Regression test: two lines crossing more than once used to raise
    ValueError('cannot handle a non-unique multi-index!') because the lookup used to
    join filled_lines back onto the intersections table was keyed only on (line,
    intersecting_line), which isn't unique when a line pair crosses more than once. Each
    crossing should now be handled independently, keeping its own distinct
    dist_along_line1/2 values instead of the lookup colliding."""
    # a "V"-shaped line A crosses the horizontal line B at two distinct points
    xa = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    ya = [100 - 2 * v if v <= 50 else 2 * v - 100 for v in xa]
    line_a = _line_points("A", xa, ya, value=xa)
    line_a["dist_along_line"] = xa

    xb = list(range(0, 101, 5))
    line_b = _line_points("B", xb, [50] * len(xb), value=[v * 2 for v in xb])
    line_b["dist_along_line"] = xb

    data = pd.concat([line_a, line_b], ignore_index=True)
    inters = create_intersection_table(data, line_column="line", method="network")
    assert len(inters) == 2

    _filled, inters_valid = interpolate_intersections(
        data,
        inters,
        line_column="line",
        to_interp="value",
        interp_on="dist_along_line",
        method="linear",
    )
    assert len(inters_valid) == 2
    dist1 = inters_valid["dist_along_line1"].to_numpy()
    dist2 = inters_valid["dist_along_line2"].to_numpy()
    assert len(set(dist1)) == 2
    assert len(set(dist2)) == 2


def test_interpolate_intersections_window_width_uses_windowed_interpolation():
    """Passing window_width should route through
    interpolate_missing_pointwise_with_windows instead of interpolate_missing_pointwise."""
    data = _crossing_lines_gdf()
    data["dist_along_line"] = data.groupby("line")["easting"].transform(
        lambda s: s - s.min()
    ) + data.groupby("line")["northing"].transform(lambda s: s - s.min())
    data["value"] = data["easting"] + data["northing"]

    inters = create_intersection_table(data, line_column="line", method="network")
    filled, inters_valid = interpolate_intersections(
        data,
        inters,
        line_column="line",
        to_interp="value",
        interp_on="dist_along_line",
        method="linear",
        window_width=50,
    )
    assert len(inters_valid) == 1
    intersection_rows = filled[filled.is_intersection]
    assert len(intersection_rows) == 2
    assert not intersection_rows.value.isna().any()


def test_interpolate_intersections_called_again_for_another_column_keeps_earlier_column():
    """Regression test: calling interpolate_intersections a second time for a different
    to_interp column used to wipe out the first column's interpolated values at the
    intersection rows, because add_intersections rebuilds those rows from scratch. Both
    columns should have real values at the intersection rows after both calls."""
    data = _crossing_lines_gdf()
    data["dist_along_line"] = data.groupby("line")["easting"].transform(
        lambda s: s - s.min()
    ) + data.groupby("line")["northing"].transform(lambda s: s - s.min())
    data["value"] = data["easting"] + data["northing"]
    data["height"] = data["easting"] - data["northing"]

    inters = create_intersection_table(data, line_column="line", method="network")

    filled_1, inters_1 = interpolate_intersections(
        data,
        inters,
        line_column="line",
        to_interp="value",
        interp_on="dist_along_line",
        method="linear",
    )
    filled_2, _inters_2 = interpolate_intersections(
        filled_1,
        inters_1,
        line_column="line",
        to_interp="height",
        interp_on="dist_along_line",
        method="linear",
    )

    intersection_rows = filled_2[filled_2.is_intersection]
    assert len(intersection_rows) == 2
    assert not intersection_rows.value.isna().any()
    assert not intersection_rows.height.isna().any()


def test_interpolate_intersections_loop_matches_isolated_calls_and_survives_later_failure(
    monkeypatch,
):
    """Chaining interpolate_intersections over several columns, one call per column (as
    in Survey's loop-based usage), should give each column the exact same interpolated
    value as calling interpolate_intersections for that column alone, straight from the
    original data. This should hold even when a later column in the loop fails to
    interpolate at an intersection an earlier column already succeeded at (e.g. not
    enough surrounding points, or the nearest points are too far from the crossing): the
    intersection and the earlier column's value must survive, only the failed column's
    value should be missing."""
    data = _crossing_lines_gdf()
    data["dist_along_line"] = data.groupby("line")["easting"].transform(
        lambda s: s - s.min()
    ) + data.groupby("line")["northing"].transform(lambda s: s - s.min())
    data["col_a"] = data["easting"] + data["northing"]
    data["col_b"] = data["easting"] - data["northing"]
    data["col_c"] = data["easting"] * 0.5 - data["northing"]

    inters = create_intersection_table(data, line_column="line", method="network")

    real_interpolate = cx_module.airbornegeo.interpolating.interpolate_missing_pointwise

    def fake_interpolate(df, *, to_interp, **kwargs):
        """Behave normally for every column except 'col_c', which always fails to
        interpolate at intersections (simulating too little/too-far surrounding data)."""
        result = real_interpolate(df, to_interp=to_interp, **kwargs)
        if to_interp == "col_c":
            result = result.copy()
            result.loc[result.is_intersection, "col_c"] = np.nan
            result.loc[result.is_intersection, "col_c_interpolation_type"] = "none"
        return result

    monkeypatch.setattr(
        cx_module.airbornegeo.interpolating,
        "interpolate_missing_pointwise",
        fake_interpolate,
    )

    # isolated reference: interpolate col_a and col_b independently from the original
    # (un-chained) data, exactly as if each were the only column ever processed
    filled_a_only, _ = interpolate_intersections(
        data,
        inters,
        line_column="line",
        to_interp="col_a",
        interp_on="dist_along_line",
        method="linear",
    )
    filled_b_only, _ = interpolate_intersections(
        data,
        inters,
        line_column="line",
        to_interp="col_b",
        interp_on="dist_along_line",
        method="linear",
    )

    # looped/chained: feed each call's output into the next, as Survey.interpolate_intersections does
    filled, current_inters = data, inters
    for col in ["col_a", "col_b", "col_c"]:
        filled, current_inters = interpolate_intersections(
            filled,
            current_inters,
            line_column="line",
            to_interp=col,
            interp_on="dist_along_line",
            method="linear",
        )

    sort_cols = ["line", "intersecting_line"]
    looped_rows = filled[filled.is_intersection].sort_values(sort_cols)
    a_only_rows = filled_a_only[filled_a_only.is_intersection].sort_values(sort_cols)
    b_only_rows = filled_b_only[filled_b_only.is_intersection].sort_values(sort_cols)

    assert looped_rows["col_a"].to_numpy() == pytest.approx(
        a_only_rows["col_a"].to_numpy()
    )
    assert looped_rows["col_b"].to_numpy() == pytest.approx(
        b_only_rows["col_b"].to_numpy()
    )

    # col_c failed at the only intersection, but col_a/col_b succeeded there earlier in
    # the loop: the intersection must still be present, with their values intact, and
    # only col_c missing.
    assert len(looped_rows) == 2
    assert looped_rows["col_c"].isna().all()
    assert len(current_inters) == 1


def test_interpolate_intersections_drops_rows_with_failed_interpolation(monkeypatch):
    """If a value can't be interpolated at an intersection (interpolation_type stays
    'none'), the intersection should be excluded from the returned intersections table
    and its intersection rows removed from the returned dataframe."""

    def fake_interpolate(df, *, to_interp, interp_on, groupby_column, **kwargs):  # noqa: ARG001
        df = df.copy()
        df[f"{to_interp}_interpolation_type"] = np.where(
            df["is_intersection"], "none", "interpolated"
        )
        return df

    monkeypatch.setattr(
        cx_module.airbornegeo.interpolating,
        "interpolate_missing_pointwise",
        fake_interpolate,
    )

    data = _crossing_lines_gdf()
    data["dist_along_line"] = data.groupby("line")["easting"].transform(
        lambda s: s - s.min()
    ) + data.groupby("line")["northing"].transform(lambda s: s - s.min())
    data["value"] = data["easting"] + data["northing"]

    inters = create_intersection_table(data, line_column="line", method="network")
    filled, inters_valid = interpolate_intersections(
        data,
        inters,
        line_column="line",
        to_interp="value",
        interp_on="dist_along_line",
        method="linear",
    )
    assert len(inters_valid) == 0
    assert filled[filled.is_intersection].empty


# ---------------------------------------------------------------------------
# calculate_crossover_errors
# ---------------------------------------------------------------------------


def _single_crossing_frames(line1_val, line2_val):
    df = pd.DataFrame(
        {
            "line": ["A", "B"],
            "intersecting_line": ["B", "A"],
            "easting": [50.0, 50.0],
            "northing": [50.0, 50.0],
            "mag": [line1_val, line2_val],
        }
    )
    inters = pd.DataFrame(
        {"line1": ["A"], "line2": ["B"], "easting": [50.0], "northing": [50.0]}
    )
    return df, inters


def test_calculate_crossover_errors_basic_mistie():
    """A single crossing should produce crossover_error_0 = line1_value - line2_value."""
    df, inters = _single_crossing_frames(12.0, 10.0)
    result = calculate_crossover_errors(df, inters, data_col="mag", line_column="line")
    assert result["crossover_error_0"].to_numpy() == pytest.approx([2.0])


def test_calculate_crossover_errors_handles_repeated_line_pair_crossings():
    """Regression test: when the same two lines cross twice, each crossing keeps its own mistie
    value instead of collapsing onto a single value (a prior `groupby(...).first()` lookup gave
    every crossing of a line pair the same value)."""
    df = pd.DataFrame(
        {
            "line": ["A", "B", "A", "B"],
            "intersecting_line": ["B", "A", "B", "A"],
            "easting": [10.0, 10.0, 90.0, 90.0],
            "northing": [10.0, 10.0, 90.0, 90.0],
            "mag": [12.0, 10.0, 5.0, 8.0],
        }
    )
    inters = pd.DataFrame(
        {
            "line1": ["A", "A"],
            "line2": ["B", "B"],
            "easting": [10.0, 90.0],
            "northing": [10.0, 90.0],
        }
    )

    result = calculate_crossover_errors(df, inters, data_col="mag", line_column="line")
    misties = result["crossover_error_0"].to_numpy()
    assert misties == pytest.approx([2.0, -3.0])


def test_calculate_crossover_errors_skips_unchanged_recompute():
    """Recomputing misties that haven't changed shouldn't add a new crossover_error column."""
    df, inters = _single_crossing_frames(12.0, 10.0)
    first = calculate_crossover_errors(df, inters, data_col="mag", line_column="line")
    second = calculate_crossover_errors(df, first, data_col="mag", line_column="line")
    assert "crossover_error_1" not in second.columns
    assert list(second.columns) == list(first.columns)


def test_calculate_crossover_errors_ignores_non_numeric_suffix_column():
    """A pre-existing column matching the 'crossover_error_' prefix but with a
    non-numeric suffix (e.g. manually added) should be ignored when determining the
    next column name, rather than raising or being treated as a numbered mistie
    column."""
    df, inters = _single_crossing_frames(12.0, 10.0)
    inters["crossover_error_manual"] = [5.0]
    result = calculate_crossover_errors(df, inters, data_col="mag", line_column="line")
    assert "crossover_error_0" in result.columns
    assert result["crossover_error_0"].to_numpy() == pytest.approx([2.0])
    assert result["crossover_error_manual"].to_numpy() == pytest.approx([5.0])


def test_calculate_crossover_errors_raises_if_unchanged_when_requested():
    """raise_error_if_unchanged=True should raise UserWarning when misties match the previous run."""
    df, inters = _single_crossing_frames(12.0, 10.0)
    first = calculate_crossover_errors(df, inters, data_col="mag", line_column="line")
    with pytest.raises(UserWarning, match="Mistie hasn't changed"):
        calculate_crossover_errors(
            df, first, data_col="mag", line_column="line", raise_error_if_unchanged=True
        )


# ---------------------------------------------------------------------------
# lines_without_intersections
# ---------------------------------------------------------------------------


def test_lines_without_intersections_returns_unmatched_lines():
    """Lines that never appear in either line1 or line2 of the intersections table should be returned."""
    data = pd.DataFrame({"line": ["A", "B", "C", "D"]})
    intersections = pd.DataFrame({"line1": ["A"], "line2": ["B"]})
    result = lines_without_intersections(data, intersections, line_column="line")
    assert sorted(result) == ["C", "D"]


# ---------------------------------------------------------------------------
# update_intersections_with_eq_sources
# ---------------------------------------------------------------------------


class _FakeEquivalentSource:
    """Stand-in for a fitted harmonica.EquivalentSources: just returns a fixed value."""

    def __init__(self, value):
        self.value = value

    def predict(self, coords):  # noqa: ARG002
        return np.array([self.value])


def test_update_intersections_with_eq_sources_writes_predicted_values():
    """Each intersection row's field value should be replaced by the fitted
    equivalent source's prediction for that line, evaluated at the intersection
    location and the higher of the two crossing lines' elevations."""
    data = pd.DataFrame(
        {
            "line": ["A", "B"],
            "intersecting_line": ["B", "A"],
            "is_intersection": [True, True],
            "dist_along_line": [10.0, 20.0],
            "height": [100.0, 120.0],
            "mag": [np.nan, np.nan],
        }
    )
    fitted = {"A": _FakeEquivalentSource(1.0), "B": _FakeEquivalentSource(2.0)}

    result = update_intersections_with_eq_sources(
        data,
        fitted_equivalent_sources=fitted,
        data_column="mag",
        line_column="line",
        distance_column="dist_along_line",
        progressbar=False,
    )
    assert result.to_numpy() == pytest.approx([1.0, 2.0])
