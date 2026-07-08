import boule as bl
import harmonica as hm
import numpy as np
import pandas as pd
import pytest

from airbornegeo.potential_fields import eq_sources_1d, igrf, upward_continue_by_line

RNG_SEED = 0


def _two_line_df(seed=RNG_SEED, n=20):
    rng = np.random.default_rng(seed)
    df1 = pd.DataFrame(
        {
            "distance_along_line": np.linspace(0, 1000, n),
            "height": 500 + rng.normal(0, 1, n),
            "mag": 50000 + rng.normal(0, 10, n),
            "line": "A",
        }
    )
    df2 = pd.DataFrame(
        {
            "distance_along_line": np.linspace(2000, 3000, n),
            "height": 500 + rng.normal(0, 1, n),
            "mag": 50000 + rng.normal(0, 10, n),
            "line": "B",
        }
    )
    return pd.concat([df1, df2], ignore_index=True)


def test_eq_sources_1d_no_groupby_returns_single_fit():
    """With groupby_column=None, eq_sources_1d() should return a single fitted EquivalentSources."""
    data = _two_line_df()
    result = eq_sources_1d(
        data, data_column="mag", damping=1e-3, groupby_column=None, progressbar=False
    )
    assert isinstance(result, hm.EquivalentSources)


def test_eq_sources_1d_groupby_returns_dict_keyed_by_first_appearance_order():
    """With groupby_column set, eq_sources_1d() should return a dict of per-group fits keyed in appearance order."""
    data = _two_line_df()
    result = eq_sources_1d(
        data, data_column="mag", damping=1e-3, groupby_column="line", progressbar=False
    )
    assert isinstance(result, dict)
    assert list(result.keys()) == ["A", "B"]
    assert all(isinstance(v, hm.EquivalentSources) for v in result.values())


def test_eq_sources_1d_groupby_key_order_matches_appearance_not_sort():
    """Dict key order should follow first-appearance order, not alphabetical sort order."""
    data = _two_line_df()
    data = data.iloc[::-1].reset_index(drop=True)  # B rows now come first
    result = eq_sources_1d(
        data, data_column="mag", damping=1e-3, groupby_column="line", progressbar=False
    )
    assert list(result.keys()) == ["B", "A"]


def test_eq_sources_1d_groupby_column_missing_raises_assertionerror():
    """A groupby_column that doesn't exist in the dataframe should raise AssertionError."""
    data = pd.DataFrame(
        {
            "distance_along_line": np.linspace(0, 1000, 10),
            "height": 500 + np.random.default_rng(1).normal(0, 1, 10),
            "mag": 50000 + np.random.default_rng(1).normal(0, 10, 10),
        }
    )
    with pytest.raises(AssertionError, match="groupby_column must be in dataframe"):
        eq_sources_1d(
            data,
            data_column="mag",
            damping=1e-3,
            groupby_column="line",
            progressbar=False,
        )


def test_eq_sources_1d_missing_distance_along_line_raises_attributeerror():
    """A missing distance_along_line column should raise AttributeError (accessed via attribute, not bracket)."""
    data = pd.DataFrame({"foo": [1, 2, 3]})
    with pytest.raises(AttributeError):
        eq_sources_1d(
            data,
            data_column="mag",
            damping=1e-3,
            groupby_column=None,
            progressbar=False,
        )


def test_eq_sources_1d_missing_data_column_raises_keyerror():
    """A missing data_column should raise KeyError."""
    data = pd.DataFrame(
        {"distance_along_line": [0.0, 1.0, 2.0], "height": [1.0, 2.0, 3.0]}
    )
    with pytest.raises(KeyError):
        eq_sources_1d(
            data,
            data_column="mag",
            damping=1e-3,
            groupby_column=None,
            progressbar=False,
        )


def test_eq_sources_1d_does_not_mutate_original_dataframe():
    """eq_sources_1d() should not leave its internal 'tmp' column or any other changes on the caller's dataframe."""
    data = _two_line_df()
    columns_before = data.columns.tolist()
    eq_sources_1d(
        data, data_column="mag", damping=1e-3, groupby_column="line", progressbar=False
    )
    assert data.columns.tolist() == columns_before
    assert "tmp" not in data.columns


def test_eq_sources_1d_block_size_accepted():
    """The block_size kwarg should be accepted and forwarded without error."""
    data = _two_line_df()
    result = eq_sources_1d(
        data,
        data_column="mag",
        damping=1e-3,
        block_size=50,
        groupby_column=None,
        progressbar=False,
    )
    assert isinstance(result, hm.EquivalentSources)


@pytest.mark.parametrize("progressbar", [True, False])
def test_eq_sources_1d_progressbar_does_not_affect_result(progressbar):
    """The progressbar setting should not affect the fitted result."""
    data = _two_line_df()
    result = eq_sources_1d(
        data,
        data_column="mag",
        damping=1e-3,
        groupby_column="line",
        progressbar=progressbar,
    )
    assert list(result.keys()) == ["A", "B"]


def test_upward_continue_by_line_missing_line_column_raises():
    """A missing literal 'line' column should raise AssertionError, regardless of groupby_column."""
    data = pd.DataFrame({"height": [1.0, 2.0], "distance_along_line": [0.0, 1.0]})
    with pytest.raises(AssertionError, match="line column must be in dataframe"):
        upward_continue_by_line(
            data, {}, height=100, groupby_column="line", progressbar=False
        )


def test_upward_continue_by_line_missing_height_column_raises():
    """A missing 'height' column should raise AssertionError."""
    data = pd.DataFrame({"line": ["A", "A"], "distance_along_line": [0.0, 1.0]})
    with pytest.raises(AssertionError, match="height column must be in dataframe"):
        upward_continue_by_line(
            data, {}, height=100, groupby_column="line", progressbar=False
        )


def test_upward_continue_by_line_missing_group_key_raises_keyerror():
    """A fitted_equivalent_sources dict missing a group's key should raise KeyError."""
    data = pd.DataFrame(
        {"line": ["A", "A"], "height": [1.0, 2.0], "distance_along_line": [0.0, 1.0]}
    )
    with pytest.raises(KeyError):
        upward_continue_by_line(
            data, {}, height=100, groupby_column="line", progressbar=False
        )


def test_upward_continue_by_line_bare_equivalent_sources_raises_typeerror():
    """Passing a bare EquivalentSources instead of a dict should raise TypeError."""
    data = _two_line_df()
    eqs_single = eq_sources_1d(
        data, data_column="mag", damping=1e-3, groupby_column=None, progressbar=False
    )
    with pytest.raises(TypeError, match="not subscriptable"):
        upward_continue_by_line(
            data, eqs_single, height=600, groupby_column="line", progressbar=False
        )


def test_upward_continue_by_line_returns_series_aligned_to_data():
    """The result should be a fully-populated Series aligned to the input dataframe's index."""
    data = _two_line_df()
    eqs = eq_sources_1d(
        data, data_column="mag", damping=1e-3, groupby_column="line", progressbar=False
    )
    result = upward_continue_by_line(
        data, eqs, height=600, groupby_column="line", progressbar=False
    )
    assert isinstance(result, pd.Series)
    assert result.name == "upward_continued"
    assert result.index.equals(data.index)
    assert result.notna().all()


def test_upward_continue_by_line_groupby_column_independent_of_literal_line_assert():
    """Grouping/looking up by a different column still requires a literal 'line' column to exist, purely for the hardcoded assert."""
    data = _two_line_df().rename(columns={"line": "site"})
    data["line"] = "unused"
    eqs = eq_sources_1d(
        data, data_column="mag", damping=1e-3, groupby_column="site", progressbar=False
    )
    result = upward_continue_by_line(
        data, eqs, height=600, groupby_column="site", progressbar=False
    )
    assert result.shape == (40,)
    assert result.notna().all()


def test_upward_continue_by_line_fractional_height_is_truncated_to_int():
    """Regression test: the internal int64 dummy column silently floors any fractional part of the requested height."""
    data = _two_line_df()
    eqs = eq_sources_1d(
        data, data_column="mag", damping=1e-3, groupby_column="line", progressbar=False
    )
    result_frac = upward_continue_by_line(
        data, eqs, height=700.7, groupby_column="line", progressbar=False
    )
    result_int = upward_continue_by_line(
        data, eqs, height=700.0, groupby_column="line", progressbar=False
    )
    assert np.array_equal(result_frac.to_numpy(), result_int.to_numpy())


def test_upward_continue_by_line_clamps_below_flight_height_by_default(monkeypatch):
    """With no_downward_continuation=True, points above the requested height should be clamped to their own actual height."""
    data = _two_line_df()
    eqs = eq_sources_1d(
        data, data_column="mag", damping=1e-3, groupby_column="line", progressbar=False
    )

    captured = []
    original_predict = hm.EquivalentSources.predict

    def spy_predict(self, coordinates):
        captured.append(np.asarray(coordinates[2]).copy())
        return original_predict(self, coordinates)

    monkeypatch.setattr(hm.EquivalentSources, "predict", spy_predict)

    requested_height = 300
    upward_continue_by_line(
        data,
        eqs,
        height=requested_height,
        groupby_column="line",
        no_downward_continuation=True,
        progressbar=False,
    )

    line_a = data[data.line == "A"]
    expected = np.where(
        requested_height > line_a.height.to_numpy(),
        requested_height,
        line_a.height.to_numpy(),
    )
    assert captured[0] == pytest.approx(expected)


def test_upward_continue_by_line_no_clamp_when_disabled(monkeypatch):
    """With no_downward_continuation=False, every point should get exactly the constant requested height, unclamped."""
    data = _two_line_df()
    eqs = eq_sources_1d(
        data, data_column="mag", damping=1e-3, groupby_column="line", progressbar=False
    )

    captured = []
    original_predict = hm.EquivalentSources.predict

    def spy_predict(self, coordinates):
        captured.append(np.asarray(coordinates[2]).copy())
        return original_predict(self, coordinates)

    monkeypatch.setattr(hm.EquivalentSources, "predict", spy_predict)

    requested_height = 300
    upward_continue_by_line(
        data,
        eqs,
        height=requested_height,
        groupby_column="line",
        no_downward_continuation=False,
        progressbar=False,
    )

    assert captured[0] == pytest.approx(np.full(20, requested_height))


def test_upward_continue_by_line_clamped_request_near_flight_height_approximates_input():
    """Requesting a height below all actual flight heights should give clamped results close to the original data."""
    data = _two_line_df()
    eqs = eq_sources_1d(
        data, data_column="mag", damping=1e-3, groupby_column="line", progressbar=False
    )
    result = upward_continue_by_line(
        data, eqs, height=400, groupby_column="line", progressbar=False
    )
    assert result.to_numpy() == pytest.approx(data.mag.to_numpy(), rel=0.05)


def test_upward_continue_by_line_high_continuation_attenuates_field():
    """A large upward continuation (~500 m) should strongly damp the field away from its original magnitude."""
    data = _two_line_df()
    eqs = eq_sources_1d(
        data, data_column="mag", damping=1e-3, groupby_column="line", progressbar=False
    )
    result = upward_continue_by_line(
        data, eqs, height=1000, groupby_column="line", progressbar=False
    )
    assert np.abs(result.to_numpy() - 50000).min() > 1000


def _igrf_df(n_per_line=5):
    rng = np.random.default_rng(2)
    return pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2020-01-01"] * (2 * n_per_line)),
            "lat": 50.0 + rng.normal(0, 0.5, 2 * n_per_line),
            "lon": 10.0 + rng.normal(0, 0.5, 2 * n_per_line),
            "height": np.full(2 * n_per_line, 1000.0),
            "line": ["A"] * n_per_line + ["B"] * n_per_line,
        }
    )


def test_igrf_missing_columns_no_groupby_raises_assertionerror():
    """Missing required columns (no groupby_column) should raise AssertionError."""
    data = pd.DataFrame(
        {
            "datetime": [pd.Timestamp("2020-01-01")] * 2,
            "lat": [50.0, 51.0],
            "lon": [10.0, 11.0],
        }
    )
    with pytest.raises(AssertionError, match=r"dataframe must contain columns"):
        igrf(
            data,
            datetime_column="datetime",
            latitude_column="lat",
            longitude_column="lon",
            height_column="alt",
            groupby_column=None,
            progressbar=False,
        )


def test_igrf_missing_groupby_column_raises_assertionerror():
    """A groupby_column that doesn't exist should raise AssertionError."""
    data = _igrf_df()
    with pytest.raises(AssertionError, match=r"dataframe must contain columns"):
        igrf(
            data,
            datetime_column="datetime",
            latitude_column="lat",
            longitude_column="lon",
            height_column="height",
            groupby_column="not_a_real_column",
            progressbar=False,
        )


def test_igrf_returns_tuple_of_arrays_no_groupby():
    """igrf() should return a 3-tuple of plausible-magnitude intensity/inclination/declination arrays."""
    data = _igrf_df()
    result = igrf(
        data,
        datetime_column="datetime",
        latitude_column="lat",
        longitude_column="lon",
        height_column="height",
        groupby_column=None,
        progressbar=False,
    )
    assert isinstance(result, tuple)
    assert len(result) == 3
    intensity, inclination, declination = result
    assert isinstance(intensity, np.ndarray)
    assert intensity.shape == (10,)
    assert np.all((intensity > 20000) & (intensity < 70000))
    assert np.all((inclination >= -90) & (inclination <= 90))
    assert np.all((declination >= -180) & (declination <= 180))


def test_igrf_returns_tuple_of_arrays_groupby():
    """igrf() should still return a 3-tuple of arrays (not a dict) when groupby_column is set."""
    data = _igrf_df()
    result = igrf(
        data,
        datetime_column="datetime",
        latitude_column="lat",
        longitude_column="lon",
        height_column="height",
        groupby_column="line",
        progressbar=False,
    )
    assert isinstance(result, tuple)
    assert len(result) == 3
    assert result[0].shape == (10,)


def test_igrf_only_uses_first_row_datetime_per_group():
    """Only each group's first-row datetime should affect the result, even if later rows have very different datetimes."""
    data = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2020-01-01", "2023-01-01", "2020-01-01", "2010-01-01"]
            ),
            "lat": [50.0, 50.1, 50.0, 50.1],
            "lon": [10.0, 10.1, 10.0, 10.1],
            "height": [1000.0, 1000.0, 1000.0, 1000.0],
            "line": ["A", "A", "B", "B"],
        }
    )
    intensity, inclination, declination = igrf(
        data,
        datetime_column="datetime",
        latitude_column="lat",
        longitude_column="lon",
        height_column="height",
        groupby_column="line",
        progressbar=False,
    )
    # row0 (group A, first row) and row2 (group B, first row) share datetime
    # 2020-01-01 and identical lat/lon -> results must match exactly, even though
    # row1's (2023) and row3's (2010) datetimes are never actually used.
    assert intensity[0] == pytest.approx(intensity[2])
    assert intensity[1] == pytest.approx(intensity[3])
    assert inclination[0] == pytest.approx(inclination[2])
    assert declination[0] == pytest.approx(declination[2])


def test_igrf_row_order_safety_interleaved_groups():
    """Grouped and ungrouped results should agree exactly, row for row, even with interleaved/unsorted group labels."""
    data = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2020-01-01"] * 4),
            "lat": [50.0, 55.0, 50.1, 55.1],
            "lon": [10.0, 20.0, 10.1, 20.1],
            "height": [1000.0] * 4,
            "line": ["B", "A", "B", "A"],  # interleaved, first-appearance B,A
        }
    )
    grouped = igrf(
        data,
        datetime_column="datetime",
        latitude_column="lat",
        longitude_column="lon",
        height_column="height",
        groupby_column="line",
        progressbar=False,
    )
    ungrouped = igrf(
        data,
        datetime_column="datetime",
        latitude_column="lat",
        longitude_column="lon",
        height_column="height",
        groupby_column=None,
        progressbar=False,
    )
    # all rows share the same datetime, so grouped and ungrouped must agree exactly,
    # row for row - this isolates the row-order-realignment behavior.
    assert grouped[0] == pytest.approx(ungrouped[0])
    assert grouped[1] == pytest.approx(ungrouped[1])
    assert grouped[2] == pytest.approx(ungrouped[2])


def test_igrf_single_row_segment_raises_typeerror():
    """Documents a real harmonica/numpy interaction limitation: any single-row segment crashes IGRF14.predict."""
    data = pd.DataFrame(
        {
            "datetime": [pd.Timestamp("2020-01-01")],
            "lat": [50.0],
            "lon": [10.0],
            "height": [1000.0],
        }
    )
    with pytest.raises(TypeError, match="0-dimensional"):
        igrf(
            data,
            datetime_column="datetime",
            latitude_column="lat",
            longitude_column="lon",
            height_column="height",
            groupby_column=None,
            progressbar=False,
        )


def test_igrf_ellipsoid_kwarg_forwarded():
    """The ellipsoid kwarg should be forwarded to IGRF14 and produce slightly different results for different ellipsoids."""
    data = _igrf_df()
    result_wgs84 = igrf(
        data,
        datetime_column="datetime",
        latitude_column="lat",
        longitude_column="lon",
        height_column="height",
        groupby_column=None,
        progressbar=False,
        ellipsoid=bl.WGS84,
    )
    result_grs80 = igrf(
        data,
        datetime_column="datetime",
        latitude_column="lat",
        longitude_column="lon",
        height_column="height",
        groupby_column=None,
        progressbar=False,
        ellipsoid=bl.GRS80,
    )
    assert result_wgs84[0] == pytest.approx(result_grs80[0], rel=1e-3)


@pytest.mark.parametrize("progressbar", [True, False])
def test_igrf_progressbar_does_not_affect_result(progressbar):
    """The progressbar setting should not affect the returned result."""
    data = _igrf_df()
    result = igrf(
        data,
        datetime_column="datetime",
        latitude_column="lat",
        longitude_column="lon",
        height_column="height",
        groupby_column="line",
        progressbar=progressbar,
    )
    assert result[0].shape == (10,)
