import re

import numpy as np
import pandas as pd
import pygmt
import pytest
import xarray as xr

from airbornegeo.filtering import _nearest_grid_fill, filter_grid, filter_line, pad1d


def test_pad1d_basic_reflect():
    """pad1d() with the default reflect mode should add symmetric padding derived from the median spacing."""
    x = np.arange(21, dtype=float)
    y = np.sin(x)
    data = pd.DataFrame({"dist": x, "val": y})
    result = pad1d(
        data, data_column="val", independent_column="dist", width_percentage=10
    )

    assert len(result) == 25  # 21 + 2*n_pad, n_pad=2
    assert list(result.columns) == ["true_index", "dist", "val"]

    lower = result.iloc[:2]
    assert lower["dist"].tolist() == pytest.approx([-2.0, -1.0])
    assert lower["true_index"].isna().all()

    real = result.iloc[2:23]
    assert real["true_index"].tolist() == pytest.approx(list(range(21)))
    assert real["dist"].tolist() == pytest.approx(x.tolist())
    assert real["val"].tolist() == pytest.approx(y.tolist())

    upper = result.iloc[23:]
    # upper pad's independent-column value duplicates the real last point's value
    assert upper["dist"].tolist() == pytest.approx([20.0, 22.0])
    assert upper["true_index"].isna().all()


def test_pad1d_bankers_rounding_n_pad():
    """The pad count should follow round-half-to-even (banker's rounding) when the raw pad width is exactly .5."""
    x = np.arange(11, dtype=float) * 2.5  # spacing 2.5, range 25.0
    data = pd.DataFrame({"dist": x, "val": np.zeros(11)})
    # pad_dist_raw = 25*0.15 = 3.75; round(3.75/2.5) = round(1.5) = 2 (round-half-to-even)
    result = pad1d(
        data, data_column="val", independent_column="dist", width_percentage=15
    )
    assert len(result) == 15  # 11 + 2*2
    lower = result.iloc[:2]
    upper = result.iloc[13:]
    assert lower["dist"].tolist() == pytest.approx([-5.0, -2.5])
    assert upper["dist"].tolist() == pytest.approx([25.0, 30.0])


def test_pad1d_n_pad_zero_is_a_no_op():
    """A width_percentage small enough to round down to zero pad points should leave the data unchanged."""
    x = np.arange(11, dtype=float)
    data = pd.DataFrame({"dist": x, "val": np.zeros(11)})
    result = pad1d(
        data, data_column="val", independent_column="dist", width_percentage=1
    )
    assert len(result) == 11
    assert result["true_index"].tolist() == pytest.approx(list(range(11)))


def test_pad1d_preserves_custom_index_as_true_index():
    """A non-default input index should be preserved as the 'true_index' column, with pad rows left as NaN."""
    data = pd.DataFrame(
        {"dist": np.arange(8, dtype=float), "val": np.arange(8, dtype=float)},
        index=range(100, 108),
    )
    result = pad1d(
        data,
        data_column="val",
        independent_column="dist",
        width_percentage=25,
        mode="constant",
        constant_values=-999,
    )
    assert len(result) == 12  # 8 + 2*2
    real = result[result["true_index"].notna()]
    assert real["true_index"].tolist() == pytest.approx(list(range(100, 108)))
    pad_rows = result[result["true_index"].isna()]
    assert pad_rows["val"].tolist() == pytest.approx([-999.0, -999.0, -999.0, -999.0])


def test_pad1d_single_row_raises():
    """A single row has no gap to compute spacing from, so np.median of an empty diff array raises (elevated by this repo's strict warnings config)."""
    # A single row has no gap to compute a spacing from: np.median of an empty
    # diff array emits `RuntimeWarning: Mean of empty slice`, which this repo's
    # pytest config (`filterwarnings = ["error"]`) turns into a real exception
    # before pad1d ever reaches its own `ValueError: cannot convert float NaN to
    # integer` (which is what a caller would see with warnings not elevated).
    data = pd.DataFrame({"dist": [5.0], "val": [1.0]})
    with pytest.raises(RuntimeWarning, match="Mean of empty slice"):
        pad1d(data, data_column="val", independent_column="dist", width_percentage=10)


def test_pad1d_two_rows_can_give_zero_pad():
    """Two rows should be sufficient to compute a spacing, and a small width_percentage can still yield zero pad points."""
    data = pd.DataFrame({"dist": [0.0, 10.0], "val": [1.0, 2.0]})
    result = pad1d(
        data, data_column="val", independent_column="dist", width_percentage=10
    )
    assert len(result) == 2
    assert result["true_index"].tolist() == pytest.approx([0.0, 1.0])


def test_pad1d_unsorted_independent_column_raises_valueerror():
    """An independent_column that isn't sorted ascending can produce a negative estimated spacing, which np.linspace rejects."""
    # pad1d assumes rows are already sorted ascending by independent_column; an
    # unsorted column can make the estimated spacing negative, and np.linspace
    # rejects a negative sample count.
    rng = np.random.default_rng(0)
    dist = rng.uniform(0, 100, 20)
    data = pd.DataFrame({"dist": dist, "val": np.zeros(20)})
    with pytest.raises(ValueError, match="must be non-negative"):
        pad1d(data, data_column="val", independent_column="dist", width_percentage=10)


def test_filter_line_lowpass_removes_short_wavelength_noise():
    """A lowpass filter should remove short-wavelength noise while preserving the long-wavelength signal."""
    x = np.linspace(0, 3000, 300)
    signal = np.sin(2 * np.pi * x / 3000)
    noise = 0.3 * np.sin(2 * np.pi * x / 50)
    data = pd.DataFrame({"dist": x, "val": signal + noise})

    result = filter_line(
        data,
        filter_type="g200",
        data_column="val",
        filter_by_column="dist",
        progressbar=False,
    )

    assert result.index.equals(data.index)
    residual = (result.to_numpy() - signal)[15:-15]
    assert np.max(np.abs(residual)) < 0.05


def test_filter_line_highpass_isolates_high_frequency():
    """A highpass filter should isolate the high-frequency component and remove the low-frequency one."""
    x = np.linspace(0, 3000, 300)
    low = np.sin(2 * np.pi * x / 2000)
    high = np.sin(2 * np.pi * x / 20)
    data = pd.DataFrame({"dist": x, "val": low + high})

    result = filter_line(
        data,
        filter_type="g100+h",
        data_column="val",
        filter_by_column="dist",
        progressbar=False,
    )

    residual = (result.to_numpy() - high)[15:-15]
    assert np.max(np.abs(residual)) < 0.05


@pytest.mark.parametrize("progressbar", [True, False])
def test_filter_line_groupby_preserves_row_alignment_interleaved(progressbar):
    """Grouped filtering on interleaved rows should return results realigned to the original row order, with or without a progress bar."""
    n = 150
    dist_b = np.linspace(0, 3000, n)
    dist_a = np.linspace(0, 3000, n)
    sig_b = np.sin(2 * np.pi * dist_b / 3000)
    sig_a = np.cos(2 * np.pi * dist_a / 3000)

    rows = []
    expected = np.empty(2 * n)
    for i in range(n):
        rows.append({"line": "B", "dist": dist_b[i], "val": sig_b[i]})
        expected[2 * i] = sig_b[i]
        rows.append({"line": "A", "dist": dist_a[i], "val": sig_a[i]})
        expected[2 * i + 1] = sig_a[i]
    data = pd.DataFrame(rows)

    result = filter_line(
        data,
        filter_type="g200",
        data_column="val",
        filter_by_column="dist",
        groupby_column="line",
        progressbar=progressbar,
    )

    assert result.index.equals(data.index)
    residual = (result.to_numpy() - expected)[20:-20]
    assert np.max(np.abs(residual)) < 0.05


def test_filter_line_groupby_labels_sort_differently_than_appearance():
    """Grouping should follow first-appearance order rather than sorted label order, so row alignment isn't scrambled."""
    # "B" appears before "A" but sorts after it - the grouped branch uses
    # `_iter_groups` (sort=False), so this must not scramble row alignment.
    n = 100
    dist = np.linspace(0, 2000, n)
    sig_b = np.sin(2 * np.pi * dist / 2000)
    sig_a = np.cos(2 * np.pi * dist / 2000)
    data = pd.DataFrame(
        {
            "line": ["B"] * n + ["A"] * n,
            "dist": np.concatenate([dist, dist]),
            "val": np.concatenate([sig_b, sig_a]),
        }
    )

    result = filter_line(
        data,
        filter_type="g200",
        data_column="val",
        filter_by_column="dist",
        groupby_column="line",
        progressbar=False,
    )

    assert result.index.equals(data.index)
    b_residual = (result.to_numpy()[:n] - sig_b)[15:-15]
    a_residual = (result.to_numpy()[n:] - sig_a)[15:-15]
    assert np.max(np.abs(b_residual)) < 0.05
    assert np.max(np.abs(a_residual)) < 0.05


def test_filter_line_pad_kwargs_pass_through():
    """Extra pad-related kwargs should be forwarded through to the padding step."""
    x = np.linspace(0, 1000, 100)
    data = pd.DataFrame({"dist": x, "val": np.sin(2 * np.pi * x / 1000)})
    result = filter_line(
        data,
        filter_type="g100",
        data_column="val",
        filter_by_column="dist",
        pad_mode="constant",
        constant_values=0.0,
        progressbar=False,
    )
    assert len(result) == 100
    assert result.index.equals(data.index)


def test_filter_line_invalid_filter_type_raises_gmt_error():
    """An invalid filter_type string should surface as a GMT CLib error from pygmt."""
    data = pd.DataFrame({"dist": np.linspace(0, 100, 20), "val": np.zeros(20)})
    with pytest.raises(pygmt.exceptions.GMTCLibError):
        filter_line(
            data,
            filter_type="bogus_filter",
            data_column="val",
            filter_by_column="dist",
            progressbar=False,
        )


def _linear_ramp_grid_with_nan_patch():
    northing = np.arange(15, dtype=float)
    easting = np.arange(15, dtype=float)
    e, n = np.meshgrid(easting, northing)
    data = e + n
    data[5:8, 5:8] = np.nan
    return xr.DataArray(
        data,
        coords={"northing": northing, "easting": easting},
        dims=("northing", "easting"),
        name="z",
    )


def test_nearest_grid_fill_verde_fills_all_nans_and_preserves_metadata():
    """method='verde' should fill every NaN in the grid while preserving its dims, shape, and name."""
    grid = _linear_ramp_grid_with_nan_patch()
    filled = _nearest_grid_fill(grid, method="verde")

    assert filled.dims == ("northing", "easting")
    assert filled.shape == (15, 15)
    assert filled.name == "z"
    assert not filled.isnull().any()  # noqa: PD003 (xarray.DataArray has no .isna())


def test_nearest_grid_fill_invalid_method_raises_valueerror():
    """An unrecognized method string should raise a ValueError naming the valid options."""
    grid = _linear_ramp_grid_with_nan_patch()
    with pytest.raises(ValueError, match="method must be 'rioxarray', or 'verde'"):
        _nearest_grid_fill(grid, method="bogus")


def test_nearest_grid_fill_rioxarray_method_unavailable(monkeypatch):
    """When rioxarray isn't installed, method='rioxarray' should raise ImportError."""
    monkeypatch.setattr("airbornegeo.filtering._HAS_RIOXARRAY", False)

    grid = _linear_ramp_grid_with_nan_patch()
    with pytest.raises(
        ImportError,
        match=re.escape(
            "The 'rioxarray' method requires the optional dependency 'rioxarray'. "
            "Install it with `pip install rioxarray` or `mamba install rioxarray`, "
            "or use method='verde' instead."
        ),
    ):
        _nearest_grid_fill(grid, method="rioxarray")


def test_nearest_grid_fill_rioxarray_method_success():
    """When rioxarray is installed, method='rioxarray' should fill every NaN in the grid."""
    pytest.importorskip("rioxarray")
    grid = _linear_ramp_grid_with_nan_patch()
    filled = _nearest_grid_fill(grid, method="rioxarray", crs="epsg:4326")
    assert not filled.isnull().any()  # noqa: PD003 (xarray.DataArray has no .isna())


def _low_high_grid(ny=41, nx=41, domain=4000.0):
    northing = np.linspace(0, domain, ny)
    easting = np.linspace(0, domain, nx)
    e, n = np.meshgrid(easting, northing)
    low = np.sin(2 * np.pi * e / domain) + np.cos(2 * np.pi * n / domain)
    high = 0.3 * np.sin(2 * np.pi * e / 200)
    return (
        xr.DataArray(
            low + high,
            coords={"northing": northing, "easting": easting},
            dims=("northing", "easting"),
            name="z",
        ),
        low,
    )


@pytest.mark.parametrize(
    ("filter_type", "kwargs"),
    [
        ("lowpass", {"filter_width": 1000}),
        ("highpass", {"filter_width": 1000}),
        ("up_deriv", {}),
        ("easting_deriv", {}),
        ("northing_deriv", {}),
        ("up_continue", {"height_displacement": 100}),
        ("total_gradient", {}),
        ("horizontal_gradient", {}),
    ],
)
def test_filter_grid_all_filter_types_run(filter_type, kwargs):
    """Every supported filter_type should run without error and return a grid of the same shape and name."""
    grid, _ = _low_high_grid()
    result = filter_grid(grid, filter_type=filter_type, **kwargs)
    assert result.shape == (41, 41)
    assert result.name == "z"
    assert not result.isnull().any()  # noqa: PD003 (xarray.DataArray has no .isna())


def test_filter_grid_lowpass_removes_short_wavelength_noise():
    """filter_grid's lowpass filter should remove short-wavelength noise while preserving the long-wavelength signal."""
    grid, low = _low_high_grid()
    result = filter_grid(grid, filter_type="lowpass", filter_width=1000)
    residual = (result.to_numpy() - low)[8:-8, 8:-8]
    assert np.max(np.abs(residual)) < 0.15


def test_filter_grid_restores_original_coordinates():
    """The filtered grid should keep the same northing/easting coordinates as the input."""
    grid, _ = _low_high_grid()
    result = filter_grid(grid, filter_type="easting_deriv")
    assert np.allclose(result.northing.to_numpy(), grid.northing.to_numpy())
    assert np.allclose(result.easting.to_numpy(), grid.easting.to_numpy())


def test_filter_grid_nan_mask_round_trips_exactly():
    """NaN cells in the input grid should remain NaN in the filtered output, in exactly the same positions."""
    ny, nx = 31, 31
    northing = np.linspace(0, 3000, ny)
    easting = np.linspace(0, 3000, nx)
    e, n = np.meshgrid(easting, northing)
    data = np.sin(2 * np.pi * e / 3000) + np.cos(2 * np.pi * n / 3000)
    data[10:15, 10:15] = np.nan
    grid = xr.DataArray(
        data,
        coords={"northing": northing, "easting": easting},
        dims=("northing", "easting"),
        name="z",
    )
    result = filter_grid(grid, filter_type="lowpass", filter_width=500)
    assert (result.isnull() == grid.isnull()).all()  # noqa: PD003 (xarray has no .isna())


@pytest.mark.parametrize(
    ("filter_type", "msg"),
    [
        ("lowpass", "filter_width must be provided if filter_type is 'lowpass'"),
        ("highpass", "filter_width must be provided if filter_type is 'highpass'"),
        (
            "up_continue",
            "height_displacement must be provided if filter_type is 'up_continue'",
        ),
    ],
)
def test_filter_grid_missing_required_arg_raises_valueerror(filter_type, msg):
    """Filter types that need an extra argument should raise ValueError when it's omitted."""
    grid, _ = _low_high_grid(ny=15, nx=15)
    with pytest.raises(ValueError, match=msg):
        filter_grid(grid, filter_type=filter_type)


def test_filter_grid_invalid_filter_type_raises_valueerror():
    """An unrecognized filter_type string should raise a ValueError listing the valid options."""
    grid, _ = _low_high_grid(ny=15, nx=15)
    with pytest.raises(
        ValueError,
        match=(
            r"filter_type must be 'lowpass', 'highpass' 'up_deriv', 'easting_deriv', "
            r"'northing_deriv', 'up_continue', or 'total_gradient'"
        ),
    ):
        filter_grid(grid, filter_type="bogus")


@pytest.mark.parametrize(
    "pad_kwargs",
    [
        {"pad_mode": "constant"},
        {"pad_mode": "constant", "pad_constant": 0.0},
        {"pad_mode": "linear_ramp", "pad_end_values": 0.0},
        {"pad_width_factor": 5},
    ],
)
def test_filter_grid_pad_mode_variants_run(pad_kwargs):
    """Every supported pad_mode/pad kwarg combination should run without error and preserve the grid shape."""
    grid, _ = _low_high_grid(ny=21, nx=21)
    result = filter_grid(grid, filter_type="easting_deriv", **pad_kwargs)
    assert result.shape == (21, 21)
