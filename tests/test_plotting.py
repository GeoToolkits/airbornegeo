import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pygmt
import pytest

import airbornegeo.plotting as plotting_mod
from airbornegeo.plotting import (
    LevellingConvergenceMonitor,
    align_yaxis,
    inspect_lines,
    plot_flightlines,
    plot_flightlines_grids,
    plot_levelling_convergence,
    plot_profiles,
    plotly_points,
    plotly_profiles,
)


@pytest.fixture(autouse=True)
def _no_gui_plots(monkeypatch):
    monkeypatch.setattr(plotting_mod.plt, "show", lambda *a, **k: None)
    yield
    plotting_mod.plt.close("all")


def test_align_yaxis_already_aligned_stays_unchanged():
    """Aligning two axes whose reference values already sit at the same relative position should leave both ylims unchanged."""
    fig, ax1 = plotting_mod.plt.subplots()
    ax2 = ax1.twinx()
    ax1.set_ylim(0, 10)
    ax2.set_ylim(0, 100)
    align_yaxis(ax1, 5, ax2, 50)
    assert ax1.get_ylim() == pytest.approx((0.0, 10.0))
    assert ax2.get_ylim() == pytest.approx((0.0, 100.0), abs=1e-6)


def test_align_yaxis_shifts_misaligned_axis():
    """Aligning two axes with misaligned reference values should shift the second axis's ylim, keeping its span constant."""
    fig, ax1 = plotting_mod.plt.subplots()
    ax2 = ax1.twinx()
    ax1.set_ylim(0, 10)
    ax2.set_ylim(0, 100)
    align_yaxis(ax1, 5, ax2, 20)
    # ax1's v1=5 sits at the midpoint (relative position 0.5); ax2 should shift
    # (keeping its span of 100) so v2=20 also sits at relative position 0.5
    new_min, new_max = ax2.get_ylim()
    relative_position = (20 - new_min) / (new_max - new_min)
    assert relative_position == pytest.approx(0.5)
    assert (new_max - new_min) == pytest.approx(100.0)


class _FakeDisplayHandle:
    def __init__(self):
        self.update_calls = 0

    def update(self, *args, **kwargs):
        self.update_calls += 1


def test_levelling_convergence_monitor_update_sets_axis_data(monkeypatch):
    """update() should store the rms/delta_rms history and plot a line on the monitor's first axis."""
    fake_handle = _FakeDisplayHandle()
    monkeypatch.setattr(plotting_mod, "display", lambda *a, **k: fake_handle)

    monitor = LevellingConvergenceMonitor(
        rms_tolerance=1.0, rms_percent_change_tolerance=5.0
    )
    monitor.update([10.0, 5.0, 2.0], [np.nan, 50.0, 10.0])

    assert monitor.rms_values == [10.0, 5.0, 2.0]
    assert monitor.delta_rms_values[1:] == pytest.approx([50.0, 10.0])
    assert fake_handle.update_calls >= 1
    # RMS line should have been plotted on ax1
    assert len(monitor.ax1.lines) > 0


def test_levelling_convergence_monitor_without_tolerances(monkeypatch):
    """update() should run without error when no tolerance lines are configured."""
    fake_handle = _FakeDisplayHandle()
    monkeypatch.setattr(plotting_mod, "display", lambda *a, **k: fake_handle)

    monitor = LevellingConvergenceMonitor()
    monitor.update([10.0, 5.0, 2.0], [np.nan, 50.0, 10.0])
    assert monitor.ax1.get_legend() is None or True  # no crash is the main assertion


def test_plot_levelling_convergence_runs_and_sets_labels():
    """The standalone plot_levelling_convergence() function should produce a figure with an 'Iteration' x-label and a plotted line."""
    plot_levelling_convergence(
        [10.0, 5.0, 2.0],
        [np.nan, 50.0, 10.0],
        rms_tolerance=1.0,
        rms_percent_change_tolerance=5.0,
    )
    fig = plotting_mod.plt.gcf()
    ax1 = fig.axes[0]
    assert ax1.get_xlabel() == "Iteration"
    assert len(ax1.lines) > 0


def test_plot_levelling_convergence_without_tolerances_runs():
    """plot_levelling_convergence() should still produce both axes when no tolerances are given."""
    plot_levelling_convergence([10.0, 5.0, 2.0], [np.nan, 50.0, 10.0])
    fig = plotting_mod.plt.gcf()
    assert len(fig.axes) == 2


def test_plotly_profiles_single_y_axis():
    """A single y column should produce one trace on the default axis, with the x-axis titled from the x column name."""
    data = pd.DataFrame({"x": [0.0, 1.0, 2.0], "a": [1.0, 2.0, 3.0]})
    fig = plotly_profiles(data, y="a", x="x")
    assert len(fig.data) == 1
    assert fig.data[0].y == pytest.approx((1.0, 2.0, 3.0))
    assert fig.layout.xaxis.title.text == "x"


def test_plotly_profiles_multiple_y_axes():
    """Multiple y columns with distinct y_axes should each be assigned their own axis and title."""
    data = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0],
            "a": [1.0, 2.0, 3.0],
            "b": [10.0, 20.0, 30.0],
            "c": [-1.0, -2.0, -3.0],
        }
    )
    fig = plotly_profiles(data, y=["a", "b", "c"], x="x", y_axes=["1", "2", "3"])
    assert len(fig.data) == 3
    assert fig.data[0].yaxis == "y"
    assert fig.data[1].yaxis == "y2"
    assert fig.data[2].yaxis == "y3"
    assert fig.layout.yaxis2.title.text == "b"
    assert fig.layout.yaxis3.title.text == "c"


def test_plotly_profiles_zero_axis_raises():
    """A y_axes value of '0' should raise an AssertionError since axis 0 is not allowed."""
    data = pd.DataFrame({"x": [0.0], "a": [1.0]})
    with pytest.raises(AssertionError, match="No '0' or 0 allowed"):
        plotly_profiles(data, y="a", x="x", y_axes=["0"])


def test_plotly_profiles_x_lims_and_y_lims_applied():
    """x_lims and y_lims should be applied to the figure's axis ranges."""
    data = pd.DataFrame({"x": [0.0, 1.0, 2.0], "a": [1.0, 2.0, 3.0]})
    fig = plotly_profiles(data, y="a", x="x", x_lims=(0.0, 5.0), y_lims=(0.0, 10.0))
    assert tuple(fig.layout.xaxis.range) == (0.0, 5.0)
    assert tuple(fig.layout.yaxis.range) == (0.0, 10.0)


def test_plotly_profiles_title_set():
    """A given title should be set on the resulting figure's layout."""
    data = pd.DataFrame({"x": [0.0], "a": [1.0]})
    fig = plotly_profiles(data, y="a", x="x", title="my title")
    assert fig.layout.title.text == "my title"


def test_plot_profiles_single_axis():
    """A single y column should produce a figure with exactly one axis."""
    data = pd.DataFrame({"x": [0.0, 1.0, 2.0], "a": [1.0, 2.0, 3.0]})
    fig = plot_profiles(data, y="a", x="x")
    assert len(fig.axes) == 1


def test_plot_profiles_two_axes():
    """Two y columns assigned to two distinct y_axes should produce a figure with two axes."""
    data = pd.DataFrame(
        {"x": [0.0, 1.0, 2.0], "a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]}
    )
    fig = plot_profiles(data, y=["a", "b"], x="x", y_axes=[1, 2])
    assert len(fig.axes) == 2


def test_plot_profiles_zero_axis_raises():
    """A y_axes value of 0 should raise an AssertionError since axis 0 is not allowed."""
    data = pd.DataFrame({"x": [0.0], "a": [1.0]})
    with pytest.raises(AssertionError, match="No 0 allowed"):
        plot_profiles(data, y="a", x="x", y_axes=[0])


def test_plot_profiles_more_than_two_axes_raises_not_implemented():
    """More than two unique y_axes values should raise NotImplementedError."""
    data = pd.DataFrame({"x": [0.0], "a": [1.0], "b": [2.0], "c": [3.0]})
    with pytest.raises(NotImplementedError, match="Only 2 unique y axes"):
        plot_profiles(data, y=["a", "b", "c"], x="x", y_axes=[1, 2, 3])


def _flightline_df():
    return pd.DataFrame(
        {
            "easting": np.concatenate(
                [np.linspace(0, 100, 10), np.linspace(0, 100, 10)]
            ),
            "northing": np.concatenate([np.zeros(10), np.full(10, 50.0)]),
            "line": [1] * 10 + [2] * 10,
            "track": [90.0] * 20,
        }
    )


def test_plot_flightlines_runs_without_error():
    """plot_flightlines() should run without error given a PyGMT figure and valid flightline data."""
    df = _flightline_df()
    fig = pygmt.Figure()
    fig.basemap(region=[0, 100, 0, 100], projection="X10c", frame=True)
    plot_flightlines(fig, df)


def test_plot_flightlines_missing_coord_columns_raises():
    """Missing easting/northing columns should raise an AssertionError."""
    df = pd.DataFrame({"x": [1.0], "y": [2.0], "line": [1]})
    fig = pygmt.Figure()
    fig.basemap(region=[0, 10, 0, 10], projection="X5c", frame=True)
    with pytest.raises(AssertionError, match="Project"):
        plot_flightlines(fig, df)


@pytest.mark.parametrize("direction", ["EW", "NS"])
def test_plot_flightlines_grids_runs_without_error(direction):
    """plot_flightlines_grids() should run without error for both supported direction strings."""
    df = _flightline_df()
    fig = pygmt.Figure()
    fig.basemap(region=[0, 100, 0, 100], projection="X10c", frame=True)
    plot_flightlines_grids(fig, df, direction=direction)


def test_plot_flightlines_grids_invalid_direction_raises():
    """An invalid direction string should raise a ValueError."""
    df = _flightline_df()
    fig = pygmt.Figure()
    fig.basemap(region=[0, 100, 0, 100], projection="X10c", frame=True)
    with pytest.raises(ValueError, match="invalid direction string"):
        plot_flightlines_grids(fig, df, direction="bogus")


def test_plotly_points_easting_northing_autodetect(monkeypatch):
    """With no explicit coord_names, plotly_points() should autodetect the easting/northing columns for the x data."""
    captured = []
    monkeypatch.setattr(go.Figure, "show", lambda self, *a, **k: captured.append(self))

    data = pd.DataFrame(
        {
            "easting": [0.0, 1.0, 2.0],
            "northing": [0.0, 1.0, 2.0],
            "val": [1.0, 2.0, 3.0],
        }
    )
    plotly_points(data, color_col="val")
    assert len(captured) == 1
    assert list(captured[0].data[0].x) == pytest.approx([0.0, 1.0, 2.0])


def test_plotly_points_all_nan_color_column_raises(monkeypatch):
    """A color_col that is entirely NaN should raise an AssertionError."""
    monkeypatch.setattr(go.Figure, "show", lambda self, *a, **k: None)
    data = pd.DataFrame(
        {"easting": [0.0, 1.0], "northing": [0.0, 1.0], "val": [np.nan, np.nan]}
    )
    with pytest.raises(AssertionError, match="no non nan values"):
        plotly_points(data, color_col="val")


def test_inspect_lines_iterates_all_lines(monkeypatch):
    """inspect_lines() should show one figure per unique line value, each titled with its line number."""
    shown = []
    monkeypatch.setattr(go.Figure, "show", lambda self, *a, **k: shown.append(self))
    monkeypatch.setattr(plotting_mod, "clear_output", lambda wait=False: None)
    monkeypatch.setattr("builtins.input", lambda prompt="": "")

    df = pd.DataFrame(
        {
            "line": [1] * 5 + [2] * 5,
            "distance_along_line": list(range(5)) * 2,
            "val": np.arange(10, dtype=float),
        }
    )
    inspect_lines(df, plot_variable="val", interp_on="distance_along_line")
    assert len(shown) == 2
    assert [f.layout.title.text for f in shown] == ["Line 1", "Line 2"]


def test_inspect_lines_accepts_list_of_plot_variables(monkeypatch):
    """A list of plot_variable columns should each be added as a trace on the shown figure."""
    shown = []
    monkeypatch.setattr(go.Figure, "show", lambda self, *a, **k: shown.append(self))
    monkeypatch.setattr(plotting_mod, "clear_output", lambda wait=False: None)
    monkeypatch.setattr("builtins.input", lambda prompt="": "")

    df = pd.DataFrame(
        {
            "line": [1] * 3,
            "distance_along_line": [0, 1, 2],
            "a": [1.0, 2.0, 3.0],
            "b": [4.0, 5.0, 6.0],
        }
    )
    inspect_lines(df, plot_variable=["a", "b"], interp_on="distance_along_line")
    assert len(shown[0].data) == 2
