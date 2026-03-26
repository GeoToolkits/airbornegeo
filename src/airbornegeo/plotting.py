import typing

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import pygmt
import seaborn as sns
import verde as vd
from IPython.display import clear_output

import airbornegeo

sns.set_theme()


def align_yaxis(
    ax1: mpl.axes.Axes,
    v1: float,
    ax2: mpl.axes.Axes,
    v2: float,
) -> None:
    """
    adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1.
    From https://stackoverflow.com/a/10482477/18686384
    """
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1 - y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny + dy, maxy + dy)


def plot_eqs_levelling_convergence(
    rms_values: list[float],
    delta_rms_values: list[float],
    rms_tolerance: float | None = None,
    rms_percent_change_tolerance: float | None = None,
) -> None:
    """
    plot a graph of RMS and delta RMS vs iteration number.
    """

    # create figure instance
    _fig, ax1 = plt.subplots(figsize=(5, 3.5))

    # make second y axis for delta RMS
    ax2 = ax1.twinx()

    # plot RMS convergence
    ax1.plot(
        range(1, len(rms_values) + 1),
        rms_values,
        "b-",
    )

    # plot delta RMS convergence
    ax2.plot(
        range(1, len(rms_values) + 1),
        delta_rms_values,
        "g-",
    )

    # set axis labels, ticks and gridlines
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("levelling correction RMS", color="b")
    ax1.tick_params(axis="y", colors="b", which="both")
    ax2.set_ylabel("RMS % change", color="g")
    ax2.tick_params(axis="y", colors="g", which="both")
    ax2.grid(False)

    ax1.set_ylim(min(rms_values), max(rms_values))
    ax2.set_ylim(
        np.nanmin(np.isfinite(delta_rms_values)),
        np.nanmax(np.isfinite(delta_rms_values)),
    )

    # set x axis to integer values
    ax1.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

    if (rms_tolerance is not None) and (rms_percent_change_tolerance is not None):
        # make both y axes align at tolerance levels
        align_yaxis(ax1, rms_tolerance, ax2, rms_percent_change_tolerance)
        # plot horizontal line of tolerances
        ax2.axhline(
            y=rms_percent_change_tolerance,
            linewidth=1,
            color="r",
            linestyle="dashed",
            label="tolerances",
        )
    elif rms_percent_change_tolerance is not None:
        ax2.axhline(
            y=rms_percent_change_tolerance,
            linewidth=1,
            color="r",
            linestyle="dashed",
            label="RMS percent change tolerance",
        )
    elif rms_tolerance is not None:
        ax1.axhline(
            y=rms_tolerance,
            linewidth=1,
            color="r",
            linestyle="dashed",
            label="RMS tolerance",
        )

    if (rms_tolerance is None) and (rms_percent_change_tolerance is None):
        pass
    else:
        # ask matplotlib for the plotted objects and their labels
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc="upper right")

    plt.title("Equivalent source iterative levelling")
    plt.tight_layout()
    plt.show()


def inspect_lines(
    df: pd.DataFrame | gpd.GeoDataFrame,
    *,
    plot_variable: str | list[str],
    interp_on: str = "distance_along_line",
) -> None:
    if isinstance(plot_variable, str):
        plot_variable = [plot_variable]

    lines = df.groupby("line")
    for line, line_df in lines:
        fig = plotly_profiles(
            line_df,
            x=interp_on,
            y=plot_variable,
            y_axes=[str(i + 1) for i in range(len(plot_variable))],
        )
        fig.update_layout(title_text=f"Line {line}")
        fig.show()
        input("Press key to continue...")
        clear_output(wait=True)


def plot_flightlines(
    fig: pygmt.Figure,
    df: pd.DataFrame,
    *,
    plot_labels: bool = True,
    plot_lines: bool = True,
    line_color: str = "gray",
    line_style: str = "p.5p",
    font: str = "8p,black",
) -> None:
    # group lines by their line number
    lines = [v for _, v in df.groupby("line")]

    # plot lines
    if plot_lines is True:
        for i in list(range(len(lines))):
            fig.plot(
                x=lines[i].easting,
                y=lines[i].northing,
                style=line_style,
                fill=line_color,
            )

    # plot labels
    if plot_labels is True:
        for i, line in enumerate(lines):
            # switch label locations for every other line
            if (i % 2) == 0:
                # plot label at max value of x-coord
                x_or_y = "easting"

                # plot label
                fig.text(
                    x=line.easting.loc[line[x_or_y].idxmax()],
                    y=line.northing.loc[line[x_or_y].idxmax()],
                    text=str(line.line.iloc[0]),
                    justify="CM",
                    font=font,
                    fill="white",
                    angle=line.bearing.iloc[0],
                )
            else:
                # plot label at max value of y-coord
                x_or_y = "northing"

                # plot label
                fig.text(
                    x=line.easting.loc[line[x_or_y].idxmin()],
                    y=line.northing.loc[line[x_or_y].idxmin()],
                    text=str(line.line.iloc[0]),
                    justify="CM",
                    font=font,
                    fill="white",
                    angle=line.bearing.iloc[0],
                )


def plot_flightlines_grids(
    fig: pygmt.Figure,
    df: pd.DataFrame,
    *,
    direction: str = "EW",
    plot_labels: bool = True,
    plot_lines: bool = True,
    font: str = "5p,black",
    fill: str = "gray",
    style: str = "p.5p",
) -> None:
    # group lines by their line number
    lines = [v for _, v in df.groupby("line")]

    # plot lines
    if plot_lines is True:
        for i in list(range(len(lines))):
            fig.plot(
                x=lines[i].easting,
                y=lines[i].northing,
                style=style,
                fill=fill,
            )

    # plot labels
    if plot_labels is True:
        for i in list(range(len(lines))):
            # switch label locations for every other line
            if (i % 2) == 0:
                if direction == "EW":
                    offset = "0.25c/0c"
                    # plot label at max value of x-coord
                    x_or_y = "easting"
                    # angle of label
                    angle = 0
                elif direction == "NS":
                    offset = "0c/0.25c"
                    # plot label at max value of y-coord
                    x_or_y = "northing"
                    # angle of label
                    angle = 90
                else:
                    msg = "invalid direction string"
                    raise ValueError(msg)
                # plot label
                fig.text(
                    x=lines[i].easting.loc[lines[i][x_or_y].idxmax()],
                    y=lines[i].northing.loc[lines[i][x_or_y].idxmax()],
                    text=str(int(lines[i].line.iloc[0])),
                    justify="CM",
                    font=font,
                    fill="white",
                    offset=offset,
                    angle=angle,
                )
            else:
                if direction == "EW":
                    offset = "-0.25c/0c"
                    # plot label at max value of x-coord
                    x_or_y = "easting"
                    # angle of label
                    angle = 0
                elif direction == "NS":
                    offset = "0c/-0.25c"
                    # plot label at max value of y-coord
                    x_or_y = "northing"
                    # angle of label
                    angle = 90
                else:
                    msg = "invalid direction string"
                    raise ValueError(msg)
                # plot label
                fig.text(
                    x=lines[i].easting.loc[lines[i][x_or_y].idxmin()],
                    y=lines[i].northing.loc[lines[i][x_or_y].idxmin()],
                    text=str(int(lines[i].line.iloc[0])),
                    justify="CM",
                    font=font,
                    fill="white",
                    offset=offset,
                    angle=angle,
                )


def plotly_points(
    df: pd.DataFrame | gpd.GeoDataFrame,
    *,
    color_col: str,
    coord_names: tuple[str, str] | None = None,
    hover_cols: list[str] | None = None,
    size: int = 4,
    edge_width: int | None = None,
    edge_color: str = "black",
    cmap: str | None = None,
    cmap_middle: float | None = None,
    cmap_lims: tuple[float, float] | None = None,
    robust: bool = True,
    absolute: bool = False,
    theme: str | None = None,
    title: str | None = None,
) -> None:
    """
    Create a scatterplot of spatial data. By default, coordinates are extracted from
    geopandas geometry column, or from user specified columns given by 'coord_names'.
    """
    data = df[df[color_col].notna()].copy()
    assert len(data) > 0, "supplied column of data has no non nan values!"
    if coord_names is None:
        try:
            x = data.geometry.x
            y = data.geometry.y
        except AttributeError:
            try:
                x = data["easting"]
                y = data["northing"]
            except KeyError:
                try:
                    x = data["x"]
                    y = data["y"]
                except AttributeError:
                    pass
        coord_names = (x, y)

    # either
    if cmap_lims is None and color_col is not None:
        vmin, vmax = airbornegeo.get_min_max(
            data[color_col],
            robust=robust,
            absolute=absolute,
        )
        cmap_lims = (vmin, vmax)
    else:
        vmin, vmax = cmap_lims

    if cmap is None:
        if (cmap_lims[0] < 0) and (cmap_lims[1] > 0):  # pylint: disable=R1716
            cmap = "balance"
            cmap_middle = 0
        else:
            cmap = None
            cmap_middle = None
    else:
        cmap_middle = None

    if cmap_middle == 0:
        max_abs = vd.maxabs((vmin, vmax))
        cmap_lims = (-max_abs, max_abs)

    fig = px.scatter(
        data,
        x=coord_names[0],
        y=coord_names[1],
        color=data[color_col],
        color_continuous_scale=cmap,
        color_continuous_midpoint=cmap_middle,
        range_color=cmap_lims,
        hover_data=hover_cols,
        template=theme,
    )
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )

    fig.update_layout(
        title_text=title,
        autosize=False,
        width=800,
        height=800,
    )

    fig.update_traces(
        marker={"size": size, "line": {"color": edge_color, "width": edge_width}}
    )

    fig.show()


def plotly_profiles(
    data: pd.DataFrame,
    *,
    y: list[str] | str,
    x: str = "dist_along_line",
    y_axes: list[str] | None = None,
    x_lims: tuple[float, float] | None = None,
    y_lims: tuple[float, float] | None = None,
    title: str | None = None,
    modes: typing.Any = None,
    marker_sizes: typing.Any = None,
    marker_symbols: typing.Any = None,
) -> go.Figure:
    """
    plot data profiles with plotly
    currently only allows 3 separate y axes, set with "y_axes", starting with 1
    """
    df = data.copy()

    # turn y column name into list
    if isinstance(y, str):
        y = [y]

    # list of y axes to use, if none, all will be same
    y_axes = ["" for _ in y] if y_axes is None else [str(x) for x in y_axes]
    assert "0" not in y_axes, "No '0' or 0 allowed, axes start with 1"
    # convert y axes to plotly expected format: "y", "y2", "y3" ...
    y_axes = [s.replace("1", "") for s in y_axes]
    y_axes = [f"y{x}" for x in y_axes]

    if y_lims is not None:
        if isinstance(y_lims[0], list | tuple):  # type: ignore [unreachable]
            assert len(y_lims) == len(y), "y_lims must be same length as y"  # type: ignore [unreachable]
        elif isinstance(y_lims[0], int | float):
            y_lims = tuple(y_lims for _ in y)  # type: ignore [assignment] # pylint: disable=R1728

    # set plotting mode
    if modes is None:
        modes = ["markers" for _ in y]

    # set marker properties
    if marker_sizes is None:
        marker_sizes = [2 for _ in y]
    if marker_symbols is None:
        marker_symbols = ["circle" for _ in y]

    fig = go.Figure()

    # iterate through data columns
    for i, col in enumerate(y):
        fig.add_trace(
            go.Scatter(
                mode=modes[i],
                x=df[x],
                y=df[col],
                name=col,
                marker_size=marker_sizes[i],
                marker_symbol=marker_symbols[i],
                marker_color=plotly.colors.DEFAULT_PLOTLY_COLORS[i],
                yaxis=y_axes[i],
            )
        )

    unique_axes = len(pd.Series(y_axes).unique())
    x_domain = [0.0, 1.0]
    if unique_axes >= 1:
        y_axes_args = {"yaxis": {"title": y[y_axes.index("y")]}}
    if unique_axes >= 2:
        y_axes_args["yaxis2"] = {  # pylint: disable=E0606
            "title": y[y_axes.index("y2")],
            "overlaying": "y",
            "side": "right",
        }  # pylint: disable=E0606
    if unique_axes >= 3:
        y_axes_args["yaxis3"] = {
            "title": y[y_axes.index("y3")],
            "anchor": "free",
            "overlaying": "y",
        }
        x_domain = [0.15, 1.0]
    else:
        pass

    if x_lims is not None:
        fig.update_layout(xaxis={"range": x_lims})
    for i, _col in enumerate(y):
        if y_lims is not None:
            if y_axes[i] == "y":
                y_axes_args["yaxis"]["range"] = y_lims[i]  # type: ignore [assignment]
            elif y_axes[i] == "y2":
                y_axes_args["yaxis2"]["range"] = y_lims[i]  # type: ignore [assignment]
            elif y_axes[i] == "y3":
                y_axes_args["yaxis3"]["range"] = y_lims[i]  # type: ignore [assignment]
            else:
                y_axes_args["yaxis3"]["range"] = y_lims[i]  # type: ignore [assignment]

    fig.update_layout(
        title_text=title,
        xaxis={
            "title": x,
            "domain": x_domain,
        },
        **y_axes_args,
    )

    return fig
