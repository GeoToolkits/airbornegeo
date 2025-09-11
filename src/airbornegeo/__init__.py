# ruff: noqa: F401, E402


import logging

from ._version import version as __version__

__all__ = ["__version__"]


logger = logging.getLogger(__name__)

from .filtering import (
    # List of functions and classes to be imported when using `import airbornegeo`
    block_reduce_by_line,
    filter_by_distance,
)
from .levelling import (
    add_intersections,
    calculate_intersection_weights,
    calculate_misties,
    create_intersection_table,
    # List of functions and classes to be imported when using `import airbornegeo`
    detect_outliers,
    distance_along_flight,
    distance_along_line,
    extend_line,
    get_line_intersections,
    get_line_tie_intersections,
    interp1d,
    interp1d_all_lines,
    interp1d_single_col,
    interp1d_windows,
    interp1d_windows_single_col,
    iterative_levelling_alternate,
    iterative_line_levelling,
    level_lines,
    level_survey_lines_to_grid,
    normalize_values,
    plot_flightlines,
    plot_levelling_convergence,
    plot_line_and_crosses,
    plotly_points,
    plotly_profiles,
    scipy_interp1d,
    skl_predict_trend,
    verde_interp1d,
    verde_predict_trend,
)
