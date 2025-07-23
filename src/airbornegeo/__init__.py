"""
Copyright (c) 2025 Matt Tankersley. All rights reserved.

airbornegeo: Tools for processing airborne geophysics data.
"""

from __future__ import annotations

from ._version import version as __version__

__all__ = ["__version__"]

import logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

logger.addHandler(logging.NullHandler())

from ._levelling import (
    detect_outliers,
    normalize_values,
    calculate_intersection_weights,
    plot_levelling_convergence,
    distance_along_line,
    distance_along_flight,
    create_intersection_table,
    add_intersections,
    extend_line,
    get_line_tie_intersections,
    get_line_intersections,
    scipy_interp1d,
    verde_interp1d,
    interp1d_single_col,
    interp1d_windows_single_col,
    interp1d_windows,
    interp1d,
    interp1d_all_lines,
    calculate_misties,
    verde_predict_trend,
    skl_predict_trend,
    level_survey_lines_to_grid,
    level_lines,
    iterative_line_levelling,
    iterative_levelling_alternate,
    plotly_points,
    plotly_profiles,
    plot_line_and_crosses,
    plot_flightlines,
)

from ._filtering import (
    block_reduce_by_line,
    filter_by_distance,
)