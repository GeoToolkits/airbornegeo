import logging

import scooby

from ._version import version as __version__

__all__ = ["__version__"]

logger = logging.getLogger(__name__)


class Report(scooby.Report):  # type: ignore[misc] # pylint: disable=missing-class-docstring
    def __init__(self, additional=None, ncol=3, text_width=80, sort=False):  # type: ignore[no-untyped-def]
        """Initiate a scooby.Report instance."""

        # Mandatory packages.
        core = [
            "numpy",
            "scipy",
            "matplotlib",
            "pandas",
            "scikit-learn",
            "ipython",
            "harmonica",
            "geopandas",
            "pygmt",
            "shapely",
            "tqdm",
            "verde",
            "ipykernel",
            "plotly",
            "seaborn",
        ]

        # Optional packages.
        optional = []

        scooby.Report.__init__(
            self,
            additional=additional,
            core=core,
            optional=optional,
            ncol=ncol,
            text_width=text_width,
            sort=sort,
        )


from .filtering import (  # noqa: E402
    filter_by_line,
)
from .levelling import (  # noqa: E402
    add_intersections,
    calculate_intersection_weights,
    calculate_misties,
    create_intersection_table,
    extend_line,
    get_line_intersections,
    get_line_tie_intersections,
    inspect_intersections,
    interp,
    interp_all_lines,
    interp_single_col,
    interp_windows,
    interp_windows_single_col,
    iterative_levelling,
    iterative_levelling_alternate,
    level_lines,
    level_survey_lines_to_grid,
    plot_levelling_convergence,
    plot_line_and_crosses,
    scipy_interp,
    skl_predict_trend,
    update_intersections_with_eq_sources,
)
from .plotting import (  # noqa: E402
    inspect_lines,
    plot_flightlines,
    plot_flightlines_grids,
    plotly_points,
    plotly_profiles,
)
from .processing import (  # noqa: E402
    detect_outliers,
    distance_along_flight,
    distance_along_line,
    eastward_velocity,
    eotvos_correction,
    eotvos_correction_full,
    eq_sources_by_line,
    line_bearing,
    northward_velocity,
    reduce_by_line,
    relative_distance,
    unique_line_id,
    upward_continue_by_line,
    vertical_acceleration,
)
from .utils import (  # noqa: E402
    get_min_max,
    normalize_values,
    rmse,
)
