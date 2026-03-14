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


from .block_reduce import (  # noqa: E402
    block_reduce,
)
from .distances import (  # noqa: E402
    along_track_distance,
    relative_distance,
)
from .filtering import (  # noqa: E402
    filter1d,
)
from .levelling import (  # noqa: E402
    calculate_misties,
    # calculate_intersection_weights,
    # level_survey_lines_to_grid,
    # iterative_levelling,
    # iterative_levelling_alternate,
    # level_lines,
    # plot_levelling_convergence,
    create_intersection_table,
    inspect_intersections,
    interpolate_intersections,
    lines_without_intersections,
    plot_line_and_crosses,
)
from .plotting import (  # noqa: E402
    inspect_lines,
    plot_flightlines,
    plot_flightlines_grids,
    plotly_points,
    plotly_profiles,
)
from .potential_fields import (  # noqa: E402
    eotvos_correction,
    eotvos_correction_full,
    eq_sources_by_line,
    update_intersections_with_eq_sources,
    upward_continue_by_line,
)
from .processing import (  # noqa: E402
    bearing,
    detect_outliers,
    eastward_velocity,
    line_bearing,
    northward_velocity,
    split_into_segments,
    unique_line_id,
    vertical_acceleration,
)
from .reproject import (  # noqa: E402
    reproject,
)
from .utils import (  # noqa: E402
    get_min_max,
    normalize_values,
    rmse,
)
