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
            "xrft",
            "geopandas",
            "pygmt",
            "shapely",
            "tqdm",
            "verde",
            "geographiclib",
            "ipykernel",
            "plotly",
            "seaborn",
            "pyproj",
            "nbformat",
            "iprogress",
            "nomkl",
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
from .filtering import (  # noqa: E402
    filter_grid,
    filter_line,
)
from .interpolating import (  # noqa: E402
    interpolate_missing,
)
from .levelling import (  # noqa: E402
    alternating_iterative_line_levelling,
    calculate_crossover_errors,
    create_intersection_table,
    equivalent_source_levelling,
    inspect_intersections,
    interpolate_intersections,
    # calculate_intersection_weights,
    iterative_line_levelling,
    level_to_grid,
    line_levelling,
    lines_without_intersections,
    plot_levelling_convergence,
    plot_line_and_crosses,
    update_intersections_with_eq_sources,
)
from .nav import (  # noqa: E402
    along_track_distance,
    directional_velocity,
    # eastward_velocity,
    ground_speed,
    # northward_velocity,
    relative_distance,
    track,
    vertical_acceleration,
)
from .plotting import (  # noqa: E402
    inspect_lines,
    # plot_flightlines,
    # plot_flightlines_grids,
    plotly_points,
    plotly_profiles,
)
from .potential_fields import (  # noqa: E402
    eotvos_correction_approx,
    eotvos_correction_full,
    eotvos_correction_glicken,
    eotvos_correction_harlan_track,
    eotvos_correction_harlan_velocity,
    eq_sources_1d,
    upward_continue_by_line,
)
from .processing import (  # noqa: E402
    # detect_outliers,
    split_into_segments,
    unique_line_id,
)
from .reproject import (  # noqa: E402
    reproject,
)
from .resample import (  # noqa: E402
    resample,
    resample_as,
)
from .utils import (  # noqa: E402
    get_min_max,
    normalize_values,
    rmse,
    sample_grid,
)
