import logging

import scooby

from ._version import version as __version__

__all__ = ["Survey", "__version__"]

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
from .crossover_levelling import (  # noqa: E402
    alternating_iterative_line_levelling,
    calculate_intersection_weights,
    crossover_network_levelling,
    crossover_pair_levelling,
    plot_levelling_convergence,
)
from .crossovers import (  # noqa: E402
    add_values_to_intersections,
    calculate_crossover_errors,
    create_intersection_table,
    inspect_intersections,
    interpolate_intersections,
    lines_without_intersections,
    plot_line_and_crosses,
    update_intersections_with_eq_sources,
)
from .eotvos import (  # noqa: E402
    eotvos_correction,
)
from .eq_source_levelling import (  # noqa: E402
    equivalent_source_levelling,
)
from .filtering import (  # noqa: E402
    filter_grid,
    filter_line,
)
from .grid_levelling import (  # noqa: E402
    level_to_grid,
)
from .grid_sample import (  # noqa: E402
    sample_grid,
)
from .interpolating import (  # noqa: E402
    interpolate_missing,
    interpolate_missing_pointwise,
    interpolate_missing_pointwise_with_windows,
    optimal_spline_damping,
)
from .nav import (  # noqa: E402
    along_track_distance,
    directional_velocity,
    ground_speed,
    relative_distance,
    track,
    vertical_acceleration,
)
from .plotting import (  # noqa: E402
    add_scalebar,
    choose_colormap,
    inspect_lines,
    nice_scalebar_width,
    plot_profiles,
    plotly_points,
    plotly_profiles,
)
from .potential_fields import (  # noqa: E402
    eq_sources_1d,
    igrf,
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
from .survey import (  # noqa: E402
    Survey,
)
from .trend import (  # noqa: E402
    trend,
)
from .utils import (  # noqa: E402
    get_min_max,
    largest_line_dimensions,
    median_line_spacing,
    normalize_values,
    rmse,
)
