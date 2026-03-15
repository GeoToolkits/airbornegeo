.. _api:

API Reference
=============


.. automodule:: airbornegeo

.. currentmodule:: airbornegeo


Geospatial data operations
--------------------------
Functions for performing geospatial operations.

.. autosummary::
    :toctree: generated/

    reproject
    block_reduce
    along_track_distance
    relative_distance


Organizing survey data
----------------------
Functions for working with and organize survey flights.

.. autosummary::
    :toctree: generated/

    split_into_segments
    unique_line_id


Quality Control (QC)
--------------------
Functions for automated and manual quality control of airborne geophysical survey data.

.. autosummary::
    :toctree: generated/

    inspect_lines
    detect_outliers

Processing
----------
Functions for processing airborne geophysical survey data.

.. autosummary::
    :toctree: generated/

    filter1d
    upward_continue_by_line
    sample_grid
    bearing


Levelling
---------
Functions for levelling airborne geophysical survey data.

.. autosummary::
    :toctree: generated/

    level_to_grid
    create_intersection_table
    interpolate_intersections
    inspect_intersections
    plot_line_and_crosses
    lines_without_intersections
    calculate_crossover_errors
    update_intersections_with_eq_sources
    line_levelling
    iterative_line_levelling
    alternating_iterative_line_levelling
    plot_levelling_convergence


Potential-fields related functions
----------------------------------
Functions related specifically to working with potential-fields data

.. autosummary::
    :toctree: generated/

    eq_sources_1d
    upward_continue_by_line

Plotting functions
------------------

.. autosummary::
    :toctree: generated/

    plotly_points
    plotly_profiles
    inspect_lines
    plot_flightlines
    plot_flightlines_grids
