.. _api:

API Reference
=============


.. automodule:: airbornegeo

.. currentmodule:: airbornegeo


Trajectories
--------------------------
Functions for calculated trajectory or navigation related fields

.. autosummary::
    :toctree: generated/

    along_track_distance
    relative_distance
    ground_speed
    vertical_acceleration
    track


Geospatial data operations
--------------------------
Functions for performing geospatial operations.

.. autosummary::
    :toctree: generated/

    reproject
    block_reduce
    filter_line
    filter_grid
    sample_grid


Organizing survey data
----------------------
Functions for working with and organizing survey data.

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
    equivalent_source_levelling
    plot_levelling_convergence


Potential-fields related functions
----------------------------------
Functions related specifically to working with potential-fields data

.. autosummary::
    :toctree: generated/

    eq_sources_1d
    upward_continue_by_line
    eotvos_correction_glicken
    eotvos_correction_harlan
    vertical_acceleration

Plotting functions
------------------

.. autosummary::
    :toctree: generated/

    plotly_points
    plotly_profiles
    inspect_lines
