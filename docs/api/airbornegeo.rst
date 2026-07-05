.. _api:

API Reference
=============


.. automodule:: airbornegeo

.. currentmodule:: airbornegeo


Trajectories
--------------------------
Functions for calculating trajectory or navigation related fields

.. autosummary::
    :toctree: generated/

    along_track_distance
    relative_distance
    ground_speed
    directional_velocity
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
    resample
    resample_as
    interpolate_missing
    interpolate_missing_pointwise
    interpolate_missing_pointwise_with_windows
    trend

Organizing survey data
----------------------
Functions for working with and organizing survey data.

.. autosummary::
    :toctree: generated/

    split_into_segments
    unique_line_id
    median_line_spacing


Quality Control (QC)
--------------------
Functions for automated and manual quality control of airborne geophysical survey data.

.. autosummary::
    :toctree: generated/

    inspect_lines


Cross-over analysis
---------
Functions for finding and examining cross-overs

.. autosummary::
    :toctree: generated/

    create_intersection_table
    interpolate_intersections
    add_values_to_intersections
    inspect_intersections
    plot_line_and_crosses
    lines_without_intersections
    calculate_crossover_errors
    update_intersections_with_eq_sources
    calculate_intersection_weights


Levelling
---------
Functions for levelling airborne geophysical survey data.

.. autosummary::
    :toctree: generated/

    level_to_grid
    crossover_pair_levelling
    crossover_network_levelling
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
    eotvos_correction_approx
    eotvos_correction_glicken
    eotvos_correction_harlan
    eotvos_correction_full
    vertical_acceleration
    igrf


Plotting functions
------------------

.. autosummary::
    :toctree: generated/

    plotly_points
    plotly_profiles
    plot_profiles
    inspect_lines


Utilities
---------

.. autosummary::
    :toctree: generated/

    get_min_max
    rmse
    normalize_values
