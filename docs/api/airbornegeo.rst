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


Processing
----------
Functions for processing airborne geophysical survey data.

.. autosummary::
    :toctree: generated/

    filter1d
    vertical_acceleration
    upward_continue_by_line


Levelling
---------
Functions for levelling airborne geophysical survey data.

.. autosummary::
    :toctree: generated/

    eq_sources_1d
    level_to_grid
    sample_grid
    create_intersection_table
    interpolate_intersections
    inspect_intersections
    lines_without_intersections
    calculate_misties
