.. _api:

API Reference
=============


.. automodule:: airbornegeo

.. currentmodule:: airbornegeo


Handling survey data
--------------------
Functions for working with entire surveys of data.

.. autosummary::
    :toctree: generated/

    along_track_distance
    relative_distance
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

    calculate_misties
    update_intersections_with_eq_sources
