.. _api:

API Reference
=============


.. automodule:: airbornegeo

.. currentmodule:: airbornegeo


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

    vertical_acceleration
    relative_distance
    distance_along_flight
    upward_continue_by_line


Levelling
---------
Functions for levelling airborne geophysical survey data.

.. autosummary::
    :toctree: generated/

    add_intersections
    calculate_intersection_weights
    calculate_misties
    update_intersections_with_eq_sources
