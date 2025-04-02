"""
Copyright (c) 2025 Matt Tankersley. All rights reserved.

airbornegeo: Tools for processing airborne geophysics data.
"""

from __future__ import annotations

from ._version import version as __version__

__all__ = ["__version__"]

import logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

logger.addHandler(logging.NullHandler())
