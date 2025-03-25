from __future__ import annotations

import importlib.metadata

import airbornegeo as m


def test_version():
    assert importlib.metadata.version("airbornegeo") == m.__version__
