import importlib.metadata

import airbornegeo


def test_version():
    assert importlib.metadata.version("airbornegeo") == airbornegeo.__version__
