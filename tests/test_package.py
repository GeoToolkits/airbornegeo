import importlib.metadata

import airbornegeo


def test_version():
    """The package's __version__ attribute should match its installed metadata version."""
    assert importlib.metadata.version("airbornegeo") == airbornegeo.__version__
