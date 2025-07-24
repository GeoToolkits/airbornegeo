import logging

from ._version import version as __version__

__all__ = ["__version__"]


logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

logger.addHandler(logging.NullHandler())

# from .module1 import (
#     # List of functions and classes to be imported when using `import samplepackagename`
#     example_function,
# )
