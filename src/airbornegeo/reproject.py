import pandas as pd
from numpy.typing import NDArray
from pyproj import Transformer


def reproject(
    x: NDArray | pd.Series,
    y: NDArray | pd.Series,
    input_crs: str,
    output_crs: str,
) -> tuple[NDArray, NDArray]:
    """
    Convert coordinates from input CRS to output CRS.

    Parameters
    ----------
    x : np.ndarray | pd.Series
        coordinates of either easting or longitude
    y : np.ndarray | pd.Series,
        coordinates of either northing or latitude
    input_crs : str
        input CRS in EPSG format, e.g. "epsg:4326" which can used for geographic data in
        degrees
    output_crs : str
        output CRS in EPSG format, e.g. "epsg:3413"

    Returns
    -------
    tuple[NDArray, NDArray]
        a tuple of two arrays in the order earthing, northing, or longitude, latitude.
    """

    # make crs lowercase
    input_crs = input_crs.lower()
    output_crs = output_crs.lower()

    transformer = Transformer.from_crs(
        input_crs,
        output_crs,
        always_xy=True,
    )

    return transformer.transform(x, y)
