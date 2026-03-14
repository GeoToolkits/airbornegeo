import pandas as pd
from pyproj import Transformer
from numpy.typing import NDArray

def reproject(
    data: pd.DataFrame,
    input_crs: str,
    output_crs: str,
    input_coord_names: tuple[str, str],
) -> tuple[NDArray, NDArray]:
    """
    Convert coordinates from input CRS to output CRS. Coordinates can be supplied as a
    dataframe with coordinate columns set by input_coord_names, or as a tuple of a list
    of x coordinates and a list of y coordinates.

    Parameters
    ----------
    data : pandas.DataFrame
        input dataframe with coordinate columns specified by input_coord_names
    input_crs : str
        input CRS in EPSG format, e.g. "epsg:4326" which can used for geographic data in 
        degrees
    output_crs : str
        output CRS in EPSG format, e.g. "epsg:3413"
    input_coord_names : tuple
        columns names for input coordinate columns in order x/lon then y/lat

    Returns
    -------
    tuple[NDArray, NDArray]
        a tuple of two arrays which can be added to the dataframe.
    """

    # make crs lowercase
    input_crs = input_crs.lower()
    output_crs = output_crs.lower()

    transformer = Transformer.from_crs(
        input_crs,
        output_crs,
        always_xy=True,
    )

    data = data.copy()

    assert all(col in data.columns for col in input_coord_names), "supplied coordinate names not in columns of dataframe"

    return transformer.transform(
        data[input_coord_names[0]].to_numpy(),
        data[input_coord_names[1]].to_numpy(),
    )
