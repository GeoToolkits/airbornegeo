import geopandas as gpd
import numpy as np
import pandas as pd
import pygmt

# Any functions or classes defined here which you want to be available when to users
# with `import airbornegeo` should be added to the list in `__init__.py`.


def filter_by_line(
    df: gpd.GeoDataFrame | pd.DataFrame,
    filt_type: str,
    data_column: str,
    filter_by_column: str,
    line_column: str = "line",
    pad_width_percentage: float = 10,
) -> gpd.GeoDataFrame | pd.DataFrame:
    """
    Individually filter each flight line in a dataframe. The data to filter is supplied
    via the data_column argument and the name of the flight lines is supplied via the
    line_column argument. Filtering can be based on distance or time. To choose,
    supply with filter_by_column argument the name of the column containing either the
    distance along the line or the time. The filter width will then be in the units of
    that column. For example, if filter_by_column is "distance_along_line" in meters,
    and filt_type is "g1000+l", then this would give a 1000 m low pass gaussian filter.
    If filter_by_column is "time" in seconds, then the filter width would be in
    1000 seconds. Ends of lines are padded to avoid edge effects, but default this is by
    10% of the total line length.

    ----------
    df : gpd.GeoDataFrame | pd.DataFrame
        _description_
    filt_type : str
        a string with format "<type><width>+h" where type is GMT filter type, width is
        the filter width in same units as distance column, and optional +h switches from
        low-pass to high-pass filter; e.g. "g10+h" is a 10m high-pass Gaussian filter.
    data_column : str
        _description_
    filter_by_column : str, optional
        _description_,
    line_column : str, optional
        _description_, by default "line"
    pad_width_percentage : float, optional
        _description_, by default 10

    Returns
    -------
    pd.Series
        _description_
    """

    df = df.copy()

    for i in df[line_column].unique():
        # subset data from 1 line
        line = df[df[line_column] == i]
        line = line[[filter_by_column, data_column]]

        # get data spacing
        distance = line[filter_by_column].to_numpy()
        data_spacing = np.median(np.diff(distance))

        # pad distance of 10% of line distance
        pad_dist = (distance.max() - distance.min()) * (pad_width_percentage / 100)
        pad_dist = round(pad_dist / data_spacing) * data_spacing

        # get the number of points to pad
        n_pad = int(pad_dist / data_spacing)

        # add pad points to distance values
        lower_pad = np.linspace(
            distance.min() - pad_dist,
            distance.min() - data_spacing,
            n_pad,
        )
        # lower_pad = np.arange(
        #     distance.min() - pad_dist,
        #     distance.min(),
        #     data_spacing,
        # )
        upper_pad = np.linspace(
            distance.max(),
            distance.max() + pad_dist,
            n_pad,
        )
        # upper_pad = np.arange(
        #     distance.max(),
        #     distance.max() + pad_dist,
        #     data_spacing,
        # )

        vals = np.concatenate((lower_pad, upper_pad))
        new_dist = pd.DataFrame({filter_by_column: vals})

        # pad the line in the front and back
        padded = (
            pd.concat(
                [line.reset_index(), new_dist],
            )
            .sort_values(by=filter_by_column)
            .set_index("index")
        ).reset_index()
        padded = padded.rename(columns={"index": "true_index"})

        # get unpadded data
        unpadded_data = line[data_column].to_numpy()

        # pad this with numpy
        padded_data = np.pad(
            unpadded_data,
            pad_width=n_pad,
            mode="reflect",
            # mode="median",
            # stat_length=n_pad/2,
        )

        # add padded data to padded dataframe
        padded[data_column] = padded_data

        # pad has nans, fill with nearest value
        # with high frequency noise, relying on the 1 nearest value can lead to some
        # weirdness at the edges.
        # padded = padded.ffill().bfill().reset_index()
        # padded = padded.rename(columns={"index": "true_index"})

        # filter the padded data
        filtered = pygmt.filter1d(
            padded[[filter_by_column, data_column]],
            end=True,
            time_col=0,
            filter_type=filt_type,
        ).rename(columns={0: filter_by_column, 1: data_column})

        # un-pad the data
        filtered["original_index"] = padded.true_index
        filtered = filtered.set_index("original_index")
        filtered = filtered[filtered.index.isin(line.index)]

        # replace original data with filtered data
        df.loc[df[line_column] == i, data_column] = filtered[data_column]

    return df[data_column]
