"""
This module provides functionality to generate a scenario DataFrame by combining population data with geographical data.

Functions:

    get_scenario(params, verbose: bool = False) -> pd.DataFrame:

        Reads population data from a CSV file and geographical data from a shapefile,
        processes and merges them, and returns a GeoDataFrame with additional latitude
        and longitude columns.

        Parameters:

            params: An object containing the file paths for the population and shape files.
            verbose (bool): If True, enables verbose output. Default is False.

        Returns:

            pd.DataFrame: A GeoDataFrame containing the merged population and geographical data.
"""

import numpy as np
import pandas as pd

from laser_measles.demographics import shapefiles


def get_scenario(params, verbose: bool = False) -> pd.DataFrame:
    """
    Generates a scenario DataFrame by merging population data with geographical shape data.

    Args:

        params: An object containing the following attributes:

            - population_file (str): Path to the CSV file containing population data.
            - shape_file (str): Path to the shapefile containing geographical data.

        verbose (bool, optional): If True, enables verbose output. Defaults to False.

    Returns:

        pd.DataFrame: A DataFrame containing the merged population and geographical data with additional latitude and longitude columns.
    """

    pops = pd.read_csv(params.population_file)
    pops.rename(columns={"county": "name"}, inplace=True)
    pops.set_index("name", inplace=True)
    gpdf = shapefiles.get_dataframe(params.shape_file).to_pandas()
    gpdf.drop(
        columns=[
            "EDIT_DATE",
            "EDIT_STATU",
            "EDIT_WHO",
            "GLOBALID",
            "JURISDICT_",
            "JURISDIC_1",
            "JURISDIC_3",
            "JURISDIC_4",
            "JURISDIC_5",
            "JURISDIC_6",
            "OBJECTID",
        ],
        inplace=True,
    )
    gpdf.rename(columns={"JURISDIC_2": "name"}, inplace=True)
    gpdf.set_index("name", inplace=True)

    gpdf = gpdf.join(pops)

    # Convert centroids from meters to degrees using NumPy
    centroids = np.array([np.array(s.points).mean(axis=0) for s in gpdf["shape"]])
    x_meters = centroids[:, 0]
    y_meters = centroids[:, 1]

    # Convert to degrees (approximate conversion)
    # This assumes the input is in EPSG:3857 (Web Mercator)
    longitude = x_meters / 20037508.34 * 180
    latitude = np.arctan(np.sinh(y_meters * np.pi / 20037508.34)) * 180 / np.pi

    gpdf["latitude"] = latitude
    gpdf["longitude"] = longitude
    # gpdf.to_crs(epsg=4326, inplace=True)
    gpdf.reset_index(inplace=True)  # return "name" to just a column

    return gpdf
