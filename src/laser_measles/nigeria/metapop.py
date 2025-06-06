"""
This module provides functionality to load and process population and location data from a shapefile.

Functions:

    get_scenario(params, verbose: bool = False) -> pd.DataFrame:

        Loads population and location data from a specified shapefile and returns it as a GeoDataFrame.

        Parameters:

            params: An object containing the path to the shapefile.
            verbose: A boolean flag to enable verbose output.

        Returns:

            A GeoDataFrame containing the population and location data.

Deprecated Functions:

    initialize_patches(verbose: bool = False) -> tuple:

        This function was used to process admin2 areas in Nigeria and load population and location data for Northern Nigeria.

        Parameters:

            verbose: A boolean flag to enable verbose output.

        Returns:

            A tuple containing arrays of names, populations, latitudes, and longitudes.
"""

import click
import pandas as pd

from laser_measles.demographics import shapefiles


def get_scenario(params, verbose: bool = False) -> pd.DataFrame:
    """
    Load population and location data from a shapefile and return it as a GeoDataFrame.

    Parameters:

        params (object): An object containing the parameters for the scenario, including the path to the shapefile.
        verbose (bool, optional): If True, prints detailed information about the loading process. Defaults to False.

    Returns:

        pd.DataFrame: A dataframe containing the population and location data from the shapefile.
    """

    # We need some patches with population data ...
    # names, populations, latitudes, longitudes = initialize_patches(verbose)
    if verbose:
        click.echo(f"Loading population and location data from '{params.shape_file}'…")
    gpdf = shapefiles.get_dataframe(params.shape_file).to_pandas()
    if verbose:
        click.echo(f"Loaded {len(gpdf):,} patches (total population {gpdf.population.sum():,}) from '{params.shape_file}'.")

    return gpdf
