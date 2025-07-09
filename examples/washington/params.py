"""
This module defines the function `get_parameters` which initializes and returns a `PropertySet` object containing
parameters for a measles simulation model. The parameters are divided into several categories: meta parameters,
measles-specific parameters, network parameters, and routine immunization (RI) parameters. The function also allows
for the optional loading of parameters from a JSON file and the overriding of parameters via command line arguments.

Functions:

    get_parameters(kwargs) -> PropertySet:

        Initializes and returns a `PropertySet` object with default parameters,
        optionally overridden by parameters from a JSON file and/or command line arguments.
"""

import re
from pathlib import Path

import numpy as np
from laser_core.propertyset import PropertySet


def get_parameters(kwargs) -> PropertySet:
    """
    Generate a set of parameters for the generic measles simulation.

    This function initializes default parameters for the simulation, including meta parameters,
    measles-specific parameters, network parameters, and routine immunization parameters. It then
    allows for these parameters to be overwritten by values from a JSON file and/or command line
    arguments.

    Args:

        kwargs (dict): A dictionary of keyword arguments that can include:

            - "params" (str): Path to a JSON file containing parameter values to overwrite defaults.
            - "param" (list): List of strings in the format "key=value" to overwrite specific parameters.

    Returns:

        PropertySet: A PropertySet object containing all the parameters for the simulation.
    """

    meta_params = PropertySet(
        {
            "nticks": 365,
            "verbose": False,
            "cbr": np.float32(13.7),
            "population_file": Path(__file__).parent.absolute() / "WA County Populations-2000.csv",
            "shape_file": Path(__file__).parent.absolute() / "WA_County_Boundaries" / "WA_County_Boundaries.shp",
            "pyramid_file": Path(__file__).parent.absolute() / "USA pyramid-2000.csv",
            "mortality_file": Path(__file__).parent.absolute() / "USA mortality-2000.csv",
        }
    )

    measles_params = PropertySet(
        {
            "exp_scale": np.float32(1.0),
            "exp_shape": np.float32(3.5),
            "inf_mean": np.float32(18.0),
            "inf_std": np.float32(2.0),
            "r_naught": np.float32(15.0),
            "seasonality_factor": np.float32(0.125),
            "seasonality_phase": np.float32(182),
        }
    )

    network_params = PropertySet(
        {
            "k": np.float32(50.0),
            "a": np.float32(1.0),
            "b": np.float32(0.0),
            "c": np.float32(1.0),
            "max_frac": np.float32(0.05),
        }
    )

    ri_params = PropertySet(
        {
            "ri_coverage": 0.7,
            "probability_mcv1_take": 0.85,
            "probability_mcv2_take": 0.95,
            "mcv1_start": int(8.5 * 365 / 12),  # 8.5 months
            "mcv1_end": int(9.5 * 365 / 12),  # 9.5 months
            "mcv2_start": int(14.5 * 365 / 12),  # 14.5 months
            "mcv2_end": int(15.5 * 365 / 12),  # 15.5 months
        }
    )

    params = PropertySet(meta_params, measles_params, network_params, ri_params)

    # Overwrite any default parameters with those from a JSON file (optional)
    if kwargs.get("params") is not None:
        paramfile = Path(kwargs.get("params"))
        params += PropertySet.load(paramfile)
        print(f"Loaded parameters from `{paramfile}`…")

    # Finally, overwrite any parameters with those from the command line (optional)
    for key, value in kwargs.items():
        if key == "params":
            continue  # handled above

        if key != "param":
            print(f"Using `{value}` for parameter `{key}` from the command line…")
            params[key] = value
        else:  # arbitrary param:value pairs from the command line
            for kvp in kwargs["param"]:
                key, value = re.split("[=:]+", kvp)
                if key not in params:
                    print(f"Unknown parameter `{key}` ({value=}). Skipping…")
                    continue
                value = type(params[key])(value)  # Cast the value to the same type as the existing parameter
                print(f"Using `{value}` for parameter `{key}` from the command line…")
                params[key] = value

    params.beta = np.float32(np.float32(params.r_naught) / np.float32(params.inf_mean))

    return params
