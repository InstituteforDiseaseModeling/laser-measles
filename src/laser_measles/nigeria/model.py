r"""
Northern Nigeria Measles Model

This script runs a simulation of measles dynamics in Northern Nigeria using a
compartmental model. The model includes various components such as births,
non-disease deaths, susceptibility, maternal antibodies, routine immunization,
infection, incubation, and transmission.

The script uses the Click library to handle command-line options for configuring
the simulation parameters, including the number of ticks to run, random seed,
verbosity, visualization options, output file, and parameter overrides.

Functions:

    run(\*\*kwargs): Main function to run the simulation with specified parameters.

Command-line Options:

    --nticks (int): Number of ticks to run the simulation (default: 365).
    --seed (int): Random seed for reproducibility (default: 20241107).
    --verbose (bool): Print verbose output if set.
    --no-viz (bool): If True, suppress visualizations to help validate the model. Default is False.
    --pdf (bool): Output visualization results as a PDF if set.
    --output: Output file for results (default: None).
    --params (filename): JSON file with parameters (default: None).
    --param (key:value pairs): Additional parameter overrides (param:value or param=value).

Usage:

    Run the script from the command line with the desired options.
    Example: ``python nn_model.py --nticks 365 --seed 20241107 --verbose --viz --pdf``
"""

import click
import numpy as np

from laser_measles import Births
from laser_measles import Incubation
from laser_measles import Infection
from laser_measles import MaternalAntibodies
from laser_measles import NonDiseaseDeaths
from laser_measles import RoutineImmunization
from laser_measles import Susceptibility
from laser_measles import Transmission
from laser_measles.model import Model
from laser_measles.nigeria import get_parameters
from laser_measles.nigeria import get_scenario
from laser_measles.utils import seed_infections_in_patch


@click.command()
@click.option("--nticks", default=365, help="Number of ticks to run the simulation")
@click.option("--seed", default=20241107, help="Random seed")
@click.option("--verbose", is_flag=True, help="Print verbose output")
@click.option("--no-viz", is_flag=True, help="Suppress validation visualizations")
@click.option("--pdf", is_flag=True, help="Output visualization results as a PDF")
@click.option("--output", default=None, help="Output file for results")
@click.option("--params", default=None, help="JSON file with parameters")
@click.option("--param", "-p", multiple=True, help="Additional parameter overrides (param:value or param=value)")
def run(**kwargs):
    """
    Run the measles simulation model with the given parameters.
    This function initializes the model with the specified parameters, sets up the
    components of the model, seeds initial infections, runs the simulation, and
    optionally visualizes the results.

    Parameters:

        **kwargs: Arbitrary keyword arguments containing the parameters for the simulation.

        Expected keys include:

            - "verbose": bool, whether to print detailed information during the simulation.
            - "no-viz": (bool) Whether to suppress visualizations.
            - "pdf": bool, whether to save the visualization as a PDF.

    Returns:

        None
    """

    parameters = get_parameters(kwargs)
    scenario = get_scenario(parameters, parameters["verbose"])
    model = Model(scenario, parameters, name="northern nigeria measles")

    # infection dynamics come _before_ incubation dynamics so newly set itimers
    # don't immediately expire
    model.components = [
        Births,
        NonDiseaseDeaths,
        Susceptibility,
        MaternalAntibodies,
        RoutineImmunization,
        Infection,
        Incubation,
        Transmission,
    ]

    # Seed initial infections in most populous patch at the start of the simulation
    ipatch = np.argmax(model.patches.populations[0, :])
    seed_infections_in_patch(model, ipatch=ipatch, ninfections=100)

    # Run the model
    model.run()

    # Visualize the results
    if not parameters["no_viz"]:
        model.visualize(pdf=parameters["pdf"])

    return


if __name__ == "__main__":
    ctx = click.Context(run)
    ctx.invoke(run, nticks=365, seed=20241107, viz=True,verbose=True, pdf=False)
