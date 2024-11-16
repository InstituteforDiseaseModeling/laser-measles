r"""
measles_model.py

This module defines the base Measles Model and provides a command-line interface (CLI) to run the simulation.

Classes:
    None

Functions:
    run(\*\*kwargs)
        Runs the measles model simulation with the specified parameters.

        Parameters:
            nticks (int): Number of ticks to run the simulation. Default is 365.
            seed (int): Random seed for the simulation. Default is 20241107.
            verbose (bool): If True, print verbose output. Default is False.
            viz (bool): If True, display visualizations to help validate the model. Default is False.
            pdf (bool): If True, output visualization results as a PDF. Default is False.
            output (str): Output file for results. Default is None.
            params (str): JSON file with parameters. Default is None.
            param (tuple): Additional parameter overrides in the form of (param:value or param=value). Default is an empty tuple.

Usage:
    To run the simulation from the command line:
        python measles_model.py --nticks 365 --seed 20241107 --verbose --viz --pdf --output results.json --params params.json -p param1=value1 -p param2=value2
"""

import click

from laser_measles.measles_births import Births
from laser_measles.measles_incubation import Incubation
from laser_measles.measles_infection import Infection
from laser_measles.measles_maternalabs import MaternalAntibodies
from laser_measles.measles_metapop import get_scenario
from laser_measles.measles_nddeaths import NonDiseaseDeaths
from laser_measles.measles_params import get_parameters
from laser_measles.measles_ri import RoutineImmunization
from laser_measles.measles_susceptibility import Susceptibility
from laser_measles.measles_transmission import Transmission
from laser_measles.model import Model
from laser_measles.utils import seed_infections_in_patch


@click.command()
@click.option("--nticks", default=365, help="Number of ticks to run the simulation")
@click.option("--seed", default=20241107, help="Random seed")
@click.option("--verbose", is_flag=True, help="Print verbose output")
@click.option("--viz", is_flag=True, help="Display visualizations  to help validate the model")
@click.option("--pdf", is_flag=True, help="Output visualization results as a PDF")
@click.option("--output", default=None, help="Output file for results")
@click.option("--params", default=None, help="JSON file with parameters")
@click.option("--param", "-p", multiple=True, help="Additional parameter overrides (param:value or param=value)")
def run(**kwargs):
    """
    Run the measles model simulation with the given parameters.
    This function initializes the model with the specified parameters, sets up the
    components of the model, seeds initial infections, runs the simulation, and
    optionally visualizes the results.
    Parameters:
    **kwargs: Arbitrary keyword arguments containing the parameters for the simulation.
        Expected keys include:
        - "verbose": (bool) Whether to print verbose output.
        - "viz": (bool) Whether to visualize the results.
        - "pdf": (str) The file path to save the visualization as a PDF.
    Returns:
    None
    """

    parameters = get_parameters(kwargs)
    scenario = get_scenario(parameters, parameters["verbose"])
    model = Model(scenario, parameters)

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

    # seed_infections_randomly(model, ninfections=100)
    # Seed initial infections in Node 13 (King County) at the start of the simulation
    # Pierce County is Node 18, Snohomish County is Node 14, Yakima County is 19
    seed_infections_in_patch(model, ipatch=13, ninfections=100)

    model.run()

    if parameters["viz"]:
        model.visualize(pdf=parameters["pdf"])

    return


if __name__ == "__main__":
    ctx = click.Context(run)
    ctx.invoke(run, nticks=365, seed=20241107, verbose=True, viz=True, pdf=False)
