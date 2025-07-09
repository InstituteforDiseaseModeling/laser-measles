r"""
measles_model.py

This module defines the base Measles Model and provides a command-line interface (CLI) to run the simulation.

Classes:

    None

Functions:

    run(\*\*kwargs)

        Runs the measles model simulation with the specified parameters.

        Parameters:

            - nticks (int): Number of ticks to run the simulation. Default is 365.
            - seed (int): Random seed for the simulation. Default is 20241107.
            - verbose (bool): If True, print verbose output. Default is False.
            - no-viz (bool): If True, suppress visualizations to help validate the model. Default is False.
            - pdf (bool): If True, output visualization results as a PDF. Default is False.
            - output (str): Output file for results. Default is None.
            - params (str): JSON file with parameters. Default is None.
            - param (tuple): Additional parameter overrides in the form of (param:value or param=value). Default is an empty tuple.

Usage:

    To run the simulation from the command line (365 ticks, 20241107 seed, show visualizations):

        ``measles``

    To run the simulation with custom parameters, e.g., 5 years, 314159265 seed, output to PDF:

        ``measles --nticks 1825 --seed 314159265 --pdf``
"""

import typer

from laser_measles import Births
from laser_measles import Incubation
from laser_measles import Infection
from laser_measles import MaternalAntibodies
from laser_measles import Model
from laser_measles import NonDiseaseDeaths
from laser_measles import RoutineImmunization
from laser_measles import Susceptibility
from laser_measles import Transmission
from laser_measles.utils import seed_infections_in_patch
from laser_measles.washington import get_parameters
from laser_measles.washington import get_scenario

app = typer.Typer()


@app.command()
def run(
    nticks: int = typer.Option(365, help="Number of ticks to run the simulation"),
    seed: int = typer.Option(20241107, help="Random seed"),
    verbose: bool = typer.Option(False, help="Print verbose output"),
    no_viz: bool = typer.Option(False, "--no-viz", help="Suppress validation visualizations"),
    pdf: bool = typer.Option(False, help="Output visualization results as a PDF"),
    output: str | None = typer.Option(None, help="Output file for results"),
    params: str | None = typer.Option(None, help="JSON file with parameters"),
    param: list[str] = typer.Option([], "-p", "--param", help="Additional parameter overrides (param:value or param=value)"),
):
    """
    Run the measles model simulation with the given parameters.

    This function initializes the model with the specified parameters, sets up the
    components of the model, seeds initial infections, runs the simulation, and
    optionally visualizes the results.

    Parameters:

        **kwargs: Arbitrary keyword arguments containing the parameters for the simulation.

            Expected keys include:

                - "verbose": (bool) Whether to print verbose output.
                - "no-viz": (bool) Whether to suppress visualizations.
                - "pdf": (str) The file path to save the visualization as a PDF.

    Returns:

        None
    """

    kwargs = {
        "nticks": nticks,
        "seed": seed,
        "verbose": verbose,
        "no_viz": no_viz,
        "pdf": pdf,
        "output": output,
        "params": params,
        "param": param,
    }
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

    # Seed initial infections in Node 13 (King County) at the start of the simulation
    # Pierce County is Node 18, Snohomish County is Node 14, Yakima County is 19
    seed_infections_in_patch(model, ipatch=13, ninfections=100)

    model.run()

    if not parameters["no_viz"]:
        model.visualize(pdf=parameters["pdf"])

    return


if __name__ == "__main__":
    app()
