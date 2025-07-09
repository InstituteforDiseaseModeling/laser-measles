r"""
Northern Nigeria Measles Model

This script runs a simulation of measles dynamics in Northern Nigeria using a
compartmental model. The model includes various components such as births,
non-disease deaths, susceptibility, maternal antibodies, routine immunization,
infection, incubation, and transmission.

The script uses the Typer library to handle command-line options for configuring
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

import numpy as np
import typer

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
    app()
