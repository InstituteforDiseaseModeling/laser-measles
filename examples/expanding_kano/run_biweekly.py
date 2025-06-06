"""
Run the biweekly model.
"""
from pathlib import Path

import numpy as np
import polars as pl

from laser_measles.biweekly import BaseScenario
from laser_measles.biweekly import BiweeklyModel
from laser_measles.biweekly import BiweeklyParams
from laser_measles.biweekly.components import FadeOutTracker
from laser_measles.biweekly.mixing import init_gravity_diffusion

THIS_DIR = Path(__file__).parent

def initialize_model(model, scenario: pl.DataFrame, params: BiweeklyParams):
        """
        Initializes the state of the model with the given settlement and parameters.

        Args:
            settlement_s (pd.DataFrame): The settlement data for the model.
            params (PropertySet): The parameters for the model.

        Returns:
            None
        """

        # initialize the compartment numbers
        population = scenario["pop"]
        if "crude_birth_rate" in scenario.columns:
            crude_birth_rates = scenario["crude_birth_rate"]
        else:
            crude_birth_rates = params.crude_birth_rate
        births = crude_birth_rates * population / 1000.0 * 26.0 / 365.0 # per biweek

        num = population
        susc = births * 2
        inf = susc / 26.0 / 2.0
        # Convert to numpy array first, then cast to the correct type
        # inf = np.array(inf).astype(model.nodes.states.dtype)

        model.nodes.states[:, :] = np.array([susc, inf, num - susc - inf], dtype=model.nodes.states.dtype)  # S

        # initialize the mixing matrix (dense)
        params.mixing = init_gravity_diffusion(scenario, params.mixing_scale, params.distance_exponent)

        return


def main() -> None:
    """
    Run the biweekly model.
    """

    # Load the scenarios
    scenarios = pl.read_parquet(THIS_DIR / "scenarios.parquet")
    scenario_names = scenarios["scenario_id"].unique().to_list()
    params = BiweeklyParams(nticks=26, crude_birth_rate=38, crude_death_rate=13)

    # Run the model
    for scenario_name in ["kano_region"]:
        scenario = scenarios.filter(pl.col("scenario_id") == scenario_name).drop(["scenario_id", "state"]).rename({"dotname": "ids"})
        model = BiweeklyModel(BaseScenario(scenario), params, name=scenario_name)
        model.components = [*model.components, FadeOutTracker]
        initialize_model(model, scenario, params)
        model.run()

    print("pause")


if __name__ == "__main__":
    main()
