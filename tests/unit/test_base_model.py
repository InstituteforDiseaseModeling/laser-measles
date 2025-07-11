import polars as pl
from laser_core import LaserFrame
from laser_core import PropertySet

import laser_measles as lm
from laser_measles.base import BaseLaserModel
from laser_measles.base import BaseScenario

VERBOSE = False


class MockScenario(BaseScenario):
    def _validate(self, df: pl.DataFrame) -> None:
        pass


# Initialize the model and its population
def test_laserframe():
    # Use the proper LaserModel base class
    class TestModel(BaseLaserModel):
        def __init__(self, scenario=None, parameters=None, name="test"):
            scenario = MockScenario(lm.scenarios.synthetic.single_patch_scenario())
            parameters = PropertySet({"verbose": VERBOSE, "start_time": "2000-01"})
            super().__init__(scenario, parameters, name)
            # Create the agent population with max size 1000
            self.population = LaserFrame(capacity=1000, initial_count=0)
            # Add our properties, which can be thought of as the columns of our dataframe.
            self.population.add_scalar_property("disease_state")
            # Explicitly add the total population size, in this case the same as our max capacity
            self.population.add(1000)

        def __call__(self, model, tick):
            pass

        def _setup_components(self) -> None:
            pass

    # Initialization test
    TestModel()


if __name__ == "__main__":
    test_laserframe()
    print("✓ test_laserframe passed")
