from laser_core import LaserFrame

from laser_measles import compute


def test_compute():
    assert compute(["a", "bc", "abc"]) == "abc"


# Initialize the model and its population
def test_laserframe():
    # Declare a very simple Model class to house our model pieces.
    class Model:
        pass

    model = Model()
    # Create the agent population with max size 1000
    model.population = LaserFrame(capacity=1000, initial_count=0)
    # Add our properties, which can be thought of as the columns of our dataframe.
    model.population.add_scalar_property("disease_state")
    # Explicitly add the total population size, in this case the same as our max capacity
    model.population.add(1000)

if __name__ == "__main__":
    test_compute()
