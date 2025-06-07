import numpy as np

from laser_measles.biweekly.base import BaseComponent


def cast_type(a, dtype):
    return a.astype(dtype) if a.dtype != dtype else a


class VitalDynamics(BaseComponent):
    """
    Component for simulating the vital dynamics in the model
    """

    def __init__(self, model, verbose: bool = False) -> None:
        super().__init__(model, verbose)

    def __call__(self, model, tick: int) -> None:
        # state counts
        states = model.nodes.states

        # model parameters
        params = model.params

        # Vital dynamics
        population = states.sum(axis=0)
        biweek_avg_births = population * (params.crude_birth_rate / 26.0 / 1000.0)
        vaccinated_births = cast_type(np.random.poisson(biweek_avg_births*np.array(model.scenario['mcv1'])), states.dtype)
        unvaccinated_births = cast_type(np.random.poisson(biweek_avg_births*(1-np.array(model.scenario['mcv1']))), states.dtype)
        # births = cast_type(np.random.poisson(biweek_avg_births)*np.array(model.scenario['mcv1']), states.dtype)

        biweek_avg_deaths = population * (params.crude_death_rate / 26.0 / 1000.0)
        deaths = cast_type(np.random.poisson(biweek_avg_deaths), states.dtype)  # number of deaths

        states[0] += unvaccinated_births  # add births to S
        states[2] += vaccinated_births  # add births to R
        states -= deaths  # remove deaths from each compartment

        # make sure that all states >= 0
        np.maximum(states, 0, out=states)
