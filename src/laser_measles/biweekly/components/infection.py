import numpy as np

from laser_measles.biweekly.base import BaseComponent


def cast_type(a, dtype):
    return a.astype(dtype) if a.dtype != dtype else a


class Infection(BaseComponent):
    """
    Component for simulating the spread of infection in the model
    """

    def __init__(self, model, verbose: bool = False) -> None:
        super().__init__(model, verbose)

    def __call__(self, model, tick: int) -> None:
        # state counts
        states = model.nodes.states

        # model parameters
        params = model.params

        # calculate the expected number of new infections
        # beta * (1 + seasonality * sin(2π(t-t0)/26)) * mixing * I
        expected = (
            params.beta
            * (1 + params.seasonality * np.sin(2 * np.pi * (tick - params.season_start) / 26.0))
            * np.matmul(params.mixing, states[1])
        )

        # probability of infection = 1 - exp(-expected/total_population)
        prob = 1 - np.exp(-expected / states.sum(axis=0))

        # sample from binomial distribution to get actual new infections
        dI = cast_type(np.random.binomial(n=states[0], p=prob), states.dtype)

        # move all currently infected to recovered (assuming 14 day recovery)
        states[2] += states[1]
        states[1] = 0

        # update susceptible and infected populations
        states[1] += dI  # add new infections to I
        states[0] -= dI  # remove new infections from S

        return
