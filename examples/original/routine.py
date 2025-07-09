"""
This module implements the Routine Immunization (RI) process for a population model. It includes the initialization
of RI coverage for patches, the assignment of MCV (Measles Containing Vaccine) status to agents, and the updating
of RI timers for agents.

Classes:

    RoutineImmunization: Manages the routine immunization process, including initialization, updating RI timers,
                         handling births, and plotting immunization status.

Functions:

    nb_update_ri_timers(count, ri_timers, susceptibility): Numba-optimized function to update RI timers and adjust
                                                           susceptibility when timers expire.
    set_mcv_status(model, istart, iend): Assigns MCV status to agents based on RI coverage and probabilities of MCV1
                                         and MCV2 take.
    set_mcv_timers(model, istart, iend): Sets the RI timers for agents based on their MCV status and predefined
                                         time ranges for MCV1 and MCV2.

Constants:

    GET_MCV1: Constant representing the status of an agent who has received an effective MCV1 vaccination.
    GET_MCV2: Constant representing the status of an agent who has received an effective MCV2 vaccination.
    GET_NONE: Constant representing the status of an unvaccinated agent or an agent with ineffective vaccination.
"""

import numba as nb
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure


class RoutineImmunization:
    """
    A component to handle the routine immunization process within a model.
    """

    def __init__(self, model, verbose: bool = False):
        """
        Initializes the routine immunization process for the model.

        Args:

            model: The model instance to which the routine immunization process is applied.
            verbose (bool, optional): If True, enables verbose output. Defaults to False.

        Attributes:

            model: The model instance to which the routine immunization process is applied.

        Initializes:

            - Adds a scalar property "ri_coverage" to the model's patches with dtype np.float32.
            - Sets the "ri_coverage" for each patch based on a Poisson distribution around the specified parameter.
            - Adds a scalar property "mcv" to the model's population with dtype np.uint8.
            - Adds a scalar property "ri_timer" to the model's population with dtype np.uint16.
        """

        self.model = model

        # Coverage by patch
        model.patches.add_scalar_property("ri_coverage", dtype=np.float32)
        # TODO - make this more robust
        # Coverage ranges around the specified parameter
        model.patches.ri_coverage[:] = model.prng.poisson(model.params.ri_coverage * 100, model.patches.count) / 100
        # Agents get an MCV status - 0 for unvaccinated _or ineffective vaccination_, 1 for effective MCV1, 2 for effective MCV2
        model.population.add_scalar_property("mcv", dtype=np.uint8)
        model.population.add_scalar_property("ri_timer", dtype=np.uint16)  # Use uint16 for timer since 15 months = ~450 > 2^8

        # TODO - initialize existing agents with MCV status and ri_timer

        return

    def __call__(self, model, tick):
        """
        Updates (decrements) the RI (Routine Immunization) timers for the population in the model.

        Sets susceptibility to 0 for agents when their RI timers hit 0.

        Args:

            model (object): The model containing the population data.
            tick (int): The current time step or tick in the simulation.

        Returns:

            None
        """

        self.nb_update_ri_timers(model.population.count, model.population.ri_timer, model.population.susceptibility)
        return

    @staticmethod
    @nb.njit((nb.uint32, nb.uint16[:], nb.uint8[:]), parallel=True, cache=True)
    def nb_update_ri_timers(count, ri_timers, susceptibility):  # pragma: no cover
        """Numba compiled function to check and update routine immunization timers for the population in parallel."""
        for i in nb.prange(count):
            timer = ri_timers[i]
            if timer > 0:
                timer -= 1
                ri_timers[i] = timer
                if timer == 0:
                    # When timer expires, vaccinated agents become immune
                    susceptibility[i] = 0

        return

    def on_birth(self, model, _tick, istart, iend):
        """
        Handles the birth event in the model by setting the MCV (Measles
        Conjugate Vaccine) status (effective MCV1, effective MCV2, or
        unvaccinated/ineffective vaccination) and initializing the MCV timers
        for the newborns.

        Parameters:

            model (object): The model instance containing the population data.
            tick (int): The current tick or time step in the simulation (unused in this function).
            istart (int): The starting index of the newborns in the population array.
            iend (int): The ending index of the newborns in the population array.

        Returns:

            None
        """

        # newborns get an MCV status and ri_timer
        set_mcv_status(model, istart, iend)
        set_mcv_timers(model, istart, iend)

        return

    def plot(self, fig: Figure = None):
        """
        Plots a pie chart representing the routine immunization status of the
        population born during the simulation (initial agent population does not
        have MCV status computed or tracked).

        Parameters:

            fig (Figure, optional): A matplotlib Figure object to plot on. If None, a new figure is created.

        Raises:

            AssertionError: If the sum of unvaccinated, MCV1, and MCV2 individuals does not match the total number of individuals.

        Yields:

            None
        """

        fig = plt.figure(figsize=(12, 9), dpi=128) if fig is None else fig

        population = self.model.population
        indices = population.dob[0 : population.count] > 0
        cindividuals = indices.sum()
        mcv = population.mcv[0 : population.count]  # just active agents
        cunvaccinated = (mcv[indices] == GET_NONE).sum()
        cmcv1 = (mcv[indices] == GET_MCV1).sum()
        cmcv2 = (mcv[indices] == GET_MCV2).sum()

        assert cindividuals == cunvaccinated + cmcv1 + cmcv2, (
            f"Mismatch in MCV status counts:\n{cindividuals=:,} != {cunvaccinated=:,} + {cmcv1=:,} + {cmcv2=:,}"
        )

        fig.suptitle(f"Routine Immunization\n{cindividuals:,} individuals")
        pct_none = 100 * cunvaccinated / cindividuals
        pct_mcv1 = 100 * cmcv1 / cindividuals
        pct_mcv2 = 100 * cmcv2 / cindividuals
        plt.pie(
            [cunvaccinated, cmcv1, cmcv2],
            labels=[
                f"Unvaccinated {cunvaccinated:,} ({pct_none:.1f}%)",
                f"MCV1 {cmcv1:,} ({pct_mcv1:.1f}%)",
                f"MCV2 {cmcv2:,} ({pct_mcv2:.1f}%)",
            ],
        )

        yield
        return


GET_MCV1 = 1
GET_MCV2 = 2
GET_NONE = 0


def set_mcv_status(model, istart, iend):
    """
    Set the MCV (Measles Conjugate Vaccine) status for a subset of the population.
    This function assigns (effective) MCV1, (effective) MCV2, or NONE (no or
    ineffective vaccination) status to individuals in the population based on
    the model's parameters and random draws. The MCV1 and MCV2 statuses are
    determined by the coverage and probability of vaccine take specified in the model.

    Parameters:

        model (object): The model containing population and parameters for MCV coverage and take probabilities.
        istart (int): The starting index of the population subset to update.
        iend (int): The ending index (exclusive) of the population subset to update.

    Returns:

        None
    """

    mcv1_cutoff = model.patches.ri_coverage * model.params.probability_mcv1_take  # probability of (MCV1 vaccination) _and_ (MCV1 take)
    mcv2_cutoff = (
        mcv1_cutoff + model.patches.ri_coverage * (1.0 - model.params.probability_mcv1_take) * model.params.probability_mcv2_take
    )  # probability of (MCV1 vaccination) _and_ (not MCV1 take) and (MCV2 take)

    draws = model.prng.random(size=(iend - istart))
    nodeids = model.population.nodeid[istart:iend]
    get_mcv1 = draws <= mcv1_cutoff[nodeids]
    get_mcv2 = (draws > mcv1_cutoff[nodeids]) & (draws <= mcv2_cutoff[nodeids])
    # get_none = (draws > mcv2_cutoff[nodeids]) # "get_none" is the default
    mcv = model.population.mcv[istart:iend]
    mcv[get_mcv1] = GET_MCV1
    mcv[get_mcv2] = GET_MCV2

    return


def set_mcv_timers(model, istart, iend):
    """
    Set the MCV (Measles Containing Vaccine) timers for a subset of the population in the model.

    This function assigns random timer values for MCV1 or MCV2 vaccinations to individuals in the population
    based on the specified start and end indices. The timer values are generated using the model's pseudo-random
    number generator (PRNG) and are constrained within the start and end parameters for MCV1 and MCV2.

    Parameters:

        model (object): The model object containing the population and parameters.
        istart (int): The starting index of the subset of the population.
        iend (int): The ending index of the subset of the population.

    Returns:

        None
    """

    count = iend - istart
    ri_timer_values_mcv1 = model.prng.integers(model.params.mcv1_start, model.params.mcv1_end, count).astype(
        model.population.ri_timer.dtype
    )
    ri_timer_values_mcv2 = model.prng.integers(model.params.mcv2_start, model.params.mcv2_end, count).astype(
        model.population.ri_timer.dtype
    )

    mcv = model.population.mcv[istart:iend]

    mask_mcv1 = mcv == GET_MCV1
    mask_mcv2 = mcv == GET_MCV2

    timers = model.population.ri_timer[istart:iend]
    timers[mask_mcv1] = ri_timer_values_mcv1[mask_mcv1]
    timers[mask_mcv2] = ri_timer_values_mcv2[mask_mcv2]

    return
