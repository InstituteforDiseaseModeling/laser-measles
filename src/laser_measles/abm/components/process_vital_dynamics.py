"""
Component defining the VitalDynamicsProcess, which simulates the vital dynamics in the ABM model with MCV1.
"""

import numpy as np
from laser_core import SortedQueue
from pydantic import Field

from laser_measles.abm.model import ABMModel
from laser_measles.components import BaseVitalDynamicsParams
from laser_measles.components import BaseVitalDynamicsProcess
from laser_measles.utils import cast_type


class VitalDynamicsParams(BaseVitalDynamicsParams):
    """
    Parameters for the vital dynamics process.
    """

    routine_immunization_delay: int = Field(default=9 * 30, description="Delay in days before routine immunization is administered")


class VitalDynamicsProcess(BaseVitalDynamicsProcess):
    """
    Process for simulating the vital dynamics in the ABM model with MCV1.
    """

    def __init__(self, model, verbose: bool = False, params: VitalDynamicsParams | None = None) -> None:
        params_: VitalDynamicsParams = params or VitalDynamicsParams()
        super().__init__(model, verbose=verbose, params=params_)

        # re-initialize people frame with correct capacity
        capacity = self.calculate_capacity(model=model)
        model.initialize_people_capacity(capacity=capacity, initial_count=model.scenario["pop"].sum())

        people = model.people
        patches = model.patches

        date_of_birth_dtype = np.int32
        self.null_value = np.iinfo(date_of_birth_dtype).max

        people.add_scalar_property("active", dtype=np.bool, default=False)
        people.add_scalar_property("date_of_birth", dtype=date_of_birth_dtype, default=self.null_value)
        people.add_scalar_property("date_of_vaccination", dtype=np.uint32, default=self.null_value)
        patches.add_scalar_property("births", dtype=np.uint32)

        self.vaccination_queue: SortedQueue = SortedQueue(capacity=capacity, values=people.date_of_vaccination)

        if model.params.num_ticks >= self.null_value:
            raise ValueError("Simulation is too long; birth and vaccination dates must be able to store the number of ticks")

    def __call__(self, model, tick: int) -> None:
        """
        Simulate the vital dynamics process.
        """
        patches = model.patches
        people = model.people
        population = patches.states.sum(axis=0)

        # Deaths
        # ------
        # Calculate number of deaths
        deaths = model.prng.poisson(population.sum() * self.mu_death)  # over all patches
        # select agents
        idx = model.prng.choice(np.where(model.people.active)[0], size=deaths, replace=False)
        # deactivate agents
        model.people.active[idx] = False
        # update state counter
        for state_idx, _ in enumerate(model.params.states):
            mask = model.people.state[idx] == state_idx  # mask of in-active agents in the current state
            cnt = np.bincount(model.people.patch_id[idx][mask], minlength=model.patches.states.shape[-1])
            model.patches.states[state_idx] -= cast_type(cnt, model.patches.states.dtype)

        # Births
        # ------
        # Calculate number of births
        births = model.prng.poisson(population * self.lambda_birth)  # in each patch
        # find indices of the people frame for initializing
        istart, iend = people.add(births.sum())
        people.date_of_birth[istart:iend] = tick  # born today
        people.susceptibility[istart:iend] = 1.0  # all newborns are susceptible TODO: add maternal immunity component
        people.date_of_vaccination[istart:iend] = tick + self._routine_immunization_delay()
        index = istart
        # update patch id
        for this_patch_id, this_patch_births in enumerate(births):
            people.patch_id[index : index + this_patch_births] = this_patch_id
            index += this_patch_births
        # update states
        patches.states.S += cast_type(births, patches.states.dtype)

        # Routine immunization
        # --------------------
        while len(self.vaccination_queue) > 0 and people.date_of_vaccination[self.vaccination_queue.peeki()] <= tick:
            i = self.vaccination_queue.popi()
            people.susceptibility[i] = 0.0  # susceptibility
            people.state[i] = model.params.states.index("R")  # move to recovered state

    def calculate_capacity(self, model) -> int:
        """Estimate the necessary capacity of the people laserframe."""
        rate = self.lambda_birth  # - self.mu_death # calculated per tick
        buffered_ticks = (model.params.num_ticks * model.params.time_step_days // 365 + 1) * 365 / model.params.time_step_days
        N = model.scenario["pop"].to_numpy() * np.exp(rate * buffered_ticks)
        return int(N.sum())

    def _routine_immunization_delay(self) -> int:
        """Delay in ticks before routine immunization is administered."""
        params: VitalDynamicsParams = self.params  # type: ignore
        return params.routine_immunization_delay * self.model.params.time_step_days

    def _initialize(self, model: ABMModel) -> None:
        # initialize the people laserframe with correct capacity
        initial_pop = model.scenario["pop"].sum()
        model.initialize_people_capacity(capacity=self.calculate_capacity(model), initial_count=initial_pop)

        # Activate population
        model.people.active[0 : model.people.count] = True
        # Simple initializer for ages where birth rate = mortality rate:
        # Initialize ages for existing population
        model.people.date_of_birth[0 : model.people.count] = cast_type(
            -1 * model.prng.exponential(1 / self.mu_death, model.people.count), model.people.date_of_birth.dtype
        )
