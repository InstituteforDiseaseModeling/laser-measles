"""
Inititializes the age distribution of the population and
enforces age structure with age based mortality
"""

from datetime import timedelta

import numpy as np
from laser_core import SortedQueue
from laser_core.demographics import AliasedDistribution
from pydantic import BaseModel
from pydantic import Field

from laser_measles.abm.model import ABMModel
from laser_measles.base import BasePhase
from laser_measles.demographics.wpp import WPP
from laser_measles.utils import cast_type

from laser_measles.events import EventMixin


class WPPVitalDynamicsParams(BaseModel):
    country_code: str = Field(default="nga", description="Country code (ISO3)")
    year: int = Field(default=2000, description="Year to initialize the age distribution")


class WPPVitalDynamicsProcess(BasePhase, EventMixin):
    def __init__(self, model, verbose: bool = False, params: WPPVitalDynamicsParams | None = None) -> None:
        super().__init__(model, verbose)
        if params is None:
            params = WPPVitalDynamicsParams()
        self.params = params
        self.wpp = WPP(self.params.country_code)
        
        # Initialize event mixin
        self.__init_event_mixin__()

        # re-initialize people frame with correct capacity
        capacity = self.calculate_capacity(model=model)
        model.initialize_people_capacity(capacity=int(capacity), initial_count=model.scenario["pop"].sum())

        people = model.people

        date_of_birth_dtype = np.int32
        self.null_value = np.iinfo(date_of_birth_dtype).max

        if model.params.num_ticks >= self.null_value:
            raise ValueError("Simulation is too long; birth dates must be able to store the number of ticks")

        people.add_scalar_property("active", dtype=np.bool, default=False)
        people.add_scalar_property("date_of_birth", dtype=date_of_birth_dtype, default=self.null_value)

        # initialize patch id
        patch_ids = np.concatenate([np.full(pop, i) for i, pop in enumerate(model.scenario["pop"].to_numpy())])
        model.prng.shuffle(patch_ids)
        people.patch_id[: len(people)] = patch_ids

        # initialize age distribution
        # TODO: Capture this in the demographics submodule
        # Interpolate to starting year
        pyramid = self.wpp.get_population_pyramid(self.params.year)
        # Create sampler
        sampler = AliasedDistribution(pyramid)
        # Sample from the distribution
        samples = sampler.sample(len(people))  # ndarray of int32
        # Accolate ages
        ages = np.zeros(len(samples), dtype=np.int32)
        for bin_idx in np.arange(len(pyramid)):
            mask = samples == bin_idx
            ages[mask] = np.random.randint(self.wpp.age_vec[bin_idx], self.wpp.age_vec[bin_idx + 1], size=mask.sum())
        people.date_of_birth[: len(people)] = -ages

        # initialize active
        people.active[: len(people)] = True

        # initialize everyone susceptible (this may be changed by another component)
        people.state[: len(people)] = model.params.states.index("S")
        people.susceptibility[: len(people)] = 1.0

        # set the state counts
        for patch_id in range(len(model.patches)):
            model.patches.states[:, patch_id] = np.bincount(
                people.state[people.active & (people.patch_id == patch_id)], minlength=len(model.params.states)
            )

    def __call__(self, model: ABMModel, tick: int) -> None:
        # get references to people and patches
        people = model.people
        patches = model.patches

        # get current population in each patch
        patch_pops = patches.states.sum(axis=0)

        # Deaths (get the index of the current year in the birth and mortality data vectors)
        # year_idx = np.argmin(np.abs(model.current_date.year  - self.year_vec)) # using NN
        year_idx = np.where(self.wpp.year_vec < model.current_date.year)[0][-1]  # using most recent entry
        # get indices of active people
        idx = np.where(people.active)[0]
        # calculate current age (ticks) of active people
        ages = tick - people.date_of_birth[idx]

        # age bin indices
        age_bin_idx = np.digitize(x=ages, bins=self.wpp.age_vec) - 1
        mort_rates = self.wpp.vd_tup.mort_mat[::2, year_idx][age_bin_idx]
        death_idx = idx[np.random.random(len(ages)) < mort_rates]

        # mark as not active
        people.active[death_idx] = False

        # update state counts
        for patch_id in range(len(model.patches)):
            patch_idx = death_idx[people.patch_id[death_idx] == patch_id]
            if len(patch_idx) > 0:
                model.patches.states[:, patch_id] -= cast_type(
                    np.bincount(people.state[patch_idx], minlength=len(model.params.states)), model.patches.states.dtype
                )
        
        # Emit death event with details about who died
        if len(death_idx) > 0:
            # Get ages at death and mortality rates for the deceased agents
            death_ages = ages[np.isin(idx, death_idx)]
            death_mort_rates = mort_rates[np.isin(idx, death_idx)]
            
            self.emit_event(
                event_type='deaths',
                tick=tick,
                data={
                    'agent_indices': death_idx.tolist(),
                    'patch_ids': people.patch_id[death_idx].tolist(),
                    'states': people.state[death_idx].tolist(),
                    'num_deaths': len(death_idx),
                    'ages': death_ages.tolist(),
                    'mortality_rates': death_mort_rates.tolist()
                }
            )

        # Births
        idx = np.where(~people.active)[0]
        i_start = 0
        i_end = 0
        patch_births = []
        birth_agent_indices = []
        
        for patch_id in range(len(model.patches)):
            patch_pop = patch_pops[patch_id]
            births = np.random.poisson(lam=patch_pop * self.wpp.vd_tup.birth_rate * self.wpp.vd_tup.br_mult_y[year_idx])
            patch_births.append(births)
            i_end = i_start + births
            
            # Store agent indices for this patch's births
            if births > 0:
                birth_agent_indices.extend(idx[i_start:i_end].tolist())
                
            people.active[idx[i_start:i_end]] = True
            people.date_of_birth[idx[i_start:i_end]] = tick
            people.state[idx[i_start:i_end]] = model.params.states.index("S")
            people.susceptibility[idx[i_start:i_end]] = 1.0
            i_start = i_end
            # update state counts
            model.patches.states.S[patch_id] += births
        
        # Emit birth event with details about new births
        total_births = sum(patch_births)
        if total_births > 0:
            self.emit_event(
                event_type='births',
                tick=tick,
                data={
                    'agent_indices': birth_agent_indices,
                    'patch_births': patch_births,
                    'total_births': total_births,
                    'birth_rate': self.wpp.vd_tup.birth_rate * self.wpp.vd_tup.br_mult_y[year_idx]
                }
            )

    def calculate_wpp_total_pop(self, year: int) -> int:
        return int(np.sum(self.wpp.get_population_pyramid(year)))

    def calculate_capacity(self, model: ABMModel, buffer: float = 0.05) -> int:
        sim_end_date = timedelta(days=model.params.num_ticks * model.params.time_step_days) + model.current_date
        return int(
            model.scenario["pop"].sum()
            * (self.calculate_wpp_total_pop(sim_end_date.year) / self.calculate_wpp_total_pop(model.current_date.year))
            * (1 + buffer)
        )

    def _initialize(self, model: ABMModel) -> None:
        # initialization is done in the __init__ method
        pass
