"""
Component for simulating MCV1 vaccination with delayed scheduling.

This component listens to birth events and schedules MCV1 vaccinations
for newborns after a delay drawn from a gamma distribution with a
configurable mean (default 9 months ≈ 270 days).
"""

import numpy as np
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, DefaultDict
from collections import defaultdict

from laser_measles.abm.model import ABMModel
from laser_measles.base import BasePhase
from laser_measles.utils import cast_type

from laser_measles.events import EventMixin, BaseEvent


class MCV1Params(BaseModel):
    """
    Parameters for MCV1 vaccination process.
    """
    
    vaccination_delay_mean: float = Field(
        default=270.0, 
        description="Mean delay in days between birth and MCV1 vaccination (≈9 months)",
        ge=1.0
    )
    vaccination_efficacy: float = Field(
        default=0.9,
        description="Efficacy of MCV1 vaccination (fraction protected)",
        ge=0.0,
        le=1.0
    )
    coverage: float = Field(
        default=1.0,
        description="Fraction of eligible newborns who will receive MCV1",
        ge=0.0,
        le=1.0
    )
    delay_distribution: str = Field(
        default="gamma",
        description="Distribution type for vaccination delays ('gamma' or 'exponential')"
    )


class ProcessMCV1(BasePhase, EventMixin):
    """
    Process for simulating MCV1 vaccination with realistic delays.
    
    This component:
    1. Listens to birth events from VitalDynamicsProcess
    2. Schedules MCV1 vaccinations with delays drawn from a gamma distribution
    3. Processes vaccinations on their scheduled dates
    4. Updates agent states from Susceptible (S) to Recovered (R) upon vaccination
    
    The vaccination delay follows a gamma distribution with specified mean,
    providing realistic variability in vaccination timing that reflects
    real-world healthcare delivery patterns.
    """
    
    def __init__(self, model: ABMModel, verbose: bool = False, params: Optional[MCV1Params] = None) -> None:
        super().__init__(model, verbose)
        self.params = params or MCV1Params()
        self.__init_event_mixin__()
        
        # Validate parameters
        if self.params.delay_distribution not in ["gamma", "exponential"]:
            raise ValueError("delay_distribution must be 'gamma' or 'exponential'")
        
        # Initialize vaccination schedule - maps tick -> list of agent indices
        self.vaccination_schedule: DefaultDict[int, List[int]] = defaultdict(list)
        
        # Statistics tracking
        self.stats = {
            'births_scheduled': 0,
            'vaccinations_completed': 0,
            'agents_protected': 0,
            'agents_not_protected': 0
        }
        
        if self.verbose:
            print(f"ProcessMCV1 initialized with {self.params.vaccination_delay_mean:.1f} day mean delay")
    
    def set_event_bus(self, event_bus) -> None:
        """Set the event bus and subscribe to birth events."""
        super().set_event_bus(event_bus)
        # Subscribe to births to schedule vaccinations
        self.subscribe_to_events(['births'], self.handle_birth_event)
    
    def handle_birth_event(self, event: BaseEvent) -> None:
        """
        Handle birth events by scheduling MCV1 vaccinations.
        
        Args:
            event: Birth event containing newborn agent information
        """
        if event.event_type != 'births':
            return
        
        birth_data = event.data
        newborn_indices = birth_data.get('agent_indices', [])
        current_tick = event.tick
        
        if not newborn_indices:
            return
        
        # Determine which newborns are eligible for vaccination based on coverage
        if self.params.coverage < 1.0:
            n_eligible = int(len(newborn_indices) * self.params.coverage)
            if n_eligible == 0:
                return
            # Randomly select eligible agents
            eligible_indices = self.model.prng.choice(
                newborn_indices, 
                size=n_eligible, 
                replace=False
            ).tolist()
        else:
            eligible_indices = newborn_indices
        
        # Schedule vaccinations for eligible newborns
        vaccination_delays = self._generate_delays(len(eligible_indices))
        
        for agent_idx, delay in zip(eligible_indices, vaccination_delays):
            vaccination_tick = current_tick + int(delay)
            self.vaccination_schedule[vaccination_tick].append(agent_idx)
        
        self.stats['births_scheduled'] += len(eligible_indices)
        
        if self.verbose and len(eligible_indices) > 0:
            mean_delay = np.mean(vaccination_delays)
            print(f"Tick {current_tick}: Scheduled {len(eligible_indices)} MCV1 vaccinations "
                  f"(mean delay: {mean_delay:.1f} days)")
    
    def _generate_delays(self, n_agents: int) -> np.ndarray:
        """
        Generate vaccination delays from the specified distribution.
        
        Args:
            n_agents: Number of delay values to generate
            
        Returns:
            Array of delay values in days
        """
        if self.params.delay_distribution == "gamma":
            # Use gamma distribution with shape parameter = 4 for realistic spread
            # This gives a reasonable coefficient of variation (~0.5)
            shape = 4.0
            scale = self.params.vaccination_delay_mean / shape
            delays = self.model.prng.gamma(shape, scale, size=n_agents)
        else:  # exponential
            delays = self.model.prng.exponential(self.params.vaccination_delay_mean, size=n_agents)
        
        # Ensure minimum delay of 1 day
        delays = np.maximum(delays, 1.0)
        
        return delays
    
    def __call__(self, model: ABMModel, tick: int) -> None:
        """
        Process vaccinations scheduled for the current tick.
        
        Args:
            model: The ABM model
            tick: Current simulation tick
        """
        # Check if there are vaccinations scheduled for this tick
        if tick not in self.vaccination_schedule:
            return
        
        # Get all agents scheduled for vaccination on this tick
        agent_indices = self.vaccination_schedule.pop(tick)
        
        if agent_indices:
            if self.verbose:
                print(f"Tick {tick}: Processing {len(agent_indices)} scheduled vaccinations")
            self._vaccinate_agents(model, agent_indices, tick)
    
    def _vaccinate_agents(self, model: ABMModel, agent_indices: List[int], tick: int) -> None:
        """
        Vaccinate the specified agents.
        
        Args:
            model: The ABM model
            agent_indices: List of agent indices to vaccinate
            tick: Current tick
        """
        if not agent_indices:
            return
        
        people = model.people
        patches = model.patches
        
        # Filter for agents that are still alive and susceptible
        valid_agents = []
        for agent_idx in agent_indices:
            agent_valid = False
            reason = ""
            
            if agent_idx >= people.count:
                reason = f"agent_idx {agent_idx} >= people.count {people.count}"
            elif not hasattr(people, 'active'):
                reason = "no 'active' property"
            elif not people.active[agent_idx]:
                reason = f"agent {agent_idx} not active"
            elif people.state[agent_idx] != 0:
                reason = f"agent {agent_idx} state={people.state[agent_idx]} (not susceptible)"
            else:
                valid_agents.append(agent_idx)
                agent_valid = True
            
            if self.verbose and not agent_valid:
                print(f"    Agent {agent_idx} invalid: {reason}")
        
        if self.verbose:
            print(f"  Valid agents for vaccination: {len(valid_agents)} out of {len(agent_indices)}")
        
        if not valid_agents:
            return
        
        valid_agents = np.array(valid_agents)
        
        # Determine vaccination outcome based on efficacy
        n_agents = len(valid_agents)
        vaccination_success = self.model.prng.random(n_agents) < self.params.vaccination_efficacy
        
        # Agents with successful vaccination move to recovered state
        protected_agents = valid_agents[vaccination_success]
        if len(protected_agents) > 0:
            # Get recovered state index
            recovered_state = model.params.states.index("R")
            
            # Update agent states
            people.state[protected_agents] = recovered_state
            
            # Update patch-level counts
            for agent_idx in protected_agents:
                patch_id = people.patch_id[agent_idx]
                patches.states.S[patch_id] -= 1
                patches.states.R[patch_id] += 1
            
            self.stats['agents_protected'] += len(protected_agents)
        
        # Track agents who received vaccination but weren't protected
        unprotected_agents = valid_agents[~vaccination_success]
        self.stats['agents_not_protected'] += len(unprotected_agents)
        
        self.stats['vaccinations_completed'] += n_agents
        
        if self.verbose and n_agents > 0:
            protection_rate = len(protected_agents) / n_agents * 100
            print(f"Tick {tick}: Vaccinated {n_agents} agents, "
                  f"{len(protected_agents)} protected ({protection_rate:.1f}%)")
        
        # Emit vaccination event for other components to potentially use
        if self.has_subscribers('vaccination'):
            self.emit_event(
                event_type='vaccination',
                tick=tick,
                data={
                    'agent_indices': valid_agents.tolist(),
                    'protected_agents': protected_agents.tolist(),
                    'unprotected_agents': unprotected_agents.tolist(),
                    'efficacy_achieved': len(protected_agents) / n_agents if n_agents > 0 else 0.0,
                    'vaccination_type': 'mcv1'
                }
            )
    
    def get_stats(self) -> dict:
        """Get vaccination statistics."""
        stats = self.stats.copy()
        if stats['vaccinations_completed'] > 0:
            stats['overall_protection_rate'] = (
                stats['agents_protected'] / stats['vaccinations_completed']
            )
        else:
            stats['overall_protection_rate'] = 0.0
        
        stats['pending_vaccinations'] = sum(len(agents) for agents in self.vaccination_schedule.values())
        return stats
    
    def initialize(self, model: ABMModel) -> None:
        """Initialize component - called before simulation starts."""
        pass