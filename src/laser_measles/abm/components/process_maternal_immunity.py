"""
Component for simulating maternal immunity in newborn agents.

This component listens to birth events and provides temporary protection
to newborns by reducing their susceptibility to 0 for a random duration
drawn from a gamma distribution (default mean: 6 months ≈ 180 days).
"""

import numpy as np
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, DefaultDict
from collections import defaultdict

from laser_measles.abm.model import ABMModel
from laser_measles.base import BasePhase
from laser_measles.utils import cast_type

from laser_measles.events import EventMixin, BaseEvent


class MaternalImmunityParams(BaseModel):
    """
    Parameters for maternal immunity process.
    """
    
    protection_duration_mean: float = Field(
        default=180.0, 
        description="Mean duration of maternal immunity in days (~6 months)",
        ge=1.0
    )
    coverage: float = Field(
        default=1.0,
        description="Fraction of newborns who receive maternal antibodies",
        ge=0.0,
        le=1.0
    )
    distribution: str = Field(
        default="gamma",
        description="Distribution type for protection duration ('gamma' or 'exponential')"
    )


class ProcessMaternalImmunity(BasePhase, EventMixin):
    """
    Process for simulating maternal immunity in newborn agents.
    
    This component:
    1. Listens to birth events from VitalDynamicsProcess
    2. Immediately sets newborn susceptibility to 0 (protected)
    3. Schedules immunity expiration with delays drawn from gamma distribution
    4. Restores susceptibility to 1.0 when maternal immunity expires
    
    The protection duration follows a gamma distribution with specified mean,
    providing realistic variability in maternal antibody decay that reflects
    real-world passive immunity patterns.
    
    Unlike vaccination, this component:
    - Does NOT change disease state (agents remain in S state)
    - Only manipulates the susceptibility property (0 = protected, 1 = susceptible)
    - Provides temporary protection that naturally wanes over time
    """
    
    def __init__(self, model: ABMModel, verbose: bool = False, params: Optional[MaternalImmunityParams] = None) -> None:
        super().__init__(model, verbose)
        self.params = params or MaternalImmunityParams()
        self.__init_event_mixin__()
        
        # Validate parameters
        if self.params.distribution not in ["gamma", "exponential"]:
            raise ValueError("distribution must be 'gamma' or 'exponential'")
        
        # Initialize immunity expiration schedule - maps tick -> list of agent indices
        self.immunity_schedule: DefaultDict[int, List[int]] = defaultdict(list)
        # Track agents that need protection applied this tick
        self.pending_protection: DefaultDict[int, List[int]] = defaultdict(list)
        
        # Statistics tracking
        self.stats = {
            'births_protected': 0,
            'immunity_expired': 0,
            'agents_currently_protected': 0
        }
        
        if self.verbose:
            print(f"ProcessMaternalImmunity initialized with {self.params.protection_duration_mean:.1f} day mean protection")
    
    def set_event_bus(self, event_bus) -> None:
        """Set the event bus and subscribe to birth events."""
        super().set_event_bus(event_bus)
        # Subscribe to births to provide maternal immunity
        self.subscribe_to_events(['births'], self.handle_birth_event)
        # Uncomment for debugging:
        # if self.verbose:
        #     print(f"ProcessMaternalImmunity: Subscribed to birth events")
    
    def handle_birth_event(self, event: BaseEvent) -> None:
        """
        Handle birth events by providing maternal immunity to newborns.
        
        Args:
            event: Birth event containing newborn agent information
        """
        if event.event_type != 'births':
            return
        
        birth_data = event.data
        newborn_indices = birth_data.get('agent_indices', [])
        current_tick = event.tick
        
        # Uncomment for debugging:
        # if self.verbose:
        #     print(f"ProcessMaternalImmunity: Received birth event at tick {current_tick} with {len(newborn_indices)} newborns")
        
        if not newborn_indices:
            return
        
        # Determine which newborns receive maternal antibodies based on coverage
        if self.params.coverage < 1.0:
            n_protected = int(len(newborn_indices) * self.params.coverage)
            if n_protected == 0:
                return
            # Randomly select protected agents
            protected_indices = self.model.prng.choice(
                newborn_indices, 
                size=n_protected, 
                replace=False
            ).tolist()
        else:
            protected_indices = newborn_indices
        
        # Schedule these agents for protection application in the next tick
        # (to ensure VitalDynamicsProcess has finished setting susceptibility to 1.0)
        self.pending_protection[current_tick].extend(protected_indices)
        
        # Generate protection durations and schedule immunity expiration
        protection_durations = self._generate_durations(len(protected_indices))
        
        for agent_idx, duration in zip(protected_indices, protection_durations):
            expiration_tick = current_tick + int(duration)
            self.immunity_schedule[expiration_tick].append(agent_idx)
        
        self.stats['births_protected'] += len(protected_indices)
        self.stats['agents_currently_protected'] += len(protected_indices)
        
        if self.verbose and len(protected_indices) > 0:
            mean_duration = np.mean(protection_durations)
            print(f"Tick {current_tick}: Scheduled {len(protected_indices)} newborns for maternal immunity "
                  f"(mean duration: {mean_duration:.1f} days)")
    
    def _generate_durations(self, n_agents: int) -> np.ndarray:
        """
        Generate protection durations from the specified distribution.
        
        Args:
            n_agents: Number of duration values to generate
            
        Returns:
            Array of duration values in days
        """
        if self.params.distribution == "gamma":
            # Use gamma distribution with shape parameter = 4 for realistic spread
            # This gives a reasonable coefficient of variation (~0.5)
            shape = 4.0
            scale = self.params.protection_duration_mean / shape
            durations = self.model.prng.gamma(shape, scale, size=n_agents)
        else:  # exponential
            durations = self.model.prng.exponential(self.params.protection_duration_mean, size=n_agents)
        
        # Ensure minimum duration of 1 day and convert to integers
        durations = np.maximum(durations, 1.0)
        
        return durations
    
    def __call__(self, model: ABMModel, tick: int) -> None:
        """
        Process maternal immunity - apply protection to newborns and expire existing protection.
        
        Args:
            model: The ABM model
            tick: Current simulation tick
        """
        # First, apply protection to agents that were born this tick
        if tick in self.pending_protection:
            protected_agents = self.pending_protection.pop(tick)
            if protected_agents:
                self._apply_protection(model, protected_agents, tick)
        
        # Then, check if there are immunity expirations scheduled for this tick
        if tick in self.immunity_schedule:
            # Get all agents whose maternal immunity expires on this tick
            agent_indices = self.immunity_schedule.pop(tick)
            
            if agent_indices:
                if self.verbose:
                    print(f"Tick {tick}: Processing {len(agent_indices)} maternal immunity expirations")
                self._expire_immunity(model, agent_indices, tick)
    
    def _apply_protection(self, model: ABMModel, agent_indices: List[int], tick: int) -> None:
        """
        Apply maternal immunity protection to the specified agents.
        
        Args:
            model: The ABM model
            agent_indices: List of agent indices to protect
            tick: Current tick
        """
        if not agent_indices:
            return
        
        people = model.people
        
        # Filter for agents that are still alive and susceptible
        valid_agents = []
        for agent_idx in agent_indices:
            if (agent_idx < people.count and 
                hasattr(people, 'active') and 
                people.active[agent_idx] and
                people.susceptibility[agent_idx] > 0):
                valid_agents.append(agent_idx)
        
        if valid_agents:
            # Apply protection by setting susceptibility to 0
            people.susceptibility[valid_agents] = 0.0
            
            if self.verbose:
                print(f"Tick {tick}: Applied maternal immunity protection to {len(valid_agents)} newborns")
            
            # Emit protection start event
            if self.has_subscribers('maternal_immunity_start'):
                self.emit_event(
                    event_type='maternal_immunity_start',
                    tick=tick,
                    data={
                        'agent_indices': valid_agents,
                        'num_protected': len(valid_agents)
                    }
                )
    
    def _expire_immunity(self, model: ABMModel, agent_indices: List[int], tick: int) -> None:
        """
        Expire maternal immunity for the specified agents.
        
        Args:
            model: The ABM model
            agent_indices: List of agent indices whose immunity expires
            tick: Current tick
        """
        if not agent_indices:
            return
        
        people = model.people
        
        # Filter for agents that are still alive and currently protected
        valid_agents = []
        for agent_idx in agent_indices:
            agent_valid = False
            reason = ""
            
            if agent_idx >= people.count:
                reason = f"agent_idx {agent_idx} >= people.count {people.count}"
            elif not hasattr(people, 'active'):
                reason = "no 'active' property"
            elif not people.active[agent_idx]:
                reason = f"agent {agent_idx} not active (died)"
            elif people.susceptibility[agent_idx] > 0:
                reason = f"agent {agent_idx} already susceptible ({people.susceptibility[agent_idx]})"
            else:
                valid_agents.append(agent_idx)
                agent_valid = True
            
            if self.verbose and not agent_valid:
                print(f"    Agent {agent_idx} immunity expiration skipped: {reason}")
        
        if self.verbose:
            print(f"  Valid agents for immunity expiration: {len(valid_agents)} out of {len(agent_indices)}")
        
        if not valid_agents:
            return
        
        valid_agents = np.array(valid_agents)
        
        # Restore susceptibility to 1.0 (fully susceptible)
        people.susceptibility[valid_agents] = 1.0
        
        self.stats['immunity_expired'] += len(valid_agents)
        self.stats['agents_currently_protected'] -= len(valid_agents)
        
        if self.verbose and len(valid_agents) > 0:
            print(f"Tick {tick}: Expired maternal immunity for {len(valid_agents)} agents, "
                  f"now susceptible to infection")
        
        # Emit maternal immunity end event
        if self.has_subscribers('maternal_immunity_end'):
            self.emit_event(
                event_type='maternal_immunity_end',
                tick=tick,
                data={
                    'agent_indices': valid_agents.tolist(),
                    'num_expired': len(valid_agents),
                    'agents_still_protected': self.stats['agents_currently_protected']
                }
            )
    
    def get_stats(self) -> dict:
        """Get maternal immunity statistics."""
        stats = self.stats.copy()
        stats['pending_expirations'] = sum(len(agents) for agents in self.immunity_schedule.values())
        return stats
    
    def initialize(self, model: ABMModel) -> None:
        """Initialize component - called before simulation starts."""
        pass