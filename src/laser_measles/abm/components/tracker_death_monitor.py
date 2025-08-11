"""
Example component that demonstrates event listening.

This component listens to death events from the vital dynamics process
and tracks death-related metrics and indices. This serves as an example
of how components can use the event system for inter-component communication.
"""

from typing import Dict, List, Optional, Set
import numpy as np
from pydantic import BaseModel, Field

from laser_measles.base import BaseComponent
from laser_measles.events import EventMixin, BaseEvent, VitalDynamicsEvent


class DeathMonitorParams(BaseModel):
    """Parameters for the DeathMonitor tracker."""
    
    track_death_locations: bool = Field(default=True, description="Whether to track patch locations of deaths")
    track_death_states: bool = Field(default=True, description="Whether to track disease states at death")
    verbose_deaths: bool = Field(default=False, description="Whether to print death information")


class DeathMonitorTracker(BaseComponent, EventMixin):
    """
    Example tracker that monitors death events and maintains death-related metrics.
    
    This component demonstrates how to use the event system to listen for
    vital dynamics events and react to them. It tracks:
    - Total deaths over time
    - Deaths by patch location
    - Deaths by disease state
    - Indices of agents who died (for other components to access)
    
    Other components can access the death indices through this tracker
    to perform their own death-related processing.
    """
    
    def __init__(self, model, verbose: bool = False, params: Optional[DeathMonitorParams] = None):
        """
        Initialize the death monitor tracker.
        
        Args:
            model: The ABM model instance
            verbose: Whether to print verbose output
            params: Parameters for the tracker
        """
        super().__init__(model, verbose=verbose)
        self.params = params or DeathMonitorParams()
        
        # Initialize event mixin
        self.__init_event_mixin__()
        
        # Initialize tracking data structures
        self.reset_metrics()
        
        # Subscribe to death events when event bus becomes available
        self._subscribed = False
    
    def reset_metrics(self):
        """Reset all tracking metrics."""
        # Track deaths over time (tick -> number of deaths)
        self.deaths_by_tick: Dict[int, int] = {}
        
        # Track deaths by patch location
        self.deaths_by_patch: Dict[int, int] = {}
        
        # Track deaths by disease state
        self.deaths_by_state: Dict[int, int] = {}
        
        # Store indices of agents who died (for access by other components)
        self.recent_death_indices: List[int] = []
        self.all_death_indices: Set[int] = set()
        
        # Total death count
        self.total_deaths: int = 0
    
    def set_event_bus(self, event_bus):
        """
        Set the event bus and subscribe to death events.
        
        This method is called by the model when setting up components.
        """
        self._event_bus = event_bus
        if not self._subscribed:
            self.subscribe_to_events(['deaths'], self.handle_death_event)
            self._subscribed = True
            if self.verbose:
                print("DeathMonitorTracker: Subscribed to death events")
    
    def handle_death_event(self, event: BaseEvent):
        """
        Handle death events from vital dynamics components.
        
        Args:
            event: The death event containing agent indices and metadata
        """
        if not isinstance(event, VitalDynamicsEvent) or event.event_type != 'deaths':
            return
        
        # Extract death data from the event
        data = event.data
        tick = event.tick
        agent_indices = data.get('agent_indices', [])
        patch_ids = data.get('patch_ids', [])
        states = data.get('states', [])
        num_deaths = data.get('num_deaths', len(agent_indices))
        
        # Update metrics
        self.deaths_by_tick[tick] = self.deaths_by_tick.get(tick, 0) + num_deaths
        self.total_deaths += num_deaths
        
        # Track death locations if enabled
        if self.params.track_death_locations:
            for patch_id in patch_ids:
                self.deaths_by_patch[patch_id] = self.deaths_by_patch.get(patch_id, 0) + 1
        
        # Track death states if enabled
        if self.params.track_death_states:
            for state in states:
                self.deaths_by_state[state] = self.deaths_by_state.get(state, 0) + 1
        
        # Store death indices for other components to access
        self.recent_death_indices = list(agent_indices)
        self.all_death_indices.update(agent_indices)
        
        # Verbose output if enabled
        if self.params.verbose_deaths or self.verbose:
            print(f"Tick {tick}: {num_deaths} deaths occurred")
            if self.params.track_death_locations and patch_ids:
                patch_counts = {}
                for pid in patch_ids:
                    patch_counts[pid] = patch_counts.get(pid, 0) + 1
                print(f"  Deaths by patch: {patch_counts}")
            if self.params.track_death_states and states:
                state_counts = {}
                for state in states:
                    state_counts[state] = state_counts.get(state, 0) + 1
                print(f"  Deaths by state: {state_counts}")
    
    def get_recent_deaths(self) -> List[int]:
        """
        Get indices of agents who died in the most recent death event.
        
        Returns:
            List of agent indices who died recently
        """
        return self.recent_death_indices.copy()
    
    def get_all_deaths(self) -> Set[int]:
        """
        Get indices of all agents who have died throughout the simulation.
        
        Returns:
            Set of all agent indices who have died
        """
        return self.all_death_indices.copy()
    
    def get_death_summary(self) -> Dict:
        """
        Get a summary of death statistics.
        
        Returns:
            Dictionary containing death statistics
        """
        return {
            'total_deaths': self.total_deaths,
            'deaths_by_tick': dict(self.deaths_by_tick),
            'deaths_by_patch': dict(self.deaths_by_patch) if self.params.track_death_locations else {},
            'deaths_by_state': dict(self.deaths_by_state) if self.params.track_death_states else {},
            'num_death_events': len(self.deaths_by_tick),
            'recent_death_count': len(self.recent_death_indices)
        }
    
    def __call__(self, model, tick: int) -> None:
        """
        Optional tick processing. 
        
        This tracker is event-driven, so it doesn't need to do anything
        on each tick, but this method is available if needed.
        """
        # Clear recent deaths at the start of each tick
        # so they don't accumulate across ticks
        if tick > 0:  # Don't clear on first tick
            self.recent_death_indices.clear()
    
    def _initialize(self, model) -> None:
        """
        Initialize the tracker.
        
        Called by the model during initialization phase.
        """
        if self.verbose:
            print(f"DeathMonitorTracker initialized with params: {self.params}")
        
        # Reset metrics at initialization
        self.reset_metrics()


# Example of another component that uses death information
class DeathAwareComponent(BaseComponent, EventMixin):
    """
    Example component that demonstrates accessing death information.
    
    This component shows how other components can listen to death events
    or access death information from the DeathMonitorTracker.
    """
    
    def __init__(self, model, verbose: bool = False):
        super().__init__(model, verbose=verbose)
        self.__init_event_mixin__()
        self.death_reactions = 0
    
    def set_event_bus(self, event_bus):
        """Subscribe to death events."""
        self._event_bus = event_bus
        self.subscribe_to_events(['deaths'], self.react_to_deaths)
        if self.verbose:
            print("DeathAwareComponent: Subscribed to death events")
    
    def react_to_deaths(self, event: BaseEvent):
        """
        React to death events - example processing.
        
        Args:
            event: The death event
        """
        if event.event_type == 'deaths':
            self.death_reactions += 1
            num_deaths = event.data.get('num_deaths', 0)
            
            if self.verbose:
                print(f"DeathAwareComponent: Reacting to {num_deaths} deaths (reaction #{self.death_reactions})")
            
            # Example: Could adjust behavior based on deaths
            # For instance, increase surveillance in patches with deaths
            patch_ids = event.data.get('patch_ids', [])
            if patch_ids and self.verbose:
                unique_patches = set(patch_ids)
                print(f"  Increasing surveillance in patches: {sorted(unique_patches)}")
    
    def get_reaction_count(self) -> int:
        """Get the number of times this component reacted to deaths."""
        return self.death_reactions