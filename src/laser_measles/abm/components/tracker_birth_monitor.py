"""
Example component that demonstrates event listening for births.

This component listens to birth events from the vital dynamics process
and tracks birth-related metrics and indices. This serves as an example
of how components can use the event system for inter-component communication.
"""

from typing import Dict, List, Optional, Set
import numpy as np
from pydantic import BaseModel, Field

from laser_measles.base import BaseComponent
from laser_measles.events import EventMixin, BaseEvent, VitalDynamicsEvent


class BirthMonitorParams(BaseModel):
    """Parameters for the BirthMonitor tracker."""
    
    track_birth_locations: bool = Field(default=True, description="Whether to track patch locations of births")
    verbose_births: bool = Field(default=False, description="Whether to print birth information")


class BirthMonitorTracker(BaseComponent, EventMixin):
    """
    Example tracker that monitors birth events and maintains birth-related metrics.
    
    This component demonstrates how to use the event system to listen for
    vital dynamics events and react to them. It tracks:
    - Total births over time
    - Births by patch location
    - Indices of agents who were born (for other components to access)
    
    Other components can access the birth indices through this tracker
    to perform their own birth-related processing.
    """
    
    def __init__(self, model, verbose: bool = False, params: Optional[BirthMonitorParams] = None):
        """
        Initialize the birth monitor tracker.
        
        Args:
            model: The ABM model instance
            verbose: Whether to print verbose output
            params: Parameters for the tracker
        """
        super().__init__(model, verbose=verbose)
        self.params = params or BirthMonitorParams()
        
        # Initialize event mixin
        self.__init_event_mixin__()
        
        # Initialize tracking data structures
        self.reset_metrics()
        
        # Subscribe to birth events when event bus becomes available
        self._subscribed = False
    
    def reset_metrics(self):
        """Reset all tracking metrics."""
        # Track births over time (tick -> number of births)
        self.births_by_tick: Dict[int, int] = {}
        
        # Track births by patch location
        self.births_by_patch: Dict[int, int] = {}
        
        # Store indices of agents who were born (for access by other components)
        self.recent_birth_indices: List[int] = []
        self.all_birth_indices: Set[int] = set()
        
        # Total birth count
        self.total_births: int = 0
    
    def set_event_bus(self, event_bus):
        """
        Set the event bus and subscribe to birth events.
        
        This method is called by the model when setting up components.
        """
        self._event_bus = event_bus
        if not self._subscribed:
            self.subscribe_to_events(['births'], self.handle_birth_event)
            self._subscribed = True
            if self.verbose:
                print("BirthMonitorTracker: Subscribed to birth events")
    
    def handle_birth_event(self, event: BaseEvent):
        """
        Handle birth events from vital dynamics components.
        
        Args:
            event: The birth event containing agent indices and metadata
        """
        if not isinstance(event, VitalDynamicsEvent) or event.event_type != 'births':
            return
        
        # Extract birth data from the event
        data = event.data
        tick = event.tick
        agent_indices = data.get('agent_indices', [])
        patch_births = data.get('patch_births', [])
        total_births = data.get('total_births', len(agent_indices))
        
        # Update metrics
        self.births_by_tick[tick] = self.births_by_tick.get(tick, 0) + total_births
        self.total_births += total_births
        
        # Track birth locations if enabled
        if self.params.track_birth_locations:
            for patch_id, patch_birth_count in enumerate(patch_births):
                if patch_birth_count > 0:
                    self.births_by_patch[patch_id] = self.births_by_patch.get(patch_id, 0) + patch_birth_count
        
        # Store birth indices for other components to access
        self.recent_birth_indices = list(agent_indices)
        self.all_birth_indices.update(agent_indices)
        
        # Verbose output if enabled
        if self.params.verbose_births or self.verbose:
            print(f"Tick {tick}: {total_births} births occurred")
            if self.params.track_birth_locations and patch_births:
                patch_counts = {pid: count for pid, count in enumerate(patch_births) if count > 0}
                print(f"  Births by patch: {patch_counts}")
    
    def get_recent_births(self) -> List[int]:
        """
        Get indices of agents who were born in the most recent birth event.
        
        Returns:
            List of agent indices who were born recently
        """
        return self.recent_birth_indices.copy()
    
    def get_all_births(self) -> Set[int]:
        """
        Get indices of all agents who have been born throughout the simulation.
        
        Returns:
            Set of all agent indices who have been born
        """
        return self.all_birth_indices.copy()
    
    def get_birth_summary(self) -> Dict:
        """
        Get a summary of birth statistics.
        
        Returns:
            Dictionary containing birth statistics
        """
        return {
            'total_births': self.total_births,
            'births_by_tick': dict(self.births_by_tick),
            'births_by_patch': dict(self.births_by_patch) if self.params.track_birth_locations else {},
            'num_birth_events': len(self.births_by_tick),
            'recent_birth_count': len(self.recent_birth_indices)
        }
    
    def __call__(self, model, tick: int) -> None:
        """
        Optional tick processing. 
        
        This tracker is event-driven, so it doesn't need to do anything
        on each tick, but this method is available if needed.
        """
        # Clear recent births at the start of each tick
        # so they don't accumulate across ticks
        if tick > 0:  # Don't clear on first tick
            self.recent_birth_indices.clear()
    
    def _initialize(self, model) -> None:
        """
        Initialize the tracker.
        
        Called by the model during initialization phase.
        """
        if self.verbose:
            print(f"BirthMonitorTracker initialized with params: {self.params}")
        
        # Reset metrics at initialization
        self.reset_metrics()


# Example of another component that uses birth information
class BirthAwareComponent(BaseComponent, EventMixin):
    """
    Example component that demonstrates accessing birth information.
    
    This component shows how other components can listen to birth events
    or access birth information from the BirthMonitorTracker.
    """
    
    def __init__(self, model, verbose: bool = False):
        super().__init__(model, verbose=verbose)
        self.__init_event_mixin__()
        self.birth_reactions = 0
    
    def set_event_bus(self, event_bus):
        """Subscribe to birth events."""
        self._event_bus = event_bus
        self.subscribe_to_events(['births'], self.react_to_births)
        if self.verbose:
            print("BirthAwareComponent: Subscribed to birth events")
    
    def react_to_births(self, event: BaseEvent):
        """
        React to birth events - example processing.
        
        Args:
            event: The birth event
        """
        if event.event_type == 'births':
            self.birth_reactions += 1
            total_births = event.data.get('total_births', 0)
            
            if self.verbose:
                print(f"BirthAwareComponent: Reacting to {total_births} births (reaction #{self.birth_reactions})")
            
            # Example: Could adjust behavior based on births
            # For instance, prepare vaccination supplies in patches with births
            patch_births = event.data.get('patch_births', [])
            if patch_births and self.verbose:
                patches_with_births = [i for i, count in enumerate(patch_births) if count > 0]
                print(f"  Preparing resources in patches: {patches_with_births}")
    
    def get_reaction_count(self) -> int:
        """Get the number of times this component reacted to births."""
        return self.birth_reactions