"""Base tracker component for laser_measles models."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

from ..base import BaseComponent


class BaseTrackerParams(BaseModel):
    """Common parameters for tracker components."""
    
    track_states: bool = Field(
        default=True,
        description="Whether to track disease states"
    )
    
    track_population: bool = Field(
        default=True,
        description="Whether to track population dynamics"
    )
    
    output_frequency: int = Field(
        default=1,
        description="Frequency of output (every N time steps)",
        gt=0
    )


class BaseTracker(BaseComponent, ABC):
    """Abstract base class for tracker components."""
    
    def __init__(self, model, verbose: bool = False, params: Optional[BaseTrackerParams] = None):
        super().__init__(model, verbose)
        self.params = params if params is not None else BaseTrackerParams()
        self.data = {}  # Storage for tracked data
    
    @abstractmethod
    def initialize(self, model):
        """Initialize tracker with model."""
        pass
    
    @abstractmethod
    def update(self, model, tick: int):
        """Update tracker with current model state."""
        pass
    
    def should_track(self, tick: int) -> bool:
        """Check if we should track data at this time step."""
        return tick % self.params.output_frequency == 0
    
    def get_data(self) -> Dict[str, Any]:
        """Get all tracked data."""
        return self.data
    
    def reset(self):
        """Reset tracked data."""
        self.data = {}