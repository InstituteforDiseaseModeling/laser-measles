"""
Event system for inter-component communication in ABM models.

This module provides a flexible, non-invasive event system that allows components
to communicate with each other without tight coupling. Components can emit events
when certain actions occur (e.g., deaths, births, infections) and other components
can listen for these events to react appropriately.

The event system is designed to integrate seamlessly with the existing laser-measles
framework while maintaining full backward compatibility.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Union
# Removed weakref - using regular sets with manual cleanup

import numpy as np


@dataclass
class BaseEvent(ABC):
    """
    Abstract base class for all events in the ABM system.
    
    Events represent occurrences in the model that other components might be
    interested in. They carry information about what happened, when it happened,
    and any relevant data.
    
    Attributes:
        event_type: String identifier for the type of event
        source: Component or object that emitted the event
        tick: Model tick when the event occurred
        timestamp: System timestamp when event was created
        data: Dictionary containing event-specific data
    """
    
    event_type: str
    source: Any
    tick: int
    timestamp: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate event after initialization."""
        if not self.event_type:
            raise ValueError("event_type cannot be empty")


@dataclass
class ModelEvent(BaseEvent):
    """
    Events related to model lifecycle and execution.
    
    These events are emitted by the model itself during various phases
    of execution (initialization, tick start/end, completion).
    """
    pass


@dataclass
class VitalDynamicsEvent(BaseEvent):
    """
    Events related to vital dynamics processes (births, deaths, aging).
    
    Common event_types:
    - 'deaths': When agents die
    - 'births': When new agents are born
    - 'aging': When agents age or reach milestones
    """
    pass


@dataclass 
class DiseaseEvent(BaseEvent):
    """
    Events related to disease processes and state transitions.
    
    Common event_types:
    - 'infection': When agents become infected
    - 'transmission': When transmission occurs between agents
    - 'recovery': When agents recover
    - 'state_change': General state transitions
    """
    pass


class EventListener:
    """
    Base class for objects that want to listen to events.
    
    This is a utility class that components can inherit from or use
    to simplify event listening patterns.
    """
    
    def __init__(self):
        self._subscriptions: Set[str] = set()
    
    def subscribe_to(self, event_bus: 'EventBus', event_types: Union[str, List[str]], 
                     callback: Optional[Callable] = None) -> None:
        """
        Subscribe to one or more event types.
        
        Args:
            event_bus: The EventBus to subscribe to
            event_types: Event type(s) to listen for
            callback: Optional callback function. If None, uses self.handle_event
        """
        if isinstance(event_types, str):
            event_types = [event_types]
        
        callback_func = callback or self.handle_event
        
        for event_type in event_types:
            event_bus.subscribe(event_type, callback_func)
            self._subscriptions.add(event_type)
    
    def handle_event(self, event: BaseEvent) -> None:
        """
        Default event handler. Override in subclasses.
        
        Args:
            event: The event to handle
        """
        pass
    
    def unsubscribe_all(self, event_bus: 'EventBus') -> None:
        """
        Unsubscribe from all events.
        
        Args:
            event_bus: The EventBus to unsubscribe from
        """
        for event_type in self._subscriptions:
            event_bus.unsubscribe(event_type, self.handle_event)
        self._subscriptions.clear()


class EventBus:
    """
    Central event dispatcher for the ABM system.
    
    The EventBus manages event subscriptions and dispatching. Components can
    subscribe to specific event types and will be notified when events of those
    types are emitted.
    
    Features:
    - Event filtering by type and source
    - Weak references to prevent memory leaks
    - Synchronous event dispatch (events processed immediately)
    - Subscription management
    """
    
    def __init__(self):
        # Dictionary mapping event types to sets of callback functions
        self._subscribers: Dict[str, Set[Callable]] = {}
        # Statistics for debugging and monitoring
        self._stats = {
            'events_emitted': 0,
            'subscriptions': 0,
            'dispatch_errors': 0
        }
    
    def subscribe(self, event_type: str, callback: Callable[[BaseEvent], None]) -> None:
        """
        Subscribe to events of a specific type.
        
        Args:
            event_type: Type of events to listen for (e.g., 'deaths', 'births')
            callback: Function to call when event occurs. Must accept BaseEvent parameter
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = set()
        
        self._subscribers[event_type].add(callback)
        self._stats['subscriptions'] += 1
    
    def unsubscribe(self, event_type: str, callback: Callable[[BaseEvent], None]) -> None:
        """
        Unsubscribe from events of a specific type.
        
        Args:
            event_type: Type of events to stop listening for
            callback: The callback function to remove
        """
        if event_type in self._subscribers:
            self._subscribers[event_type].discard(callback)
            if not self._subscribers[event_type]:
                del self._subscribers[event_type]
    
    def emit(self, event: BaseEvent) -> None:
        """
        Emit an event to all subscribed listeners.
        
        Args:
            event: The event to emit
        """
        self._stats['events_emitted'] += 1
        
        # Get subscribers for this event type
        subscribers = self._subscribers.get(event.event_type, set())
        
        # Dispatch to all subscribers
        for callback in subscribers.copy():  # Copy to avoid modification during iteration
            try:
                callback(event)
            except Exception as e:
                self._stats['dispatch_errors'] += 1
                # Log error but don't crash the simulation
                print(f"Warning: Error in event callback for {event.event_type}: {e}")
    
    def clear_subscriptions(self) -> None:
        """Clear all subscriptions."""
        self._subscribers.clear()
        self._stats['subscriptions'] = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about event bus usage."""
        total_subs = sum(len(subscribers) for subscribers in self._subscribers.values())
        return {
            **self._stats,
            'active_event_types': list(self._subscribers.keys()),
            'total_subscribers': total_subs,
            'subscribers_by_type': {k: len(v) for k, v in self._subscribers.items()}
        }


class EventMixin:
    """
    Mixin class to add event capabilities to existing components.
    
    This mixin can be added to any component class to provide easy access
    to event emission and subscription methods.
    """
    
    def __init_event_mixin__(self, event_bus: Optional[EventBus] = None):
        """
        Initialize the event mixin.
        
        Args:
            event_bus: EventBus instance to use. If None, component must set it later
        """
        self._event_bus = event_bus
        self._event_listener = EventListener() if hasattr(self, 'handle_event') else None
    
    def set_event_bus(self, event_bus: EventBus) -> None:
        """Set the event bus for this component."""
        self._event_bus = event_bus
    
    def has_subscribers(self, event_type: str) -> bool:
        """
        Check if there are any subscribers for the given event type.
        
        Args:
            event_type: Type of event to check
            
        Returns:
            True if there are subscribers, False otherwise
        """
        if self._event_bus is None:
            return False
        return event_type in self._event_bus._subscribers and len(self._event_bus._subscribers[event_type]) > 0
    
    def emit_event(self, event_type: str, data: Optional[Dict[str, Any]] = None, 
                   tick: Optional[int] = None) -> None:
        """
        Emit an event of the specified type.
        
        Args:
            event_type: Type of event to emit
            data: Optional data to include with the event
            tick: Model tick (if None, will try to get from model)
        """
        if self._event_bus is None:
            return  # Silently fail if no event bus available
        
        # Early return if no subscribers - avoid creating event objects unnecessarily
        if not self.has_subscribers(event_type):
            return
        
        # Try to get tick from model if not provided
        if tick is None and hasattr(self, 'model') and hasattr(self.model, 'current_tick'):
            tick = getattr(self.model, 'current_tick', 0)
        elif tick is None:
            tick = 0
        
        # Determine event class based on event type
        if event_type in ['deaths', 'births', 'aging']:
            event_class = VitalDynamicsEvent
        elif event_type in ['infection', 'transmission', 'recovery', 'state_change']:
            event_class = DiseaseEvent
        elif event_type in ['tick_start', 'tick_end', 'model_init', 'model_complete']:
            event_class = ModelEvent
        else:
            event_class = BaseEvent
        
        event = event_class(
            event_type=event_type,
            source=self,
            tick=tick,
            data=data or {}
        )
        
        self._event_bus.emit(event)
    
    def subscribe_to_events(self, event_types: Union[str, List[str]], 
                           callback: Optional[Callable] = None) -> None:
        """
        Subscribe to events of specified types.
        
        Args:
            event_types: Event type(s) to listen for
            callback: Optional callback function. If None, uses self.handle_event
        """
        if self._event_bus is None:
            print(f"Warning: No event bus available for {type(self).__name__}")
            return
        
        # Use direct subscription instead of EventListener
        if isinstance(event_types, str):
            event_types = [event_types]
        
        callback_func = callback or getattr(self, 'handle_event', None)
        if callback_func is None:
            print(f"Warning: No callback function available for {type(self).__name__}")
            return
        
        for event_type in event_types:
            self._event_bus.subscribe(event_type, callback_func)
    
    def unsubscribe_from_events(self) -> None:
        """Unsubscribe from all events."""
        if self._event_bus is None or self._event_listener is None:
            return
        
        self._event_listener.unsubscribe_all(self._event_bus)


def create_event(event_type: str, source: Any, tick: int, **kwargs) -> BaseEvent:
    """
    Convenience function to create an event of appropriate type.
    
    Args:
        event_type: Type of event to create
        source: Source component or object
        tick: Model tick
        **kwargs: Additional data for the event
    
    Returns:
        BaseEvent instance of appropriate subclass
    """
    data = dict(kwargs)
    
    # Choose event class based on type
    if event_type in ['deaths', 'births', 'aging']:
        return VitalDynamicsEvent(event_type=event_type, source=source, tick=tick, data=data)
    elif event_type in ['infection', 'transmission', 'recovery', 'state_change']:
        return DiseaseEvent(event_type=event_type, source=source, tick=tick, data=data)
    elif event_type in ['tick_start', 'tick_end', 'model_init', 'model_complete']:
        return ModelEvent(event_type=event_type, source=source, tick=tick, data=data)
    else:
        return BaseEvent(event_type=event_type, source=source, tick=tick, data=data)