"""
Tests for the BaseComponent class and its functionality.
"""
import pytest
from unittest.mock import Mock

from laser_measles.base import BaseComponent


class MockModel:
    """Mock model class for testing."""
    pass


class TestComponentWithDocstring(BaseComponent):
    """
    A test component with a custom docstring.
    
    This component is used for testing the __str__ method
    functionality with child class docstrings.
    """
    
    def __call__(self, model, tick: int) -> None:
        return None


class TestComponentWithoutDocstring(BaseComponent):
    
    def __call__(self, model, tick: int) -> None:
        return None


class TestBaseComponent:
    """Test suite for BaseComponent class."""
    
    def test_initialization_default_params(self):
        """Test BaseComponent initialization with default parameters."""
        model = MockModel()
        component = BaseComponent(model)
        
        assert component.model is model
        assert component.verbose is False
        assert component.initialized is True
    
    def test_initialization_with_verbose(self):
        """Test BaseComponent initialization with verbose=True."""
        model = MockModel()
        component = BaseComponent(model, verbose=True)
        
        assert component.model is model
        assert component.verbose is True
        assert component.initialized is True
    
    def test_call_method_exists(self):
        """Test that __call__ method exists and can be called."""
        model = MockModel()
        component = BaseComponent(model)
        
        # Should not raise an exception (returns ellipsis/None)
        result = component(model, 0)
        assert result is ...
    
    def test_plot_method_exists(self):
        """Test that plot method exists and returns a generator."""
        model = MockModel()
        component = BaseComponent(model)
        
        plot_gen = component.plot()
        assert hasattr(plot_gen, '__next__')  # Is a generator
        
        # Should yield None
        result = next(plot_gen)
        assert result is None
    
    def test_plot_method_with_figure(self):
        """Test that plot method accepts a figure parameter."""
        model = MockModel()
        component = BaseComponent(model)
        
        mock_fig = Mock()
        plot_gen = component.plot(mock_fig)
        result = next(plot_gen)
        assert result is None
    
    def test_str_method_base_class(self):
        """Test __str__ method returns base class docstring."""
        model = MockModel()
        component = BaseComponent(model)
        
        str_repr = str(component)
        expected = ("Base class for all laser-measles components.\n    \n    "
                   "Components follow a uniform interface with __call__(model, tick) method\n    "
                   "for execution during simulation loops.")
        assert str_repr == expected
    
    def test_str_method_child_class_with_docstring(self):
        """Test __str__ method returns child class docstring when available."""
        model = MockModel()
        component = TestComponentWithDocstring(model)
        
        str_repr = str(component)
        expected = ("A test component with a custom docstring.\n    \n    "
                   "This component is used for testing the __str__ method\n    "
                   "functionality with child class docstrings.")
        assert str_repr == expected
    
    def test_str_method_child_class_without_docstring(self):
        """Test __str__ method falls back to base class docstring when child has none."""
        model = MockModel()
        component = TestComponentWithoutDocstring(model)
        
        str_repr = str(component)
        expected = ("Base class for all laser-measles components.\n    \n    "
                   "Components follow a uniform interface with __call__(model, tick) method\n    "
                   "for execution during simulation loops.")
        assert str_repr == expected
    
    def test_str_method_no_docstring_fallback(self):
        """Test __str__ method fallback when no docstring is available."""
        model = MockModel()
        component = BaseComponent(model)
        
        # Temporarily remove docstring
        original_doc = BaseComponent.__doc__
        BaseComponent.__doc__ = None
        
        try:
            str_repr = str(component)
            assert str_repr == "BaseComponent component"
        finally:
            # Restore original docstring
            BaseComponent.__doc__ = original_doc


class TestBaseComponentIntegration:
    """Integration tests for BaseComponent with actual laser-measles components."""
    
    def test_import_from_new_location(self):
        """Test that BaseComponent can be imported from the new location."""
        from laser_measles.base import BaseComponent as ImportedBaseComponent
        
        model = MockModel()
        component = ImportedBaseComponent(model)
        assert isinstance(component, BaseComponent)
    
    def test_biweekly_components_still_work(self):
        """Test that existing biweekly components still work with refactored BaseComponent."""
        try:
            # Test importing some biweekly components
            from laser_measles.biweekly.components.tracker_state import StateTracker
            from laser_measles.biweekly.components.process_infection import InfectionProcess
            
            model = MockModel()
            
            # These should not raise import errors
            assert hasattr(StateTracker, '__init__')
            assert hasattr(InfectionProcess, '__init__')
            
        except ImportError as e:
            pytest.fail(f"Failed to import biweekly components: {e}")
    
    def test_components_utility_still_works(self):
        """Test that the components utility module still works with the refactored BaseComponent."""
        try:
            from laser_measles.components import component, create_component
            from laser_measles.base import BaseComponent
            
            # These should not raise import errors
            assert callable(component)
            assert callable(create_component)
            
        except ImportError as e:
            pytest.fail(f"Failed to import components utilities: {e}")


if __name__ == "__main__":
    pytest.main([__file__])