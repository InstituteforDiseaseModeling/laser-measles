# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

laser-measles is a spatial measles modeling framework built on the LASER toolkit. It provides three modeling approaches:

1. **ABM (Agent-Based Model)**: Individual-level simulation with stochastic agents
2. **Biweekly Compartmental Model**: Population-level SEIR dynamics with 2-week timesteps
3. **Compartmental Model**: Population-level SEIR dynamics with daily timesteps

## Common Development Commands

### Testing

Use pytest for writing tests.

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov --cov-report=term-missing --cov-report=xml -vv tests

# Run specific test patterns
pytest tests/test_*.py
pytest -m "not slow"  # Skip slow tests
```

### Code Quality
```bash
# Lint code
ruff check src/ tests/

# Format code
ruff format src/ tests/

# Type checking (if mypy/pyright available)
mypy src/
pyright src/
```

### Documentation

Use google docstring convections.

```bash
# Generate API documentation
tox -e docs

# Build documentation locally
sphinx-build -b html docs dist/docs
```

### Environment Management
```bash
# Install dependencies with uv
uv pip install -e .

# Install development dependencies
uv pip install -e ".[dev]"

# Install full dependencies (including examples)
uv pip install -e ".[full]"
```

## Architecture Overview

### Core Base Classes

- **BaseLaserModel**: Abstract base class for all model types with common functionality:
  - Component management (`components`, `instances`, `phases`)
  - Execution loop with timing metrics
  - Initialization and cleanup
  - Progress tracking with `alive_progress`

- **BaseComponent**: Base class for all components with standardized interface:
  - `__init__(model, verbose=False)` constructor
  - `initialize(model)` method called before simulation
  - Optional `__call__(model, tick)` method for phases

- **BasePhase**: Components that execute every tick (inherit from BaseComponent)

### Model Types

Each model type has its own directory with similar structure:
- `base.py`: Model-specific base classes and LaserFrame definitions
- `model.py`: Main model implementation
- `params.py`: Pydantic parameter classes
- `components/`: Model-specific component implementations

### Component Architecture

Components are organized into categories:
- **Process components**: Modify model state (births, deaths, infection, transmission)
- **Tracker components**: Record metrics and state over time
- **Initialization components**: Set up initial conditions

### Base Components (New Architecture)

The `base_components/` directory contains abstract base classes that define:
- Common parameters using Pydantic BaseModel
- Shared functionality (seasonality, spatial mixing)
- Abstract interfaces for model-specific implementations

Key base components:
- `base_transmission.py`: Base transmission/infection logic
- `base_vital_dynamics.py`: Base births/deaths logic
- `base_importation.py`: Base importation pressure logic
- `base_tracker.py`: Base tracking/metrics logic

## Key Patterns

### Component Creation
```python
# Simple component
model.components = [ComponentClass1, ComponentClass2]

# Component with parameters using component decorator
@component(param1=10, param2=20)
class MyComponent(BaseComponent):
    def __init__(self, model, verbose=False, param1=1, param2=2):
        super().__init__(model, verbose)
        self.param1 = param1
        self.param2 = param2

# Component with Pydantic parameters
component_instance = create_component(ComponentClass, params=ParamsModel())
```

### Model Execution
```python
# Standard model execution pattern
model = ModelClass(scenario, params, name="model_name")
model.components = [Component1, Component2]
model.run()  # Runs for params.num_ticks

# Access component instances
state_tracker = model.get_instance(StateTracker)[0]
```

### LaserFrame Usage
- **BasePatchLaserFrame**: For patch-level data with `states` property (StateArray)
- **BasePeopleLaserFrame**: For agent-level data with factory methods
- StateArray provides attribute access to disease states (S, E, I, R)

## Development Guidelines

### Component Development
1. Inherit from appropriate base class (BaseComponent or BasePhase)
2. Use Pydantic for parameter validation
3. Implement `initialize()` for setup that depends on other components
4. Use `__call__(model, tick)` for per-tick execution
5. Include docstrings following Google style format

### Testing
- Unit tests in `tests/unit/`
- Integration tests in `tests/integration/`
- Scientific validation tests in `tests/scientific/`
- Use pytest fixtures for common setup
- Mark slow tests with `@pytest.mark.slow`

### Parameter Management
- Use Pydantic BaseModel for all parameter classes
- Inherit from base parameter classes when possible
- Include field descriptions and validation
- Default values should be epidemiologically reasonable

### Model-Specific Notes

#### ABM Models
- Use `BasePeopleLaserFrame` for agent data
- Individual-level stochastic processes
- Agents have properties like `patch_id`, `age`, `state`

#### Biweekly Models
- Use 2-week timesteps (26 ticks per year)
- Compartmental dynamics with binomial sampling
- Recommended for scenario building and policy analysis

#### Compartmental Models  
- Use daily timesteps (365 ticks per year)
- SEIR dynamics with detailed temporal resolution
- Recommended for parameter estimation and outbreak modeling

## CLI Commands

The package provides multiple CLI entry points:
- `cli`: Main laser-measles CLI
- `nigeria`: Nigeria-specific model CLI
- `measles`: ABM-specific CLI

## Dependencies

Core dependencies:
- `laser-core>=0.5.1`: Core LASER framework
- `pydantic>=2.11.5`: Parameter validation
- `polars>=1.30.0`: DataFrame operations
- `alive-progress>=3.2.0`: Progress bars
- `typer>=0.12.0`: CLI framework

## Common Issues

1. **Memory management**: Use `model.cleanup()` after large simulations
2. **Component order**: Some components depend on others being initialized first
3. **Parameter validation**: Pydantic will raise errors for invalid parameters
4. **LaserFrame capacity**: Pre-allocate sufficient capacity for dynamic populations

## File Naming Conventions

- Components: `process_*.py` for processes, `tracker_*.py` for trackers
- Tests: `test_*.py` pattern
- Use snake_case for Python files
- Use PascalCase for class names