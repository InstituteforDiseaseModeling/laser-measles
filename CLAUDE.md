# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

laser-measles is a spatial epidemiological modeling toolkit for measles transmission dynamics, built on the LASER framework. It provides three different model types and a flexible component-based architecture for disease simulation.

## Common Development Commands

### Testing
```bash
# Run all tests
pytest
pytest -vv  # verbose output

# Run tests with coverage
pytest --cov --cov-report=term-missing

# Run specific test file
pytest tests/test_core.py

# Run using tox (full test matrix)
tox
```

### Code Quality
```bash
# Lint and format code
ruff check .
ruff format .

# Run pre-commit hooks
pre-commit run --all-files
```

### Documentation

Use google style docstrings.

```bash
# Build documentation (requires tox)
tox -e docs

# The docs environment automatically installs pandoc and builds HTML docs
```

### Package Management
```bash
# Install in development mode with dependencies
pip install -e .

# Install with development dependencies
pip install -e .[dev]

# Install with documentation dependencies
pip install -e .[docs]

# Install with example notebook dependencies
pip install -e .[examples]
```

## Architecture Overview

### Core Model Types
- **Generic Model**: General-purpose implementation for any geographic region (uses Washington State as example)
- **Nigeria Model**: Specialized for Northern Nigeria with pre-configured demographic data
- **Biweekly Model**: Compartmental (SIR-style) approach using Polars DataFrames for performance

### Component Architecture
Models are composed of interchangeable components that implement specific disease dynamics:
- `Births`, `NonDiseaseDeaths`, `MaternalAntibodies`, `Susceptibility`
- `RoutineImmunization`, `Infection`, `Incubation`, `Transmission`
- Components follow a uniform `__call__(model, tick)` interface

### Key Data Structures
- **LaserFrame**: High-performance array-based structure for agent populations
- **PropertySet**: Parameter management system for model configuration
- **Demographics System**: Handles shapefiles, raster data, and administrative boundaries

### Execution Flow
1. Load scenario data (demographics + geography)
2. Initialize model with LaserFrame and components
3. Run simulation loop (component ticks)
4. Collect metrics and generate visualizations

## Entry Points
- `cli`: Main CLI interface
- `nigeria`: Nigeria-specific model runner
- `measles`: Generic model runner

## Key Directories
- `src/laser_measles/`: Core model implementation
- `src/laser_measles/biweekly/`: Compartmental model implementation
- `src/laser_measles/demographics/`: Geographic data handling
- `src/laser_measles/generic/`: General-purpose model
- `src/laser_measles/nigeria/`: Nigeria-specific model
- `tests/`: Test suite
- `docs/`: Documentation and tutorials
- `examples/`: Example scenarios and data

## Dependencies
Built on `laser-core` with key dependencies: `pydantic`, `polars`, `sciris`, `diskcache`, `requests`

## Testing Notes
- Uses pytest with doctests enabled
- Warnings treated as errors in test configuration
- C extension coverage supported on Linux via tox
- Test files follow `test_*.py` pattern in `tests/` directory

## Development Memories
- Skip ruff checks
- Use "rg" (ripgrep) instead of "grep"