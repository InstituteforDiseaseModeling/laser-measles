=====
Usage
=====

Overview
--------

laser-measles is a spatial epidemiological modeling toolkit for measles transmission dynamics, built on the `LASER framework <https://github.com/InstituteforDiseaseModeling/laser>`_.
It provides a flexible, component-based architecture for disease simulation with support for multiple geographic scales and demographic configurations.

Key features include:

* **Spatial modeling**: Support for geographic regions with administrative boundaries and population distributions
* **Multiple model types**: Biweekly and Generic models for different use cases
* **Component-based architecture**: Interchangeable disease dynamics components
* **High-performance computing**: Optimized data structures and Numba JIT compilation
* **Type-safe parameters**: Pydantic-based configuration management

Installation and Setup
----------------------

Install laser-measles using pip (requires Python 3.10+):

.. code-block:: bash

    pip install laser-measles

For development installation with all dependencies (recommended: use `uv` for faster package management):

.. code-block:: bash

    # Using uv (recommended)
    uv pip install -e ".[dev]"
    # or for full installation including examples
    uv pip install -e ".[full]"

    # Alternative: using pip
    pip install -e ".[dev]"

**Major Dependencies:**

* ``laser-core>=0.5.1``: Core LASER framework
* ``pydantic>=2.11.5``: Parameter validation and serialization
* ``polars>=1.30.0``: High-performance data manipulation
* ``alive-progress>=3.2.0``: Progress bars and status indicators
* ``rastertoolkit>=0.3.11``: Raster data processing utilities
* ``patito>=0.8.3``: Polars DataFrame validation

----------


Model Types
-----------

laser-measles provides three complementary modeling approaches, each optimized for different use cases:

1. **ABM (Agent-Based Model)**: Individual-level simulation with stochastic agents
2. **Biweekly Compartmental Model**: Population-level SEIR dynamics with 2-week timesteps
3. **Compartmental Model**: Population-level SEIR dynamics with daily timesteps

Each model type offers different trade-offs between computational efficiency, temporal resolution, and modeling detail.

----------

ABM (Agent-Based Model)
~~~~~~~~~~~~~~~~~~~~~~~

The ABM model provides individual-level simulation with stochastic agents, allowing for detailed tracking of disease dynamics at the person level.

**Key Characteristics:**

* **Individual agents**: Each person is represented as a discrete agent with properties like age, location, and disease state
* **Daily timesteps**: Fine-grained temporal resolution for precise modeling
* **Stochastic processes**: Individual-level probabilistic events for realistic variability
* **Spatial heterogeneity**: Agents can move between patches and have location-specific interactions
* **Flexible demographics**: Full support for births, deaths, aging, and migration

**Example usage:**

.. code-block:: python

    from laser_measles.abm import ABMModel, ABMParams

    # Configure model parameters
    params = ABMParams(
        num_ticks=7300,  # 20 years of daily timesteps
        seed=12345
    )

    # Initialize and run model
    model = ABMModel(scenario_data, params)
    model.run()

----------

Biweekly Model
~~~~~~~~~~~~~~

The Biweekly Model is a compartmental model optimized for fast simulation and parameter exploration with 2-week timesteps.

**Key Characteristics:**

* **Compartmental approach**: SEIR (Susceptible-Exposed-Infected-Recovered) compartmental model structure
* **Time resolution**: 14-day fixed time steps (26 ticks per year)
* **High performance**: Uses Polars DataFrames for efficient data manipulation
* **Stochastic sampling**: Binomial sampling for realistic variability
* **Policy analysis**: Recommended for scenario building and intervention assessment

**Example usage:**

.. code-block:: python

    from laser_measles.biweekly import BiweeklyModel, BiweeklyParams

    # Configure model parameters
    params = BiweeklyParams(
        num_ticks=520,  # 20 years of bi-weekly time steps
        seed=12345
    )

    # Initialize and run model
    model = BiweeklyModel(scenario_data, params)
    model.run()

----------

Compartmental Model
~~~~~~~~~~~~~~~~~~~

The Compartmental Model provides population-level SEIR dynamics with daily timesteps, optimized for parameter estimation and detailed outbreak modeling.

**Key Characteristics:**

* **Daily timesteps**: Fine-grained temporal resolution (365 ticks per year)
* **SEIR dynamics**: Detailed compartmental structure with exposed compartment
* **Parameter estimation**: Recommended for fitting to surveillance data
* **Outbreak modeling**: Ideal for detailed temporal analysis of disease dynamics
* **Deterministic core**: Efficient ODE-based dynamics with optional stochastic elements

**Example usage:**

.. code-block:: python

    from laser_measles.compartmental import CompartmentalModel, CompartmentalParams

    # Configure model parameters
    params = CompartmentalParams(
        num_ticks=7300,  # 20 years of daily time steps
        seed=12345
    )

    # Initialize and run model
    model = CompartmentalModel(scenario_data, params)
    model.run()

----------

Demographics Package
--------------------

The demographics package provides comprehensive geographic data handling capabilities for spatial epidemiological modeling.

**Core Features:**

* **GADM Integration**: ``GADMShapefile`` class for administrative boundary management
* **Raster Processing**: ``RasterPatchGenerator`` for population distribution handling
* **Shapefile Utilities**: Functions for geographic data visualization and analysis
* **Flexible Geographic Scales**: Support from national to sub-district administrative levels

**Key Classes:**

* ``GADMShapefile``: Manages administrative boundaries from GADM database
* ``RasterPatchParams``: Configuration for raster-based population patches
* ``RasterPatchGenerator``: Creates population patches from raster data
* ``get_shapefile_dataframe``: Utility for shapefile data manipulation
* ``plot_shapefile_dataframe``: Visualization functions for geographic data

**Example usage:**

.. code-block:: python

    from laser_measles.demographics import GADMShapefile, RasterPatchGenerator, RasterPatchParams

    # Load administrative boundaries
    shapefile = GADMShapefile("ETH", admin_level=1)  # Ethiopia, admin level 1

    # Generate population patches
    params = RasterPatchParams(
        shapefile_path="path/to/shapefile.shp",
        raster_path="path/to/population.tif",
        patch_size=1000  # 1km patches
    )
    generator = RasterPatchGenerator(params)
    patches = generator.generate_patches()

Technical Features
------------------

Pydantic Integration
~~~~~~~~~~~~~~~~~~~~

laser-measles uses Pydantic for type-safe parameter management, providing automatic validation and documentation.

**Parameter Classes:**

* ``ABMParams``: Configuration for agent-based models with individual-level parameters
* ``BiweeklyParams``: Configuration for biweekly models with epidemiological parameters
* ``CompartmentalParams``: Configuration for compartmental models with daily dynamics

**Component Classes:**
Components come in "process" and "tracker" categories and each component has a corresponding parameter class.
Each model (ABM, Biweekly, or Compartmental) has its own set of components. See the :doc:`API documentation <api/index>` for more details.

**Benefits:**

* **Type safety**: Automatic validation of parameter types and ranges
* **Documentation**: Built-in parameter descriptions and constraints
* **Serialization**: JSON export/import of model configurations
* **IDE support**: Enhanced autocomplete and error detection

**Example:**

.. code-block:: python

    from laser_measles.biweekly import BiweeklyParams

    params = BiweeklyParams(
        num_ticks=520,  # Validated as positive integer
        seed=12345      # Random seed for reproducibility
    )

    # Export configuration
    config_json = params.model_dump_json()

High-Performance Computing
~~~~~~~~~~~~~~~~~~~~~~~~~~

laser-measles is optimized for performance through several technical approaches:

**LaserFrame Architecture:**
    High-performance array-based structure for agent populations, built on the LASER framework

**numba JIT Compilation:**
    Performance-critical operations implemented in numba for maximum speed

**Polars DataFrames:**
    Efficient data manipulation using Polars for biweekly model operations with Arrow backend

**Component Modularity:**
    Modular architecture allows for selective component usage and optimization

**Progress Tracking:**
    Integrated progress bars using alive-progress for long-running simulations

**Python 3.10+ Support:**
    Optimized for modern Python features and performance improvements

Component System
~~~~~~~~~~~~~~~~

The component system provides a uniform interface for disease dynamics with interchangeable modules built on a hierarchical base class architecture.

**Base Architecture:**

* **BaseLaserModel**: Abstract base class for all model types with common functionality
* **BaseComponent**: Base class for all components with standardized interface
* **BasePhase**: Components that execute every tick (inherit from BaseComponent)
* **Inheritance-based design**: Base components define shared functionality and abstract interfaces

**Base Component Classes:**

* ``base_transmission.py``: Base transmission/infection logic
* ``base_vital_dynamics.py``: Base births/deaths logic
* ``base_importation.py``: Base importation pressure logic
* ``base_tracker.py``: Base tracking/metrics logic
* ``base_infection.py``: Base infection state transitions
* ``base_tracker_state.py``: Base state tracking functionality

**Component Naming Convention:**

* **Process components**: ``process_*.py`` - Modify model state (births, deaths, infection, transmission)
* **Tracker components**: ``tracker_*.py`` - Record metrics and state over time

**Component Creation Patterns:**

.. code-block:: python

    # Component with parameters using Pydantic
    from laser_measles.components.base_infection import BaseInfectionProcess

    class MyInfectionProcess(BaseInfectionProcess):
        def __init__(self, model, verbose=False, **params):
            super().__init__(model, verbose)
            # Initialize with validated parameters

    # Add to model
    model.components = [MyInfectionProcess]
