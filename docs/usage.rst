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

Install laser-measles using pip:

.. code-block:: bash

    pip install laser-measles

For development installation with all dependencies:

.. code-block:: bash

    pip install -e .[dev,docs,examples]

**Dependencies:**

* ``laser-core``: Core LASER framework
* ``pydantic``: Parameter validation and serialization
* ``polars``: High-performance data manipulation
* ``sciris``: Scientific computing utilities
* ``requests``: HTTP requests for data fetching

----------


Model Types
-----------

There are two main model types in laser-measles: the Biweekly Model and the Generic Model. 
The Biweekly Model is focused to be fast and easy to use. It is a stochastic compartmental model 
designed to be used for quick simulations and parameter exploration.
However, it has some important limitations and is not suitable for all use cases.
The Generic Model is a more flexible and powerful agent-based model that can be used for a wide range of scenarios.
It is designed to be used for detailed simulations and research.

----------

Biweekly Model
~~~~~~~~~~~~~~

The Biweekly Model is focused to be fast and easy to use. It is designed to be used for quick simulations and parameter exploration.
However, it has some important limitations and is not suitable for all use cases.

**Key Characteristics:**

* **Compartmental approach**: SIR (Susceptible-Infected-Recovered) compartmental model structure
* **Time resolution**: 14-day fixed time steps
* **High performance**: Uses Polars DataFrames for efficient data manipulation
* **Configurable components**: Modular disease processes including infection, vital dynamics, and vaccination campaigns.

**Components include:**

* ``InfectionProcess``: Handles disease transmission dynamics
* ``VitalDynamicsProcess``: Manages births, deaths, and aging
* ``ImportationPressureProcess``: Models external infection sources
* ``SIACalendarProcess``: Supplementary immunization activities
* ``CaseSurveillanceTracker``: Monitors disease incidence
* ``StateTracker``: Records population state changes

**Example usage:**

.. code-block:: python

    from laser_measles.biweekly import BiweeklyModel, BiweeklyParams
    
    # Configure model parameters
    params = BiweeklyParams(
        nticks=520,  # 20 years of bi-weekly time steps
        crude_birth_rate=35.0,  # births per 1000 per year
        crude_death_rate=10.0,  # deaths per 1000 per year
        seed=12345
    )
    
    # Initialize and run model
    model = BiweeklyModel(scenario_data, params)
    model.run()

----------

Generic Model
~~~~~~~~~~~~~

The Generic Model provides a general-purpose agent-based implementation.

**Key Characteristics:**

* **Daily time steps**: Fine-grained temporal resolution
* **Geographic flexibility**: Adaptable to regions using the demographics package
* **Comprehensive demographics**: Births, deaths, aging, and migration processes

**Components include:**

* ``BirthsProcess``: Population birth dynamics
* ``ExposureProcess``: Disease exposure modeling
* ``InfectionProcess``: Infection state transitions
* ``TransmissionProcess``: Spatial transmission dynamics
* ``SusceptibilityProcess``: Immunity and susceptibility management

**Example usage:**

.. code-block:: python

    from laser_measles.generic import Model, GenericParams
    
    # Configure model parameters
    params = GenericParams(
        nticks=7300,  # 20 years of daily time steps
        seed=12345
    )
    
    # Initialize model with custom components
    model = Model(scenario_data, params)
    model.components = [
        BirthsProcess(model),
        ExposureProcess(model),
        InfectionProcess(model),
        TransmissionProcess(model)
    ]
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
* ``RasterPatchConfig``: Configuration for raster-based population patches
* ``RasterPatchGenerator``: Creates population patches from raster data

**Example usage:**

.. code-block:: python

    from laser_measles.demographics import GADMShapefile, RasterPatchGenerator
    
    # Load administrative boundaries
    shapefile = GADMShapefile("ETH", admin_level=1)  # Ethiopia, admin level 1
    
    # Generate population patches
    config = RasterPatchConfig(
        shapefile_path="path/to/shapefile.shp",
        raster_path="path/to/population.tif",
        patch_size=1000  # 1km patches
    )
    generator = RasterPatchGenerator(config)
    patches = generator.generate_patches()

Technical Features
------------------

Pydantic Integration
~~~~~~~~~~~~~~~~~~~~

laser-measles uses Pydantic for type-safe parameter management, providing automatic validation and documentation.

**Parameter Classes:**

* ``BiweeklyParams``: Configuration for biweekly models with epidemiological parameters
* ``GenericParams``: Flexible parameters for generic model implementations

**Component Classes:**
Components come in "process" and "tracker" categories and each component has a corresponding parameter class. 
Each model (Biweekly or Generic) has its own set of components. See the :doc:`API documentation <api/index>` for more details.

**Benefits:**

* **Type safety**: Automatic validation of parameter types and ranges
* **Documentation**: Built-in parameter descriptions and constraints
* **Serialization**: JSON export/import of model configurations
* **IDE support**: Enhanced autocomplete and error detection

**Example:**

.. code-block:: python

    from laser_measles.biweekly import BiweeklyParams
    
    params = BiweeklyParams(
        crude_birth_rate=35.0,  # Validated as positive float
        nticks=520,             # Validated as positive integer
        seed=12345
    )
    
    # Export configuration
    config_json = params.model_dump_json()

High-Performance Computing
~~~~~~~~~~~~~~~~~~~~~~~~~~

laser-measles is optimized for performance through several technical approaches:

**LaserFrame Architecture:**
    High-performance array-based structure for agent populations, built on the LASER framework

**Numba JIT Compilation:**
    Performance-critical operations implemented in Numba for maximum speed

**Polars DataFrames:**
    Efficient data manipulation using Polars for biweekly model operations

**Component Modularity:**
    Modular architecture allows for selective component usage and optimization

Component System
~~~~~~~~~~~~~~~~

The component system provides a uniform interface for disease dynamics with interchangeable modules.

**Base Architecture:**

* **Uniform Interface**: All components implement ``__call__(model, tick)`` method
* **Modular Design**: Components can be mixed and matched for different scenarios
* **Extensibility**: Easy to create custom components for specific research needs

**Component Categories:**

* **Demographic**: Births, deaths, aging, migration
* **Epidemiological**: Infection, transmission, immunity, incubation
* **Intervention**: Vaccination, case management, surveillance
* **Environmental**: Importation, seasonal effects, spatial mixing
