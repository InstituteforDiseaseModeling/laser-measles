=============
API reference
=============

.. currentmodule:: laser_measles

This page lists laser-measles's API.

Base Components
===============

.. currentmodule:: laser_measles.components

Components (e.g., infection, aging, vital dynamics, etc.) setup and affect the model, its structure and dynamics.
Shared base classes that provide common interfaces and functionality across all model types.
Model-specific implementations inherit from these base classes.

Component Architecture
----------------------

The laser-measles framework follows a hierarchical component architecture:

1. **Base Components** (``laser_measles.components``) define abstract interfaces and common functionality
2. **Model-Specific Implementations** (``laser_measles.{model}.components``) inherit from base classes and implement model-specific behavior
3. **Parameter Classes** use Pydantic for validation and are paired with their respective component classes
4. **Numpy/Numba Pattern** - Components can provide both numpy and numba implementations for performance optimization

**Inheritance Pattern Example:**

.. code-block:: python

   # Base class in laser_measles.components
   class BaseInitializeEquilibriumStatesProcess(BasePhase):
       # Common interface and default behavior
       pass

   # Model-specific implementation in laser_measles.abm.components
   class InitializeEquilibriumStatesProcess(BaseInitializeEquilibriumStatesProcess):
       # ABM-specific implementation
       pass

Process Base Classes
--------------------

Abstract base classes for components that modify population states and drive model dynamics:

.. autosummary::
   :toctree: _autosummary
   :template: custom-function-template.rst
   :nosignatures:

   BaseInitializeEquilibriumStatesProcess
   BaseVitalDynamicsProcess
   BaseConstantPopProcess
   BaseInfectionProcess

Note that each component has a corresponding parameter class that is used to configure the component.
For example, the ``BaseInfectionProcess`` has a corresponding ``BaseInfectionParams`` class.
The parameter class is used to configure the component and defaults can be overridden using the
``create_component`` function.

Tracker Base Classes
--------------------

Abstract base classes for components that monitor and record model state for analysis:

.. autosummary::
   :toctree: _autosummary
   :template: custom-function-template.rst
   :nosignatures:

   BaseStateTracker
   BaseCaseSurveillanceTracker
   BasePopulationTracker
   BaseFadeOutTracker

Utilities
---------

Component creation and management utilities:

.. autosummary::
   :toctree: _autosummary
   :template: custom-function-template.rst
   :nosignatures:

   component
   create_component

----

ABM Model
=========

.. currentmodule:: laser_measles.abm

Core Model
----------

Agent based model:

.. autosummary::
   :toctree: _autosummary
   :template: custom-function-template.rst
   :nosignatures:

   model
   params
   base
   utils
   cli

Process Components
------------------

Components that modify population states and drive model dynamics.
Most inherit from base classes in ``laser_measles.components``:

.. autosummary::
   :toctree: _autosummary
   :template: custom-function-template.rst
   :nosignatures:

   components.InitializeEquilibriumStatesProcess
   components.VitalDynamicsProcess
   components.ConstantPopProcess
   components.InfectionProcess
   components.InfectionSeedingProcess
   components.ImportationPressureProcess
   components.SIACalendarProcess

ABM-Specific Process Components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Components unique to the ABM model:

.. autosummary::
   :toctree: _autosummary
   :template: custom-function-template.rst
   :nosignatures:

   components.NoBirthsProcess
   components.TransmissionProcess
   components.DiseaseProcess

Tracker Components
------------------

Components that monitor and record model state for analysis.
All inherit from base classes in ``laser_measles.components``:

.. autosummary::
   :toctree: _autosummary
   :template: custom-function-template.rst
   :nosignatures:

   components.StateTracker
   components.CaseSurveillanceTracker
   components.PopulationTracker
   components.FadeOutTracker

----

Compartmental Model
===================

.. currentmodule:: laser_measles.compartmental

Core Model
----------

Compartmental model with daily timesteps:

.. autosummary::
   :toctree: _autosummary
   :template: custom-function-template.rst
   :nosignatures:

   model
   params
   base

Process Components
------------------

Components that modify population states and drive model dynamics.
All inherit from base classes in ``laser_measles.components``:

.. autosummary::
   :toctree: _autosummary
   :template: custom-function-template.rst
   :nosignatures:

   components.InitializeEquilibriumStatesProcess
   components.InfectionSeedingProcess
   components.InfectionProcess
   components.ImportationPressureProcess
   components.VitalDynamicsProcess
   components.ConstantPopProcess
   components.SIACalendarProcess

Tracker Components
------------------

Components that monitor and record model state for analysis.
All inherit from base classes in ``laser_measles.components``:

.. autosummary::
   :toctree: _autosummary
   :template: custom-function-template.rst
   :nosignatures:

   components.StateTracker
   components.CaseSurveillanceTracker
   components.PopulationTracker
   components.FadeOutTracker

----

Biweekly Model
==============

.. currentmodule:: laser_measles.biweekly

Core Model
----------

Compartmental model with 2-week timesteps

.. autosummary::
   :toctree: _autosummary
   :template: custom-function-template.rst
   :nosignatures:

   model
   params
   base

Process Components
------------------

Components that modify population states and drive model dynamics.
All inherit from base classes in ``laser_measles.components``:

.. autosummary::
   :toctree: _autosummary
   :template: custom-function-template.rst
   :nosignatures:

   components.InitializeEquilibriumStatesProcess
   components.InfectionSeedingProcess
   components.InfectionProcess
   components.VitalDynamicsProcess
   components.ImportationPressureProcess
   components.ConstantPopProcess
   components.SIACalendarProcess

Tracker Components
------------------

Components that monitor and record model state for analysis.
Most inherit from base classes in ``laser_measles.components``:

.. autosummary::
   :toctree: _autosummary
   :template: custom-function-template.rst
   :nosignatures:

   components.StateTracker
   components.CaseSurveillanceTracker
   components.PopulationTracker
   components.FadeOutTracker

Biweekly-Specific Tracker Components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Components unique to the Biweekly model:

.. autosummary::
   :toctree: _autosummary
   :template: custom-function-template.rst
   :nosignatures:

   components.FadeOutTracker

Utilities
---------

Core Framework
==============

.. currentmodule:: laser_measles

Base Classes
------------

Foundation classes that provide the component architecture:

.. autosummary::
   :toctree: _autosummary
   :template: custom-function-template.rst
   :nosignatures:

   base.BaseComponent
   base.BaseLaserModel

Utilities
---------

Core utilities and computation functions:

.. autosummary::
   :toctree: _autosummary
   :template: custom-function-template.rst
   :nosignatures:

   create_component
   pretty_laserframe

----

Demographics Package
====================

.. currentmodule:: laser_measles.demographics

Geographic data handling for spatial epidemiological modeling:

Shapefile Utilities
-------------------

Functions for processing and visualizing geographic shapefiles:

.. autosummary::
   :toctree: _autosummary
   :template: custom-function-template.rst
   :nosignatures:

   get_shapefile_dataframe
   plot_shapefile_dataframe
   GADMShapefile

Raster Processing
-----------------

Tools for handling raster data and patch generation:

.. autosummary::
   :toctree: _autosummary
   :template: custom-function-template.rst
   :nosignatures:

   RasterPatchParams
   RasterPatchGenerator
