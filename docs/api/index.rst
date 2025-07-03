=============
API reference
=============

.. currentmodule:: laser_measles

This page lists laser-measles's API.

Core Framework
==============

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

Biweekly Model
==============

.. currentmodule:: laser_measles.biweekly

Core Model
----------

.. autosummary::
   :toctree: _autosummary
   :template: custom-function-template.rst
   :nosignatures:

   model
   params
   base

Processes
---------

Components that modify population states and drive model dynamics:

.. autosummary::
   :toctree: _autosummary
   :template: custom-function-template.rst
   :nosignatures:

   components.InfectionProcess
   components.VitalDynamicsProcess
   components.ImportationPressureProcess
   components.SIACalendarProcess

Trackers
--------

Components that monitor and record model state for analysis:

.. autosummary::
   :toctree: _autosummary
   :template: custom-function-template.rst
   :nosignatures:

   components.StateTracker
   components.FadeOutTracker
   components.CaseSurveillanceTracker

Utilities
---------

Biweekly model utilities and mixing functions:

.. autosummary::
   :toctree: _autosummary
   :template: custom-function-template.rst
   :nosignatures:

   mixing

ABM Model
=============

.. currentmodule:: laser_measles.abm

Core Model
----------

Agent based model

.. autosummary::
   :toctree: _autosummary
   :template: custom-function-template.rst
   :nosignatures:

   model
   params
   core
   utils
   cli


Processes
---------

Components that modify population states and drive model dynamics:

.. autosummary::
   :toctree: _autosummary
   :template: custom-function-template.rst
   :nosignatures:

   components.BirthsProcess
   components.BirthsConstantPopProcess
   components.DiseaseProcess
   components.TransmissionProcess
   components.InfectRandomAgentsProcess
   components.InfectAgentsInPatchProcess

Trackers
--------

Components that monitor and record model state for analysis:

.. autosummary::
   :toctree: _autosummary
   :template: custom-function-template.rst
   :nosignatures:

   components.StatesTracker
   components.PopulationTracker

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