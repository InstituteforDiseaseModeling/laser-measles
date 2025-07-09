# ruff: noqa: I001, E402
# %% [markdown]
# # Model structure
#
# This tutorial compares the structure of compartmental and agent-based models,
# focusing on their LaserFrame data structures and how they operate.

# %% [markdown]
# ## Overview
#
# laser-measles provides two primary modeling approaches:
# - **Compartmental Model**: Population-level SEIR dynamics using aggregated patch data
# - **ABM Model**: Individual-level simulation with stochastic agents
#
# The key difference lies in their data organization and LaserFrame structures.

# %% [markdown]
# ## Patches
#
# Patches exist for both the compartmental and ABM models and track the spatial
# data and aggregates in the model.
# The `patches` use a `BasePatchLaserFrame` (or child class) for population-level aggregates:

# %%
from laser_measles.compartmental import CompartmentalModel
from laser_measles.compartmental.params import CompartmentalParams
import polars as pl
# Create a simple scenario
scenario = pl.DataFrame({
    'patch_id': [1, 2, 3],
    'pop': [1000, 2000, 1500],
    'lat': [40.0, 41.0, 42.0],
    'lon': [-74.0, -73.0, -72.0]
})

# Initialize compartmental model
params = CompartmentalParams(num_ticks=100)
comp_model = CompartmentalModel(scenario, params)

# Examine the patch structure
print("Compartmental model patches:")
print(f"Shape: {comp_model.patches.states.shape}")
print(f"State names: {comp_model.patches.states.state_names}")
print(f"Initial S compartment: {comp_model.patches.states.S}")
print(f"Total population: {comp_model.patches.states.S.sum()}")

# You can also print the model to get some info:
print("Compartmental model 'out of the box':")
print(comp_model)

# %% [markdown]
# ### Key Features of patches (e.g., BasePatchLaserFrame):
# - **`states` property**: StateArray with shape `(num_states, num_patches)`
# - **Attribute access**: `states.S`, `states.E`, `states.I`, `states.R`
# - **Population aggregates**: Each patch contains total counts by disease state
# - **Spatial organization**: Patches represent geographic locations

# %% [markdown]
# ## People
#
# In addition to a `patch`, the ABM uses `people` (e.g., `BasePeopleLaserFrame`) for individual agents:

# %%
from laser_measles.abm import ABMModel
from laser_measles.abm.params import ABMParams
from laser_measles.abm.components import TransmissionProcess

# Initialize ABM model
abm_params = ABMParams(num_ticks=100)
abm_model = ABMModel(scenario, abm_params)

# Examine the model
print("ABM model 'out of the box':")
print(abm_model)

# Now what if add a transmission?
abm_model.add_component(TransmissionProcess)
print("ABM model after adding transmission:")
print(abm_model)

# %% [markdown]
# ### Key Features of BasePeopleLaserFrame:
# - **Individual agents**: Each row represents one person
# - **Agent properties**: `patch_id`, `state`, `susceptibility`, `active`
# - **Dynamic capacity**: Can grow/shrink as agents are born/die
# - **Stochastic processes**: Each agent processed individually

# %% [markdown]
# ## Key Differences
#
# | Aspect | Compartmental | ABM |
# |--------|---------------|-----|
# | **Data Structure** | `BasePatchLaserFrame` | `BasePeopleLaserFrame` |
# | **Population Storage** | Aggregated counts by patch | Individual agents |
# | **State Representation** | `states.S[patch_id]` | `people.state[agent_id]` |
# | **Spatial Organization** | Patch-level mixing matrix | Agent patch assignment |
# | **Transitions** | Binomial sampling | Individual stochastic events |
# | **Performance** | Faster (fewer calculations) | Slower (more detailed) |
# | **Memory Usage** | Lower (aggregates) | Higher (individual records) |

# %% [markdown]
# ## When to Use Each Model
#
# **Use Compartmental Model when:**
# - Analyzing population-level dynamics
# - Running many scenarios quickly
# - Interested in aggregate outcomes
# - Working with large populations
#
# **Use ABM Model when:**
# - Modeling individual heterogeneity
# - Studying contact networks
# - Tracking individual histories
# - Need detailed stochastic processes
#
# Both models share the same component architecture and can use similar
# initialization and analysis tools, making it easy to switch between approaches.