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
import polars as pl

from laser_measles.compartmental import CompartmentalModel
from laser_measles.compartmental.components import CaseSurveillanceParams
from laser_measles.compartmental.components import CaseSurveillanceTracker
from laser_measles.compartmental.params import CompartmentalParams

# Create a simple scenario
scenario = pl.DataFrame(
    {"id": ["1", "2", "3"], "pop": [1000, 2000, 1500], "lat": [40.0, 41.0, 42.0], "lon": [-74.0, -73.0, -72.0], "mcv1": [0.0, 0.0, 0.0]}
)

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

# Create a CaseSurveillanceTracker to monitor infections
case_tracker = lm.create_component(
    CaseSurveillanceTracker,
    CaseSurveillanceParams(detection_rate=1.0),  # 100% detection for accurate infection counting
)

# Add transmission and surveillance to the model
from laser_measles.compartmental.components import TransmissionProcess

comp_model.add_component(TransmissionProcess)
comp_model.add_component(case_tracker)

print("\nCompartmental model with surveillance:")
print(comp_model)

# Run the simulation
comp_model.run()

# Access infection data
case_tracker_instance = comp_model.get_instance(CaseSurveillanceTracker)[0]
comp_infections_df = case_tracker_instance.get_dataframe()
print(f"\nCompartmental model total infections: {comp_infections_df['cases'].sum()}")

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
import laser_measles as lm
from laser_measles.abm import ABMModel
from laser_measles.abm.components import TransmissionProcess
from laser_measles.abm.params import ABMParams

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

# Add CaseSurveillanceTracker to ABM model
abm_case_tracker = lm.create_component(
    lm.abm.components.CaseSurveillanceTracker, lm.abm.components.CaseSurveillanceParams(detection_rate=1.0)
)
abm_model.add_component(abm_case_tracker)

print("\nABM model with surveillance:")
print(abm_model)

# Run the simulation
abm_model.run()

# Access infection data
abm_case_tracker_instance = abm_model.get_instance(lm.abm.components.CaseSurveillanceTracker)[0]
abm_infections_df = abm_case_tracker_instance.get_dataframe()
print(f"\nABM model total infections: {abm_infections_df['cases'].sum()}")

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

# %% [markdown]
# ## Using CaseSurveillanceTracker for Infection Monitoring
#
# The CaseSurveillanceTracker component provides a powerful way to monitor disease
# dynamics and evaluate intervention effectiveness. Unlike StateTracker which tracks
# population compartments, CaseSurveillanceTracker specifically monitors new infections
# (cases) over time.

# %%
# Compare infection patterns between models
print("=== Infection Analysis ===")
print(f"Compartmental model total infections: {comp_infections_df['cases'].sum()}")
print(f"ABM model total infections: {abm_infections_df['cases'].sum()}")

# Show infection timeline for compartmental model
comp_timeline = comp_infections_df.group_by("tick").agg(pl.col("cases").sum().alias("daily_cases"))
print("\nCompartmental model infection timeline (first 20 days):")
print(comp_timeline.head(20))

# Calculate cumulative infections
comp_timeline = comp_timeline.with_columns(pl.col("daily_cases").cumsum().alias("cumulative_cases"))
print(f"\nPeak daily infections (compartmental): {comp_timeline['daily_cases'].max()}")
print(f"Days to peak: {comp_timeline.filter(pl.col('daily_cases') == comp_timeline['daily_cases'].max())['tick'].to_list()[0]}")

# %% [markdown]
# ### Key Benefits of CaseSurveillanceTracker:
# - **Detection simulation**: Models realistic case reporting with configurable detection rates
# - **Geographic aggregation**: Groups cases by administrative levels
# - **Time-series data**: Provides detailed infection timeline for analysis
# - **Intervention evaluation**: Perfect for assessing PIRIProcess or SIA effectiveness
# - **Surveillance modeling**: Simulates real-world surveillance system limitations
#
# ### Evaluating Intervention Effectiveness:
# To evaluate PIRIProcess effectiveness, you would:
# 1. Run baseline simulation without PIRIProcess
# 2. Run intervention simulation with PIRIProcess
# 3. Compare total infections using CaseSurveillanceTracker data
# 4. Calculate percent reduction: `(baseline - intervention) / baseline * 100`
