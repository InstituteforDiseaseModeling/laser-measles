# %% [markdown]
# # Spatial Mixing Models Tutorial
#
# This tutorial demonstrates how to choose and configure different spatial mixing models
# in the laser-measles framework and shows how they affect disease transmission patterns.
#
# ## What are Spatial Mixing Models?
#
# Spatial mixing models determine how infectious individuals in one location can infect
# susceptible individuals in other locations. They create a **mixing matrix** that quantifies
# the probability of contact between people from different patches (spatial locations).
#
# The tutorial covers:
# - Overview of available mixing models (gravity, radiation, competing destinations, stouffer)
# - Configuring models with different mixing patterns
# - Comparing spatial disease spread patterns
# - Understanding how mixing matrices affect transmission dynamics
#
# ## Available Mixing Models
#
# laser-measles provides four spatial mixing models:
#
# 1. **Gravity Model**: Based on gravitational attraction, depends on population sizes and distance
#    - Formula: `k * (pop_source^(a-1)) * (pop_target^b) * (distance^(-c))`
#    - Good for modeling general mobility patterns
#
# 2. **Radiation Model**: Based on radiation theory of human mobility 
#    - Less dependent on specific parameter tuning
#    - Often performs well for real-world mobility data
#
# 3. **Competing Destinations**: Extension of gravity model with destination competition
#    - Includes delta parameter for destination selection
#
# 4. **Stouffer Model**: Based on intervening opportunities theory
#    - Considers intermediate locations between origin and destination

# %% [markdown]
# ## Setting up the scenario
#
# We'll create a scenario with multiple spatial nodes to demonstrate the effects
# of different mixing models on disease transmission patterns.

# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle

from laser_measles.scenarios import synthetic
from laser_measles.compartmental import BaseScenario, CompartmentalParams, Model, components
from laser_measles.components import create_component
from laser_measles.mixing.gravity import GravityMixing, GravityParams
from laser_measles.mixing.radiation import RadiationMixing, RadiationParams

# Set up matplotlib for better plots
plt.style.use('default')
sns.set_palette("husl")

# Create a scenario with spatial structure
scenario = synthetic.two_cluster_scenario(
    n_nodes_per_cluster=20,  # Smaller for clearer visualization
    cluster_size_std=2.0,    # More spread out clusters
    seed=42
)

print(f"Scenario created with {len(scenario)} patches")
print(f"Total population: {scenario['pop'].sum():,}")

# Visualize the scenario
plt.figure(figsize=(8, 6))
scatter = plt.scatter(scenario["lon"], scenario["lat"], 
                     c=scenario["pop"], s=scenario["pop"]/10, 
                     cmap="viridis", alpha=0.7, edgecolors="black")
plt.colorbar(scatter, label="Population")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Spatial Distribution of Populations")
plt.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# ## How Mixing Matrices Work in Disease Transmission
#
# Before we compare models, let's understand how the mixing matrix is used in the infection process.
# In the compartmental model's infection component (`process_infection.py:114`), the force of infection
# is calculated as:
#
# ```python
# lambda_i = (beta * seasonal_factor * prevalence) @ mixer.mixing_matrix
# ```
#
# This matrix multiplication means:
# - `prevalence` is a vector of infectious individuals per patch
# - `mixing_matrix[i,j]` is the probability that someone from patch i mixes with patch j
# - The result `lambda_i[i]` is the total infectious pressure on patch i from all patches
#
# Key insight: **The mixing matrix determines which patches can "seed" infections in other patches**

# %% [markdown]
# ## Model 1: Gravity Mixing
#
# The gravity model assumes that mixing between patches follows a gravitational law:
# stronger attraction between larger populations, weaker with greater distance.

# %%
# Configure model parameters
years = 5
num_ticks = years * 365  # Daily timesteps

params = CompartmentalParams(
    num_ticks=num_ticks,
    seed=42,
    verbose=False,  # Set to True to see timing information
    start_time="2000-01"
)

# Create gravity mixing with specific parameters
gravity_params = GravityParams(
    a=1.0,    # Source population exponent  
    b=1.0,    # Target population exponent
    c=2.0,    # Distance decay exponent (higher = more local mixing)
    k=0.005   # Overall mixing scale
)
gravity_mixer = GravityMixing(params=gravity_params)

# Create infection parameters with gravity mixing
infection_params = components.InfectionParams(
    beta=0.8,           # Transmission rate
    seasonality=0.2,    # Seasonal variation
    mixer=gravity_mixer
)

# Create and configure the model
gravity_model = Model(scenario, params, name="gravity_mixing_demo")
gravity_model.components = [
    components.InitializeEquilibriumStatesProcess,
    components.ImportationPressureProcess,
    create_component(components.InfectionProcess, params=infection_params),
    components.VitalDynamicsProcess,
    components.StateTracker
]

print("Running gravity model simulation...")
gravity_model.run()
print("Gravity model completed!")

# Get results
gravity_tracker = gravity_model.get_instance("StateTracker")[0]
gravity_final_R = gravity_model.patches.states.R.copy()
gravity_mixing_matrix = gravity_mixer.mixing_matrix.copy()

# %% [markdown]
# ## Model 2: Radiation Mixing
#
# The radiation model is based on a different theory of human mobility.
# It tends to be less parameter-dependent and often captures real mobility patterns well.

# %%
# Create radiation mixing
radiation_params = RadiationParams(
    k=0.005,          # Same overall mixing scale as gravity
    include_home=True # Include self-mixing (staying home)
)
radiation_mixer = RadiationMixing(params=radiation_params)

# Create infection parameters with radiation mixing
infection_params_rad = components.InfectionParams(
    beta=0.8,              # Same transmission rate
    seasonality=0.2,       # Same seasonal variation
    mixer=radiation_mixer
)

# Create new model instance for radiation
radiation_model = Model(scenario, params, name="radiation_mixing_demo")
radiation_model.components = [
    components.InitializeEquilibriumStatesProcess,
    components.ImportationPressureProcess,
    create_component(components.InfectionProcess, params=infection_params_rad),
    components.VitalDynamicsProcess,
    components.StateTracker
]

print("Running radiation model simulation...")
radiation_model.run()
print("Radiation model completed!")

# Get results
radiation_tracker = radiation_model.get_instance("StateTracker")[0]
radiation_final_R = radiation_model.patches.states.R.copy()
radiation_mixing_matrix = radiation_mixer.mixing_matrix.copy()

# %% [markdown]
# ## Comparing Mixing Matrices
#
# Let's visualize the actual mixing matrices to understand how differently
# the two models structure spatial interactions.

# %%
# Create comparison of mixing matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot gravity mixing matrix
im1 = axes[0].imshow(gravity_mixing_matrix, cmap='Blues', aspect='auto')
axes[0].set_title('Gravity Mixing Matrix')
axes[0].set_xlabel('Destination Patch')
axes[0].set_ylabel('Source Patch')
plt.colorbar(im1, ax=axes[0], label='Mixing Probability')

# Plot radiation mixing matrix  
im2 = axes[1].imshow(radiation_mixing_matrix, cmap='Reds', aspect='auto')
axes[1].set_title('Radiation Mixing Matrix')
axes[1].set_xlabel('Destination Patch')
axes[1].set_ylabel('Source Patch')
plt.colorbar(im2, ax=axes[1], label='Mixing Probability')

# Plot difference
diff_matrix = gravity_mixing_matrix - radiation_mixing_matrix
im3 = axes[2].imshow(diff_matrix, cmap='RdBu_r', aspect='auto', 
                     vmin=-np.max(np.abs(diff_matrix)), 
                     vmax=np.max(np.abs(diff_matrix)))
axes[2].set_title('Difference (Gravity - Radiation)')
axes[2].set_xlabel('Destination Patch')
axes[2].set_ylabel('Source Patch')
plt.colorbar(im3, ax=axes[2], label='Probability Difference')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Spatial Attack Rate Comparison
#
# Now let's see how the different mixing patterns affect the final spatial
# distribution of the epidemic.

# %%
# Calculate attack rates (proportion who got infected)
initial_pop = scenario["pop"].to_numpy()
gravity_attack_rates = (gravity_final_R / initial_pop) * 100
radiation_attack_rates = (radiation_final_R / initial_pop) * 100

# Create spatial comparison plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot gravity results
coords = scenario[["lon", "lat"]].to_numpy()
sc1 = axes[0].scatter(coords[:, 0], coords[:, 1], 
                     c=gravity_attack_rates, s=initial_pop/10,
                     cmap='Reds', alpha=0.7, edgecolors='black',
                     vmin=0, vmax=max(gravity_attack_rates.max(), radiation_attack_rates.max()))
axes[0].set_title('Gravity Model: Attack Rates (%)')
axes[0].set_xlabel('Longitude')
axes[0].set_ylabel('Latitude')
axes[0].grid(True, alpha=0.3)
plt.colorbar(sc1, ax=axes[0], label='Attack Rate (%)')

# Plot radiation results
sc2 = axes[1].scatter(coords[:, 0], coords[:, 1], 
                     c=radiation_attack_rates, s=initial_pop/10,
                     cmap='Blues', alpha=0.7, edgecolors='black',
                     vmin=0, vmax=max(gravity_attack_rates.max(), radiation_attack_rates.max()))
axes[1].set_title('Radiation Model: Attack Rates (%)')
axes[1].set_xlabel('Longitude')
axes[1].set_ylabel('Latitude')
axes[1].grid(True, alpha=0.3)
plt.colorbar(sc2, ax=axes[1], label='Attack Rate (%)')

# Plot difference
attack_rate_diff = gravity_attack_rates - radiation_attack_rates
max_diff = np.max(np.abs(attack_rate_diff))
sc3 = axes[2].scatter(coords[:, 0], coords[:, 1], 
                     c=attack_rate_diff, s=initial_pop/10,
                     cmap='RdBu_r', alpha=0.7, edgecolors='black',
                     vmin=-max_diff, vmax=max_diff)
axes[2].set_title('Difference (Gravity - Radiation)')
axes[2].set_xlabel('Longitude')
axes[2].set_ylabel('Latitude')
axes[2].grid(True, alpha=0.3)
plt.colorbar(sc3, ax=axes[2], label='Attack Rate Difference (%)')

plt.tight_layout()
plt.show()

# Print summary statistics
print(f"Gravity model - Mean attack rate: {gravity_attack_rates.mean():.1f}% (std: {gravity_attack_rates.std():.1f}%)")
print(f"Radiation model - Mean attack rate: {radiation_attack_rates.mean():.1f}% (std: {radiation_attack_rates.std():.1f}%)")
print(f"Maximum difference: {np.max(np.abs(attack_rate_diff)):.1f} percentage points")

# %% [markdown]
# ## Epidemic Curves Comparison
#
# Let's compare how the two mixing models affect the temporal dynamics of the epidemic.

# %%
# Create epidemic curves comparison
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Time vector
time_days = np.arange(num_ticks)

# Plot total infectious over time
axes[0, 0].plot(time_days, gravity_tracker.I, label='Gravity', color='red', linewidth=2)
axes[0, 0].plot(time_days, radiation_tracker.I, label='Radiation', color='blue', linewidth=2)
axes[0, 0].set_xlabel('Days')
axes[0, 0].set_ylabel('Total Infectious')
axes[0, 0].set_title('Epidemic Curves: Infectious Individuals')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot susceptible fraction over time
gravity_total_pop = gravity_tracker.state_tracker.sum(axis=0)
radiation_total_pop = radiation_tracker.state_tracker.sum(axis=0)

axes[0, 1].plot(time_days, gravity_tracker.S / gravity_total_pop, 
                label='Gravity', color='red', linewidth=2)
axes[0, 1].plot(time_days, radiation_tracker.S / radiation_total_pop, 
                label='Radiation', color='blue', linewidth=2)
axes[0, 1].set_xlabel('Days')
axes[0, 1].set_ylabel('Susceptible Fraction')
axes[0, 1].set_title('Susceptible Depletion')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot cumulative attack rates over time
gravity_cum_attack = (gravity_tracker.R / gravity_total_pop) * 100
radiation_cum_attack = (radiation_tracker.R / radiation_total_pop) * 100

axes[1, 0].plot(time_days, gravity_cum_attack, 
                label='Gravity', color='red', linewidth=2)
axes[1, 0].plot(time_days, radiation_cum_attack, 
                label='Radiation', color='blue', linewidth=2)
axes[1, 0].set_xlabel('Days')
axes[1, 0].set_ylabel('Cumulative Attack Rate (%)')
axes[1, 0].set_title('Cumulative Attack Rate Over Time')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot daily incidence
gravity_daily_incidence = np.diff(gravity_tracker.R, prepend=0)
radiation_daily_incidence = np.diff(radiation_tracker.R, prepend=0)

axes[1, 1].plot(time_days, gravity_daily_incidence, 
                label='Gravity', color='red', alpha=0.7)
axes[1, 1].plot(time_days, radiation_daily_incidence, 
                label='Radiation', color='blue', alpha=0.7)
axes[1, 1].set_xlabel('Days')
axes[1, 1].set_ylabel('Daily New Recoveries')
axes[1, 1].set_title('Daily Incidence (New Recoveries)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Key Insights and Summary
#
# ### Mixing Matrix Differences
# - **Gravity models** tend to create stronger long-range connections proportional to population sizes
# - **Radiation models** tend to favor shorter-range connections and intermediate populations
# - The diagonal elements (self-mixing) can differ significantly between models
#
# ### Spatial Pattern Differences  
# - Different mixing models can lead to different spatial patterns of attack rates
# - Some patches may be more "protected" under one model vs another
# - The overall epidemic size may be similar, but spatial heterogeneity can vary
#
# ### Temporal Dynamics
# - Peak timing and height can vary between mixing models
# - The shape of the epidemic curve reflects different connectivity patterns
# - Final attack rates may converge but the path to get there differs
#
# ### Practical Implications
# When choosing a mixing model for your analysis:
#
# 1. **Gravity**: Good default choice, interpretable parameters, works well for general mobility
# 2. **Radiation**: Less parameter-dependent, often matches real mobility data well  
# 3. **Competing Destinations**: For complex urban environments with destination competition
# 4. **Stouffer**: When intervening opportunities are important for mobility patterns
#
# ### Model Configuration
# To use different mixing models in your own analysis:
#
# ```python
# # Import the mixing model
# from laser_measles.mixing.radiation import RadiationMixing, RadiationParams
#
# # Configure parameters
# mixing_params = RadiationParams(k=0.01, include_home=True)
# mixer = RadiationMixing(params=mixing_params)
#
# # Use in infection parameters
# infection_params = components.InfectionParams(
#     beta=0.8,
#     mixer=mixer  # Pass the configured mixer
# )
# ```
#
# The mixing object will automatically receive the scenario data when the model is initialized,
# and the mixing matrix will be computed based on the spatial coordinates and populations
# in your scenario.

# %% [markdown]
# ## Exploring Parameter Sensitivity
#
# Let's briefly explore how changing mixing parameters affects the results.

# %%
# Test different gravity model parameters
distance_exponents = [1.0, 2.0, 3.0]  # Different distance decay rates
final_attack_rates = []
mixing_scales = []

fig, axes = plt.subplots(1, len(distance_exponents), figsize=(15, 5))

for i, c_param in enumerate(distance_exponents):
    # Create new gravity mixer with different distance exponent
    test_params = GravityParams(a=1.0, b=1.0, c=c_param, k=0.005)
    test_mixer = GravityMixing(params=test_params)
    
    # Quick model run
    test_infection_params = components.InfectionParams(
        beta=0.8, seasonality=0.2, mixer=test_mixer
    )
    
    test_model = Model(scenario, params, name=f"gravity_c{c_param}")
    test_model.components = [
        components.InitializeEquilibriumStatesProcess,
        components.ImportationPressureProcess,
        create_component(components.InfectionProcess, params=test_infection_params),
        components.VitalDynamicsProcess,
        components.StateTracker
    ]
    
    test_model.run()
    test_final_R = test_model.patches.states.R
    test_attack_rates = (test_final_R / initial_pop) * 100
    
    # Visualize
    sc = axes[i].scatter(coords[:, 0], coords[:, 1], 
                        c=test_attack_rates, s=initial_pop/10,
                        cmap='plasma', alpha=0.7, edgecolors='black',
                        vmin=0, vmax=100)
    axes[i].set_title(f'Distance Exponent c={c_param}')
    axes[i].set_xlabel('Longitude')
    axes[i].set_ylabel('Latitude')
    axes[i].grid(True, alpha=0.3)
    plt.colorbar(sc, ax=axes[i], label='Attack Rate (%)')
    
    final_attack_rates.append(test_attack_rates)
    
    # Calculate mixing scale (average off-diagonal mixing probability)
    test_mixing_matrix = test_mixer.mixing_matrix
    off_diag_mean = (test_mixing_matrix.sum() - np.trace(test_mixing_matrix)) / (len(test_mixing_matrix)**2 - len(test_mixing_matrix))
    mixing_scales.append(off_diag_mean)
    
    print(f"c={c_param}: Mean attack rate = {test_attack_rates.mean():.1f}%, "
          f"Std = {test_attack_rates.std():.1f}%, "
          f"Avg off-diagonal mixing = {off_diag_mean:.4f}")

plt.tight_layout()
plt.show()

print("\nSummary: Higher distance exponents (c) lead to more localized mixing and greater spatial heterogeneity in attack rates.")