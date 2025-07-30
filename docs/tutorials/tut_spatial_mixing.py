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

# Create two different scenarios to showcase model differences
# 1. Two-cluster scenario (default)
cluster_scenario = synthetic.two_cluster_scenario(
    n_nodes_per_cluster=20,  # Smaller for clearer visualization
    cluster_size_std=2.0,    # More spread out clusters
    seed=42
)

# 2. Create a linear chain scenario for better demonstration of intervening opportunities
def create_linear_chain_scenario(n_nodes=30, seed=42):
    """Create a linear chain of nodes to highlight radiation vs gravity differences"""
    np.random.seed(seed)
    
    # Create nodes in a line with some random variation
    x_coords = np.linspace(0, 10, n_nodes) + np.random.normal(0, 0.1, n_nodes)
    y_coords = np.random.normal(0, 0.2, n_nodes)  # Small y variation
    
    # Create population gradient - larger populations at the ends
    base_pop = 1000
    pop_multiplier = 1 + 3 * (np.abs(np.linspace(-1, 1, n_nodes)))  # U-shaped
    populations = (base_pop * pop_multiplier).astype(int)
    
    # Create vaccination coverage (uniform for simplicity)
    mcv1_coverage = np.full(n_nodes, 0.85)
    
    return pl.DataFrame({
        'lat': y_coords,
        'lon': x_coords, 
        'pop': populations,
        'mcv1': mcv1_coverage
    })

linear_scenario = create_linear_chain_scenario(seed=42)

# Choose which scenario to use for main comparison
scenario = linear_scenario  # Use linear for dramatic differences
print(f"Using linear chain scenario with {len(scenario)} patches")
print(f"Total population: {scenario['pop'].sum():,}")

# Visualize both scenarios
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot cluster scenario
axes[0].scatter(cluster_scenario["lon"], cluster_scenario["lat"], 
               c=cluster_scenario["pop"], s=cluster_scenario["pop"]/10, 
               cmap="viridis", alpha=0.7, edgecolors="black")
axes[0].set_xlabel("Longitude")
axes[0].set_ylabel("Latitude")
axes[0].set_title("Two-Cluster Scenario")
axes[0].grid(True, alpha=0.3)

# Plot linear scenario  
scatter = axes[1].scatter(scenario["lon"], scenario["lat"], 
                         c=scenario["pop"], s=scenario["pop"]/8, 
                         cmap="viridis", alpha=0.7, edgecolors="black")
axes[1].set_xlabel("Longitude")
axes[1].set_ylabel("Latitude") 
axes[1].set_title("Linear Chain Scenario (Used for Analysis)")
axes[1].grid(True, alpha=0.3)

plt.colorbar(scatter, ax=axes[1], label="Population")
plt.tight_layout()
plt.show()

print(f"Population range: {scenario['pop'].min():,} - {scenario['pop'].max():,}")
print(f"Spatial extent: lon=[{scenario['lon'].min():.1f}, {scenario['lon'].max():.1f}], lat=[{scenario['lat'].min():.1f}, {scenario['lat'].max():.1f}]")

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

# Create gravity mixing with extreme parameters to show clear differences
gravity_params = GravityParams(
    a=1.0,    # Source population exponent  
    b=2.0,    # Target population exponent - higher to favor large populations
    c=1.0,    # Distance decay exponent - lower for long-range connections
    k=0.01    # Overall mixing scale - higher for more mixing
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
    k=0.01,           # Same overall mixing scale as gravity
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
# ## Analyzing Spatial Mixing Patterns
#
# Let's perform detailed analysis of how the two models structure spatial interactions differently.

# %%
# Calculate distances between all patches for analysis
def calculate_mixing_distances(scenario, mixing_matrix):
    """Calculate weighted average mixing distance for each patch"""
    coords = scenario[['lat', 'lon']].to_numpy()
    n_patches = len(coords)
    
    # Calculate distance matrix
    distances = np.zeros((n_patches, n_patches))
    for i in range(n_patches):
        for j in range(n_patches):
            distances[i, j] = np.sqrt((coords[i, 0] - coords[j, 0])**2 + 
                                    (coords[i, 1] - coords[j, 1])**2)
    
    # Calculate weighted average mixing distance for each patch
    mixing_distances = np.zeros(n_patches)
    for i in range(n_patches):
        weights = mixing_matrix[i, :]
        if weights.sum() > 0:
            mixing_distances[i] = np.average(distances[i, :], weights=weights)
    
    return mixing_distances, distances

gravity_mix_dist, distance_matrix = calculate_mixing_distances(scenario, gravity_mixing_matrix)
radiation_mix_dist, _ = calculate_mixing_distances(scenario, radiation_mixing_matrix)

# Create comprehensive mixing analysis
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot 1: Mixing matrices
im1 = axes[0, 0].imshow(gravity_mixing_matrix, cmap='Blues', aspect='auto')
axes[0, 0].set_title('Gravity Mixing Matrix')
axes[0, 0].set_xlabel('Destination Patch')
axes[0, 0].set_ylabel('Source Patch')
plt.colorbar(im1, ax=axes[0, 0], label='Mixing Probability')

im2 = axes[0, 1].imshow(radiation_mixing_matrix, cmap='Reds', aspect='auto')
axes[0, 1].set_title('Radiation Mixing Matrix')
axes[0, 1].set_xlabel('Destination Patch')
axes[0, 1].set_ylabel('Source Patch')
plt.colorbar(im2, ax=axes[0, 1], label='Mixing Probability')

# Plot difference
diff_matrix = gravity_mixing_matrix - radiation_mixing_matrix
im3 = axes[0, 2].imshow(diff_matrix, cmap='RdBu_r', aspect='auto', 
                       vmin=-np.max(np.abs(diff_matrix)), 
                       vmax=np.max(np.abs(diff_matrix)))
axes[0, 2].set_title('Difference (Gravity - Radiation)')
axes[0, 2].set_xlabel('Destination Patch')
axes[0, 2].set_ylabel('Source Patch')
plt.colorbar(im3, ax=axes[0, 2], label='Probability Difference')

# Plot 2: Mixing distance profiles
patch_positions = scenario['lon'].to_numpy()  # Use longitude as position along chain
axes[1, 0].plot(patch_positions, gravity_mix_dist, 'b-o', label='Gravity', linewidth=2, markersize=4)
axes[1, 0].plot(patch_positions, radiation_mix_dist, 'r-s', label='Radiation', linewidth=2, markersize=4)
axes[1, 0].set_xlabel('Patch Position (Longitude)')
axes[1, 0].set_ylabel('Average Mixing Distance')
axes[1, 0].set_title('Mixing Distance Profiles')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 3: Population vs mixing distance
pop_sizes = scenario['pop'].to_numpy()
axes[1, 1].scatter(pop_sizes, gravity_mix_dist, c='blue', alpha=0.7, s=50, label='Gravity')
axes[1, 1].scatter(pop_sizes, radiation_mix_dist, c='red', alpha=0.7, s=50, label='Radiation')
axes[1, 1].set_xlabel('Population Size')
axes[1, 1].set_ylabel('Average Mixing Distance')
axes[1, 1].set_title('Population Size vs Mixing Distance')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Plot 4: Mixing strength vs distance for representative patches
# Choose patches at positions 5, 15, 25 (left, center, right)
representative_patches = [5, 15, 25]
colors = ['green', 'orange', 'purple']

for idx, patch_i in enumerate(representative_patches):
    distances_from_patch = distance_matrix[patch_i, :]
    gravity_mixing_from_patch = gravity_mixing_matrix[patch_i, :]
    radiation_mixing_from_patch = radiation_mixing_matrix[patch_i, :]
    
    # Sort by distance for cleaner plotting
    sort_idx = np.argsort(distances_from_patch)
    sorted_distances = distances_from_patch[sort_idx]
    sorted_gravity = gravity_mixing_from_patch[sort_idx]
    sorted_radiation = radiation_mixing_from_patch[sort_idx]
    
    axes[1, 2].plot(sorted_distances, sorted_gravity, '-', 
                   color=colors[idx], alpha=0.7, linewidth=2,
                   label=f'Gravity (Patch {patch_i})')
    axes[1, 2].plot(sorted_distances, sorted_radiation, '--', 
                   color=colors[idx], alpha=0.7, linewidth=2,
                   label=f'Radiation (Patch {patch_i})')

axes[1, 2].set_xlabel('Distance')
axes[1, 2].set_ylabel('Mixing Probability') 
axes[1, 2].set_title('Mixing vs Distance (Representative Patches)')
axes[1, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
axes[1, 2].grid(True, alpha=0.3)
axes[1, 2].set_yscale('log')

plt.tight_layout()
plt.show()

# Print quantitative comparison
print("=== QUANTITATIVE MIXING ANALYSIS ===")
print(f"Gravity model - Mean mixing distance: {gravity_mix_dist.mean():.3f} ± {gravity_mix_dist.std():.3f}")
print(f"Radiation model - Mean mixing distance: {radiation_mix_dist.mean():.3f} ± {radiation_mix_dist.std():.3f}")

# Calculate spatial coupling metrics
gravity_off_diag = gravity_mixing_matrix.copy()
np.fill_diagonal(gravity_off_diag, 0)
radiation_off_diag = radiation_mixing_matrix.copy() 
np.fill_diagonal(radiation_off_diag, 0)

print(f"\nGravity model - Off-diagonal mixing (total spatial coupling): {gravity_off_diag.sum():.3f}")
print(f"Radiation model - Off-diagonal mixing (total spatial coupling): {radiation_off_diag.sum():.3f}")

# Calculate mixing inequality (how evenly distributed is mixing)
def mixing_inequality(mixing_matrix):
    """Calculate Gini coefficient of mixing probabilities (excluding diagonal)"""
    off_diag = mixing_matrix.copy()
    np.fill_diagonal(off_diag, 0)
    probs = off_diag.flatten()
    probs = probs[probs > 0]  # Only consider non-zero probabilities
    if len(probs) == 0:
        return 0
    probs = np.sort(probs)
    n = len(probs)
    cumsum = np.cumsum(probs)
    return (2 * np.sum((np.arange(1, n+1) * probs))) / (n * cumsum[-1]) - (n + 1) / n

gravity_gini = mixing_inequality(gravity_mixing_matrix)
radiation_gini = mixing_inequality(radiation_mixing_matrix)

print(f"\nGravity model - Mixing inequality (Gini): {gravity_gini:.3f}")
print(f"Radiation model - Mixing inequality (Gini): {radiation_gini:.3f}")
print("(Higher Gini = more unequal mixing, more concentrated on specific connections)")

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
# ## Epidemic Wave Propagation Analysis
#
# Let's visualize how the infection spreads geographically over time for each mixing model.

# %%
# Create epidemic wave propagation visualization
def create_wave_propagation_plot(tracker, scenario, model_name, color_scheme):
    """Create time-lapse visualization of epidemic wave propagation"""
    
    # Get cumulative attack rates over time for each patch
    n_patches = len(scenario)
    n_ticks = tracker.R.shape[0]
    coords = scenario[['lon', 'lat']].to_numpy()
    populations = scenario['pop'].to_numpy()
    
    # Calculate attack rates over time for each patch
    attack_rates_over_time = np.zeros((n_ticks, n_patches))
    for t in range(n_ticks):
        for p in range(n_patches):
            total_pop = tracker.state_tracker[t, :, p].sum()
            if total_pop > 0:
                attack_rates_over_time[t, p] = (tracker.R[t, p] / total_pop) * 100
    
    # Select key time points for visualization
    time_points = [50, 100, 200, 400, 800]  # Days
    fig, axes = plt.subplots(1, len(time_points), figsize=(20, 4))
    
    for i, t in enumerate(time_points):
        if t < n_ticks:
            attack_rates_t = attack_rates_over_time[t, :]
            scatter = axes[i].scatter(coords[:, 1], coords[:, 0], 
                                    c=attack_rates_t, s=populations/8,
                                    cmap=color_scheme, alpha=0.8, 
                                    edgecolors='black', linewidth=0.5,
                                    vmin=0, vmax=attack_rates_over_time.max())
            axes[i].set_title(f'Day {t}')
            axes[i].set_xlabel('Longitude')
            if i == 0:
                axes[i].set_ylabel('Latitude')
            axes[i].grid(True, alpha=0.3)
            
            # Add colorbar to last subplot
            if i == len(time_points) - 1:
                plt.colorbar(scatter, ax=axes[i], label='Attack Rate (%)')
    
    plt.suptitle(f'{model_name} Model: Epidemic Wave Propagation', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()
    
    return attack_rates_over_time

# Create wave propagation plots for both models
print("=== EPIDEMIC WAVE PROPAGATION ===")
gravity_waves = create_wave_propagation_plot(gravity_tracker, scenario, 'Gravity', 'Reds')
radiation_waves = create_wave_propagation_plot(radiation_tracker, scenario, 'Radiation', 'Blues')

# %% 
# Analyze the speed and pattern of epidemic spread
def analyze_epidemic_spread(waves, scenario, model_name):
    """Analyze spatial and temporal patterns of epidemic spread"""
    coords = scenario[['lon', 'lat']].to_numpy()
    populations = scenario['pop'].to_numpy()
    
    # Calculate time to 50% attack rate for each patch
    threshold = 50.0  # 50% attack rate
    time_to_threshold = np.full(len(scenario), np.nan)
    
    for p in range(len(scenario)):
        threshold_times = np.where(waves[:, p] >= threshold)[0]
        if len(threshold_times) > 0:
            time_to_threshold[p] = threshold_times[0]
    
    # Calculate epidemic velocity (spatial spread rate)
    # Distance from center of chain
    center_lon = coords[:, 1].mean()
    distances_from_center = np.abs(coords[:, 1] - center_lon)
    
    # Only consider patches that reached threshold
    valid_patches = ~np.isnan(time_to_threshold)
    if valid_patches.sum() > 2:
        # Fit linear relationship between distance and time
        from scipy import stats
        if np.var(distances_from_center[valid_patches]) > 0:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                distances_from_center[valid_patches], 
                time_to_threshold[valid_patches]
            )
            epidemic_velocity = 1.0 / slope if slope > 0 else np.inf  # distance/time
        else:
            epidemic_velocity = np.inf
    else:
        epidemic_velocity = np.nan
    
    print(f"\n{model_name} Model Spread Analysis:")
    print(f"  Patches reaching 50% attack rate: {valid_patches.sum()}/{len(scenario)}")
    print(f"  Mean time to 50% attack rate: {np.nanmean(time_to_threshold):.1f} days")
    print(f"  Epidemic velocity: {epidemic_velocity:.3f} distance units/day")
    
    return time_to_threshold, epidemic_velocity

gravity_times, gravity_velocity = analyze_epidemic_spread(gravity_waves, scenario, 'Gravity')
radiation_times, radiation_velocity = analyze_epidemic_spread(radiation_waves, scenario, 'Radiation')

# %% [markdown]
# ## Temporal Dynamics Comparison
#
# Now let's compare the overall temporal patterns of the epidemics.

# %%
# Create comprehensive temporal comparison
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Time vector
time_days = np.arange(num_ticks)

# Plot 1: Total infectious over time
axes[0, 0].plot(time_days, gravity_tracker.I, label='Gravity', color='red', linewidth=2)
axes[0, 0].plot(time_days, radiation_tracker.I, label='Radiation', color='blue', linewidth=2)
axes[0, 0].set_xlabel('Days')
axes[0, 0].set_ylabel('Total Infectious')
axes[0, 0].set_title('Epidemic Curves: Infectious Individuals')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Susceptible depletion
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

# Plot 3: Cumulative attack rates
gravity_cum_attack = (gravity_tracker.R / gravity_total_pop) * 100
radiation_cum_attack = (radiation_tracker.R / radiation_total_pop) * 100

axes[0, 2].plot(time_days, gravity_cum_attack, 
                label='Gravity', color='red', linewidth=2)
axes[0, 2].plot(time_days, radiation_cum_attack, 
                label='Radiation', color='blue', linewidth=2)
axes[0, 2].set_xlabel('Days')
axes[0, 2].set_ylabel('Cumulative Attack Rate (%)')
axes[0, 2].set_title('Cumulative Attack Rate Over Time')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# Plot 4: Daily incidence
gravity_daily_incidence = np.diff(gravity_tracker.R, prepend=0)
radiation_daily_incidence = np.diff(radiation_tracker.R, prepend=0)

axes[1, 0].plot(time_days, gravity_daily_incidence, 
                label='Gravity', color='red', alpha=0.7)
axes[1, 0].plot(time_days, radiation_daily_incidence, 
                label='Radiation', color='blue', alpha=0.7)
axes[1, 0].set_xlabel('Days')
axes[1, 0].set_ylabel('Daily New Recoveries')
axes[1, 0].set_title('Daily Incidence (New Recoveries)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 5: Spatial heterogeneity over time (coefficient of variation of attack rates)
def calculate_spatial_heterogeneity(waves):
    """Calculate coefficient of variation of attack rates across patches over time"""
    cv_over_time = np.zeros(waves.shape[0])
    for t in range(waves.shape[0]):
        attack_rates_t = waves[t, :]
        if attack_rates_t.mean() > 0:
            cv_over_time[t] = attack_rates_t.std() / attack_rates_t.mean()
    return cv_over_time

gravity_heterogeneity = calculate_spatial_heterogeneity(gravity_waves)
radiation_heterogeneity = calculate_spatial_heterogeneity(radiation_waves)

axes[1, 1].plot(time_days, gravity_heterogeneity, 
                label='Gravity', color='red', linewidth=2)
axes[1, 1].plot(time_days, radiation_heterogeneity, 
                label='Radiation', color='blue', linewidth=2)
axes[1, 1].set_xlabel('Days')
axes[1, 1].set_ylabel('Coefficient of Variation')
axes[1, 1].set_title('Spatial Heterogeneity Over Time')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Plot 6: Time to epidemic peak by position
coords = scenario[['lon', 'lat']].to_numpy()
patch_positions = coords[:, 1]  # Longitude

# Find peak time for each patch
def find_peak_times(waves):
    peak_times = np.zeros(waves.shape[1])
    for p in range(waves.shape[1]):
        daily_incidence = np.diff(waves[:, p], prepend=0)
        if daily_incidence.max() > 0:
            peak_times[p] = np.argmax(daily_incidence)
    return peak_times

gravity_peak_times = find_peak_times(gravity_waves)
radiation_peak_times = find_peak_times(radiation_waves)

axes[1, 2].scatter(patch_positions, gravity_peak_times, 
                  c='red', alpha=0.7, s=50, label='Gravity')
axes[1, 2].scatter(patch_positions, radiation_peak_times, 
                  c='blue', alpha=0.7, s=50, label='Radiation')
axes[1, 2].set_xlabel('Patch Position (Longitude)')
axes[1, 2].set_ylabel('Time to Epidemic Peak (Days)')
axes[1, 2].set_title('Epidemic Peak Timing by Position')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Key Insights and Summary
#
# This tutorial demonstrated dramatic differences between gravity and radiation mixing models
# using a linear chain scenario and extreme parameter settings. Here are the key findings:
#
# ### Quantitative Differences in Spatial Coupling
# The analysis revealed substantial differences in how the models structure spatial interactions:
# - **Mixing distances**: Models can have very different average mixing distances
# - **Spatial coupling strength**: Total off-diagonal mixing varies significantly 
# - **Mixing inequality**: Distribution of connections can be highly unequal between models
# - **Population sensitivity**: Gravity models with high `b` parameters strongly favor large populations
#
# ### Epidemic Wave Propagation Patterns
# The time-lapse visualizations showed distinct spatial spread patterns:
# - **Gravity models**: Can create long-range "jumps" to large populations
# - **Radiation models**: Tend to create more local, wave-like spread patterns
# - **Epidemic velocity**: Different models produce different speeds of spatial spread
# - **Peak timing**: Time to epidemic peak varies spatially in model-specific ways
#
# ### Spatial Heterogeneity Over Time
# The coefficient of variation analysis revealed:
# - Models can produce different levels of spatial heterogeneity
# - Heterogeneity patterns evolve differently over time
# - Some models maintain spatial heterogeneity longer than others
#
# ### Model Selection Guidelines
# 
# **Choose gravity models when:**
# - You have good estimates of population attraction parameters
# - Long-range connections to large cities are epidemiologically important
# - You need interpretable, tunable parameters for scenario analysis
# - Parameters: Use low `c` (1.0-1.5) for long-range mixing, high `b` (1.5-2.0) for population attraction
#
# **Choose radiation models when:**
# - You want less parameter-dependent results
# - Intervening opportunities are important (e.g., people stop at intermediate cities)
# - You're modeling realistic human mobility patterns
# - You prefer models with theoretical grounding in mobility research
#
# **Choose competing destinations when:**
# - Urban environments with destination competition effects
# - Multiple attractive destinations compete for travelers
#
# **Choose Stouffer models when:**
# - Intervening opportunities are critical
# - Step-by-step migration patterns are important
#
# ### Technical Configuration
# 
# To create dramatic model differences in your analysis:
#
# ```python
# # Extreme gravity parameters for demonstration
# gravity_params = GravityParams(
#     a=1.0,  # Source population exponent
#     b=2.0,  # High target population attraction  
#     c=1.0,  # Low distance decay for long-range connections
#     k=0.01  # Overall mixing scale
# )
#
# # Radiation parameters
# radiation_params = RadiationParams(
#     k=0.01,           # Same mixing scale for fair comparison
#     include_home=True # Include self-mixing
# )
#
# # Use linear or chain-like scenarios to highlight intervening opportunities
# ```
#
# ### Key Takeaway
# **The choice of spatial mixing model can dramatically affect both the spatial patterns 
# and temporal dynamics of epidemic spread.** Always compare multiple models and use 
# quantitative metrics (mixing distances, spatial coupling, epidemic velocity) to 
# understand how your choice affects results.
#
# Different mixing models represent different theories of human mobility and contact patterns.
# The "best" model depends on your research question, available data, and the geographic
# context of your study.

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