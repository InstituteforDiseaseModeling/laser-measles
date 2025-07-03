# %% [markdown]
# # Biweekly Model Tutorial
# 
# This tutorial demonstrates how to initialize and run a biweekly epidemiological model
# using the laser-measles framework. The biweekly model uses a compartmental (SIR) approach
# with discrete time steps of 14 days.
#
# The tutorial covers:
# - Setting up scenario data with multiple spatial nodes
# - Configuring model parameters including transmission and vital dynamics
# - Adding components for disease transmission and state tracking
# - Running the simulation and visualizing results

# %%
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

from laser_measles.biweekly.model import BiweeklyModel
from laser_measles.biweekly.base import BaseScenario
from laser_measles.biweekly.params import BiweeklyParams
from laser_measles.biweekly.components import (
    InfectionProcess, 
    InfectionParams,
    VitalDynamicsProcess,
    StateTracker
)
from laser_measles.components import create_component

print("All imports successful!")

# %% [markdown]
# ## Create Sample Scenario Data
# 
# First, we'll create a scenario with multiple spatial nodes (patches) representing
# different communities. Each node has population, geographic coordinates, and
# MCV1 vaccination coverage.

# %%
# Set random seed for reproducibility
np.random.seed(42)

# Create scenario data for 9 nodes arranged in a 3x3 grid
n_nodes = 9
node_ids = [f"Node_{i+1}" for i in range(n_nodes)]

# Create a 3x3 grid of coordinates
grid_size = 3
coordinates = []
for i in range(grid_size):
    for j in range(grid_size):
        lat = 40.0 + i * 0.5  # Latitude from 40.0 to 41.0
        lon = -74.0 + j * 0.5  # Longitude from -74.0 to -73.0
        coordinates.append((lat, lon))

lats, lons = zip(*coordinates)

# Generate population sizes (10,000 to 100,000 per node)
populations = np.random.randint(10000, 100000, n_nodes)

# Generate MCV1 coverage (60% to 90%)
mcv1_coverage = np.random.uniform(0.6, 0.9, n_nodes)

# Create scenario DataFrame
scenario_data = pl.DataFrame({
    "ids": node_ids,
    "pop": populations,
    "lat": lats,
    "lon": lons,
    "mcv1": mcv1_coverage
})

print("Scenario data:")
print(scenario_data)

# Create BaseScenario object
scenario = BaseScenario(scenario_data)

# %% [markdown]
# ## Initialize Model Parameters
# 
# Configure the biweekly model parameters including demographic rates,
# simulation duration, and spatial mixing patterns.

# %%
# Calculate number of time steps (bi-weekly for 5 years)
years = 5
nticks = years * 26  # 26 bi-weekly periods per year

# Create model parameters
params = BiweeklyParams(
    nticks=nticks,
    crude_birth_rate=12.0,  # births per 1000 per year
    crude_death_rate=8.0,   # deaths per 1000 per year
    distance_exponent=1.5,  # how transmission decays with distance
    mixing_scale=0.001,     # scale parameter for spatial mixing
    seed=42,
    verbose=True
)

print(f"Model configured for {nticks} time steps ({years} years)")
print(f"Parameters: {params}")

# %% [markdown]
# ## Create Spatial Mixing Matrix
# 
# Generate a distance-based mixing matrix that determines how individuals
# from different nodes interact and transmit disease.

# %%
# Calculate distances between all pairs of nodes
coords = np.array(list(zip(lats, lons)))
distances = squareform(pdist(coords, metric='euclidean'))

# Create mixing matrix based on distance decay
# Higher mixing for closer nodes, lower for distant ones
mixing_matrix = params.mixing_scale * np.exp(-distances ** params.distance_exponent)

# Ensure diagonal elements represent within-node mixing
np.fill_diagonal(mixing_matrix, 1.0)

# Set the mixing matrix in parameters
params.mixing = mixing_matrix

print("Mixing matrix shape:", mixing_matrix.shape)
print("Mixing matrix (first 3x3):")
print(mixing_matrix[:3, :3])

# %% [markdown]
# ## Initialize the Biweekly Model
# 
# Create the model instance and add components for disease transmission,
# vital dynamics, and state tracking.

# %%
# Create the biweekly model
model = BiweeklyModel(scenario, params, name="biweekly_tutorial")

# Initialize states (S, I, R) for all nodes
# Start with most population susceptible, small fraction recovered (vaccinated at birth)
for i in range(n_nodes):
    pop = populations[i]
    # Start with 95% susceptible, 0% infected, 5% recovered
    model.patches.states[0, i] = int(pop * 0.95)  # Susceptible
    model.patches.states[1, i] = 0                # Infected
    model.patches.states[2, i] = int(pop * 0.05)  # Recovered

print("Initial state distribution:")
print(f"Total Susceptible: {model.patches.states[0].sum():,}")
print(f"Total Infected: {model.patches.states[1].sum():,}")
print(f"Total Recovered: {model.patches.states[2].sum():,}")

# %% [markdown]
# ## Add Model Components
# 
# Configure and add the essential components for disease dynamics:
# - InfectionProcess: Handles disease transmission between individuals
# - VitalDynamicsProcess: Manages births and deaths
# - StateTracker: Records population states over time for analysis

# %%
# Create infection parameters with seasonal transmission
infection_params = InfectionParams(
    beta=25.0,        # Base transmission rate
    seasonality=0.15, # 15% seasonal variation
    season_start=13   # Peak transmission at week 26 (mid-year)
)

# Add components to model using create_component
model.components = [
    create_component(VitalDynamicsProcess),  # Process births/deaths first
    create_component(InfectionProcess, params=infection_params),  # Then handle disease transmission
    create_component(StateTracker)  # Finally track the resulting states
]

print(f"Added {len(model.components)} components to the model")

# %% [markdown]
# ## Seed Initial Infections
# 
# Introduce initial infections in a few nodes to start the epidemic.

# %%
# Seed infections in the first 3 nodes
initial_infections_per_node = 50

for i in range(3):  # First 3 nodes
    # Move individuals from Susceptible to Infected
    infections = min(initial_infections_per_node, model.patches.states[0, i])
    model.patches.states[0, i] -= infections  # Remove from Susceptible
    model.patches.states[1, i] += infections  # Add to Infected

print("Seeded initial infections:")
print(f"Total Susceptible: {model.patches.states[0].sum():,}")
print(f"Total Infected: {model.patches.states[1].sum():,}")
print(f"Total Recovered: {model.patches.states[2].sum():,}")

# %% [markdown]
# ## Run the Simulation
# 
# Execute the model for the specified number of time steps.

# %%
print("Starting simulation...")
model.run()
print("Simulation completed!")

# Print final state summary
print("\nFinal state distribution:")
print(f"Total Susceptible: {model.patches.states[0].sum():,}")
print(f"Total Infected: {model.patches.states[1].sum():,}")
print(f"Total Recovered: {model.patches.states[2].sum():,}")
print(f"Total Population: {model.patches.states.sum():,}")

# %% [markdown]
# ## Visualize Results
# 
# Generate plots to analyze the simulation results, including time series
# of disease states and spatial distribution of the final epidemic.

# %%
# Get the state tracker instance from the model
state_tracker = None
for instance in model.instances:
    if isinstance(instance, StateTracker):
        state_tracker = instance
        break

if state_tracker is None:
    raise RuntimeError("StateTracker not found in model instances")

# Create comprehensive visualization
fig = plt.figure(figsize=(15, 12))

# Plot 1: Time series of total S, I, R across all nodes
ax1 = plt.subplot(2, 3, 1)
time_steps = np.arange(nticks)
ax1.plot(time_steps, state_tracker.S, 'b-', label='Susceptible', linewidth=2)
ax1.plot(time_steps, state_tracker.I, 'r-', label='Infected', linewidth=2)
ax1.plot(time_steps, state_tracker.R, 'g-', label='Recovered', linewidth=2)
ax1.set_xlabel('Time (bi-weekly periods)')
ax1.set_ylabel('Number of Individuals')
ax1.set_title('Disease Dynamics Over Time')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Cumulative incidence (total who got infected)
ax2 = plt.subplot(2, 3, 2)
cumulative_incidence = state_tracker.R + state_tracker.I
ax2.plot(time_steps, cumulative_incidence, 'purple', linewidth=2)
ax2.set_xlabel('Time (bi-weekly periods)')
ax2.set_ylabel('Cumulative Cases')
ax2.set_title('Cumulative Incidence')
ax2.grid(True, alpha=0.3)

# Plot 3: Attack rate by node (final % infected)
ax3 = plt.subplot(2, 3, 3)
final_recovered = model.patches.states[2] + model.patches.states[1]  # R + I
initial_population = scenario_data['pop'].to_numpy()
attack_rates = (final_recovered / initial_population) * 100
bars = ax3.bar(range(n_nodes), attack_rates, color='coral')
ax3.set_xlabel('Node ID')
ax3.set_ylabel('Attack Rate (%)')
ax3.set_title('Final Attack Rate by Node')
ax3.set_xticks(range(n_nodes))
ax3.set_xticklabels([f'N{i+1}' for i in range(n_nodes)], rotation=45)
ax3.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

# Plot 4: Spatial distribution of final states
ax4 = plt.subplot(2, 3, 4)
coords_array = np.array(coordinates)
# Size points by population, color by attack rate
scatter = ax4.scatter(coords_array[:, 1], coords_array[:, 0], 
                     s=populations/500, c=attack_rates, 
                     cmap='Reds', alpha=0.7, edgecolors='black')
ax4.set_xlabel('Longitude')
ax4.set_ylabel('Latitude')
ax4.set_title('Spatial Attack Rate Distribution')
plt.colorbar(scatter, ax=ax4, label='Attack Rate (%)')

# Add node labels
for i, (lon, lat) in enumerate(coordinates):
    ax4.annotate(f'N{i+1}', (lon, lat), xytext=(5, 5), 
                textcoords='offset points', fontsize=8)

# Plot 5: Epidemic curve (new infections per time step)
ax5 = plt.subplot(2, 3, 5)
new_infections = np.diff(state_tracker.R + state_tracker.I, prepend=150)  # Initial infections
ax5.plot(time_steps, new_infections, 'red', linewidth=1)
ax5.set_xlabel('Time (bi-weekly periods)')
ax5.set_ylabel('New Infections')
ax5.set_title('Epidemic Curve')
ax5.grid(True, alpha=0.3)

# Plot 6: Population dynamics over time
ax6 = plt.subplot(2, 3, 6)
total_pop_over_time = state_tracker.S + state_tracker.I + state_tracker.R
ax6.plot(time_steps, total_pop_over_time, 'black', linewidth=2)
ax6.set_xlabel('Time (bi-weekly periods)')
ax6.set_ylabel('Total Population')
ax6.set_title('Population Growth')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary Statistics
# 
# Calculate and display key epidemiological metrics from the simulation.

# %%
print("=== SIMULATION SUMMARY ===")
print(f"Simulation period: {years} years ({nticks} bi-weekly time steps)")
print(f"Number of nodes: {n_nodes}")
print(f"Initial population: {initial_population.sum():,}")
print(f"Final population: {model.patches.states.sum():,}")

print("\n=== EPIDEMIC METRICS ===")
total_final_recovered = model.patches.states[2].sum()
total_final_infected = model.patches.states[1].sum()
total_cases = total_final_recovered + total_final_infected
overall_attack_rate = (total_cases / initial_population.sum()) * 100

print(f"Total cases: {total_cases:,}")
print(f"Overall attack rate: {overall_attack_rate:.1f}%")
print(f"Peak infected: {state_tracker.I.max():,}")
print(f"Peak time: Bi-week {np.argmax(state_tracker.I)}")

print(f"\n=== NODE-SPECIFIC ATTACK RATES ===")
for i in range(n_nodes):
    node_attack_rate = attack_rates[i]
    print(f"Node {i+1}: {node_attack_rate:.1f}% (pop: {initial_population[i]:,})")

print("\n=== TRANSMISSION PARAMETERS ===")
print(f"Base transmission rate (beta): {infection_params.beta}")
print(f"Seasonality factor: {infection_params.seasonality}")
print(f"Seasonal peak: Bi-week {infection_params.season_start}")

print("\nTutorial completed successfully!")

# %%