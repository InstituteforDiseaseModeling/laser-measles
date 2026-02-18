# README

Compartmental SEIR model using daily timesteps. This model implements stochastic disease transmission dynamics with four compartments: Susceptible, Exposed, Infected, and Recovered.

## Model Overview

The compartmental model uses:
- **Daily timesteps** (1-day intervals) for fine-grained temporal resolution
- **SEIR compartments**: S → E → I → R disease progression
- **Stochastic transitions** using binomial sampling for realistic variability
- **Spatial mixing** via gravity diffusion based on population and distance
- **Seasonal transmission** with configurable amplitude and timing

## Key Features

### SEIR Dynamics
- **S → E**: Susceptible individuals become exposed based on force of infection
- **E → I**: Exposed individuals become infectious at rate σ (1/incubation_period)
- **I → R**: Infectious individuals recover at rate γ (1/infectious_period)
- **Basic reproduction number**: R₀ = β/γ

### Parameters
- **β (beta)**: Transmission rate per day
- **σ (sigma)**: Progression rate from exposed to infectious (1/incubation_period)
- **γ (gamma)**: Recovery rate from infection (1/infectious_period)
- **Seasonality**: Optional seasonal variation in transmission
- **Spatial mixing**: Gravity model with configurable distance decay

## Example Usage

```python
import polars as pl
import numpy as np
from laser_measles.compartmental import BaseScenario, CompartmentalModel, CompartmentalParams
from laser_measles.compartmental.components import InfectionProcess, StateTracker

# Create scenario data
scenario_data = pl.DataFrame({
    'ids': ['patch_1', 'patch_2', 'patch_3'],
    'lat': [11.0, 12.0, 13.0],
    'lon': [8.0, 9.0, 10.0],
    'pop': [10000, 15000, 12000],
    'mcv1': [0.8, 0.75, 0.85]
})

# Set up parameters
params = CompartmentalParams(
    num_ticks=365,  # 1 year simulation
    beta=0.5,       # Transmission rate per day
    sigma=1.0/8.0,  # 8-day incubation period
    gamma=1.0/5.0,  # 5-day infectious period
    seasonality=0.15,  # 15% seasonal variation
    season_start=90    # Peak in spring (day 90)
)

# Create model
scenario = BaseScenario(scenario_data)
model = CompartmentalModel(scenario, params, name="seir_example")

# Add components
model.components = [
    InfectionProcess,
    StateTracker
]

# Initialize with some initial infections
model.patches.states[2, :] = 10  # Start with 10 infected per patch

# Run simulation
model.run()

# Access results
for instance in model.instances:
    if isinstance(instance, StateTracker):
        # Plot SEIR dynamics
        for _ in instance.plot_combined():
            plt.show()

        # Access individual compartments
        print(f"Final susceptible: {instance.S[-1]}")
        print(f"Peak infections: {instance.I.max()}")
```

## Epidemiological Interpretation

- **Incubation period**: Time from exposure to becoming infectious (1/σ days)
- **Infectious period**: Time spent infectious before recovery (1/γ days)
- **R₀**: Expected number of secondary infections from one infectious individual
- **Seasonality**: Models seasonal variation in transmission (e.g., school terms, weather)
- **Spatial spread**: Disease propagates between patches based on population mixing

## Recommended for

- **Policy analysis**: Evaluate intervention strategies with daily precision
- **Outbreak modeling**: Track disease spread with realistic incubation dynamics
- **Parameter estimation**: Calibrate epidemiological parameters against data
- **Scenario planning**: Compare different control strategies and timing
