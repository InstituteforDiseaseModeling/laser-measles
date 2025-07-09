# ruff: noqa: I001
# %% [markdown]
# # Creating Custom Components
#
# This tutorial demonstrates how to create custom components for the compartmental model.
# We'll build an MCV1 booster component that periodically strengthens vaccination coverage.

# %%
import polars as pl
from pydantic import BaseModel, Field

import laser_measles as lm
from laser_measles.base import BasePhase
from laser_measles.utils import cast_type


# %% [markdown]
# ## Component Architecture
#
# Components in laser-measles follow a standard pattern:
# 1. **Parameter Class**: Pydantic model defining component parameters
# 2. **Component Class**: Inherits from `BasePhase` and implements the logic
# 3. **Integration**: Add to model's component list and run simulation

# %% [markdown]
# ## Creating the MCV1 Booster Component
#
# Let's create a component that periodically boosts MCV1 coverage to simulate
# vaccination campaigns that occur every year for one month.


# %%
class MCV1BoosterParams(BaseModel):
    """Parameters for the MCV1 booster component."""

    boost_strength: float = Field(default=0.15, description="Percentage increase in vaccination coverage during boost", ge=0.0, le=1.0)
    boost_interval: int = Field(default=365, description="Days between vaccination campaigns", gt=0)
    boost_duration: int = Field(default=30, description="Duration of vaccination campaign in days", gt=0)
    start_day: int = Field(default=90, description="Day to start first vaccination campaign", ge=0)


class MCV1BoosterProcess(BasePhase):
    """
    Component that implements periodic MCV1 vaccination campaigns.

    This component simulates vaccination campaigns that occur at regular intervals,
    moving susceptible individuals to the recovered compartment based on boost strength.
    """

    def __init__(self, model, params: MCV1BoosterParams | None = None, verbose: bool = False):
        super().__init__(model, verbose)
        self.params = params if params is not None else MCV1BoosterParams()

    def __call__(self, model, tick: int):
        """Execute vaccination campaign if within boost period."""
        # Check if we're in a boost period
        if not self._is_boost_active(tick):
            return

        # Get current states
        states = model.patches.states

        # Calculate number to vaccinate in each patch
        # Use binomial sampling for realistic vaccination
        susceptible_counts = states.S
        vaccinated = cast_type(model.prng.binomial(susceptible_counts, self.params.boost_strength), states.dtype, round=True)

        # Move vaccinated individuals from S to R
        states.S -= vaccinated
        states.R += vaccinated

        if self.verbose and vaccinated.sum() > 0:
            print(f"Day {tick}: Vaccinated {vaccinated.sum()} individuals")

    def _is_boost_active(self, tick: int) -> bool:
        """Check if current tick is within an active boost period."""
        if tick < self.params.start_day:
            return False

        # Calculate days since start
        days_since_start = tick - self.params.start_day

        # Check if we're in a boost period
        cycle_position = days_since_start % self.params.boost_interval
        return cycle_position < self.params.boost_duration

    def initialize(self, model):
        """Initialize component (no special initialization needed)."""
        return


# %% [markdown]
# ## Testing the Component
#
# Let's create two simulations: one with the booster component and one without,
# to see the impact on disease transmission.


# %%
def run_simulation(use_booster: bool = True, num_ticks: int = 730) -> tuple:
    """Run a simulation with or without the MCV1 booster component."""

    # Create scenario with low initial MCV1 coverage
    scenario = lm.compartmental.BaseScenario(lm.scenarios.synthetic.single_patch_scenario(population=100_000, mcv1_coverage=0.3))

    # Create model parameters
    params = lm.compartmental.CompartmentalParams(num_ticks=num_ticks, verbose=False, seed=42)

    # Create and configure model
    model = lm.compartmental.CompartmentalModel(scenario, params)

    # Base components for all simulations
    components = [
        # Initialize with some immune individuals based on MCV1 coverage
        lm.create_component(
            lm.compartmental.components.InitializeEquilibriumStatesProcess, lm.compartmental.components.InitializeEquilibriumStatesParams()
        ),
        # Seed infection to start outbreak
        lm.create_component(
            lm.compartmental.components.InfectionSeedingProcess, lm.compartmental.components.InfectionSeedingParams(num_infections=50)
        ),
        # Disease transmission
        lm.compartmental.components.InfectionProcess(model),
        # Track states over time
        lm.compartmental.components.StateTracker(model),
    ]

    # Add booster component if requested
    if use_booster:
        booster_params = MCV1BoosterParams(
            boost_strength=0.20,  # 20% of susceptibles vaccinated
            boost_interval=365,  # Annual campaigns
            boost_duration=30,  # Month-long campaigns
            start_day=90,  # Start after 3 months
        )
        components.append(MCV1BoosterProcess(model, booster_params))

    model.components = components

    # Run simulation
    model.run()

    # Get results
    state_tracker = model.get_instance(lm.compartmental.components.StateTracker)[0]
    results = state_tracker.get_state_counts()

    return model, results


# %% [markdown]
# ## Comparison of Results
#
# Let's run both simulations and compare the outcomes.

# %%
print("Running simulation without MCV1 booster...")
model_no_booster, results_no_booster = run_simulation(use_booster=False)

print("Running simulation with MCV1 booster...")
model_with_booster, results_with_booster = run_simulation(use_booster=True)

print("\n" + "=" * 50)
print("SIMULATION RESULTS COMPARISON")
print("=" * 50)

# %%
# Compare final outcomes
final_no_booster = results_no_booster.tail(1)
final_with_booster = results_with_booster.tail(1)

print(f"\nFinal Results (Day {final_no_booster['tick'][0]}):")
print(f"{'Metric':<20} {'No Booster':<15} {'With Booster':<15} {'Difference':<15}")
print("-" * 65)

for state in ["S", "E", "I", "R"]:
    no_boost_val = final_no_booster[state][0]
    with_boost_val = final_with_booster[state][0]
    difference = with_boost_val - no_boost_val

    print(f"{state + ' (final)':<20} {no_boost_val:<15,} {with_boost_val:<15,} {difference:<15,}")

# %%
# Calculate attack rates (fraction of population that got infected)
total_pop = 100_000

attack_rate_no_booster = (final_no_booster["R"][0] / total_pop) * 100
attack_rate_with_booster = (final_with_booster["R"][0] / total_pop) * 100

print("\nAttack Rates:")
print(f"No Booster:    {attack_rate_no_booster:.1f}%")
print(f"With Booster:  {attack_rate_with_booster:.1f}%")
print(f"Reduction:     {attack_rate_no_booster - attack_rate_with_booster:.1f} percentage points")

# %%
# Find peak infections
peak_no_booster = results_no_booster.select(pl.col("I").max()).item()
peak_with_booster = results_with_booster.select(pl.col("I").max()).item()

print("\nPeak Infections:")
print(f"No Booster:    {peak_no_booster:,}")
print(f"With Booster:  {peak_with_booster:,}")
print(f"Reduction:     {peak_no_booster - peak_with_booster:,} ({100 * (peak_no_booster - peak_with_booster) / peak_no_booster:.1f}%)")

# %% [markdown]
# ## Key Insights
#
# This tutorial demonstrates:
#
# 1. **Component Structure**: Parameter class + component class inheriting from `BasePhase`
# 2. **Pydantic Validation**: Use Field() with constraints for robust parameter handling
# 3. **Model Integration**: Components are added to model.components list
# 4. **State Manipulation**: Direct access to model.patches.states for SEIR compartments
# 5. **Stochastic Sampling**: Use binomial distribution for realistic vaccination
# 6. **Timing Logic**: Implement periodic behavior using modulo arithmetic
#
# The MCV1 booster component successfully reduces both peak infections and overall
# attack rates, demonstrating the public health impact of vaccination campaigns.

# %% [markdown]
# ## Best Practices
#
# When creating components:
# - Use Pydantic BaseModel for parameters with proper validation
# - Inherit from BasePhase for components that run each tick
# - Use model.prng for reproducible random number generation
# - Include verbose logging for debugging
# - Follow Google docstring conventions
# - Test components with simple scenarios first
