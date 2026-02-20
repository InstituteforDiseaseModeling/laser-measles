Quick Start: Spatial ABM 
========================

This tutorial introduces the Agent-Based Model (ABM) in
``laser-measles`` using a minimal spatial example.

This is the **first example script**, so we explain:

- How scenarios are constructed
- How an ABM model is created
- How components define model behavior
- How spatial mixing works
- How to track and visualize results

The model we build is intentionally simple:

- 8 spatial patches arranged in a line
- Heterogeneous population sizes
- Infection seeded in one patch
- Gravity-based spatial mixing
- No births, no waning immunity
- 50 simulated days

The goal is clarity, not realism.

------------------------------------------------------------
1. Scenario Construction
------------------------------------------------------------

Every laser-measles model begins with a **scenario**.

A scenario is a Polars DataFrame with one row per spatial patch.

Required columns:

- ``id`` — unique patch identifier
- ``lat`` — latitude
- ``lon`` — longitude
- ``pop`` — population size
- ``mcv1`` — routine vaccination coverage

In this example, we create 8 patches arranged along a straight line.

.. code-block:: python

    import polars as pl
    import numpy as np

    def create_linear_scenario():

        n_patches = 8

        populations = np.array([
            50_000,
            80_000,
            120_000,
            200_000,
            150_000,
            100_000,
            70_000,
            40_000,
        ])

        scenario = pl.DataFrame({
            "id": [f"patch_{i}" for i in range(n_patches)],
            "lat": np.zeros(n_patches),
            "lon": np.linspace(0, 7, n_patches),
            "pop": populations,
            "mcv1": np.zeros(n_patches),
        })

        return scenario

Here:

- Latitude is fixed at 0.
- Longitude increases linearly.
- Populations vary substantially.

This creates a simple 1-dimensional spatial structure.

------------------------------------------------------------
2. Creating the ABM Model
------------------------------------------------------------

We now instantiate the ABM model.

.. code-block:: python

    from laser.measles.abm import ABMModel, ABMParams

    params = ABMParams(
        num_ticks=50,
        seed=42,
        start_time="2000-01"
    )

    model = ABMModel(
        scenario=scenario,
        params=params
    )

Key parameters:

``num_ticks``
    Number of days to simulate.

``seed``
    Ensures reproducibility.

``start_time``
    Calendar start date (used by seasonal forcing and campaigns).

Unlike compartmental models, the ABM maintains
an explicit individual-level population internally.

------------------------------------------------------------
3. Components: The Engine of the Model
------------------------------------------------------------

In laser-measles, **components define model behavior**.

Each tick, the model executes its components in order.

We will use:

- ``NoBirthsProcess`` — disable births
- ``InfectionSeedingProcess`` — seed infection
- ``InfectionProcess`` — handle transmission and progression
- ``StateTracker`` — record SEIR states

------------------------------------------------------------
4. Spatial Mixing in the ABM
------------------------------------------------------------

This is the most important new concept.

Spatial mixing determines how infectious individuals
in one patch influence infection risk in other patches.

Important distinction:

Compartmental model
    Accepts an explicit ``mixer=`` object.

ABM model
    Does NOT accept ``mixer=`` in ``InfectionParams``.

Instead, the ABM automatically constructs a
gravity mixing model using two parameters:

``distance_exponent``
    Controls how quickly mixing declines with distance.

``mixing_scale``
    Controls the overall magnitude of cross-patch mixing.

This design simplifies usage but reduces flexibility.

Here is the correct ABM configuration:

.. code-block:: python

    infection_params = components.InfectionParams(
        beta=20.0,
        seasonality=0.0,
        distance_exponent=20.0,
        mixing_scale=0.01
    )

Interpretation:

Large ``distance_exponent`` (20)
    Very strong local transmission. Patches mostly mix with neighbors.

Small ``mixing_scale`` (0.01)
    Moderate cross-patch infection pressure.

Mechanically, the ABM does the following each day:

1. Count infectious agents in each patch.
2. Construct a gravity-based mixing matrix.
3. Compute force of infection per patch.
4. Apply infection probability to each susceptible agent.

Agents do not move between patches.
Mixing changes infection probability only.

------------------------------------------------------------
5. Seeding Infection
------------------------------------------------------------

We seed 5 infections in the last patch:

.. code-block:: python

    seeding_params = components.InfectionSeedingParams(
        target_patches=["patch_7"],
        infections_per_patch=5
    )

This allows us to observe spatial spread outward from one location.

------------------------------------------------------------
6. Assembling the Model
------------------------------------------------------------

We now attach components in execution order.

.. code-block:: python

    model.components = [

        components.NoBirthsProcess,

        create_component(
            components.InfectionSeedingProcess,
            seeding_params
        ),

        create_component(
            components.InfectionProcess,
            infection_params
        ),

        components.StateTracker,

        create_component(
            components.StateTracker,
            BaseStateTrackerParams(aggregation_level=0)
        ),
    ]

The second ``StateTracker`` records patch-level data.

------------------------------------------------------------
7. Running the Simulation
------------------------------------------------------------

.. code-block:: python

    model.run()

This advances the simulation for 50 days.

------------------------------------------------------------
8. Extracting Results
------------------------------------------------------------

StateTracker stores results in a structured array:

Shape:
    (num_states, num_ticks, num_patches)

We extract:

- Global SEIR curves
- Final recovered counts
- Patch-level infectious trajectories

------------------------------------------------------------
9. Visualization
------------------------------------------------------------

We generate three panels:

1. Global SEIR fractions
2. Spatial attack rates
3. Patch-level infectious curves

The spatial attack rate shows which patches experienced
the highest cumulative infection.

The patch-level infectious curves show the
temporal spread of infection across space.

------------------------------------------------------------
What This Example Demonstrates
------------------------------------------------------------

This script introduces:

- Scenario construction
- ABM instantiation
- Component architecture
- Individual-level transmission
- Spatial mixing mechanics
- State tracking
- Visualization

While remaining:

- Deterministic in structure
- Simple in design
- Free of demographic complexity

------------------------------------------------------------
How This Differs from the Compartmental Model
------------------------------------------------------------

Mixing configuration differs:

Compartmental:
    ``InfectionParams(mixer=GravityMixing(...))``

ABM:
    ``InfectionParams(distance_exponent=..., mixing_scale=...)``

Mechanically:

Compartmental:
    Mixing redistributes infection pressure across aggregated compartments.

ABM:
    Mixing modifies per-agent infection probability.

------------------------------------------------------------
Next Steps
------------------------------------------------------------

You could extend this model by:

- Adding births (``VitalDynamicsProcess``)
- Adding vaccination campaigns (``SIACalendarProcess``)
- Increasing spatial resolution
- Comparing ABM vs compartmental mixing behavior

This concludes the spatial ABM quick start.
