Spatial ABM “Hello World”
==========================

This tutorial presents a minimal but fully spatial Agent-Based Model (ABM)
measles simulation using ``laser-measles``.

Unlike a well-mixed “hello world,” this example demonstrates:

- Multiple spatial patches
- Heterogeneous population sizes
- Explicit gravity-based spatial mixing
- Infection seeded in a single patch
- Global and patch-level SEIR tracking
- Spatial visualization of epidemic spread

This example is intentionally pedagogical rather than epidemiologically calibrated.

Overview
--------

We construct:

- 8 patches arranged in a linear chain
- Different population sizes per patch
- Moderate gravity-based mixing (not fully well-mixed)
- No births and no waning immunity
- A short simulation (50 days)

The goal is to illustrate **spatial propagation of infection**
without introducing demographic complexity.

Key Concept: Mixing in the ABM Model
-------------------------------------

In the ABM model, spatial mixing is handled differently than in the
compartmental model.

In the **compartmental model**, mixing is configured by passing an explicit
``mixer=`` object (e.g. ``GravityMixing``) into ``InfectionParams``.

In contrast, the **ABM model does not accept a ``mixer=`` argument**
in ``InfectionParams``.

Instead:

- ``distance_exponent`` controls the distance decay of gravity mixing.
- ``mixing_scale`` controls the overall magnitude of cross-patch mixing.

Internally, the ABM model automatically constructs a gravity mixing model
using those parameters.

This is an important distinction:

- Compartmental → You pass a mixer explicitly.
- ABM → Gravity mixing is built automatically from parameters.

What Mixing Means in ABM
------------------------

In the ABM:

- Individuals have a ``patch_id``.
- They do **not physically move** between patches.
- Instead, infectious pressure is redistributed via a mixing matrix.

Mechanically:

1. Infectious agents are counted in each patch.
2. A gravity-based mixing matrix distributes infectious pressure across patches.
3. Each susceptible agent experiences a probability of infection based on
   the mixed force of infection for their patch.

Thus:

Mixing modifies **infection probability**, not agent location.

Script
------

Below is the complete working example.

.. code-block:: python

    """
    hello_world_spatial_abm.py

    Minimal spatial ABM measles demo with multiple patches and gravity-based mixing.

    Features:
    - 8 spatial patches arranged in a line
    - Heterogeneous population sizes
    - Infection seeded in one patch
    - Gravity-based mixing (configured via distance_exponent + mixing_scale)
    - Patch-level state tracking
    - Clean SEIR + spatial visualization
    """

    import polars as pl
    import numpy as np
    import matplotlib.pyplot as plt

    from laser.measles.abm import ABMModel, ABMParams, components
    from laser.measles import create_component
    from laser.measles.components.base_tracker_state import BaseStateTrackerParams


    # ---------------------------------------------------------
    # Create simple spatial scenario
    # ---------------------------------------------------------

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


    # ---------------------------------------------------------
    # Main simulation
    # ---------------------------------------------------------

    def main():

        print("🌍 Creating spatial scenario...")

        scenario = create_linear_scenario()
        total_pop = scenario["pop"].sum()

        print(f"Total population: {int(total_pop):,}")

        params = ABMParams(
            num_ticks=50,
            seed=42,
            start_time="2000-01"
        )

        model = ABMModel(
            scenario=scenario,
            params=params
        )

        # -----------------------------------------------------
        # Gravity mixing configured CORRECTLY for ABM
        # -----------------------------------------------------
        #
        # In ABM, InfectionParams does NOT accept `mixer=`.
        #
        # Gravity mixing is controlled via:
        #   - distance_exponent
        #   - mixing_scale
        #
        # These parameters internally configure a GravityMixing model.
        # -----------------------------------------------------

        infection_params = components.InfectionParams(
            beta=20.0,
            seasonality=0.0,
            distance_exponent=20.0,   # strong distance decay
            mixing_scale=0.01         # moderate mixing scale
        )

        seeding_params = components.InfectionSeedingParams(
            target_patches=["patch_7"],
            infections_per_patch=5
        )

        model.components = [

            components.NoBirthsProcess,

            create_component(components.InfectionSeedingProcess, seeding_params),

            create_component(components.InfectionProcess, infection_params),

            components.StateTracker,

            create_component(
                components.StateTracker,
                BaseStateTrackerParams(aggregation_level=0)
            ),
        ]

        print("🚀 Running spatial epidemic...")
        model.run()
        print("✅ Simulation complete!")

        trackers = model.get_instance(components.StateTracker)
        global_tracker = trackers[0]
        patch_tracker = trackers[1]

        global_df = global_tracker.get_dataframe()
        global_results = (
            global_df
            .pivot(index="tick", on="state", values="count")
            .sort("tick")
        )

        time = global_results["tick"].to_numpy()
        S = global_results["S"].to_numpy()
        E = global_results["E"].to_numpy()
        I = global_results["I"].to_numpy()
        R = global_results["R"].to_numpy()

        final_R = model.patches.states.R
        patch_pops = scenario["pop"].to_numpy()
        attack_rates = final_R / patch_pops

        state_array = patch_tracker.state_tracker
        I_index = model.params.states.index("I")
        num_patches = state_array.shape[2]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].plot(time, S / total_pop, label="S")
        axes[0].plot(time, E / total_pop, label="E")
        axes[0].plot(time, I / total_pop, label="I")
        axes[0].plot(time, R / total_pop, label="R")
        axes[0].set_title("Global SEIR")
        axes[0].set_xlabel("Days")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].scatter(
            scenario["lon"],
            np.zeros(len(scenario)),
            c=attack_rates,
            s=patch_pops / 2000,
            cmap="Reds",
            edgecolors="black"
        )
        axes[1].set_title("Spatial Attack Rates")
        axes[1].set_xlabel("Patch Position")
        axes[1].set_yticks([])
        axes[1].grid(alpha=0.3)

        for patch_i in range(num_patches):
            axes[2].plot(
                state_array[I_index, :, patch_i],
                label=f"P{patch_i}",
                alpha=0.8
            )

        axes[2].set_title("Patch-Level Infectious Curves")
        axes[2].set_xlabel("Days")
        axes[2].grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

        print("\n🎉 Spatial hello world complete!")


    if __name__ == "__main__":
        main()


Interpreting the Mixing Parameters
----------------------------------

``distance_exponent`` (gravity decay)
    Controls how strongly distance suppresses mixing.

    Higher values → infection stays local.

    Lower values → infection spreads more easily across space.

``mixing_scale``
    Controls overall cross-patch contact rate.

    Larger values → more spatial coupling.

    Smaller values → more isolated patches.

Because the ABM operates at the individual level,
mixing influences **per-agent infection probability**,
rather than directly transferring compartment counts.

Comparison with Compartmental Model
------------------------------------

In the compartmental model:

.. code-block:: python

    InfectionParams(
        beta=...,
        mixer=GravityMixing(...)
    )

You explicitly pass a mixer.

In ABM:

.. code-block:: python

    InfectionParams(
        beta=...,
        distance_exponent=...,
        mixing_scale=...
    )

No ``mixer=`` argument exists.

Internally, ABM constructs its own gravity mixer.

This design simplifies usage but reduces flexibility.

Summary
-------

This example demonstrates:

- Spatial structure
- Gravity mixing in ABM
- Patch-level tracking
- Visual epidemic wave propagation

It is minimal but truly spatial.

To extend this example, you could:

- Add births (``VitalDynamicsProcess``)
- Add SIA campaigns
- Compare different gravity decay parameters
- Switch to the compartmental model for comparison

``laser-measles`` supports all of these within the same component architecture.