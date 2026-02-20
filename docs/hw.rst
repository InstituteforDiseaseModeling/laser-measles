Spatial ABM "Hello World"
==========================

This tutorial presents a minimal but fully spatial **Agent-Based Model (ABM)** measles simulation using *laser-measles*.

Unlike a well-mixed “hello world,” this example demonstrates:

* Multiple spatial patches
* Heterogeneous population sizes
* Explicit migration via a gravity mixing model
* Infection seeded in a single patch
* Global and patch-level SEIR tracking
* Spatial visualization of epidemic spread

This is intended to be pedagogical rather than epidemiologically calibrated.

Overview
--------

We construct:

* 8 patches arranged in a linear chain
* Different population sizes per patch
* Moderate gravity-based migration (not fully well-mixed)
* No births, no immune decay
* A short simulation (50 days)

The goal is to illustrate **spatial propagation of infection** without introducing demographic complexity.

Full Script
-----------

.. code-block:: python

    """
    hello_world_spatial_abm.py
    """

    import polars as pl
    import numpy as np
    import matplotlib.pyplot as plt

    from laser.measles.abm import ABMModel, ABMParams, components
    from laser.measles.mixing.gravity import GravityMixing, GravityParams
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

        gravity_params = GravityParams(
            a=1.0,
            b=1.0,
            c=20.0,
            k=0.01
        )

        mixer = GravityMixing(
            scenario=scenario,
            params=gravity_params
        )

        infection_params = components.InfectionParams(
            beta=20.0,
            seasonality=0.0,
            mixer=mixer
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

Scenario Construction
---------------------

The scenario consists of 8 patches arranged in a line:

* Longitude varies linearly
* Latitude is fixed
* Population sizes are heterogeneous

This creates a one-dimensional spatial chain that makes wave propagation easy to visualize.

Gravity Mixing Model
--------------------

Migration between patches is controlled by a gravity mixing matrix:

.. math::

    M_{ij} \propto p_i^{a-1} p_j^{b} d_{ij}^{-c}

Where:

* :math:`p_i` is origin population
* :math:`p_j` is destination population
* :math:`d_{ij}` is distance
* :math:`c` controls distance decay

We set:

* Strong distance decay (:math:`c=20`)
* Moderate mixing scale (:math:`k=0.01`)

This ensures the system is **not well-mixed**.

Epidemiological Assumptions
---------------------------

We deliberately remove demographic processes:

* No births
* No deaths
* No waning immunity

This is achieved using:

.. code-block:: python

    components.NoBirthsProcess

The infection process is SEIR-based with:

* Constant transmission rate (:math:`\beta=20`)
* No seasonality

State Tracking
--------------

We use two trackers:

1. Global tracker (default aggregation)
2. Patch-level tracker (aggregation_level=0)

This produces a state array with shape:

.. code-block:: text

    (num_states, num_ticks, num_patches)

Which allows patch-level epidemic curves to be plotted.

Visualization
-------------

Three panels are produced:

1. Global SEIR fractions
2. Spatial attack rate (bubble plot)
3. Patch-level infectious curves

This clearly shows:

* Initial outbreak location
* Directional spread
* Differential attack rates
* Timing differences across patches

Why This is a Proper Spatial "Hello World"
------------------------------------------

This example demonstrates:

* Multi-node spatial modeling
* Explicit mixing matrices
* Heterogeneous populations
* Local seeding
* Patch-level tracking
* Clean visualization

While avoiding:

* Demographic complexity
* Vaccination dynamics
* Parameter calibration
* Stochastic edge cases

It is minimal but truly spatial.

