import os

import numpy as np
import pandas as pd
from laser_core.propertyset import PropertySet

from laser_measles.abm import Model
from laser_measles.abm.components import BirthsConstantPopProcess
from laser_measles.abm.components import BirthsParams
from laser_measles.abm.components import ExposureProcess
from laser_measles.abm.components import ImportationParams
from laser_measles.abm.components import InfectAgentsInPatchProcess
from laser_measles.abm.components import InfectionParams
from laser_measles.abm.components import InfectionProcess
from laser_measles.abm.components import SusceptibilityParams
from laser_measles.abm.components import SusceptibilityProcess
from laser_measles.abm.components import TransmissionParams
from laser_measles.abm.components import TransmissionProcess
from laser_measles.abm.utils import set_initial_susceptibility_in_patch
from laser_measles.components import create_component

# %load_ext line_profiler

f"{np.__version__=}"

# %%
# %%capture

nticks = 3 * 365
npatches = 61
pops = np.logspace(3, 6, npatches)
scenario = pd.DataFrame({"name": [str(i) for i in range(npatches)], "population": pops})

# np.random.seed(5)  # Ensure reproducibility
nsims = 1
R0_samples = np.random.uniform(3, 16, nsims)
infmean_samples = 5 + np.random.gamma(2, 10, nsims)
cbr_samples = 10 + np.random.gamma(2, 20, nsims)
i = 0
outputs = np.zeros((nsims, nticks + 1, npatches))
# Create a folder to store the outputs
output_folder = os.path.abspath(os.path.join(os.getcwd(), "CCS"))
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
for R0, infmean, cbr in zip(R0_samples, infmean_samples, cbr_samples, strict=False):
    parameters = PropertySet(
        {
            "seed": np.random.randint(0, 1000000),
            "nticks": nticks,
            "verbose": True,
            "beta": R0 / infmean,
            "inf_mean": infmean,
            "cbr": cbr,
            "importation_period": 180,
            "importation_end": 20 * 365,
        }
    )

    mu = (1 + parameters.cbr / 1000) ** (1 / 365) - 1

    model = Model(scenario, parameters)
    # Create component parameters
    births_params = BirthsParams(cbr=parameters.cbr, nticks=parameters.nticks)
    susceptibility_params = SusceptibilityParams(nticks=parameters.nticks)
    transmission_params = TransmissionParams(
        beta=parameters.beta,
        inf_mean=parameters.inf_mean,
    )
    infection_params = InfectionParams(nticks=parameters.nticks)
    importation_params = ImportationParams(
        nticks=parameters.nticks,
        importation_period=parameters.importation_period,
        importation_count=1,
        importation_end=parameters.importation_end,
    )

    model.components = [
        create_component(BirthsConstantPopProcess, params=births_params),
        create_component(SusceptibilityProcess, params=susceptibility_params),
        create_component(TransmissionProcess, params=transmission_params),
        create_component(InfectionProcess, params=infection_params),
        ExposureProcess,
        create_component(InfectAgentsInPatchProcess, params=importation_params),
    ]

    # Start them slightly asynchronously - different initial susceptibilities, infection only in 1 patch
    # Want to see how connectivity drives correlation over time.
    for j in range(npatches):
        set_initial_susceptibility_in_patch(model, j, 1 / R0 + 0.1 / R0 * np.random.normal())

    model.run()
    outputs[i, :, :] = model.patches.test_cases
    np.save(f"{output_folder}/CCSSIRoutputs_{i}.npy", outputs[i, :, :])
    i += 1
