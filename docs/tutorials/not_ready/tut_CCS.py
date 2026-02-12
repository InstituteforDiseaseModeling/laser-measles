# %% [markdown]
# # Exploring the critical community size of an SIR model
#
# Use multiple nodes with no connection to identify the critical community size and its dependence on disease and demographic parameters

# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from laser.core.propertyset import PropertySet

from laser_measles.abm import Model
from laser_measles.abm.components import BirthsConstantPopProcess
from laser_measles.abm.components import BirthsParams
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

nticks = 50 * 365
npatches = 61
pops = np.logspace(3, 6, npatches)
scenario = pd.DataFrame({"name": [str(i) for i in range(npatches)], "population": pops})

# np.random.seed(5)  # Ensure reproducibility
nsims = 200
R0_samples = np.random.uniform(3, 16, nsims)
infmean_samples = 5 + np.random.gamma(2, 10, nsims)
cbr_samples = 10 + np.random.gamma(2, 20, nsims)
i = 0
outputs = np.zeros((nsims, nticks, npatches))
# Create a folder to store the outputs
output_folder = os.path.abspath(os.path.join(os.getcwd(), "CCSSIRoutputs2"))
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
            "exp_mu": np.log(8),  # exposure parameters for SEIR model
            "exp_sigma": 0.5,
            "inf_shape": 1.0,  # shape parameter for gamma distribution
        }
    )

    mu = (1 + parameters.cbr / 1000) ** (1 / 365) - 1

    model = Model(scenario, parameters)
    # Create component parameters
    births_params = BirthsParams(cbr=parameters.cbr, nticks=parameters.nticks)
    susceptibility_params = SusceptibilityParams(nticks=parameters.nticks)
    transmission_params = TransmissionParams(
        nticks=parameters.nticks,
        beta=parameters.beta,
        exp_mu=parameters.exp_mu,
        exp_sigma=parameters.exp_sigma,
        inf_mean=parameters.inf_mean,
        inf_shape=parameters.inf_shape,
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
        create_component(InfectAgentsInPatchProcess, params=importation_params),
    ]

    # Start them slightly asynchronously - different initial susceptibilities, infection only in 1 patch
    # Want to see how connectivity drives correlation over time.
    for j in range(npatches):
        set_initial_susceptibility_in_patch(model, j, 1 / R0 + 0.1 / R0 * np.random.normal())

    model.run()
    outputs[i, :, :] = model.patches.cases
    np.save(f"{output_folder}/CCSSIRoutputs_{i}.npy", outputs[i, :, :])
    i += 1

# %%
params_df = pd.DataFrame({"R0": R0_samples, "infmean": infmean_samples, "cbr": cbr_samples})

params_df.to_csv(os.path.join(output_folder, "params.csv"), index=False)

# %%
plt.imshow(outputs[26, 7300:, :].T / pops[:, np.newaxis], aspect="auto", origin="lower")
plt.colorbar(label="Cases")
plt.xlabel("Time (days)")
plt.ylabel("Patch")
plt.yticks(range(0, npatches, 10), np.log10(pops[::10]))
plt.title("Infection spread over time across patches")
plt.show()

# %%
output_folder = os.path.abspath(os.path.join(os.getcwd(), "..", "..", "laser-generic-outputs", "CCSSIRoutputs2"))
params_df = pd.read_csv(os.path.join(output_folder, "params.csv"))

outputs = []
nsims = 200
npatches = 61
pops = np.logspace(3, 6, npatches)

for i in range(nsims):
    output_file = os.path.join(output_folder, f"CCSSIRoutputs_{i}.npy")
    outputs.append(np.load(output_file))

outputs = np.array(outputs)

# %%
CCS1 = []
CCS2 = []

for sim in range(nsims):
    end_output = outputs[sim, -1, :]
    zero_pops = pops[end_output == 0]
    nonzero_pops = pops[end_output != 0]

    if len(zero_pops) > 0:
        CCS2.append(np.max(zero_pops))
    else:
        CCS2.append(None)

    if len(nonzero_pops) > 0:
        CCS1.append(np.min(nonzero_pops))
    else:
        CCS1.append(None)

results_df = pd.DataFrame({"largest_zero_pop": CCS2, "smallest_nonzero_pop": CCS1})

print(results_df)

# %%
fig, axs = plt.subplots(3, 2, figsize=(15, 15))

# Plot largest_zero_pop against R0, infmean, and cbr
axs[0, 0].scatter(params_df["R0"], results_df["largest_zero_pop"])
axs[0, 0].set_xlabel("R0")
axs[0, 0].set_ylabel("Largest Zero Pop")
axs[0, 0].set_title("Largest Zero Pop vs R0")
axs[0, 0].set_yscale("log")

axs[1, 0].scatter(params_df["infmean"], results_df["largest_zero_pop"])
axs[1, 0].set_xlabel("Infectious Mean Period")
axs[1, 0].set_ylabel("Largest Zero Pop")
axs[1, 0].set_title("Largest Zero Pop vs Infectious Mean Period")
axs[1, 0].set_yscale("log")

axs[2, 0].scatter(params_df["cbr"], results_df["largest_zero_pop"])
axs[2, 0].set_xlabel("Contact Birth Rate")
axs[2, 0].set_ylabel("Largest Zero Pop")
axs[2, 0].set_title("Largest Zero Pop vs Contact Birth Rate")
axs[2, 0].set_yscale("log")

# Plot smallest_nonzero_pop against R0, infmean, and cbr
axs[0, 1].scatter(params_df["R0"], results_df["smallest_nonzero_pop"])
axs[0, 1].set_xlabel("R0")
axs[0, 1].set_ylabel("Smallest Nonzero Pop")
axs[0, 1].set_title("Smallest Nonzero Pop vs R0")
axs[0, 1].set_yscale("log")

axs[1, 1].scatter(params_df["infmean"], results_df["smallest_nonzero_pop"])
axs[1, 1].set_xlabel("Infectious Mean Period")
axs[1, 1].set_ylabel("Smallest Nonzero Pop")
axs[1, 1].set_title("Smallest Nonzero Pop vs Infectious Mean Period")
axs[1, 1].set_yscale("log")

axs[2, 1].scatter(params_df["cbr"], results_df["smallest_nonzero_pop"])
axs[2, 1].set_xlabel("Contact Birth Rate")
axs[2, 1].set_ylabel("Smallest Nonzero Pop")
axs[2, 1].set_title("Smallest Nonzero Pop vs Contact Birth Rate")
axs[2, 1].set_yscale("log")

plt.tight_layout()
plt.show()

# %%

fig = plt.figure(figsize=(14, 7))

# Surface plot for largest_zero_pop
ax1 = fig.add_subplot(121, projection="3d")
ax1.plot_trisurf(params_df["infmean"], params_df["cbr"], results_df["largest_zero_pop"], cmap="viridis")
ax1.set_xlabel("Infectious Mean Period")
ax1.set_ylabel("Contact Birth Rate")
ax1.set_zlabel("Largest Zero Pop")
ax1.set_zscale("log")

ax1.set_title("Largest Zero Pop vs Infectious Mean Period and Contact Birth Rate")

# Surface plot for smallest_nonzero_pop
ax2 = fig.add_subplot(122, projection="3d")
ax2.plot_trisurf(params_df["infmean"], params_df["cbr"], results_df["smallest_nonzero_pop"], cmap="viridis")
ax2.set_xlabel("Infectious Mean Period")
ax2.set_ylabel("Contact Birth Rate")
ax2.set_zlabel("Smallest Nonzero Pop")
ax2.set_title("Smallest Nonzero Pop vs Infectious Mean Period and Contact Birth Rate")
ax2.set_zscale("log")
plt.tight_layout()
plt.show()

# %%
# from scipy.optimize import curve_fit

# # Calculate alpha
# alpha = params_df['infmean'] * params_df['cbr']
# R0 = params_df['R0']

# # Define the fitting function
# def fitting_function(alpha, R0, constant, a, b, c):
#     return constant * alpha**a * (R0)**b * (R0-1)**c

# # Prepare the data for fitting'
# # Drop NA values from smallest_nonzero_pop and corresponding entries from alpha and R0
# valid_indices = ~results_df['smallest_nonzero_pop'].isna()
# alpha_values = alpha[valid_indices].values
# R0_values = params_df['R0'][valid_indices].values
# smallest_nonzero_pop_values = results_df['smallest_nonzero_pop'][valid_indices].values


# # Fit the function to the data
# popt, pcov = curve_fit(lambda alpha, constant, a, b, c: fitting_function(alpha, R0_values, constant, a, b, c), alpha_values, smallest_nonzero_pop_values)

# # Extract the optimal parameters
# constant_opt, a_opt, b_opt, c_opt = popt
print(f"Optimal parameters: constant = {constant_opt}, a = {a_opt}, b = {b_opt}")
# # Plot smallest_nonzero_pop against alpha
# plt.figure()

# # Plot the best fit line
# #
# # Create a meshgrid for alpha and R0 values
# alpha_fit = np.linspace(min(alpha_values), max(alpha_values), 100)
# R0_fit = np.linspace(min(R0_values), max(R0_values), 100)
# alpha_fit, R0_fit = np.meshgrid(alpha_fit, R0_fit)

# # Calculate the best fit surface
# best_fit_surface = fitting_function(alpha_fit, R0_fit, constant_opt, a_opt, b_opt)

# Plot the best fit surface
fig = plt.figure(figsize=(14, 7))
ax = fig.add_subplot(111, projection="3d")

# Calculate 1/alpha
inv_alpha_fit = 1 / alpha_fit
inv_alpha_values = 1 / alpha_values

# Plot the best fit surface
ax.plot_surface(inv_alpha_fit, R0_fit, np.log10(best_fit_surface), cmap="viridis", alpha=0.7)

# Scatter the real values for comparison
ax.scatter(inv_alpha_values, R0_values, np.log10(smallest_nonzero_pop_values), color="red", label="Real Values")

ax.set_xlabel("1/Alpha (1/(inf_mean * cbr))")
ax.set_ylabel("R0")
ax.set_zlabel("Log(Smallest Nonzero Pop)")
ax.set_title("Best Fit Surface and Real Values")

# Add the best fit equation as a textbox
equation_text = f"Best fit: y = {constant_opt:.2e} * alpha^{a_opt:.2f} * (R0/(R0-1))^{b_opt:.2f}"
# plt.text(0.05, 0.95, equation_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

plt.tight_layout()
# Rotate the camera for a better viewing angle
ax.view_init(elev=20, azim=255)
plt.show()

# %%
plt.figure(figsize=(10, 6))
plt.scatter(inv_alpha_values, smallest_nonzero_pop_values)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("1/Alpha (1/(inf_mean * cbr))")
plt.ylabel("Smallest Nonzero Pop")
plt.title("Smallest Nonzero Pop vs 1/Alpha")
plt.grid(True, which="both", ls="--")
plt.show()

# %%
output_folder = r"..\..\laser-generic-outputs\CCSSIRoutputs"

# %%


# %%
