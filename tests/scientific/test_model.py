import itertools

import numpy as np
import pandas as pd
import pytest
from laser_core import PropertySet
from scipy.optimize import curve_fit
from scipy.optimize import fsolve

from laser_generic import Births_ConstantPop
from laser_generic import Model
from laser_generic import Susceptibility
from laser_generic import Transmission
from laser_generic.infection import Infection
from laser_generic.infection import Infection_SIS
from laser_generic.utils import seed_infections_randomly
from laser_generic.utils import seed_infections_randomly_SI
from laser_generic.utils import set_initial_susceptibility_randomly


def SI_logistic(t, beta, size, t0):
    return size / (1 + (size - 1) * np.exp(-beta * (t - t0)))


def SI_logistic_cbr(t, beta, popsize, cbr, t0):
    mu = (1 + cbr / 1000) ** (1 / 365) - 1
    x = 1 - mu / beta
    return popsize * x / (1 + (popsize * x - 1) * np.exp(-beta * x * (t - t0)))


def SIS_logistic(t, beta, popsize, gamma, t0):
    x = 1 - gamma / beta
    return popsize * x / (1 + (popsize * x - 1) * np.exp(-beta * x * (t - t0)))


def KM_limit(z, R0, S0, I0):
    if R0 * S0 < 1:
        return 0
    else:
        return z - S0 * (1 - np.exp(-R0 * (z + I0)))


@pytest.mark.skip("long running test")
@pytest.mark.modeltest
def test_si_model_nobirths():
    """
    Test the SI model without births.

    This test simulates an SI (Susceptible-Infected) model for a population of 100,000.
    The number of cases is fitted to a logistic function, and the fitted beta values are compared
    to the original beta values to ensure they are within 5% of each other.

    Steps:
    1. Initialize parameters and scenario.
    2. Run the model for each combination of seed and beta.
    3. Fit the number of cases to a logistic function.
    4. Assert that the fitted beta values to the original beta values.

    Asserts:
    - The relative difference between the original and fitted beta values is less than 5%.

    """
    nticks = 730
    t = np.arange(730)
    pop = 1e5

    seeds = list(range(10))
    betas = [0.01 * i for i in range(1, 11)]
    scenario = pd.DataFrame(data=[["homenode", pop, "47°36′35″N 122°19′59″W"]], columns=["name", "population", "location"])
    for seed, beta in zip(seeds, betas):
        parameters = PropertySet({"seed": seed, "nticks": nticks, "verbose": True, "beta": beta})
        model = Model(scenario, parameters)
        model.components = [
            Susceptibility,
            Transmission,
        ]
        seed_infections_randomly_SI(model, ninfections=1)
        model.run()
        cases = [model.patches.cases[i][0] for i in range(nticks)]
        popt, pcov = curve_fit(SI_logistic, t, cases, p0=[0.05, 1.1e5, 1])

        output = pd.DataFrame.from_dict(
            {
                "seed": seed,
                "beta": beta,
                "cases": [np.array(cases)],
                "fitted_beta": popt[0],
                "fitted_size": popt[1],
                "fitted_t0": popt[2],
            }
        )

    assert np.all(np.abs((output["beta"] - output["fitted_beta"]) / output["beta"]) < 0.05)


@pytest.mark.skip("long running test")
@pytest.mark.modeltest
def test_si_model_wbirths():
    """
    Test the SI model with births.

    This test initializes a scenario with a population of 1 million.
    The number of cases is fitted to a logistic function, and the fitted beta values are compared
    to the original beta and crude birth rates (cbr) values to ensure they are within 5% of each other.

    The test asserts that the fitted beta and cbr values are within 5% of the original values.

    Steps:
    1. Initialize parameters and scenario.
    2. Run the model for each combination of seed, beta, and cbr.
    3. Fit the number of cases to a logistic function.
    4. Assert that the fitted beta values to the original beta values.

    Raises:
    AssertionError: If the fitted beta or cbr values deviate more than 5% from the original values.
    """
    seeds = [42 + i for i in range(10)]
    pop = 1e6
    nticks = 3650
    betas = [0.002 + 0.005 * i for i in range(1, 11)]
    cbrs = np.random.randint(10, 120, 10)
    scenario = pd.DataFrame(data=[["homenode", pop]], columns=["name", "population"])

    for seed, beta, cbr in zip(seeds, betas, cbrs):
        parameters = PropertySet({"seed": seed, "nticks": nticks, "verbose": True, "beta": beta, "cbr": cbr})
        model = Model(scenario, parameters)
        model.components = [
            Births_ConstantPop,
            Susceptibility,
            Transmission,
        ]
        seed_infections_randomly_SI(model, ninfections=3)
        model.run()
        cases = [model.patches.cases[i][0] for i in range(nticks)]
        popt, pcov = curve_fit(
            SI_logistic_cbr,
            np.arange(nticks),
            cases,
            p0=[beta * (1 + 0.1 * np.random.normal()), pop, cbr * (1 + 0.1 * np.random.normal()), 1],
            bounds=([0, pop - 1, 0, -100], [1, pop + 1, 600, 100]),
        )
        output = pd.DataFrame.from_dict(
            {
                "seed": seed,
                "beta": beta,
                "cbr": cbr,
                "cases": [np.array(cases)],
                "fitted_beta": popt[0],
                "fitted_size": popt[1],
                "fitted_cbr": popt[2],
                "fitted_t0": popt[3],
            }
        )

    assert np.all(np.abs((output["beta"] - output["fitted_beta"]) / output["beta"]) < 0.05)
    assert np.all(np.abs((output["cbr"] - output["fitted_cbr"]) / output["cbr"]) < 0.05)


@pytest.mark.skip("long running test")
@pytest.mark.modeltest
def test_sir_nobirths():
    """
    Test the SIR model without births.

    This test simulates the SIR model over a specified number of ticks, with a range of beta and gamma values,
    and verifies that the fitted parameters are within acceptable bounds of the original parameters.

    The test performs the following steps:
    1. Initialize parameters and scenario.
    2. Run the model for each combination of seed, beta, and gamma.
    3. Assert that the relative error between the original and fitted beta values is less than 5%.
    4. Assert that the relative error between the original and fitted gamma values is less than 10%.

    Raises:
        AssertionError: If the relative error between the original and fitted parameters exceeds the specified bounds.
    """
    nticks = 3000
    t = np.arange(nticks)
    betarange = [0.03, 0.1]
    gammarange = [1 / 200, 1 / 50]
    seeds = list(range(10))
    pop = 3e5
    betas = np.random.uniform(betarange[0], betarange[1], 10)
    gammas = np.random.uniform(gammarange[0], gammarange[1], 10)
    output = pd.DataFrame(columns=["seed", "beta", "gamma", "cases", "fitted_beta", "fitted_gamma", "fitted_t0"])
    scenario = pd.DataFrame(data=[["homenode", pop]], columns=["name", "population"])

    for seed, beta, gamma in zip(seeds, betas, gammas):
        parameters = PropertySet({"seed": seed, "nticks": nticks, "verbose": True, "beta": beta, "inf_mean": 1 / gamma})
        model = Model(scenario, parameters)
        model.components = [
            Infection_SIS,
            Susceptibility,
            Transmission,
        ]
        seed_infections_randomly(model, ninfections=3)
        model.run()
        cases = [model.patches.cases[i][0] for i in range(nticks)]
        popt, pcov = curve_fit(
            SIS_logistic,
            t,
            cases,
            p0=[np.mean(betarange), pop, np.mean(gammarange), 1],
            bounds=([betarange[0] / 2, pop - 1, gammarange[0] / 2, -300], [betarange[1] * 2, pop + 1, gammarange[1] * 2, 300]),
        )

        output = pd.DataFrame.from_dict(
            {
                "seed": seed,
                "beta": beta,
                "gamma": gamma,
                "cases": [np.array(cases)],
                "fitted_beta": popt[0],
                "fitted_gamma": popt[2],
                "fitted_t0": popt[3],
            }
        )

    assert np.all(np.abs((output["beta"] - output["fitted_beta"]) / output["beta"]) < 0.05)
    assert np.all(np.abs((output["gamma"] - output["fitted_gamma"]) / output["gamma"]) < 0.1)


@pytest.mark.skip("long running test")
@pytest.mark.modeltest
def test_sir_nobirths_outbreak():
    """
    Test the SIR model without births during an outbreak scenario.

    This test simulates an SIR (Susceptible-Infectious-Recovered) model without births
    to evaluate the expected and observed number of infections and susceptible individuals
    at the end of the outbreak. The test uses a range of basic reproduction numbers (R0)
    and initial susceptible fractions (S0) to generate different scenarios.

    The test performs the following steps:
    1. Initialize population size, infection mean duration, and initial number of infections.
    2. Calculate the expected final number of infections (I_inf_exp) and susceptibles (S_inf_exp)
       using the Kermack-McKendrick limit.
    3. Assert that the expected and observed values are close within a tolerance of 0.05.

    Raises:
        AssertionError: If the expected and observed values for infections or susceptibles
                        are not within the specified tolerance.
    """
    population = 1e5
    inf_mean = 20
    init_inf = 20

    R0s = np.concatenate((np.linspace(0.2, 1.0, 5), np.linspace(1.5, 10.0, 25)))
    S0s = [1.0, 0.8, 0.6, 0.4, 0.2]
    output = pd.DataFrame(list(itertools.product(R0s, S0s)), columns=["R0", "S0"])
    output["I_inf_exp"] = [
        fsolve(KM_limit, 0.5 * (R0 * S0 >= 1), args=(R0, S0, init_inf / population))[0] for R0, S0 in zip(output["R0"], output["S0"])
    ]
    output["S_inf_exp"] = output["S0"] - output["I_inf_exp"]
    output["I_inf_obs"] = np.nan
    output["S_inf_obs"] = np.nan

    for index, row in output.iterrows():
        scenario = pd.DataFrame(data=[["homenode", population]], columns=["name", "population"])
        parameters = PropertySet({"seed": 2, "nticks": 1460, "verbose": True, "inf_mean": inf_mean, "beta": row["R0"] / inf_mean})

        model = Model(scenario, parameters)
        model.components = [
            Susceptibility,
            Transmission,
            Infection,
        ]
        set_initial_susceptibility_randomly(model, row["S0"])
        seed_infections_randomly(model, ninfections=init_inf)
        model.run()

        output.loc[index, "I_inf_obs"] = (
            np.sum(model.patches.incidence) + init_inf
        ) / population  # incidence doesn't count the imported infections
        output.loc[index, "S_inf_obs"] = model.patches.susceptibility[-1] / population

    assert (np.isclose(output["S_inf_exp"], output["S_inf_obs"], atol=0.05)).all()
    assert (np.isclose(output["I_inf_exp"], output["I_inf_obs"], atol=0.05)).all()