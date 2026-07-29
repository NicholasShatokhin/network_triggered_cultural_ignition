"""Embed the local ignition curve in a conditional Poisson population model.

The conditional mean eligible exposure is read from ``data/cosmological_scenario.json``.
It is deliberately treated as an illustrative scenario parameter rather than
as an empirical estimate produced by this repository. It is defined conditional
on a planet already having reached complex animal-grade life, so that
``f_complex`` is not counted twice. A sensitivity table is
written for all exposure values listed in that file.
"""
from __future__ import annotations

from pathlib import Path
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT = Path(os.environ.get("CULTURAL_IGNITION_OUT", ROOT / "results" / "reproduced"))
OUT.mkdir(parents=True, exist_ok=True)
DENSE = DATA_DIR / "dense_phase_grid_stage9.csv"
SCENARIO_FILE = DATA_DIR / "cosmological_scenario.json"

scenario = json.loads(SCENARIO_FILE.read_text(encoding="utf-8"))
E_BAR = float(scenario["fiducial_conditional_mean_eligible_exposure_gyr"])
EXPOSURE_SENSITIVITY = [float(value) for value in scenario["exposure_sensitivity_gyr"]]
if E_BAR <= 0 or any(value <= 0 for value in EXPOSURE_SENSITIVITY):
    raise ValueError("Eligible-exposure scenarios must be positive")

df = pd.read_csv(DENSE)
g = df[df.criterion == "primary"].copy()
x = g.contact_probability.to_numpy()
duration_scaled = g.duration_generations.to_numpy() / 60.0
X = sm.add_constant(np.c_[x, duration_scaled, x * duration_scaled])
k = g.ignitions.to_numpy()
n = g.replicates.to_numpy()
fit = sm.GLM(
    np.column_stack([k, n - k]),
    X,
    family=sm.families.Binomial(),
).fit()
b0, b_contact, b_duration, b_interaction = fit.params
FIDUCIAL_DURATION_SCALED = 1.0  # 60 generations = 1,500 years


def pignite(p: np.ndarray | float) -> np.ndarray:
    p = np.asarray(p)
    linear_predictor = (
        b0
        + b_contact * p
        + b_duration * FIDUCIAL_DURATION_SCALED
        + b_interaction * p * FIDUCIAL_DURATION_SCALED
    )
    return 1.0 / (1.0 + np.exp(-linear_predictor))


P50 = float(
    -(b0 + b_duration * FIDUCIAL_DURATION_SCALED)
    / (b_contact + b_interaction * FIDUCIAL_DURATION_SCALED)
)
NHZ = [1e8, 1e9, 1e10]
contacts = [0.02, P50, 0.12]
opportunities = [0.01, 0.1, 1.0]
rows: list[dict[str, float]] = []
for planet_count in NHZ:
    for contact in contacts:
        ignition_probability = float(pignite(contact))
        for opportunity_rate in opportunities:
            f50 = -math.log(0.5) / (
                planet_count * E_BAR * opportunity_rate * ignition_probability
            )
            f95 = -math.log(0.95) / (
                planet_count * E_BAR * opportunity_rate * ignition_probability
            )
            rows.append(
                {
                    "N_rocky_HZ_planets": planet_count,
                    "conditional_mean_eligible_exposure_Gyr": E_BAR,
                    "external_model_probability": contact,
                    "ABM_ignition_probability": ignition_probability,
                    "opportunity_rate_per_Gyr": opportunity_rate,
                    "max_f_complex_for_50pct_first": f50,
                    "max_f_complex_for_95pct_first": f95,
                    "max_expected_complex_biospheres_50pct": planet_count * f50,
                }
            )
pd.DataFrame(rows).to_csv(OUT / "cosmological_firstness_scenarios.csv", index=False)

constraint: list[dict[str, float | str]] = []
for planet_count in NHZ:
    for target, mu in [
        ("50% first", -math.log(0.5)),
        ("95% first", -math.log(0.95)),
        ("Expected prior <1", 1.0),
    ]:
        constraint.append(
            {
                "N_rocky_HZ_planets": planet_count,
                "conditional_mean_eligible_exposure_Gyr": E_BAR,
                "criterion": target,
                "max_f_complex_times_opportunity_rate_per_Gyr": mu
                / (planet_count * E_BAR * 0.5),
            }
        )
pd.DataFrame(constraint).to_csv(OUT / "cosmological_filter_constraints.csv", index=False)

exposure_rows: list[dict[str, float]] = []
for exposure in EXPOSURE_SENSITIVITY:
    for planet_count in NHZ:
        exposure_rows.append(
            {
                "N_rocky_HZ_planets": planet_count,
                "conditional_mean_eligible_exposure_Gyr": exposure,
                "ABM_ignition_probability": 0.5,
                "max_f_complex_times_opportunity_rate_for_50pct_first_per_Gyr":
                    -math.log(0.5) / (planet_count * exposure * 0.5),
                "max_f_complex_times_opportunity_rate_for_95pct_first_per_Gyr":
                    -math.log(0.95) / (planet_count * exposure * 0.5),
            }
        )
pd.DataFrame(exposure_rows).to_csv(
    OUT / "cosmological_exposure_sensitivity.csv", index=False
)

# Contour for the fiducial planet count and P_ignite at its 50% boundary.
planet_count = 1e10
ignition_probability = 0.5
f_complex = np.logspace(-14, -1, 260)
opportunity_rate = np.logspace(-14, 1, 260)
FF, RR = np.meshgrid(f_complex, opportunity_rate)
lambda_prior = planet_count * E_BAR * ignition_probability * FF * RR
p_no_prior = np.exp(-lambda_prior)
plt.figure(figsize=(8.8, 5.9))
mesh = plt.pcolormesh(f_complex, opportunity_rate, p_no_prior, shading="auto", vmin=0, vmax=1)
plt.xscale("log")
plt.yscale("log")
plt.colorbar(mesh, label="Probability of no earlier successful ignition event")
plt.contour(f_complex, opportunity_rate, p_no_prior, levels=[0.05, 0.5, 0.95])
plt.xlabel("Fraction of HZ rocky planets reaching complex animal life")
plt.ylabel("Network-opportunity rate per complex biosphere (Gyr$^{-1}$)")
plt.title(f"Conditional cultural-ignition constraint ($\\bar{{E}}={E_BAR:g}$ Gyr)")
plt.tight_layout()
plt.savefig(OUT / "cosmological_firstness_contour.png", dpi=220)
plt.close()

# Probability versus external contact for scenario products.
pgrid = np.linspace(0.001, 0.16, 220)
pig = pignite(pgrid)
plt.figure(figsize=(8.8, 5.6))
for product, label in [
    (1e-12, "$f_{complex}r_{opp}=10^{-12}$ Gyr$^{-1}$"),
    (1e-11, "$10^{-11}$ Gyr$^{-1}$"),
    (1e-10, "$10^{-10}$ Gyr$^{-1}$"),
]:
    plt.plot(
        pgrid,
        np.exp(-1e10 * E_BAR * product * pig),
        label=label,
    )
plt.axvline(P50, ls="--", label=f"ABM $p_{{50}}={P50:.3f}$")
plt.xlabel("External-model probability during a 1,500-year opportunity")
plt.ylabel("Probability of no earlier successful ignition event")
plt.title(f"Local ignition curve in a conditional Galactic model ($\\bar{{E}}={E_BAR:g}$ Gyr)")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "cosmological_firstness_vs_contact.png", dpi=220)
plt.close()

summary = {
    "glm_intercept": float(b0),
    "glm_contact_coefficient": float(b_contact),
    "glm_duration_scaled_coefficient": float(b_duration),
    "glm_contact_duration_interaction": float(b_interaction),
    "ABM_p50_1500_years": P50,
    "conditional_mean_eligible_exposure_given_complex_life_Gyr": E_BAR,
    "eligible_exposure_status": "illustrative conditional scenario parameter, not empirically inferred",
    "eligible_exposure_sensitivity_Gyr": EXPOSURE_SENSITIVITY,
    "fiducial_constraint_fcomplex_times_opportunity_rate_50pct_first_at_pignite_0.5":
        -math.log(0.5) / (1e10 * E_BAR * 0.5),
}
(OUT / "cosmological_embedding_summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)
print(json.dumps(summary, indent=2))
print(pd.DataFrame(constraint).to_string(index=False))
