# Reproducibility guide

## Level 1: reproduce the reported analyses from frozen ensembles

Run:

```bash
python code/reproduce_published_analyses.py
```

Expected outputs appear in `results/reproduced/`.

The script regenerates:

- the filtered archaeological single-chain table from the bundled Windows-1252 source CSV;
- archaeological change-point robustness, permutation and threshold analyses;
- ablation phase summaries and boundaries;
- topology curves and bootstrap threshold intervals;
- random-forest global-sensitivity importance from the frozen Latin-hypercube table;
- a transparent refit of the exploratory population-architecture ensembles (`scaling_summary_refit.csv`); the exact low-replicate table cited in the manuscript is frozen at `results/published/scaling_summary.csv`;
- the conditional Galactic firstness constraints and eligible-exposure sensitivity table.

## Level 2: compare against frozen published outputs

The reference outputs used in the submitted manuscript are in `results/published/`. Numerical differences can occur in random-forest permutation importance across platforms or library versions, but the ranking and qualitative conclusions should remain stable under the pinned environment.

## Level 3: rerun stochastic experiments

The complete stochastic ensembles are much more expensive than Level 1. Use the `run_*` scripts in `code/`. New runs are intentionally separated from the frozen publication data.

All principal experiment seeds are deterministic and begin from the fixed base seed 20260728 or documented offsets in the scripts.

## Portability

All repository scripts resolve paths relative to the repository root. They do not require `/mnt/data` or any ChatGPT-specific directory.

## Declared cosmological scenario parameter

`data/cosmological_scenario.json` contains the fiducial conditional mean eligible exposure and the sensitivity values. The manuscript uses 4 Gyr as an illustrative scenario, not as an estimate inferred from the archaeological data or the agent-based model. All Poisson bounds scale inversely with this parameter.
