# Data dictionary

## Primary frozen inputs

- `procedural_units_obfus.csv`: upstream stone-tool manufacturing sequences and metadata (Windows-1252 encoded).
- `archaeological_single_chain_data.csv`: filtered dated single-chain sequences used in the change-point analysis; exactly rebuilt by `code/prepare_archaeological_data.py`.
- `dense_phase_grid_stage9.csv`: dense phase experiment aggregated by contact probability, duration and outcome criterion.
- `ablation_phase_raw.csv`: seed-level ablation results across contact-duration grids.
- `topology_independent_realisations_raw.csv`: topology experiments containing independent graph realisations and agent replicates.
- `global_lhs_samples.csv`: Latin-hypercube parameter sets and stochastic outcomes.
- `scaling_ensemble.csv`: population-architecture ensembles.
- `cosmological_scenario.json`: declared fiducial and sensitivity values for conditional mean eligible exposure; these are scenario assumptions, not fitted data.

## Key outcome variables

- `contact_probability`: probability that a learner samples an external cultural model during a contact phase.
- `duration_years`: duration of the contact phase.
- `ignitions`: number of runs reaching persistent cumulative-cultural ignition.
- `replicates`: number of stochastic runs in the cell.
- `ignition_probability`: `ignitions / replicates`.
- `estimated_p50`: estimated external-model probability at 50% persistent ignition.
- `primary_probability`: ignition probability under the primary outcome definition.
- `procedural_units`: number of coded operations in a manufacturing sequence.
- `age_mid_ka`: midpoint age in thousands of years before present.
- `above_noncumulative_baseline`: indicator that procedural complexity exceeds the comparison baseline.

Detailed column meanings for specialised tables are documented in the code that creates them and in the Supplementary Information.
