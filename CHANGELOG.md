# Changelog

## 1.0.2 - 2026-07-28

- Fixed missing `os` imports in the ablation and topology stochastic-run scripts.
- Corrected the recombination ablation so it disables both cross-model skill assembly and diversity-dependent innovation synergy, then regenerated the ablation ensemble.
- Renamed the implemented hub-plus-peripheral-ring topology from `Hub-and-spoke` to `Wheel`.
- Aligned the topology rerun script with the manuscript by generating independent stochastic graph realisations at each contact level.
- Defined mean eligible exposure conditionally on a planet having reached complex life, eliminating ambiguity about double counting.
- Tightened exploratory sensitivity and population-scaling language.

## 1.0.1 - 2026-07-28

- Distinguished dense-phase thresholds from the separate coarser ablation estimates.
- Added an explicit archaeological preprocessing script and documented source encoding and filters.
- Replaced the undocumented eligible-exposure constant with a declared 4 Gyr fiducial scenario and 1/4/10 Gyr sensitivity outputs.
- Made the Supplementary Information portable by bundling Figure S9 and removing an absolute path.
- Softened causal and universality claims and clarified the Poisson-event interpretation.

## 1.0.0 - 2026-07-28

- Initial public reproducibility release.
- Portable relative-path analysis scripts.
- Frozen phase, ablation, topology, scaling and sensitivity ensembles.
- Archaeological robustness and conditional Galactic embedding.
- SerAJ manuscript and Supplementary Information sources.
