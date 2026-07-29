# Network-triggered cultural ignition as a candidate Great Filter

Reproducibility repository for the manuscript by **Mykola Shatokhin**, National University "Kyiv Aviation Institute".

The project combines:

1. a continuous-feedback agent-based model of cumulative-cultural ignition;
2. phase-boundary, ablation, sensitivity, scaling and network-topology experiments;
3. a robustness analysis of an open archaeological procedural-complexity dataset; and
4. a conditional Galactic population embedding relevant to the Great Filter and the Fermi paradox.

The repository supports the numerical claims in the manuscript. It does **not** claim that humanity has been empirically shown to be the first technological civilisation.

## Repository layout

- `code/` - model implementations, explicit data-preparation and frozen-data analysis scripts, and optional high-cost simulation runners.
- `data/` - archived source/derived inputs and frozen stochastic ensembles used in the manuscript.
- `results/published/` - publication tables, summaries and figures.
- `results/reproduced/` - created locally by the reproduction script; not required to be committed.
- `paper/seraj/` - Serbian Astronomical Journal LaTeX source, compiled manuscript and Supplementary Information.
- `docs/` - data dictionary, provenance, licensing and detailed reproduction notes.

## Quick reproduction from frozen ensembles

Python 3.13 is recommended.

```bash
conda env create -f environment.yml
conda activate cultural-ignition-stage10
python code/reproduce_published_analyses.py
```

Alternatively:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python code/reproduce_published_analyses.py
```

The script creates `results/reproduced/`, rebuilds the 64-row archaeological analysis table from the bundled source CSV, and regenerates the archaeological robustness analysis, ablation summaries, topology summaries, global-sensitivity ranking, scaling summary and cosmological constraints from the frozen input ensembles. It does not rerun every expensive agent simulation.

## Full stochastic reruns

The following scripts generate new stochastic ensembles and can require substantial computation:

- `code/run_phase.py`
- `code/run_sensitivity.py`
- `code/ablation_chunk.py`
- `code/run_topology_type.py`
- `code/run_scaling_reduced.py`

By default they write to `results/new_runs/`. Set `CULTURAL_IGNITION_OUT` to choose another output directory.

## Data provenance

The bundled `data/procedural_units_obfus.csv` is derived from the open Paige and Perreault stone-tool analysis release, archived at DOI **10.5281/zenodo.11398650** under CC BY 4.0. The original data paper is DOI **10.5334/joad.114**, and the associated analysis paper is DOI **10.1073/pnas.2319175121**. See `docs/THIRD_PARTY_NOTICES.md` before redistribution or reuse.

All other CSV files are either derived analysis tables, declared scenario inputs or frozen outputs of the simulations described in the manuscript. The cosmological calculation reads `data/cosmological_scenario.json`; its 4 Gyr fiducial conditional eligible exposure is illustrative rather than empirically inferred, and 1/4/10 Gyr sensitivity results are generated automatically.

## Citation

Until the journal article receives a DOI, cite this repository using `CITATION.cff` and the version-specific Zenodo DOI assigned after deposit. After publication, cite both the article and the repository.

## Licensing

- Original source code: MIT License (`LICENSE`).
- Original derived tables, figures and documentation: CC BY 4.0 (`LICENSE-DATA.md`).
- Third-party source data retain their upstream licence and attribution requirements (`docs/THIRD_PARTY_NOTICES.md`).

## Contact

Mykola Shatokhin  
National University "Kyiv Aviation Institute", Kyiv, Ukraine  
ORCID: 0000-0003-0028-6208  
Email: n.shatokhin@gmail.com
