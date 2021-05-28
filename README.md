# Code for The Contagion of Mass Shootings: The Interdependence of Large-Scale Massacres and Mass Media Coverage

## Purpose

This code performs self-exciting point process modeling and analysis of the dataset of US Mass Shootings discussed in the paper ["The Contagion of Mass Shootings: The Interdependence of Large-Scale Massacres and Mass Media Coverage" published in **Statistics and Public Policy**](https://amstat.tandfonline.com/doi/full/10.1080/2330443X.2021.1932645) by James Alan Fox, [Nathan E. Sanders](https://github.com/nesanders), Emma E. Fridel, Grant Duwe, and Michael Rocque. It reproduces all figures and statistics presented in that paper, as well as some additional related outputs.

## Usage

To run this code,

1. Establish a conda environment using the environment.yml file.
2. Run hawkes_contagion.py with python3.

## Contents

* `hawkes_contagion.py` - Primary entry point.  This script reproduces all analysis and outputs associated with the paper.
* `util.py` - Utilities and function definitions used by hawkes_contagion.py
* `model_params.py` - Specification of data I/O paths and model configurations associated with the analysis.  You can edit this to run the self-exciting point process model in additional configurations, e.g. with other variables.
* `environment.yml` - Conda environment specification
* `data/2020-07-17_contagion_data.csv` - Final input data file for hawkes_contagion.py used in the published version of the analysis.
* `data/grid_search_model_results.jl` - Serialized output of fitted models from the parameter grid simulation.
* `copy_figures.sh` - Map (copy) figures from the output of hawkes_contagion.py to their names/positions from the final published paper.
* `submission_plots/` - Final versions of figures included in the published paper, as mapped by `copy_figures.sh`

