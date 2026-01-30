# A Computational Assessment of McNeill and Stark on Christian Expansion during the Antonine Plague and the Plague of Cyprian

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17070198.svg)](https://doi.org/10.5281/zenodo.17070198)

This repository contains the code for a computational historiographical study of the Antonine Plague and Plague of Cyprian and their impact on the Christian and Pagan subpopulations of the Roman Empire. The study employs system dynamics and network models to assess whether the plausibility of scenarios hypothesized by William H. McNeill and Rodney Stark. The dissertation _A Computational Assessment of McNeill and Stark on Christian Expansion during the Antonine Plague and the Plague of Cyprian_ contains the theoretical and methodological background of the code in this repository and the analysis of simulations performed with the functionality provided here.

## Repository Contents
### Models and plotting
- `src/models/type_1_models/seir.py`: Implementation of the type 1 models for simulating disease spread.
- `src/analysis/plotting.py`: Functions for executing simulations based on the type 1 models and visualization of the results.
- `src/models/type_2_models/network_model.py`: Implementation of the type 2 network models for simulating disease spread and visualization of the results.

### Input data
- `src/parameters/params.py`: Parameter sets for all models.

### Utility scripts
- `src/analysis/ode_model_stiffness_analysis.py`: Script for analyzing the stiffness of the type 1 models.
