# Deconomix

## Overview

Deconomix is a Python library aimed at the bioinformatics community, offering methods to estimate cell type compositions, hidden background contributions and gene regulation factors of bulk RNA mixtures. Visit the documentation [here](https://deconomix.github.io/Deconomix/).

## Features

- **Data Simulation**: Generate artificial bulk mixtures from single-cell data in an efficient way to provide training data for your models.

- **Gene Weighting**: Learn gene weights from artifical bulk mixtures to optimize the cellular composition estimation of real bulk RNA mixtures.

- **Cellular Composition**: Estimate the cellular composition of your bulk RNA profiles or Spatial Transcriptomics spots.

- **Background Estimation**: Refine the composition estimation by estimate a hidden background contribution and profile, which cannot be explained by the cell types featured in the reference.

- **Gene Regulation**: Find out, how cell types in your bulk data is regulated in relation to your reference profiles, for instance in a disease context.

- **Visualization**: Visualize your results with predefined functions.

- **Evaluation**: Perform basic enrichment analysis for the estimated gene regulatory factors.

## Updates
This package will eventually be updated with our recent developments. The version provided for the biorXiv preprint is tagged accordingly. The version at first submission will be tagged and released as version 1.0.0. 

## Installation

Deconomix is added to the official PyPI repositories and can be installed from there directly:

```
pip install deconomix
```

Alternatively, this git repository can be cloned to install the latest version:

```
pip install git+https://github.com/Deconomix/Deconomix.git
```

## Getting Started

Upon successful installation, users are encouraged to explore the curated examples provided in this repository. The `examples` directory contains Jupyter notebooks showcasing various example workflows. Start out with `Example-Standard_Workflow.ipynb` to get an introduction to the standard workflow with the package. Continue with `Example-Hyperparameter_Gridsearch.ipynb` to learn how to conduct a hyperparameter search for our advanced models, if you have ground truth available. A low-cost alternative to the grid search is featured in `Example-Standard_Workflow.ipynb` aswell. In `Example-Plots_for_Application_Note.ipynb` you can find some visualizations which are featured in the overview figure for the article.


## GUI Application

For easier usage, we also distribute a graphical user interface for our package, available for all common operating systems. It is an application built with Dash/Plotly and can be ran locally or on a webserver.
Check out the corresponding repository of [Deconomix GUI](https://gitlab.gwdg.de/MedBioinf/MedicalDataScience/Deconomix-GUI).

## For Developers
For feature requests and bug reports do not hesitate to contact us via a gitlab issue or email.

If you want to generate or update the html documentation yourself, follow these steps:

Install `sphinx` and `sphinx-rtd-theme`:

```
pip install sphinx
pip install sphinx-rtd-theme
```

Then move into the `docs` directory and execute the make command:
```
make html
```
