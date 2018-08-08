iapetus
==============================
[//]: # (Badges)
[![Travis Build Status](https://travis-ci.org/choderalab/iapetus.png)](https://travis-ci.org/choderalab/iapetus)
[![codecov](https://codecov.io/gh/choderalab/iapetus/branch/master/graph/badge.svg)](https://codecov.io/gh/choderalab/iapetus/branch/master)

![Image of Yaktocat](https://raw.githubusercontent.com/choderalab/iapetus/master/iapetus-logo.png)

`iapetus`: An open source toolkit for predicting bacterial porin permeation

## Installation

### Installing the release version

0. If you don't already have Anaconda or Miniconda installed, install it from [here](https://conda.io/miniconda.html).
1. Next, install the release version of `iapetus` from the `omnia` Anaconda Cloud channel (check out our detailed installation section):
```bash
conda install -c conda-forge -c omnia iapetus
```

### Installing the development version

0. If you don't already have Anaconda or Miniconda installed, install it from [here](https://conda.io/miniconda.html).
1. Uninstall `iapetus` if you already have it installed
```bash
pip uninstall --yes iapetus
```
2. Check out the github repository
```bash
git clone https://github.com/choderalab/iapetus.git
```
3. Enter the `iapetus` directory
```bash
cd iapetus
```
4. Install `conda-build` and build/install the dev version and dependencies
```bash
conda install --yes conda-build
conda build devtools/conda-recipe
conda install --use-local iapetus
```

## Examples

See [`iapetus-examples`](https://github.com/choderalab/iapetus-examples) for example input files.

## Copyright

Copyright (c) 2018, Chodera lab // MSKCC

## Acknowledgements

Project based on the
[Computational Chemistry Python Cookiecutter](https://github.com/choderalab/cookiecutter-python-comp-chem)
