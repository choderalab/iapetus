iapetus
==============================
[//]: # (Badges)
[![Travis Build Status](https://travis-ci.org/choderalab/iapetus.png)](https://travis-ci.org/choderalab/iapetus)
[![codecov](https://codecov.io/gh/choderalab/iapetus/branch/master/graph/badge.svg)](https://codecov.io/gh/choderalab/iapetus/branch/master)

![Image of Yaktocat](https://raw.githubusercontent.com/choderalab/iapetus/master/iapetus-logo.png)

`iapetus`: An open source toolkit for predicting bacterial porin permeation

## Installation

### Installing the release version

1. If you don't already have Anaconda or Miniconda installed, install it from [here](https://conda.io/miniconda.html).
2. Next, install the release version of `iapetus` from the `omnia` Anaconda Cloud channel (check out our detailed installation section):
```bash
$ conda install -c conda-forge -c omnia iapetus
```

### Installing the development version

1. If you don't already have Anaconda or Miniconda installed, install it from [here](https://conda.io/miniconda.html).
2. Next, install the release version of `iapetus` from the `omnia` Anaconda Cloud channel (check out our detailed installation section):
```bash
$ conda install --yes -c conda-forge -c omnia iapetus
```
3. Uninstall the release version and install the GitHub dev version via `pip`:
```bash
conda remove --yes iapetus
pip install git+https://github.com/choderalab/iapetus.git
```

## Examples

See [`iapetus-examples`](https://github.com/choderalab/iapetus-examples) for example input files.

## Copyright

Copyright (c) 2018, Chodera lab // MSKCC

## Acknowledgements

Project based on the
[Computational Chemistry Python Cookiecutter](https://github.com/choderalab/cookiecutter-python-comp-chem)
