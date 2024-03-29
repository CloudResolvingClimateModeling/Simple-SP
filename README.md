[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5511753.svg)](https://doi.org/10.5281/zenodo.5511753)

# Simple-SP
Simple superparameterization example in Python using the DALES model

This is a simplified superparameterization setup, where the global
model consists of only advection. The purpose of this example is
showing a lack of cloud advection in a superparameterized model, and
exploring schemes to improve cloud advection by adjusting the
small-scale variability of the total humidity in the local models.

## Installation and Use

* Install OMUSE (recommended in a virtual environment)
and the DALES model within OMUSE. See the
[OMUSE installation instructions](https://omuse.readthedocs.io/en/latest/installing.html)

* Other requirements: scipy, matplotlib, netCDF4 (`pip install netCDF4`)

* Activate the OMUSE virtual environment

```
python simple-SP.py # run simulations, 10 minutes on 4-core desktop
python plot-lwp.py  # plot result
```

(Instructions tested on Ubuntu 21.04, with OMUSE 2021.6.2.dev14+gf2bbc23).

The script is set up to perform three simulations with a moist bubble perturbation in the initial state:

* a single, wide LES

* a traditional superparameterization with four LES domains

* a modified superparameterization with four LES domains, where the moisture variability is coupled


## References

DALES model description article: Formulation of the Dutch Atmospheric Large-Eddy Simulation (DALES) and overview of its applications, T. Heus et al, [Geosci. Model Dev., 3, 415-444, 2010](https://doi.org/10.5194/gmd-3-415-2010)

Superparamerization of OpenIFS with DALES: [code repository](https://github.com/CloudResolvingClimateModeling/sp-coupler), article:
Regional Superparameterization in a Global Circulation Model Using Large Eddy Simulations, F. Jansson et al, [Journal of Advances in Modeling Earth Systems, 11, 2958– 2979.]( https://doi.org/10.1029/2018MS001600)






