# ePop!

Package to infer population level eccentricity distributions using hierarchical MCMC. 

## Installation

First, (make sure you have orbitize! installed) [https://orbitize.readthedocs.io/en/latest/installation.html].

Next, from the commmand line:

$ git clone https://github.com/vighnesh-nagpal/ePop.git

$ cd ePop

$ pip install -e . --upgrade

## Quick start code

```
import ePop.simulate
import ePop.hier_sim

# simulate a forward modelled sample of 10 imaged companion eccentricity posteriors 
# drawn from the RV exoplanet distribution from Kipping (2010)


a, b = 0.87, 3.03


# this step simulates realistic eccentricity posteriors for a set of systems with 
# eccentricities drawn from the (0.87, 3.03) Beta Distribution. (Time-Intensive)

ecc_posteriors=ePop.simulate.simulate_sample((a,b))


# create Likelihood object and choose hyperprior
like=ePop.hier_sim.Pop_Likelihood(posteriors=ecc_posteriors,prior='Gaussian')


# NOTE: you can also load in samples from already saved eccentricity posteriors and
# initialise a likelihod object as above. In this case, posteriors must be a list of 
# 1D arrays, where the arrays are the 1D eccentricity posteriors for each system in the sapmle.  


# sample the hyperparameters using MCMC using 1000 steps.
beta_samples=like.sample(1000,burn_steps=500,nwalkers=30)


```
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![DOI](https://zenodo.org/badge/371124285.svg)](https://zenodo.org/badge/latestdoi/371124285)

