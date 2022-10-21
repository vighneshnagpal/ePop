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
# eccentricities drawn from the (0.87, 3.03) Beta Distribution. 

ecc_posteriors=ePop.simulate.simulate_sample((a,b))


# create Likelihood object and choose hyperprior
like=ePop.hier_sim.Pop_Likelihood(posteriors=ecc_posteriors,prior='Gaussian')


# NOTE: you can also load in samples from already saved eccentricity posteriors and
# initialise a likelihod object as above. In this case, ecc_posteriors must be a list
# of 1D eccentricity samples. 


# sample the hyperparameters using MCMC
beta_samples=like.sample(1000,burn_steps=500,nwalkers=30)


```
