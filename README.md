# ePop!

Package to infer population level eccentricity distributions using hierarchical MCMC. 

## Quick start code

```
import ePop.simulate
import ePop.hier_sim

# simulate a forward modelled sample of 10 imaged companion eccentricity posteriors 
# drawn from the RV exoplanet distribution from Kipping (2010)


a, b = 0.87, 3.03
ecc_posteriors=ePop.simulate.simulate_sample((a,b))


# create Likelihood object and choose hyperprior

like=ePop.hier_sim.Pop_Likelihood(posteriors=ecc_posteriors,prior='Gaussian')

# NOTE: you can also load in samples from already saved eccentricity posteriors and
# initialise a likelihod object as above.


# sample the hyperparameters using MCMC
beta_samples=like.sample(1000,burn_steps=500,nwalkers=30)


```
