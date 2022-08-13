# ePop!

Package to infer population level eccentricity distributions using hierarchical MCMC. 

## Quick start code

```
import ePop 
    
fnames=sorted(glob.glob('./posteriors/*'))

# load individual eccentricity distributions
posts=[np.load(f) for f in fnames]

# create Likelihood object and choose prior
like=ePop.hier_sim.Pop_Likelihood(posteriors=posts,prior='log_uniform')

# sample the hyperparameters using MCMC
beta_samples=like.sample(2000,burn_steps=500,nwalkers=30)
```
