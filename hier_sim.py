import numpy as np
from scipy.stats import beta 
import matplotlib.pyplot as plt
from orbitize import priors, read_input, kepler, results
import emcee



class Pop_Likelihood(object):
    '''
    A class to compute and store the hierarchical eccentricity likelihood for a set 
    of orbitize posteriors
    '''
    def __init__(self,filenames):
        
        self.system_results = [results.Results().load_results(fname) for fname in filenames]
        self.posteriors =     [sys.post for sys in self.system_results]
        self.ecc_posteriors = [post[:,1] for post in self.posteriors]

    def calc_likelihood(self,beta_params):
        '''
        Returns the log likelihood that a pair of beta distribution parameters
        are the correct parametrisation for a population level eccentricity distribution.
        '''
        a , b = beta_params
        if a<=0 or b<=0:
            return -np.inf
        system_sums = np.array([np.sum( beta.pdf(ecc_post,a,b) )/(np.shape(ecc_post)[0])
                        for ecc_post in self.ecc_posteriors])
        log_likelihood = np.sum(np.log(system_sums))
        return log_likelihood

    def sample(self,nsteps,burn_steps=200,nwalkers=100):
        '''
        Samples from the hierarchical likelihood using MCMC and returns the samples.
        '''

        ndim = 2
        p0 = 9*np.random.rand(nwalkers, ndim) + 1

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.calc_likelihood)
        state = sampler.run_mcmc(p0, burn_steps)
        
        sampler.reset() 
        sampler.run_mcmc(state, nsteps)

        beta_post=sampler.get_chain(flat=True)

        return beta_post

    

        
    

