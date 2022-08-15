import numpy as np
from scipy.stats import beta 
import matplotlib.pyplot as plt
from orbitize import results
import emcee
from scipy.optimize import minimize
import glob
import corner
import priors




class Pop_Likelihood(object):
    '''
    A class to compute and store the hierarchical eccentricity likelihood for
    a set of orbitize posteriors. 
    Args:
        
        fnames (str or bool) : List of strings, where each string is a file 
                               path for a saved posterior. If None, the code
                               expects the posteriors to be handed in directly
                               through the 'posteriors' keyword
        
        posteriors (bool or list of arrays) : List of arrays, where each array 
                                              is a posterior.
        
        posterior_format (str): Should be either 'orbitize' or 'np'. Corresponding
                                to orbitize results objects and npy files respectively
        prior (str or None): The prior to place on the beta distribution 
                             parameters during MCMC.
                             Valid options include: 'log-uniform'
                             Defaults to None, in which case there is no prior.
        
        beta_max (float/int > 0.1): Maximum value the beta parameters are 
                                    allowed to take on. Defaults to 100.
                                                    
    '''
    def __init__(self,fnames=None,posteriors=None, prior=None,beta_max=100,mu=0.69,std=1.0):
        
        if fnames is not None:

            self.system_results = [results.Results() for fname in fnames]
            for i,res_obj in enumerate(self.system_results):
                res_obj.load_results(fnames[i])
            self.posteriors =     [sys.post for sys in self.system_results]
            self.ecc_posteriors = [post[:,1] for post in self.posteriors]
            self.beta_max=beta_max
            
            
            self.prior_type=prior

            if self.prior_type is None:
                pass

            elif self.prior_type.lower()=='gaussian':
                self.prior=priors.GaussianPrior(mu,std)

            elif self.prior_type.lower()=='log-uniform':
                self.prior=priors.LogUniformPrior(0.01,beta_max)

            elif self.prior_type.lower()=='lognormal':
                self.prior=priors.LogNormalPrior(4,0.5)


        elif posteriors is not None:

            self.system_results=None
            self.posteriors=None
            self.ecc_posteriors=posteriors

            self.beta_max=beta_max
            self.prior_type=prior

            if self.prior_type is None:
                self.prior=priors.UniformPrior(0.01,beta_max)

            elif self.prior_type.lower()=='gaussian':
                self.prior=priors.GaussianPrior(mu,std)

            elif self.prior_type.lower()=='log-uniform':
                self.prior=priors.LogUniformPrior(0.01,beta_max)

            elif self.prior_type.lower()=='lognormal':
                self.prior=priors.LogNormalPrior(4,0.5)
                        

        else:
            raise ValueError('Must provide either the file path to the posteriors, or the posteriors themselves')  
        


    def calc_likelihood(self,beta_params):
        '''
        Returns the log likelihood that a pair of beta distribution parameters
        are the correct parametrisation for a population level eccentricity 
        distribution. This likelihood function is defined as in 
        (Hogg, Myers & Bovy, 2010). Specifically, we use eqs. (9), (12) & (13) 
        from that paper. 
        Args: 
            beta_params (2-tuple of floats): Values of the beta parameters for 
                                           which to evaluate the likelihood. 
        returns:
            log_likelihood (float): The logarithm of the value of the 
                                    likelihood function evaluated at the 
                                    input beta function parameters. 
                                    
        '''
        a , b = beta_params

        if a<0.01 or b<0.01 or a>=self.beta_max or b>=self.beta_max:
            return -np.inf

        system_sums = np.array([np.sum(beta.pdf(ecc_post,a,b))/(np.shape(ecc_post)[0])
                        for ecc_post in self.ecc_posteriors])


        log_likelihood = np.sum(np.log(system_sums))

        log_prior_prob=self.prior.compute_logprob(a,b)
        
        return log_likelihood + log_prior_prob



    def sample(self,nsteps,burn_steps=200,nwalkers=100):
        '''
        Samples from the hierarchical likelihood using MCMC and returns 
        the samples.
        Args: 
            nsteps (int) : Number of steps for the MCMC walkers to take
            burn_steps (int): Number of steps to discard.
            nwalkers (int): Number of walkers
        '''

        ndim = 2
        p0=np.random.uniform(0.01,self.beta_max-0.001,size=(nwalkers, ndim))
        

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.calc_likelihood)
        state = sampler.run_mcmc(p0, burn_steps)
        

        sampler.reset() 
        sampler.run_mcmc(state, nsteps)

        beta_post=sampler.get_chain(flat=True)

        return beta_post

  
   
