import numpy as np
from scipy.stats import beta 
import matplotlib.pyplot as plt
from orbitize import results
import emcee
from scipy.optimize import minimize
import glob
import corner



class Pop_Likelihood(object):
    '''
    A class to compute and store the hierarchical eccentricity likelihood for a set 
    of orbitize posteriors. 

    Args:
        
        fnames (str or bool) : List of strings, where each string is a file path for a
                               saved posterior. If None, the code expects the posteriors
                               to be handed in directly through the 'posteriors' keyword
        
        posteriors (bool or list of arrays) : List of arrays, where each array is a posterior.
                                                    
    '''
    def __init__(self,fnames=None,posteriors=None):
        
        if fnames is not None:
            self.system_results = [results.Results() for fname in fnames]
            for i,res_obj in enumerate(self.system_results):
                res_obj.load_results(fnames[i])
            self.posteriors =     [sys.post for sys in self.system_results]
            self.ecc_posteriors = [post[:,1] for post in self.posteriors]
        elif posteriors is not None:
            self.system_results=None
            self.posteriors=None
            self.ecc_posteriors=posteriors
        else:
            raise ValueError('Must provide either fnames or posteriors')


    def calc_likelihood(self,beta_params):
        '''
        Returns the log likelihood that a pair of beta distribution parameters
        are the correct parametrisation for a population level eccentricity distribution.
        This likelihood function is defined as in (Hogg, Myers & Bovy, 2010).
        Specifically, we use eqs. (9), (12) & (13) from that paper. 

        Args: 

            beta_params (tuple of floats): The values of the beta parameters for which 
                                           to evaluate the likelihood. 
        
        returns:

            log_likelihood (float): The logarithm of the value of thelikelihood function
                                    evaluated at the input beta function parameters. 
                                    
        '''
        a , b = beta_params
        if a<=0 or b<=0 or a>=1000 or b>=1000:
            return -np.inf

        system_sums = np.array([np.sum( beta.pdf(ecc_post,a,b) )/(np.shape(ecc_post)[0])
                        for ecc_post in self.ecc_posteriors])
        log_likelihood = np.sum(np.log(system_sums))
        return log_likelihood

    def map_fit(self):
        '''
        Calculate max likelihood value for the beta distribution parameters. 

        '''
        p0 = (20,20)
        
        map_fit = minimize(-1*self.calc_likelihood,p0,method='Powell',bounds=( (0,1000), (0,1000) ) )

        return map_fit.x


    def sample(self,nsteps,burn_steps=200,nwalkers=100):
        '''
        Samples from the hierarchical likelihood using MCMC and returns the samples.

        Args: 

            nsteps (int) : Number of steps for the MCMC walkers to take
            burn_steps (int): Number of steps to discard.
            nwalkers (int): Number of walkers
        '''

        ndim = 2
        p0 = 100*np.random.rand(nwalkers, ndim) + 1

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.calc_likelihood)
        state = sampler.run_mcmc(p0, burn_steps)
        
        sampler.reset() 
        sampler.run_mcmc(state, nsteps)

        beta_post=sampler.get_chain(flat=True)

        return beta_post


def experiment():

    ns=[5,10,20,50]
    sigs=[0.2,0.05,0.01]

    a = 0.867
    b = 3.03

    for N in ns:
    
        eccentricities   = np.random.beta(a,b,size=N)

        for sig in sigs:

            ecc_posts=[np.random.normal(ecc, sig,5000) for ecc in eccentricities]
            like=Pop_Likelihood(posteriors=ecc_posts)
            beta_samples=like.sample(10,burn_steps=2,nwalkers=10)
            np.save(f'./gaussian_experiment/beta_posts/{str(N)}_{str(int(100*sig))}',beta_samples)

            fig=corner.corner(beta_samples,labels=['a','b'])
            plt.savefig(f'./gaussian_experiment/beta_corners/{str(N)}_{str(int(100*sig))}.png')




if __name__ == '__main__':
    
    fnames=sorted(glob.glob('./bd_run/posteriors/*'))
    like=Pop_Likelihood(fnames)
    beta_samples=like.sample(1000,burn_steps=200,nwalkers=10)
    np.save('samples',beta_samples)
    fig=corner.corner(beta_samples,labels=['a','b'])
    plt.savefig('./bd_run/beta_sampling.png')

