import numpy as np
from scipy.stats import beta 
import matplotlib.pyplot as plt
import corner


def gaussian_experiment(beta_params=None,save_samples=None,savename=None):
    
    '''
    This function follows the method outlined in section 4.3.1 of (Bowler, Blunt & Nielsen,2020)
    to try and recover a given population level eccentricity distribution (which is in this case
    a beta distribution) by carrying out hierarchical bayesian analysis on a set of gaussian
    eccentricity posteriors sampled from the parent population distribution. 
    
    The analysis is conducted for the 12 combinations of N (number of systems/eccentricity 
    posteriors) and sigma (standard deviation of each eccentricity posterior)
    
    N = 5, 10, 20, 50 
    sigma = 0.2 , 0.1, 0.05   
    
    Following the run through, corner plots of the beta sampling and the samples themselves 
    are saved (if desired)
    
    args: 
        beta_params (tuple of floats) : A two-tuple containing the values parametrising the parent 
                                        beta distribution. If set to None, the code defaults to the
                                        parameters of the Warm Jupiter Distrbition. 
                                        
        save_samples (bool) : Set to True in order to save the samples from the MCMC run. 
        
        savename (str or bool) : Path to where the samples should be saved, if save_samples is True.
                                         
    '''
    
    if beta_params==None:
        a = 0.867
        b = 3.03
    else:
        a , b = beta_params
        
    ns=[5,10,20,50]
    sigs=[0.2,0.05,0.01]
    
    for N in ns:
    
        eccentricities   = np.random.beta(a,b,size=N)

        for sig in sigs:

            ecc_posts=[np.random.normal(ecc, sig,5000) for ecc in eccentricities]
            like=Pop_Likelihood(posteriors=ecc_posts)
            beta_samples=like.sample(10,burn_steps=2,nwalkers=10)
            
            if save_samples:
                if savename==None:
                    np.save(f'./beta_posts/{str(N)}_{str(int(100*sig))}',beta_samples)
                else:
                    np.save(savename,beta_samples)

            fig=corner.corner(beta_samples,labels=['a','b'])
            plt.savefig(f'./plots/{str(N)}_{str(int(100*sig))}.png')
           
if __name__ == '__main__':
    experiment()

