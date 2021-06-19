
import numpy as np
from scipy.stats import beta 
import matplotlib.pyplot as plt
import corner


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

