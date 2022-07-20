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
    def __init__(self,fnames=None,posteriors=None, prior=None,beta_max=100,mu=1.0,std=0.5):
        
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

            for post in self.ecc_posteriors:
                print(post)
                # post[np.where(post==0)]=0.001
                # post[np.where(post==1)]=0.999

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

            beta_params (tuple of floats): Values of the beta parameters for 
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

        # print(beta.pdf(self.ecc_posteriors[3],a,b))
        # print(self.ecc_posteriors[3][np.where(beta.pdf(self.ecc_posteriors[3],a,b)==np.inf)])

        # print(system_sums)

        log_likelihood = np.sum(np.log(system_sums))

        log_prior_prob=self.prior.compute_logprob(a,b)
        
        return log_likelihood + log_prior_prob

    def map_fit(self):
        '''
        Calculate max likelihood value for the beta distribution 
        parameters. 

        '''
        p0 = (20,20)
        
        map_fit = minimize(-1*self.calc_likelihood,p0,method='Powell',
                            bounds=( (0,1000), (0,1000) ) )

        return map_fit.x


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



def experiment():

    ns=[5,10,20,50]
    sigs=[0.2,0.05,0.01]

    a = 0.867
    b = 3.03

    for N in ns:
    
        eccentricities   = np.random.beta(a,b,size=N)

        for sig in sigs:

            ecc_posts=[np.random.normal(ecc, sig,5000) for ecc in eccentricities]

            like=Pop_Likelihood(posteriors=ecc_posts,prior='Gaussian',mu=0.6867,std=1.0)
            beta_samples=like.sample(2000,burn_steps=200,nwalkers=20)
            np.save(f'./gaussian_experiment/wj/shifted_gaussian_bigstd/beta_posts/{str(N)}_{str(int(100*sig))}',beta_samples)

            fig=corner.corner(beta_samples,labels=['a','b'])
            plt.savefig(f'./gaussian_experiment/wj/shifted_gaussian_bigstd/beta_corners/{str(N)}_{str(int(100*sig))}.png')

            # like=Pop_Likelihood(posteriors=ecc_posts)
            # beta_samples=like.sample(2000,burn_steps=200,nwalkers=20)
            # np.save(f'./gaussian_experiment/a4b2/uniform/beta_posts/{str(N)}_{str(int(100*sig))}',beta_samples)

            # fig=corner.corner(beta_samples,labels=['a','b'])
            # plt.savefig(f'./gaussian_experiment/a4b2/uniform/beta_corners/{str(N)}_{str(int(100*sig))}.png')

            # like=Pop_Likelihood(posteriors=ecc_posts,prior='log-uniform')
            # beta_samples=like.sample(2000,burn_steps=200,nwalkers=20)
            # np.save(f'./gaussian_experiment/a4b2/log_uniform/beta_posts/{str(N)}_{str(int(100*sig))}',beta_samples)

            # fig=corner.corner(beta_samples,labels=['a','b'])
            # plt.savefig(f'./gaussian_experiment/a4b2/log_uniform/beta_corners/{str(N)}_{str(int(100*sig))}.png')

def four_priors(fnames,dir,nsteps=15000,burn_steps=2000,test=False):

    if test:
        nsteps=50
        burn_steps=20

        
    # # log-uniform prior
    # like=Pop_Likelihood(fnames=fnames,prior='log-uniform',beta_max=100)
    # beta_samples=like.sample(nsteps,burn_steps=burn_steps,nwalkers=20)
    # np.save(f'{dir}/loguniform_100',beta_samples)
    # fig=corner.corner(beta_samples,labels=['a','b'])
    # plt.savefig(f'{dir}/loguniform_100_corner.png')

    # # Log Normal prior

    # like=Pop_Likelihood(fnames=fnames,prior='LogNormal',beta_max=100)
    # beta_samples=like.sample(nsteps,burn_steps=burn_steps,nwalkers=20)
    # np.save(f'{dir}/lognormal',beta_samples)
    # fig=corner.corner(beta_samples,labels=['a','b'])
    # plt.savefig(f'{dir}/lognormal_corner.png')

    # # Uniform Prior

    # like=Pop_Likelihood(fnames=fnames,prior=None,beta_max=100)
    # beta_samples=like.sample(nsteps,burn_steps=burn_steps,nwalkers=20)
    # np.save(f'{dir}/uniform',beta_samples)
    # fig=corner.corner(beta_samples,labels=['a','b'])
    # plt.savefig(f'{dir}/uniform_corner.png')

    # Gaussian Prior

    like=Pop_Likelihood(fnames=fnames,prior='Gaussian',beta_max=100,mu=0.6867,std=1.0)
    beta_samples=like.sample(nsteps,burn_steps=burn_steps,nwalkers=20)
    np.save(f'{dir}/shifted_gaussian_bigstd',beta_samples)
    fig=corner.corner(beta_samples,labels=['a','b'])
    plt.savefig(f'{dir}/shifted_gaussian_bigstd_corner.png')



def sim_forward_modelling():

    ### ALL SYSTEMS: uniform DISTRIBUTION
    
    uniform_all_fnames=sorted(glob.glob('/data/user/vnagpal/eccentricities/e2esims/days/uniform_e2e_sims/fixed_inc/*'))
    
    # 5 systems:
    n_systems=5
    sample_systems=np.random.randint(0,high=len(uniform_all_fnames)-1,size=n_systems)
    sys_fnames=[uniform_all_fnames[sys] for sys in sample_systems]

    four_priors(sys_fnames,dir='/home/vnagpal/eccentricities/e2esims/days/uniform_e2e_sims/5_5/fixed_inc/5systems')

    # 10 systems
    n_systems=10
    sample_systems=np.random.randint(0,high=len(uniform_all_fnames)-1,size=n_systems)
    sys_fnames=[uniform_all_fnames[sys] for sys in sample_systems]

    four_priors(sys_fnames,dir='/home/vnagpal/eccentricities/e2esims/days/uniform_e2e_sims/5_5/fixed_inc/10systems')


    # 20 systems
    n_systems=20
    sample_systems=np.random.randint(0,high=len(uniform_all_fnames)-1,size=n_systems)
    sys_fnames=[uniform_all_fnames[sys] for sys in sample_systems]

    four_priors(sys_fnames,dir='/home/vnagpal/eccentricities/e2esims/days/uniform_e2e_sims/5_5/fixed_inc/20systems')
    
    # all systems
    n_systems=45
    sample_systems=np.random.randint(0,high=len(uniform_all_fnames)-1,size=n_systems)
    sys_fnames=[uniform_all_fnames[sys] for sys in sample_systems]
    
    four_priors(sys_fnames,dir='/home/vnagpal/eccentricities/e2esims/days/uniform_e2e_sims/5_5/fixed_inc/allsystems')


def sim_observational_sample(nsteps,burn_steps):

    # construct samples

    fnames=sorted(glob.glob('/home/vnagpal/eccentricities/ogpaper3/individual_posts/*.npy'))
    # print(fnames)
    planets=['51erib','hr8799b','hr8799c','hr8799d','hr8799e','pds70b','hd95086b','picb','hip65426b']
    planet_fnames={}
    bd_fnames=[]
    for f in fnames:
        is_planet=False
        for p in planets:
            if p in f:
                planet_fnames[p]=f
                is_planet=True
        if not is_planet:
            bd_fnames.append(f)

    for planet in planet_fnames:
        planet_post=np.load(planet_fnames[planet],allow_pickle=True)
        fig=plt.figure()
        plt.hist(planet_post,bins=50)
        plt.savefig(f'/home/vnagpal/eccentricities/ogpaper3/individual_posts/{planet}_post')

    bd_posts=[np.load(bd) for bd in bd_fnames]
    print(planet_fnames)
    planet_posts=[np.load(planet_fnames[planet]) for planet in planet_fnames]


    like=Pop_Likelihood(posteriors=planet_posts,prior='Gaussian',beta_max=100,mu=0.6867,std=1.0)
    beta_samples=like.sample(nsteps,burn_steps=burn_steps,nwalkers=20)
    np.save('/home/vnagpal/eccentricities/ogpaper3/gp_shifted_bigstd_gaussian',beta_samples)
    fig=corner.corner(beta_samples,labels=['a','b'])
    plt.savefig('/home/vnagpal/eccentricities/ogpaper3/gp_shifted_bigstd_gaussian_corner.png')

    # like=Pop_Likelihood(posteriors=bd_posts,prior='log-uniform',beta_max=100)
    # beta_samples=like.sample(nsteps,burn_steps=burn_steps,nwalkers=20)
    # np.save('/home/vnagpal/eccentricities/ogpaper3/bd_log_uniform',beta_samples)
    # fig=corner.corner(beta_samples,labels=['a','b'])
    # plt.savefig('/home/vnagpal/eccentricities/ogpaper3/bd_log_uniform_corner.png')

    like=Pop_Likelihood(posteriors=planet_posts,prior=None,beta_max=10)
    beta_samples=like.sample(nsteps,burn_steps=burn_steps,nwalkers=20)
    np.save('/home/vnagpal/eccentricities/ogpaper3/gp_uniform_10',beta_samples)
    fig=corner.corner(beta_samples,labels=['a','b'])
    plt.savefig('/home/vnagpal/eccentricities/ogpaper3/gp_uniform_10_corner.png')

if __name__ == '__main__':
    # experiment()
    # sim_forward_modelling()

    sim_observational_sample(nsteps=15000,burn_steps=2000)




    
   


