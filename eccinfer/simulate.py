import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from astropy.table import Table
import orbitize
from orbitize import kepler
from orbitize import sampler
from orbitize import results


def generate_orbits(systems):
    '''
    Generates orbits for multiple systems in one go.

    args: 

        systems (dict): Dictionary ordered by system index that contains
                        the eccentricties and inclinations for each.
    
    returns:

        astrometry (dict): Dictionary ordered by system index that contains
                           the epochs and corresponding sep/pa of each obs-
                           -ervation. 
    '''
    astrometry={}
    
    for sys in systems:
        
        # eccentricities and inclinations
        ecc,inc=systems[sys]
        
        ### other orbital parameters: MOST ARE FIXED FOR NOW, WILL VARY LATER 
        sma=20
        aop=2*np.pi*np.random.rand()
        pan=2*np.pi*np.random.rand()
        tau=np.random.rand()
        plx=40
        mass_for_kamp=0.01
        mtot=1.01
        
        #epochs and errors for astrometry
        astro_epochs=np.linspace(51500,52000,2)

        # SETTING UNCERTAINTY TO 0 FOR NOW, CHANGE LATER
        sep_err=10
        pa_err=0.1
        
        #generate predictions for astrometric epochs
        astro_set=kepler.calc_orbit(astro_epochs,sma,ecc,inc,aop,pan,tau,plx,mtot,mass_for_Kamp=mass_for_kamp)
        ras,decs=astro_set[0],astro_set[1]
        sep,pa=orbitize.system.radec2seppa(ras,decs)
        
        # Add gaussian noise to the model predictions
        sep_jit=np.random.randn(len(sep))*sep_err
        pa_jit=np.random.randn(len(pa))*pa_err

        sep+=sep_jit
        pa+=pa_jit
        
        # add these measurements (now incorporating noise) to the astrometry
        astrometry[sys]=[astro_epochs,sep,pa]
    
    return astrometry


def draw_eccentricities(N,beta_params):
    '''
    Randomly draws N eccentricities from a beta distribution.

    args:

        N (int) : number of eccentricities to draw
        beta_params (tuple of floats) : Tuple of shape (a,b), where a, b >= 1.
                                        and are the parametrisation for the
                                        beta distribution
    
    returns:
        
        ecc_set (np.ndarray): Array of shape (N,) containing the eccentricities

    '''
    a , b = beta_params
    e_set   = np.random.beta(a,b,size=N)
    return e_set

def create_data_tables(system_observations,save=False):
    '''
    Creates astropy data tables that can be used as input for orbitize.

    args: 
        system_observations (dict): Dictionary indexed by system number 
                                    where each element is a tuple of the
                                    form (epochs, sep, pa). epochs is an 
                                    array of observation times, and sep 
                                    and pa contain the corresponding ast-
                                    -rometry for each date.
        
        save (Bool): If set to true, the data is saved as csvs for the 
                     selected data.

    '''
    dataframes=[]

    for i in range(len(system_observations)):
        frame=[]
        epochs,seps,pas=system_observations[i]
        for mjd,sep,pa in zip(epochs,seps,pas):
            obs=[mjd,1,sep,10,pa,0.1,'seppa','test',np.nan]
            frame.append(obs)
        df=pd.DataFrame(frame, columns = ['epoch', 'object','quant1','quant1_err','quant2','quant2_err','quant_type','instrument','quant12_corr'])
        dataframes.append(df)

    astropy_tables=[Table.from_pandas(df) for df in dataframes]
    
    return astropy_tables


def driver(beta_params,N,post_savedir,fig_savedir):
    '''
    A driver function that forward models the whole process. It first draws eccentricities 
    from the underlying distribution (assumed to be a beta distribution parametrised by a and b)
    for N systems and then simulates astrometry for each. Following this, orbits are fit to the
    simulated astrometry and the resulting eccentricity posteriors are saved to the desired 
    directory.
    
    args: 
    
        beta_params (2-tuple of floats>0): contains the two beta distribution parameters, a and b 
                                         which must both be floats>0
        
        N (int): number of systems
        
        post_savedir (str): path to the desired save directory for the posteriors
        
        fig_savedir (str): path to the desired save directory for the figures
    
    '''
    
    a,b= beta_params
    
    e_set   = draw_eccentricities(N,(a,b))
    inc_set = np.zeros(N)


    systems={i: [e_set[i],inc_set[i]] for i in range(N)}

    orbital_data=generate_orbits(systems)


    tables=create_data_tables(orbital_data)


    for i in range(N):
        orb_sys = orbitize.system.System(1, tables[i], 1.0,
                                    40, mass_err=0.05, plx_err=0.1)
        ofti_sampler = sampler.OFTI(orb_sys)
        n_orbs = 10
        _ = ofti_sampler.run_sampler(n_orbs)
        accepted_eccentricities = ofti_sampler.results.post[:, 1]
        
        ## make histograms for the eccentricties
        fig=plt.figure()
        plt.hist(accepted_eccentricities,bins=50)
        plt.xlabel('ecc'); plt.ylabel('number of orbits')
        fig_filename = 'sys_{np.round(systems[i][0],3)}.png'
        fig_filename =  os.path.join(fig_savedir,fig_filename)
        plt.savefig(fig_filename)

        ### save posteriors
        post_filename = f'sys_{np.round(systems[i][0],3)}.hdf5'
        post_filename = os.path.join(post_savedir,post_filename)

        # hdf5_filename=os.path.join(save_path,filename)
        ofti_sampler.results.save_results(post_filename)  # saves results object as an hdf5 file



if __name__ == '__main__':
    
    beta_params=(0.867,3.03)
    N=50
    post_savedir='../posteriors/warm_jupiter'
    fig_savedir='../ecc_plots/warm_jupiter'
    driver(beta_params, N, post_savedir, fig_savedir)


 

