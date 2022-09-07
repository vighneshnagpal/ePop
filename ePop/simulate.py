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

import astropy.units as u
import astropy.constants as consts


def generate_orbits(systems,orb_fraction=0.05,npoints=3,start_mjd=51500,err_level=None):
    '''
    Generates orbits for multiple systems in one go.
    args: 
        systems (dict): Dictionary ordered by system index that contains
                        the eccentricties and inclinations for each.
                        
        orb_fraction (float betweem 0 and 1): The fraction of orbital coverage your data points cover. 
                                     Default is 0.05
        
        npoints (int): Number of data points spanning the observation window. Assumed to be evenly spaced. Defaults to 3.
        err_level (Tuple or None): Tuple of the form (sep_err, pa_err) for astrometric data. None setting defaults to (10mas, 0.1 deg)
    
    returns:
        astrometry (dict): Dictionary ordered by system index that contains
                           the epochs and corresponding sep/pa of each obs-
                           -ervation. 
    '''
    astrometry={}
    
    for sys in systems:
        
        # eccentricities and inclinations
        ecc,inc=systems[sys]
        
        # other orbital parameters: MOST ARE FIXED FOR NOW, WILL VARY LATER 
        aop=2*np.pi*np.random.rand()
        pan=2*np.pi*np.random.rand()
        tau=np.random.rand()
        plx=40
        mass_for_kamp=0.01
        mtot=1.01

        # semi-major axis
        min_sma, max_sma = 10, 100
        log_sma=np.random.uniform(np.log(min_sma),np.log(max_sma))
        sma=np.exp(log_sma)

        # compute orbital period
        period = np.sqrt(4 * np.pi**2.0 * (sma * u.AU)**3 / (consts.G * (mtot * u.Msun)))
        period = period.to(u.day).value
        
        #epochs and errors for astrometry
        end_mjd=start_mjd+orb_fraction*period

        astro_epochs=np.linspace(start_mjd,end_mjd,npoints)

        # dealing with uncertainties in Data
        if err_level is None:
            sep_err=10
            pa_err=0.1
        else:
            sep_err, pa_err = err_level
        
        # generate predictions for astrometric epochs
        astro_set=kepler.calc_orbit(astro_epochs,sma,ecc,inc,aop,pan,tau,plx,mtot,mass_for_Kamp=mass_for_kamp)
        ras,decs=astro_set[0],astro_set[1]
        sep,pa=orbitize.system.radec2seppa(ras,decs)

        # draw uncertainties and add to data
        sep_jit=np.random.randn(len(sep))*sep_err
        pa_jit=np.random.randn(len(pa))*pa_err

        sep+=sep_jit
        pa+=pa_jit

        # save these measurements to the astrometry dictionary
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

def create_data_tables(system_observations,err_level=None,save=False):
    '''
    Creates astropy data tables that can be used as input for orbitize.
    args: 
        system_observations (dict): Dictionary indexed by system number 
                                    where each element is a tuple of the
                                    form (epochs, sep, pa). epochs is an 
                                    array of observation times, and sep 
                                    and pa contain the corresponding ast-
                                    -rometry for each date.
        err_level (Tuple or None): Tuple of the form (sep_err, pa_err) for 
                                   astrometric data. None setting defaults 
                                   to (10mas, 0.1 deg)
        
        save (Bool): If set to true, the data is saved as csvs for the 
                     selected data.
    '''
    if err_level is None:
        sep_err=10
        pa_err=0.1
    else:
        sep_err, pa_err = err_level
    
    dataframes=[]

    for i in range(len(system_observations)):
        frame=[]
        epochs,seps,pas=system_observations[i]
        for mjd,sep,pa in zip(epochs,seps,pas):
            obs=[mjd,1,sep,sep_err,pa,pa_err,'seppa','test',np.nan]
            frame.append(obs)
        df=pd.DataFrame(frame, columns = ['epoch', 'object','quant1','quant1_err','quant2','quant2_err','quant_type','instrument','quant12_corr'])
        dataframes.append(df)

    astropy_tables=[Table.from_pandas(df) for df in dataframes]
    
    return astropy_tables


def simulate_sample(beta_params, N=10, orb_fraction=0.05, npoints=3):
    '''
    Code to forward model the eccentricity posteriors of a sample of imaged companions drawn from 
    an underlying population level eccentricity distribution. 
    
    Args: 
    
        beta_params (tuple of floats): A tuple of the form (a, b), where a and b are positive 
                                       floats that a re the Beta distribution hyperparameters
                                       corresponding to the underlying population level
                                       eccentricity distribution of the simulated sample.
                                       
        N (int): Number of objects in the forward modeled sample. Defaults to 10.
        
        orb_fraction (float between 0 and 1): Orbital fraction covered by the simulated 
                                              astrometry for each object in the forward
                                              modeled sample. Defaults to 0.05.
                                              
        npoints (int): Number of simulated astrometric points for each system in the forward
                       modeled sample. Defaults to 3. 
                       
    Returns:
    
        ecc_posteriors (list of 1D arrays): Set of N eccentricity posteriors, one for each object
                                            in the simulated sample.
    '''
    a,b = beta_params
    
    e_set   = draw_eccentricities(N,(a,b))
    inc_set = np.zeros(N)

    systems={i: [e_set[i],inc_set[i]] for i in range(N)}

    orbital_data=generate_orbits(systems,orb_fraction=orb_fraction,npoints=npoints)

    tables=create_data_tables(orbital_data)
    
    ecc_posteriors=[]
    for i in range(N):
        orb_sys = orbitize.system.System(1, tables[i], 1.0,
                                    40, mass_err=0.05, plx_err=0.1)
        ofti_sampler = sampler.OFTI(orb_sys)
        n_orbs = 1000
        _ = ofti_sampler.run_sampler(n_orbs,num_cores=1)
        accepted_eccentricities = ofti_sampler.results.post[:, 1]
        
        ecc_posteriors.append(accepted_eccentricities)
        
        
    return ecc_posteriors

