from os import name
import numpy as np
from scipy.optimize.optimize import main
from scipy.stats import beta 
import matplotlib.pyplot as plt
import glob
from pdb import set_trace


def load_posteriors(fnames):
    '''
    Loads the posteriors for the beta parameters that are saved 
    as npy files into a dictionary format indexed by strings like 
    '5_20'.

    args:
        fnames: filenames
    returns:
        posts
    '''
    print([f[45:-4] for f in fnames])
    posts={f[45:-4]: np.load(f) for f in fnames}
    return posts

def truth_v_inferred(posts,true_beta,title,savename,first_row=False):
    '''
    Function to create a multipanel plot like (INSERT path).

    Args:

        posts (dict): A dictionary containing the hierarchical MCMC samples for 
                      multiple runs. 

        true_beta (tuple of positive floats): Tuple of the form (a,b), where a,b 
                                              are the beta parameters corresponding 
                                              to the second distribution you are 
                                              plotting


        title (str): Title of the plot. 

        savename (str): Savepath for the generated plot

        first_row (Bool): Only plots the first row of the panels if set to True.

    
    Returns:

        fig (matplotlib.pyplot.Figure): A figure object that has all the plot information



    '''
    a,b=true_beta

    rng  = np.linspace(0.0001,0.9999,10000)
    func = beta(a,b)

    if not first_row:
        
        fig=plt.figure(figsize=(20,20))

        fig.patch.set_facecolor('navy')

        gs = fig.add_gridspec(nrows=5,ncols=4,hspace=0.1,wspace=0.1)

        fig_ax1=fig.add_subplot(gs[0:2,:])

        fig_ax1.set_facecolor('midnightblue')

        fig_ax1.plot(rng,func.pdf(rng),c='yellow',linestyle='dashed',linewidth=3)
        fig_ax1.set_title('Underlying Distribution',size=20, c='white')
        fig_ax1.set_xlabel('Eccentricity',size=15,c='white')
        fig_ax1.set_ylabel('Probability Density',size=15,c='white')

        fig_ax1.spines['bottom'].set_color('white')
        fig_ax1.spines['top'].set_color('white')
        fig_ax1.spines['left'].set_color('white')
        fig_ax1.spines['right'].set_color('white')
        fig_ax1.xaxis.label.set_color('white')
        fig_ax1.tick_params(axis='x', colors='white')
        fig_ax1.tick_params(axis='y', colors='white')

        fig_ax1.annotate(f'a = {a}, b = {b}', (0.5,6),c='yellow',size=15)
        
        fig_ax1.set_ylim([0,7])
        fig_ax1.set_xlim([0,1])

        for j,val in enumerate([5,10,20,50]):
            
            for i,sig in enumerate([20,5,1]):
                
                print(i,j)
  
                ax=fig.add_subplot(gs[i+2,j])
                ax.set_facecolor('midnightblue')

                ax.spines['bottom'].set_color('white')
                ax.spines['top'].set_color('white')
                ax.spines['left'].set_color('white')
                ax.spines['right'].set_color('white')
                ax.xaxis.label.set_color('white')
                ax.tick_params(axis='x', colors='white')
                ax.tick_params(axis='y', colors='white')

                sim_name=f'{val}_{sig}'

                beta_post=posts[sim_name]
                
                med_a=np.median(beta_post[:,0])
                med_b=np.median(beta_post[:,1])

                med_func = beta(med_a,med_b)

                ax.plot(rng,med_func.pdf(rng),c='yellow',linewidth=5,alpha=0.8)
                ax.annotate(f'N = {val}, $\\sigma$ = {sig/100}', (0.5,6),c='yellow',size=15)

                nrandom=50
                for k in range(nrandom):
                    idx=np.random.randint(0,beta_post.shape[0]-1)
                    rnd_a,rnd_b=beta_post[idx]
                    rnd_func=beta(rnd_a,rnd_b)
                    ax.plot(rng,rnd_func.pdf(rng),c='pink',alpha=0.2)
                
                ax.set_ylim([0,7])
                ax.set_xlim([0,1])

        fig.supxlabel('Eccentricity',size=20,c='white')
        fig.supylabel('Probability Density',size=20,c='white')
        
        plt.suptitle(title,size=40,c='white')
        plt.savefig(savename)

    
    else:
           
        fig=plt.figure(figsize=(20,20))

        fig.patch.set_facecolor('navy')

        gs = fig.add_gridspec(nrows=2,ncols=8,hspace=0.1,wspace=0.1)

        fig_ax1=fig.add_subplot(gs[0,:])

        fig_ax1.set_facecolor('midnightblue')

        fig_ax1.plot(rng,func.pdf(rng),c='yellow',linestyle='dashed',linewidth=3)
        fig_ax1.set_title('Underlying Distribution',size=20, c='white')
        fig_ax1.set_xlabel('Eccentricity',size=15,c='white')
        fig_ax1.set_ylabel('Probability Density',size=15,c='white')

        fig_ax1.spines['bottom'].set_color('white')
        fig_ax1.spines['top'].set_color('white')
        fig_ax1.spines['left'].set_color('white')
        fig_ax1.spines['right'].set_color('white')
        fig_ax1.xaxis.label.set_color('white')
        fig_ax1.tick_params(axis='x', colors='white')
        fig_ax1.tick_params(axis='y', colors='white')

        fig_ax1.annotate(f'a = {a}, b = {b}', (0.5,6),c='yellow',size=15)
        
        fig_ax1.set_ylim([0,7])
        fig_ax1.set_xlim([0,1])

        for j,val in enumerate([5,10,20,50]):
            
            for i,sig in enumerate([20]):
                
                print(i,j)
  
                ax=fig.add_subplot(gs[1:,j*2:(j+1)*2])
                ax.set_facecolor('midnightblue')

                ax.spines['bottom'].set_color('white')
                ax.spines['top'].set_color('white')
                ax.spines['left'].set_color('white')
                ax.spines['right'].set_color('white')
                ax.xaxis.label.set_color('white')
                ax.tick_params(axis='x', colors='white')
                ax.tick_params(axis='y', colors='white')

                sim_name=f'{val}_{sig}'

                beta_post=posts[sim_name]
                
                med_a=np.median(beta_post[:,0])
                med_b=np.median(beta_post[:,1])

                med_func = beta(med_a,med_b)

                ax.plot(rng,med_func.pdf(rng),c='yellow',linewidth=5,alpha=0.8)
                ax.annotate(f'N = {val}',(0.5,4.5),c='yellow',size=20)
                ax.annotate(f'$\\sigma$ = {sig/100}', (0.5,4),c='yellow',size=20)

                nrandom=50
                for k in range(nrandom):
                    idx=np.random.randint(0,beta_post.shape[0]-1)
                    rnd_a,rnd_b=beta_post[idx]
                    rnd_func=beta(rnd_a,rnd_b)
                    ax.plot(rng,rnd_func.pdf(rng),c='pink',alpha=0.2)
                
                ax.set_ylim([0,7])
                ax.set_xlim([0,1])

        fig.supxlabel('Eccentricity',size=20,c='white')
        fig.supylabel('Probability Density',size=20,c='white')
        
        plt.suptitle('Experiment with Gaussian Posteriors using a log-uniform prior',size=40,c='white')
        plt.savefig(savename)

def plot_single(fname,true_beta,savename,title=None,true_label='underlying'):
    '''
    A function to plot the inferred beta distributions from hierarchical MCMC samples
    as well as a second beta distribution (which could be any other beta distribution you 
    want to contrast with the inferred, such as an underlying distribution or a previous result)

    Args:

        fname (str): Path to the file containing the hierarchical MCMC samples

        true_beta (tuple of positive floats): Tuple of the form (a,b), where a,b 
                                              are the beta parameters corresponding 
                                              to the second distribution you are 
                                              plotting

        savename (str): Savepath for the generated plot

        title (str): Title of the plot. Defaults to None

        true_label (str): The label you want for the second beta distribution on the 
                          plot. Defaults to 'underlying'.

    
    Returns:

        fig (matplotlib.pyplot.Figure): A figure object that has all the plot information

    '''
    rng  = np.linspace(0.00001,0.99999,10000)
    a,b = true_beta
    func = beta(a,b)
    fig=plt.figure(figsize=(8,6))
    plt.plot(rng,func.pdf(rng),c='black',linestyle='dashed',linewidth=3,label=true_label)
    beta_samples=np.load(fname)
    nrandom=30

    for i in range(nrandom):
        idx=np.random.randint(0,beta_samples.shape[0]-1)
        rnd_a,rnd_b=beta_samples[idx]
        rnd_func=beta(rnd_a,rnd_b)
        plt.plot(rng,rnd_func.pdf(rng),c='grey',alpha=0.2)
    
    med_a,med_b=np.median(beta_samples,axis=0)[0],np.median(beta_samples,axis=0)[1]
    med_func=beta(med_a,med_b)
    plt.plot(rng,med_func.pdf(rng),c='red',linewidth=5,alpha=0.8,label='Log-uniform prior')
    
    plt.ylim([0,7])
    plt.xlim([0,1])

    plt.xlabel('Eccentricity')
    plt.ylabel('Probability Density')

    if title is None:
        plt.title('50 systems, 4 points')
    else:
        plt.title(title)
    plt.legend()
    plt.savefig(savename)
    plt.close()
    return fig

if __name__=='__main__':
    files=sorted(glob.glob('./gaussian_experiment/log_uniform/beta_posts/*'))
    print(files[0][-14:])
    posts=load_posteriors(files)
    truth_v_inferred(posts,'onerow_log_uniform',first_row=True)

    # plot_single('./ogpaper/gp_loguniform_10.npy','./ogpaper/gp_loguniform_inferred','Comparing the eccentricity distributions of warm jupiters and imaged giants')
    # plot_single('./ogpaper/gp_uniform_1000.npy','./ogpaper/gp_uniform_inferred','A uniform prior applied to the giant planet sample')

    # plot_single('./ogpaper/bd_loguniform_10.npy','./ogpaper/bd_loguniform_inferred','Inferred population level distribution for the brown dwarf sample')
    # plot_single('./ogpaper/bd_uniform_1000.npy','./ogpaper/bd_uniform_inferred','A uniform prior applied to the brown dwarf sample')

    # plot_single('./ogpaper/overall_loguniform_10.npy','./ogpaper/overall_loguniform_inferred','Inferred distribution for the overall substellar companion sample')
    # plot_single('./ogpaper/overall_uniform_1000.npy','./ogpaper/overall_uniform_inferred','A uniform prior applied to the overall sample')


    # fnames=sorted(glob.glob('./gaussian_experiment/uniform/beta_posts/*'))
    # posts=load_posteriors(fnames)
    # print(posts.keys())
    # truth_v_inferred(posts,'./gaussian_experiment/uniform/panel.png')




