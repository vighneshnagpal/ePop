from os import name
import numpy as np
from scipy.optimize.optimize import main
from scipy.stats import beta 
import matplotlib.pyplot as plt
import glob
from pdb import set_trace

def truth_v_inferred(posts,true_beta,title,savename,first_row=False):
    '''
    Function to create a multipanel plot that shows the result of varying the
    sample size and observational precision on the ability to recover an underlying
    distribution via hierarchical MCMC. 
    
    See '../example_plots/summary.png' for a example
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
        


def plot_single(fname,compare_beta,savename=None,title=None,label='Underlying Distribution',ax=None,c=None,
                nrandom=2000,show_metric=True,show_underlying=True):
    '''
    A function to plot the inferred beta distributions from hierarchical MCMC samples
    as well as a second beta distribution (which could be any other beta distribution you 
    want to contrast with the inferred, such as an underlying distribution or a previous result)

    Args:

        fname (str): Path to the file containing the hierarchical MCMC samples

        compare_beta (tuple of positive floats): Tuple of the form (a,b), where a,b 
                                              are the beta parameters corresponding 
                                              to the second distribution you are 
                                              plotting

        savename (str): Savepath for the generated plot

        title (str): Title of the plot. Defaults to None

        label (str): The label to assign to the second beta distribution on the 
                     plot in the legend. Defaults to 'underlying'.
        
        ax  : Keyword that allows you to pass in a matplotlib axes object to plot_single.
              Defaults to None. If ax is not None, then plot_single creates and 
              returns a figure object. Handing in axes objects to plot_single using this 
              keyword can allow the creation of multipanel plots such as Figures 2 and 8 
              in Nagpal et. al (2022). These can be also be found in the 'example_plots' 
              subdirectory of ePop! and are an example of the sorts of plots that can be made
              using plot_single.  
        
        c : Color to plot the eccentricity distributions in. Must be a valid argument for 'c'
            in matplotlib. Defaults to None, in which case the color defaults to pink.
            
        nrandom
    
    Returns:
        
        if ax is None:
        
            fig (matplotlib.pyplot.Figure): A figure object that has all the plot information
        
        else:
            
            Nothing is returned, the input axes object is plotted on directly.

    '''
    rng  = np.linspace(0.00001,0.99999,10000)
    a,b = compare_beta[0],compare_beta[1]
    func = beta(a,b)


    if c==None:
        c='pink'

    if ax is None:
        plt.style.use('seaborn-bright')

        fig,ax=plt.subplots(1,1,figsize=(16,12))

        
        beta_samples=np.load(fname)
        
        if nrandom>beta_samples.shape[0]:
            print(f'There are only {beta_samples.shape[0]} eccentricity distributions to plot!')
            nrandom=beta_samples.shape[0]
            
        idx=np.random.randint(0,beta_samples.shape[0]-1,nrandom)
        random_samples=beta_samples[idx]
        random_samples=random_samples[random_samples[:,0].argsort()][::-1]
        
        for sample in random_samples:
            rnd_a,rnd_b=sample
            rnd_func=beta(rnd_a,rnd_b)
            ax.plot(rng,rnd_func.pdf(rng),c=c,alpha=0.2)
            
            resid=np.sum(np.abs(rnd_func.pdf(rng)-func.pdf(rng))*np.median(np.diff(rng)))
            metric+=resid/nrandom
        

        med_a,med_b=np.median(beta_samples,axis=0)[0],np.median(beta_samples,axis=0)[1]
        med_func=beta(med_a,med_b)

        ax.set_ylim([0,8])
        ax.set_xlim([0,1])
        
        
        if show_metric:
            ax.text(7.5,7.3,'$\\mathcal{M}$='+f'{np.round(metric,2)}',fontsize=30)
        
        
        #  beautification
        [x.set_linewidth(5.0) for x in ax.spines.values()]

        ax.xaxis.set_minor_locator(tck.AutoMinorLocator())
        ax.yaxis.set_minor_locator(tck.AutoMinorLocator())

        ax.tick_params(which='minor', length=6)
        ax.xaxis.set_tick_params(which='minor',top=True,direction='inout',length=12)
        ax.yaxis.set_tick_params(which='minor',right=True,direction='inout',length=12)

        ax.tick_params(axis="x", direction="inout")
        ax.tick_params(axis="y", direction="inout")
        ax.tick_params(axis='x',labelsize=24,length=20)#, colors='white')
        ax.tick_params(axis='y',labelsize=24,length=20)#, colors='white')
        ax.tick_params(bottom=True, top=True, left=True, right=True)


        ax.set_xlabel('Eccentricity',size=30)
        ax.set_ylabel('Probability Density',size=30)

        if title is not None:
            ax.set_title(title,size=24)
            
        plt.legend(prop={'size': 16})
        plt.savefig(savename,bbox_inches='tight')
        plt.close()
        return fig
    
    else:
        
        plt.rc('legend',fontsize=18)
        plt.style.use('seaborn-bright')
        
        beta_samples=np.load(fname)
        
        if nrandom>beta_samples.shape[0]:
            print(f'There are only {beta_samples.shape[0]} eccentricity distributions to plot!')
            nrandom=beta_samples.shape[0]
        
        # plots nrandom eccentricity distributions drawn from the posterior
        for i in range(nrandom):
            idx=np.random.randint(0,beta_samples.shape[0]-1)
            rnd_a,rnd_b=beta_samples[idx]
            rnd_func=beta(rnd_a,rnd_b)
            ax.plot(rng,rnd_func.pdf(rng),c=c,alpha=0.05)

            resid=np.sum(np.abs(rnd_func.pdf(rng)-func.pdf(rng))*np.median(np.diff(rng)))
            metric+=resid/nrandom
        

        med_a,med_b=np.median(beta_samples,axis=0)[0],np.median(beta_samples,axis=0)[1]
        med_func=beta(med_a,med_b)
        
        # plots the median distribution
        ax.plot(rng,med_func.pdf(rng),linewidth=5,alpha=0.9,c='black',label='$\\mathcal{M}$='+f'{np.round(metric,2)}')

        ax.set_xlim([0,1])
        ax.set_ylim([0,8])
        
        if show_metric:
            ax.text(7.5,7.3,'$\\mathcal{M}$='+f'{np.round(metric,2)}',fontsize=30)


        #  beautification
        [x.set_linewidth(5.0) for x in ax.spines.values()]
        ax.xaxis.set_minor_locator(tck.AutoMinorLocator())
        ax.yaxis.set_minor_locator(tck.AutoMinorLocator())

        ax.tick_params(which='minor', length=6)
        ax.xaxis.set_tick_params(which='minor',top=True,direction='inout',length=12)
        ax.yaxis.set_tick_params(which='minor',right=True,direction='inout',length=12)

        ax.tick_params(axis="x", direction="inout")
        ax.tick_params(axis="y", direction="inout")
        ax.tick_params(axis='x',labelsize=24,length=20)#, colors='white')
        ax.tick_params(axis='y',labelsize=24,length=20)#, colors='white')
        ax.tick_params(bottom=True, top=True, left=True, right=True)

        if show_underlying:
            ax.plot(rng,func.pdf(rng),c='black',linestyle='dashed',linewidth=3)#,label='Warm Jupiter')


        # copy-pasted END

        ax.set_xlabel('Eccentricity',fontsize=25)
        ax.set_ylabel('Probability Density',fontsize=25)

        if title is None:
            pass
        else:
            ax.set_title(title,fontsize=48,pad=40)
