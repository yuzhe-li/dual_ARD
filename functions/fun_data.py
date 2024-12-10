import matplotlib.pyplot as plt
import numpy as np
import os
from utility import create_folder



# from fun_artificial_data_methods import create_folder
# from fun_models_class import Model
class Data_calcium():
    def __init__(self, n0=3,n1=7,T=3000,firing_rate0=3,firing_rate1=0.15,frame_rate=30,
                 calcium_gamma=0.95, calcium_alpha=1,y_a = 1, y_b = 0.1,
                 poisson_rho =0.01,gaussian_sn_kappa=0.1,type ='kappa',seed = 0):
        self.n0 = n0
        self.n1 = n1
        self.T = T
        self.spike_lam0 = firing_rate0/frame_rate
        self.spike_lam1 = firing_rate1/frame_rate
        self.calcium_gamma = calcium_gamma
        self.calcium_alpha = calcium_alpha
        self.y_a = y_a
        self.y_b = y_b
        self.poisson_rho = poisson_rho
        self.gaussian_sn_kappa = gaussian_sn_kappa
        self.seed = seed
        self.type = type


    def gen_spike(self):
        np.random.seed(self.seed)
        s0 = np.random.poisson(lam = self.spike_lam0,size =(self.n0,self.T))
        s1 = np.random.poisson(lam = self.spike_lam1,size =(self.n1,self.T))
        s = np.vstack([s0,s1])
        self.spike = s

    def gen_calcium(self):

        s =self.spike
        c = np.zeros((self.n0 + self.n1,self.T))
        c[:,0] =self.calcium_alpha*s[:,0]
        for i in range(1,self.T):
            c[:,i] = self.calcium_gamma*c[:,i-1] + self.calcium_alpha*s[:,i]
        self.calcium = c

    def gen_observation_poisson(self):
        """
        _lambda: n_featuers * n_samples
        """
        np.random.seed(self.seed)

        c = self.calcium
        a = self.y_a
        b = self.y_b
        z =  a*c+b
        self.z = z
        rho = self.poisson_rho
        if rho == 0:
            self.y_poisson = c
            return c
        else:
            y = np.zeros(c.shape)
            for i in range(c.shape[0]):
                for j in range(c.shape[1]):
                    y[i,j] = rho*np.random.poisson(z[i,j]/rho)
            self.y_poisson=y
            return y


    def gen_observation_gaussian(self):
        np.random.seed(self.seed)
        y_poisson = self.y_poisson

        D,T = y_poisson.shape
        if self.type =='sn':
            sn = self.gaussian_sn_kappa
            y =y_poisson+np.random.randn(D,T)*sn
        elif self.type =='kappa':
            kappa = self.gaussian_sn_kappa
            y =y_poisson+np.random.randn(D,T)*kappa*np.mean(self.calcium)
        self.y_poisson_gaussian = y

        self.y = y

    def gen_data(self):
        seed=self.seed
        # np.random.seed(seed = seed)
        self.gen_spike()
        self.gen_calcium()
        y_poisson = self.gen_observation_poisson()
        self.gen_observation_gaussian()

        return self.y_poisson_gaussian,self.calcium,self.spike



    def imshow_data(self,savepath=None, figsize = (20,6)):
        s = self.spike
        c = self.calcium
        y = self.y_poisson_gaussian
        fig,ax = plt.subplots(figsize = figsize,ncols = 3, nrows = 1)
        # ax0=ax[0].imshow(s,aspect = 'auto',interpolation='none', vmax = 1);
        # ax1 =ax[1].imshow(c,aspect = 'auto',interpolation='none');
        # ax2=ax[2].imshow(y,aspect = 'auto',interpolation='none');
        ax0=ax[0].imshow(s,aspect = 'auto', vmax = 1);
        ax1 =ax[1].imshow(c,aspect = 'auto',vmax = 6);
        ax2=ax[2].imshow(y,aspect = 'auto',vmax = 6, vmin  = None);
        fig.colorbar(ax0,ax = ax[0])
        fig.colorbar(ax1,ax = ax[1])
        fig.colorbar(ax2,ax = ax[2])
        ax[0].set_title('Spike')
        ax[1].set_title('True Calcium')
        ax[2].set_title('Calcium Observation')

        ax[0].set_xlabel('Time (Frame)')
        ax[0].set_ylabel('Spike')
        ax[1].set_xlabel('Time (Frame)')
        ax[1].set_ylabel('Calcium intensity')
        ax[2].set_xlabel('Time (Frame)')
        ax[2].set_ylabel('Calcium intensity')
        fig.set_tight_layout(True)
        if savepath is not None:
            figname = 'imshow_data.pdf'
            figname = os.path.join(savepath,figname)
            fig.savefig(figname)
        plt.show()

def imshow_data_c(s,c,y,savepath=None, figsize = (20,6), xtick_s  = False, 
                  framerate = None, xtickgap = 10, vmax = 6, vmin = None):

    fig,ax = plt.subplots(figsize = figsize,ncols = 2, nrows = 1)
    ax0 =ax[0].imshow(c,aspect = 'auto',vmax = vmax)
    ax1=ax[1].imshow(y,aspect = 'auto',vmax = vmax, vmin  = vmin)
    fig.colorbar(ax0,ax = ax[0])
    fig.colorbar(ax1,ax = ax[1])
    ax[0].set_title('True Calcium')
    ax[1].set_title('Calcium Observation')

    
    ax[0].set_ylabel('Calcium intensity')
    ax[1].set_ylabel('Calcium intensity')



    if xtick_s: 
        if framerate is None:
            framerate = 3.86
        
        xticks = np.arange(0,y.shape[1]+1,framerate*xtickgap)
        xticklabels = np.floor((xticks)/framerate).astype(int)
        for i in [0,1]:
            ax[i].set_xticks(xticks)
            ax[i].set_xticklabels(xticklabels)
            ax[i].set_xlabel('Time (s)')
    else:
        ax[0].set_xlabel('Time (Frame)')
        ax[1].set_xlabel('Time (Frame)')

    fig.set_tight_layout(True)
    if savepath is not None:
        create_folder(savepath)
        figname = 'imshow_data_c.pdf'
        figname = os.path.join(savepath,figname)
        fig.savefig(figname)
    plt.show()





def plot_data_i(Y,C,S,i= 0,title=None,ymax = 4.5,ymin =-1.5,savepath = None
                , figsize = (8,6), linewidth= 2,titlesize=None,
                xtick_s = False, framerate = 30,xtickgap= 10 ):
    y = Y[i,:]
    c = C[i,:]
    s = S[i,:]
    n = Y.shape[1]
    fig, ax = plt.subplots(figsize = figsize)
    ax.plot(y, linewidth = linewidth);
    ax.plot(c,linewidth =linewidth);




    (markerline, stemlines, baseline) =ax.stem(np.arange(0,n), s*0.8-1.5,'g', markerfmt='none',basefmt='none',bottom = -1.5)   
                                            #   , use_line_collection=True)
    plt.setp(stemlines, 'linewidth',  linewidth)
    # plt.line(np.linspace(0,n-1,n),s*0.5-1,width=0.01)
    ax.legend(['Calcium Observation','True calcium', 'Spike'],loc = 'upper right')
    if title is not None:
        ax.set_title(title, fontsize = titlesize)
    ax.set_xlabel('Time (Frame)')

    if xtick_s: 

        xticks = np.arange(0,len(y)+1,framerate*xtickgap)
        xticklabels = np.floor((xticks)/framerate).astype(int)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_xlabel('Time (s)')
    else:
        ax.set_xlabel('Time (Frame)')




    ax.set_ylabel('Activity')
    ax.set_ylim([ymin,ymax])
    # fig.tight_layout(rect=[0, 0, 0.95, 1])
    fig.tight_layout()
    
    if savepath is not None:
        create_folder(savepath)
        figname='data_'+str(i)+'.pdf'
        figname = os.path.join(savepath, figname)
        fig.savefig(figname)

    plt.show()