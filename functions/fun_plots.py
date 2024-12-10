# fun_my_latent_state_space_community.py

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.insert(1, './')

def create_folder(my_folder):
    from pathlib import Path
    Path(my_folder).mkdir(parents=True, exist_ok=True)

def plot_hinton(matrix, max_weight=None, ax=None):
    import numpy as np
    import matplotlib.pyplot as plt
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log2(np.abs(matrix).max()))

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(abs(w) / max_weight)
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()


def _blob(axes, x, y, area, colour):
    """
    Draws a square-shaped blob with the given area (< 1) at
    the given coordinates.
    """
    import numpy as np
    hs = np.sqrt(area) / 2
    xcorners = np.array([x - hs, x + hs, x + hs, x - hs])
    ycorners = np.array([y - hs, y - hs, y + hs, y + hs])
    axes.fill(xcorners, ycorners, colour, edgecolor=colour)

def atleast_nd(X, d):
    import numpy as np
    if np.ndim(X) < d:
        sh = (d-np.ndim(X))*(1,) + np.shape(X)
        X = np.reshape(X, sh)
    return X
def hinton(W, error=None, vmax=None, square=False, axes=None):
    """
    Draws a Hinton diagram for visualizing a weight matrix.
    Temporarily disables matplotlib interactive mode if it is on,
    otherwise this takes forever.
    Originally copied from
    http://wiki.scipy.org/Cookbook/Matplotlib/HintonDiagrams
    """
    import numpy as np
    import matplotlib.pyplot as plt

    if axes is None:
        axes = plt.gca()

    # W = atleast_nd(W, 2)
    (height, width) = W.shape
    if not vmax:
        #vmax = 2**np.ceil(np.log(np.max(np.abs(W)))/np.log(2))
        if error is not None:
            vmax = np.max(np.abs(W) + error)
        else:
            vmax = np.max(np.abs(W))

    axes.fill(0.5+np.array([0,width,width,0]),
              0.5+np.array([0,0,height,height]),
              'gray')
    if square:
        axes.set_aspect('equal')
    axes.set_ylim(0.5, height+0.5)
    axes.set_xlim(0.5, width+0.5)
    axes.set_xticks([])
    axes.set_yticks([])
    axes.invert_yaxis()
    for x in range(width):
        for y in range(height):
            _x = x+1
            _y = y+1
            w = W[y,x]
            _w = np.abs(w)
            if w > 0:
                _c = 'white'
            else:
                _c = 'black'
            if error is not None:
                e = error[y,x]
                if e < 0:
                    print(e, _w, vmax)
                    raise Exception("BUG? Negative error")
                if _w + e > vmax:
                    print(e, _w, vmax)
                    raise Exception("BUG? Value+error greater than max")
                _rectangle(axes,
                           _x,
                           _y,
                           min(1, np.sqrt((_w+e)/vmax)),
                           min(1, np.sqrt((_w+e)/vmax)),
                           edgecolor=_c,
                           fill=False)
            _blob(axes, _x, _y, min(1, _w/vmax), _c)


# read and plot nodes of bayespy


def plot_Gamma(x,i,color = 'r',xmax = 100,name =None,n =1000,ax=None):
    """
    Params:
    x: node, with shape (multivariable)
    ---------------
    """
    from scipy.stats import gamma
    import matplotlib.pyplot as plt
    if ax is None:
        fig,ax = plt.subplots()
    if x.phi[1].shape[0]==1:
        a = x.phi[1]
    else:
        a = x.phi[1][i]
    if x.phi[0].shape[0]==1:
        b = -1/x.phi[0]
    else:
        b = -1/x.phi[0][i]
    # print('scipy:',a,b)
    xx = np.linspace(1e-6,xmax, n)
    yy =np.squeeze(gamma.pdf(xx, a,loc =0,scale=b))
    ax.plot(xx, yy,color =color)
    if name is not None:
        ax.set_title(name)

def imshow_X(X,figsize = (10,6)):
    plt.figure(figsize = figsize)
    X_ = X.get_moments()[0]
    X_ = np.squeeze(X_)
    plt.imshow(X_.T,aspect='auto',cmap='RdBu_r');
    plt.colorbar();
    plt.title('latent factors');
    plt.xlabel('Time');
    plt.ylabel('# latent factor');
    return X_


def imshow_W(W,figsize = (10,6)):
    plt.figure(figsize = figsize)
    W_ = W.get_moments()[0]
    W_ = np.squeeze(W_)
    plt.imshow(W_,aspect='auto',cmap='RdBu_r');
    plt.colorbar();
    plt.title('Loading matrix');
    plt.xlabel('# latent factor');
    plt.ylabel('# neuron');
    plt.show()
    return W_

def imshow_F(F,y, figsize=(18,6)):
    from  scipy.stats import pearsonr
    from sklearn.metrics import mean_squared_error


    F_ = F.get_moments()[0]
    mse=mean_squared_error(F_,y)
    var_F = np.var(F_,axis =1).sum()
    var_y = np.var(y,axis =1).sum()
    var_explained = var_F/var_y*100
    cc = pearsonr(np.ravel(F_),np.ravel(y))[0]
    plt.figure(figsize = figsize)
    plt.subplot(1,2,1)
    plt.imshow(F_,aspect='auto')
    plt.colorbar();
    plt.xlabel('Time')
    plt.ylabel('Neurons')
    plt.title(r'$\hat{y}$ (variance explained:'+str(int(var_explained))+'%'+'\n'+'corrlation with y: '+str(round(cc, 2))+', MSE: '+str(round(mse,4)));
    plt.subplot(1,2,2)
    plt.imshow(y,aspect='auto')
    plt.colorbar();
    plt.xlabel('Time')
    plt.ylabel('Neurons')
    plt.title(r'${y}$')
    plt.show()
    return F_,mse,cc
def plot_example_yi(W,X,y,mu= None,idx=None,irange=range(5),figsize=(10,6),savepath = None,figtype ='.png'):
    if type(W) is np.ndarray:
        W_ = W
    else:
        W_= np.squeeze(W.get_moments()[0])
    if type(X) is np.ndarray:
        X_ = X
    else:
        X_ = np.squeeze(X.get_moments()[0])
    if mu is None:
        y_ = np.matmul(W_,X_.T)
    else:
        y_ = np.matmul(W_,X_.T)+mu.reshape(-1,1)

    if idx is not None:
         W_effective = W_[:,idx]
         X_effective = X_[:,idx]
         y_effective = np.matmul(W_effective,X_effective.T)
    n = len(irange)
    fig,ax = plt.subplots(n,1,figsize = figsize)
    for i in irange:
        ax[i].plot(y[i,:],label='y');
        if idx is not None:
            ax[i].plot(y_effective[i,:],label =r'$\hat y_{effective}$');
        else:
            ax[i].plot(y_[i,:],label =r'$\hat y$');

    ax[0].legend();
    plt.xlabel('Time')
    fig.suptitle('Example neuron activity')
    if savepath is not None:
        filename = savepath + 'example_yi'+figtype
        fig.savefig(filename)
    plt.show(block = False);plt.pause(1);plt.close();plt.clf()


def plot_facotr_explained(W,X,y):
    W_ = W.get_moments()[0]
    X_ = X.get_moments()[0]
    nf = W_.shape[1]
    print(nf)
    var_explained = np.zeros(nf)
    for i in range(nf):
        print(i)
        yi= np.multiply(W_[:,i].reshape(-1,1),X_[:,i].reshape(-1,1).T)
        var_explained[i]=np.var(yi,axis =1).sum()/np.var(y,axis =1).sum()*100
    plt.plot(var_explained);
    plt.title('explained variance (%)');
    plt.xlabel('Latent factor');
    plt.ylabel(' explained variance (%)')
    plt.show()
    return var_explained
def imshow_effective_W(W,X,A = None,thre = 1e-3):
    plt.rcParams["figure.figsize"] = (10,6)
    W_ = np.squeeze(W.get_moments()[0])
    X_ = np.squeeze(X.get_moments()[0])
    w_var = np.var(W_,axis =0)
    D = W_.shape[1]  # nf
    idx_effective_w = np.where(np.var(W_,axis =0)>thre)[0]

    X_effective =X_[:,idx_effective_w]
    W_effective = W_[:,idx_effective_w]
    plt.plot(w_var,'k');
    plt.plot(np.linspace(0,D,D),thre*np.ones(D),'r--');
    plt.xlabel('latent factors');
    plt.ylabel('variance');
    plt.title('variance of $w_d$ (column of loading matrix $W$)(D='+str(len(idx_effective_w))+')');
    plt.show()


    plt.imshow(W_effective,aspect='auto',cmap='RdBu_r');
    plt.colorbar();plt.title('Effective loading matrix (D='+str(len(idx_effective_w))+')');
    plt.xlabel('Time');plt.ylabel('#');
    plt.show()
    plt.imshow(X_effective.T,aspect='auto',cmap='RdBu_r');
    plt.colorbar();plt.title('Effective latent factors');
    plt.xlabel('Time');plt.ylabel('#');
    plt.show()
    if A is None:
        return W_effective, X_effective
    else:
        A_ = np.squeeze(A.get_moments()[0])
        A_effective = A[:,idex_effective_w]

def plot_X(X,D=None,figsize = (10,12),ylim=[-1.2,1.2],savepath =None,figtype='.png'):
    """
    X: node

    """
    X_ = X.get_moments()[0]
    X_ = np.squeeze(X_)
    nf = X_.shape[1]
    fig, ax = plt.subplots(nrows=nf, ncols=1,figsize=figsize)
    for i in range(nf):
        ax[i].plot(X_[:,i],'k')
        if ylim is not None:
            ax[i].set_ylim(ylim)
    ax[nf-1].set_xlabel('Time (frame)')

    # fig.supylabel('latent factor')
    if nf == D:
        title ='Latent factors(D='+str(nf)+')'
        filename = 'X'+figtype
    else:
        title = 'Effective latent factors(D='+str(nf)+')'
        filename= 'X_effective'+figtype
    fig.suptitle(title)
    fig.subplots_adjust(top=0.95)
    if savepath is not None:
        fig.savefig(savepath +filename)
    plt.show(block = False);plt.pause(1);plt.close();plt.clf()
def plot_X_(X_,D=None,figsize = (10,12),ylim=[-1.2,1.2],savepath =None,figtype='.png'):
    """
    X: node

    """
    X_ = np.squeeze(X_)
    nf = X_.shape[1]
    fig, ax = plt.subplots(nrows=nf, ncols=1,figsize=figsize)
    for i in range(nf):
        ax[i].plot(X_[:,i],'k')
        if ylim is not None:
            ax[i].set_ylim(ylim)
    ax[nf-1].set_xlabel('Time (frame)')

    # fig.supylabel('latent factor')
    if nf == D:
        title ='Latent factors(D='+str(nf)+')'
        filename = 'X'+figtype
    else:
        title = 'Effective latent factors(D='+str(nf)+')'
        filename= 'X_effective'+figtype
    fig.suptitle(title)
    fig.subplots_adjust(top=0.95)
    if savepath is not None:
        fig.savefig(savepath +filename)
    plt.show(block = False);plt.pause(1);plt.close();plt.clf()
def plot_effective_X(X_effective,D=None,figsize = (10,20),ylim=[-1.2,1.2],savepath=None,figtype='.png'):
    nf = X_effective.shape[1]
    fig, ax = plt.subplots(nrows=nf, ncols=1,figsize=figsize)
    if nf == 1:
        ax.plot(X_effective,'k')
        ax.set_xlabel('Time (frame)')
    else:
        for i in range(nf):
            ax[i].plot(X_effective[:,i],'k')
            if ylim is not None:
                ax[i].set_ylim(ylim)
        ax[nf-1].set_xlabel('Time (frame)')
    if nf == D:
        title ='Latent factors(D='+str(nf)+')'
        filename = 'X'+figtype
    else:
        title = 'Effective latent factors(D='+str(nf)+')'
        filename= 'X_effective'+figtype
    fig.suptitle(title);
    fig.subplots_adjust(top=0.95)
    if savepath is not None:
        fig.savefig(savepath+filename)
    plt.show(block = False);plt.pause(1);plt.close();plt.clf()
def plot_gard(x,i,xrange=[-1,1],name=None,n = 1000):
    # GaussianARD
    from scipy.stats import norm
    import matplotlib.pyplot as plt
    import numpy as np
    mu = np.squeeze(x.get_moments()[0])

    alpha = -2*np.squeeze(x.phi[1])

    # cov = x.u[1]-_outer(mu,mu,ndim =0)
    # print(mu[i],np.sqrt(1/alpha[i]))

    xx = np.linspace(xrange[0],xrange[1],n)
    if len(alpha.shape) ==2:
        yy = norm.pdf(xx,mu[i],np.sqrt(1/alpha[i,i]))
    else:
        yy = norm.pdf(xx,mu[i],np.sqrt(1/alpha[i]))

    plt.plot(xx,yy)
    if name is not None:
        plt.xlabel('p('+name+')');
        plt.ylabel('p('+name+')');
        plt.title(name)

def plot_mu(mu,figsize = (10,20),xrange = [-1,2], irange=None,savepath = None,figtype='.png'):
    mu_ = np.squeeze(mu.get_moments()[0])
    D = mu_.shape[0]
    if D > 50:
        D = 50
    fig =plt.figure(figsize=figsize);
    if irange is None:
        irange = range(D)
    for i in irange:
        plt.subplot(len(irange),1,i+1)
        plot_gard(mu,i,xrange =xrange)#,name='$\mu$'+str(i));
    plt.xlabel(r'$\mu$')
    fig.suptitle(r'$P(\mu)$')
    fig.subplots_adjust(top=0.95)
    if savepath is not None:
        fig.savefig(savepath+'mu'+figtype)
    plt.show(block = False);plt.pause(1);plt.close();plt.clf()
def plot_gammas_node(x,figsize = (10,20),xmax =1,irange = None):
    fig =plt.figure(figsize=figsize);
    if irange is None:
        irange = range(x.plates[0])
    if x.plates[0] > 50:
        irange =range(50)
    for i in irange:
        plt.subplot(len(irange),1,i+1)
        plot_Gamma(x,i,c='y',xmax = xmax)
    fig.subplots_adjust(top=0.96)

    return fig

def plot_alpha(alpha,figsize = (10,20),xmax =500,irange = None,savepath = None,figtype='.png'):
    fig =plot_gammas_node(alpha,figsize = figsize,xmax = xmax,irange = irange)
    plt.xlabel(r'$\alpha$')
    fig.suptitle(r'$P(\alpha)$')
    if savepath is not None:
        filename = savepath + 'alpha'+figtype
        fig.savefig(filename)
    plt.show(block = False);plt.pause(1);plt.close();plt.clf()

def plot_beta(beta,figsize=(10,20),xmax =1,irange = None,savepath=None,figtype='.png'):
    fig = plot_gammas_node(beta,figsize = figsize,xmax = xmax,irange = irange)
    plt.xlabel(r'$\beta$')
    fig.suptitle(r'$P(\beta)$')
    if savepath is not None:
        filename = savepath + 'beta'+figtype
        fig.savefig(filename)
    plt.show(block = False);plt.pause(1);plt.close();plt.clf()
def plot_gamma(gamma, figsize=(10,20),xmax = 1,irange = None,savepath=None,figtype='.png'):
    fig = plot_gammas_node(gamma,figsize = figsize,xmax = xmax,irange = irange)
    plt.xlabel(r'$\gamma$')
    fig.suptitle(r'$P(\gamma)$')
    if savepath is not None:
        filename = savepath + 'gamma'+figtype
        fig.savefig(filename)
    plt.show(block = False);plt.pause(1);plt.close();plt.clf()
def plot_tau(tau, figsize=(10,20),xmax = 1,irange = None,savepath=None,figtype='.png'):
    fig = plot_gammas_node(tau,figsize = figsize,xmax = xmax,irange = irange)
    plt.xlabel(r'$\tau$')
    fig.suptitle(r'$P(\tau)$')
    if savepath is not None:
        filename = savepath + 'tau'+figtype
        fig.savefig(filename)
    plt.show(block = False);plt.pause(1);plt.close();plt.clf()
def hinton_W(W_,W_effective=None, range=30,figsize =(10,10),type='W',savepath = None,figtype='.png'):
    """
    Parameters:
    ----------
    W_: matrix of W_(loading matrix), or A_ (transition matrix)
    W_effective:

    type: 'W', 'A', matters title and labels

    -----------
    Out:
    hinton plot of W or A or W + W_effective
    """
    plt.rcParams["figure.figsize"] = figsize
    if range is None:
        range = W_.shape[0]


    hinton(W_[0:range,:]);
    if type =='W':
        plt.title('Loading matrix(example neurons: 0-'+str(range)+')');
        plt.xlabel('Latent factors');
        plt.ylabel('Features (neurons)');
    elif type =='A':
        plt.title('State transition matrix  (A)')
        plt.xlabel('state')
        plt.ylabel('state')
    if savepath is not None:
        filename =savepath +'hinton_W'+figtype
        plt.savefig(filename)
    plt.show(block = False);plt.pause(1);plt.close();plt.clf()
    if W_effective is not None:
        hinton(W_effective[0:range,:],square=True);
        D = W_effective.shape[1]
        plt.title('Effective loading matrix (D='+str(D)+'),  neurons: 0-'+str(range)+')');
        plt.xlabel('Latent factors');
        plt.ylabel('Features (neurons)');
        if savepath is not None:
            filename =savepath +'hinton_W_effective'+figtype
            plt.savefig(filename)

        plt.show(block = False);plt.pause(1);plt.close();plt.clf()

def find_effective_W(W,thre_mean = 1e-3,thre_std = 1e-2):
    W_ = np.squeeze(W.get_moments()[0])
    abs_mean = np.squeeze(np.mean(abs(W_),axis = 0))
    std_w = np.squeeze(np.std(W_,axis = 0))
    idx =[]
    for i in range(W_.shape[1]):
        if std_w[i] > thre_std and abs_mean[i] > thre_mean:
            idx.append(i)
    return idx


def plot_A(A,figsize=(10,8),savepath = None,figtype='.png'):
    A_ = A.get_moments()[0]
    plt.figure(figsize=figsize)
    plt.imshow(A_,aspect='auto',cmap='viridis');
    plt.title('State transition matrix')
    plt.xlabel('latent states')
    plt.ylabel('latent states')
    plt.colorbar();
    if savepath is not None:
        plt.savefig(savepath+'A'+figtype)
    plt.show(block = False);plt.pause(1);plt.close();plt.clf()



def read_behavior(matfile):
    from scipy.io import loadmat
    data = loadmat(matfile)
    move = np.squeeze(data['all_move_scan'])
    speed = np.squeeze(data['all_speed_scan'])
    spk_on= np.squeeze(data['all_spk_scan'])
    spk_off= np.squeeze(data['all_spk_scan_down'])
    behavior = dict()
    behavior['move'] = move
    behavior['speed'] = speed
    behavior['spk_on'] = spk_on
    behavior['spk_off'] = spk_off
    return behavior


def plot_X_y_behavoir(X,y,behavior,figsize = (10,12),ylim = None):
    X_ = X.get_moments()[0]
    X_ = np.squeeze(X_)
    nf = X_.shape[1]
    nb = len(behavior.keys())
    keys = list(behavior.keys())

    fig, ax = plt.subplots(nrows=nf+nb+1, ncols=1,figsize=figsize)
    for i in range(nb):
        ax[i].plot(behavior[keys[i]].T)
    ax[nb].plot(np.mean(y,axis = 0),'b')

    for i in range(nf):
        ax[i+nb+1].plot(X_[:,i],'k')
        if ylim is not None:
            ax[i+nf].set_ylim(ylim)
    ax[nf-1].set_xlabel('Time (frame)')


    fig.suptitle('move,speed,spk_, y_mean, X')
    fig.subplots_adjust(top=0.95)



def plot_Q(Q,y,savepath = None,model =None,hyperparameters=None,figtype ='.png'):
    from fun_effective_factors import find_effective_x, find_effective_x_Q

    if savepath is not None:
        from pathlib import Path
        Path(savepath).mkdir(parents=True, exist_ok=True)


    X = np.squeeze(Q['X'].get_moments()[0])
    W = np.squeeze(Q['W'].get_moments()[0])
    D = X.shape[1]
    X_effective,idx_effective = find_effective_x_Q(Q)


    if model is None:
        hyperparameters=hyperparameters
    elif model == 'wxmu' :
        hyperparameters=['mu','tau']
    elif model =='wxmu_alpha':
        hyperparameters =['mu','alpha','tau']
    elif model =='wxmu_alpha_beta':
        hyperparameters =['mu','alpha','beta','tau']
    elif model == 'wxmu_alpha_beta_gamma':
        hyperparameters =['mu','alpha','beta','gamma','tau']
    elif model=='wx':
        hyperparameters =['tau']
    elif model=='wx_alpha':
        hyperparameters =['alpha','tau']
    elif model=='wx_gamma':
        hyperparameters =['gamma','tau']
    elif model=='wx_alpha_gamma':
        hyperparameters =['alpha','gamma','tau']
    elif model=='wx_alpha_gamma_beta':
        hyperparameters =['alpha','gamma','beta','tau']
    elif model =='wx_mu_alpha':
        hyperparameters =['mu','alpha','tau']
    elif model =='wx_mu_alpha_gamma':
        hyperparameters =['mu','alpha','gamma','tau']
    elif model == 'wx_mu_alpha_gamma_beta':
        hyperparameters =['mu','alpha','gamma','beta','tau']
    elif model =='wxA_alpha':
        hyperparameters =['A','alpha']
    elif model =='wxA_alpha_gamma':
        hyperparameters =['A','alpha','gamma']
    elif model =='wxA_mu_Lambda_alpha':
        hyperparameters =['A','mu','alpha','Lambda']
    elif model =='wxA_mu_alpha_beta_gamma':
        hyperparameters =['A','mu','alpha','beta']
    elif model =='wxA_mu_Lambda_alpha_beta_gamma':
        hyperparameters =['A','mu','alpha','Lambda','beta']
    elif model =='wx_alpha_gamma':
        hyperparameters =['alpha','gamma','tau']
    elif model =='wx_mu':
        hyperparameters =['mu','tau']
    print(hyperparameters)
    plot_X_(X,D = D,figsize = (10,20),ylim=None,savepath = savepath,figtype=figtype);
    plot_effective_X(X_effective,ylim=None, savepath = savepath,figtype=figtype)
    # plot_example_yi(W,X,y,idx=idx_effective,irange = range(30),figsize = (10,15),savepath = savepath,figtype=figtype)

    plot_example_yi(W,X,y,idx=idx_effective,irange = range(10),figsize = (10,8),savepath = savepath,figtype=figtype)
    hinton_W(W,range =30,savepath = savepath,figtype=figtype)

    # plot hyperparameters

    if 'mu' in hyperparameters:
        mu = Q['mu']
        plot_mu(mu,savepath = savepath,figtype=figtype)
    if 'alpha' in hyperparameters:
        alpha = Q['alpha']
        plot_alpha(alpha,xmax =1000,irange =range(30),savepath = savepath,figtype=figtype)

    if 'gamma' in hyperparameters:
        gamma = Q['gamma']
        plot_gamma(gamma,savepath = savepath,figtype=figtype)

    if 'beta' in hyperparameters:
        beta = Q['beta']
        plot_beta(beta,savepath = savepath,figtype=figtype)

    if 'tau' in hyperparameters:
        tau = Q['tau']
        plot_tau(tau,xmax = 100,irange = range(30),savepath = savepath,figtype=figtype)
    if 'A' in hyperparameters:
        A = Q['A']
        plot_A(A,savepath = savepath,figtype=figtype)



def pointplot_twinx(y1,y2,models,labels, title=None,ylims = [None,None], figfile= None,rotation = 0,loc = 1):
    """
    plot two objects [y1,y2] with two yaixs
    models:  xticks
    labels: labels of [y1, y2]
    title:
    """
    import seaborn as sns
    import pandas as pd
    y1 = pd.DataFrame(np.array(y1).reshape(1,-1),columns=models)
    y2 = pd.DataFrame(np.array(y2).reshape(1,-1),columns=models)
    """
    plot elbo and number of effective latent factors
    """
    fig = plt.figure(figsize = (10,8));ax = fig.add_subplot(111);
    sns.pointplot(data=y1,color="b",ax =ax);
    ax.set_ylim(ylims[0])
    ax2 = ax.twinx();
    sns.pointplot(data=y2, color="g", ax=ax2);
    ax.set_xticklabels(ax.get_xticklabels(),rotation=rotation,fontsize=16);
    ax.set_ylabel(labels[0],fontsize =16);
    ax2.set_ylabel(labels[1],fontsize=16);
    ax2.set_ylim(ylims[1])
    ax2.legend(handles=ax.lines[::15]+ax2.lines[::15],labels=labels ,loc =loc)
    ax.set_title(title)

    if figfile is not None:
        plt.savefig(figfile,bbox_inches = 'tight')
    plt.show()

def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)




def pointplot_3axis(y1,y2,y3,
                    x1 = None, x2 = None, x3 = None,
                    title='',figfile = None,
                    xticklabels =None,
                    ylims=[None]*3,ylabels = [None]*3,colors = [None]*3,labels = [None]*3,
                    figsize = (10,8), lw = 3,
                    savepath = None, figname = None,
                    legend_loc = 'best'):
    """
    plot 3 objects [y1,y2,y3] with two yaixs
    models:  xticks
    labels: labels of [y1, y2,y3]
    title:
    """

    fig, host = plt.subplots(figsize=figsize)
    # fig.subplots_adjust(right=0.9)
    par1 = host.twinx()
    par2 = host.twinx()
    par2.spines["right"].set_position(("axes", 1.15))

    make_patch_spines_invisible(par2)
    par2.spines["right"].set_visible(True)
    if x1 is None:
        x1 = np.arange(0, len(y1))
    if x2 is None:
        x2 = np.arange(0, len(y2))
    if x3 is None:
        x3 = np.arange(0,len(y3))
    p1, = host.plot(x1,y1, "o-", color = colors[0], label=labels[0],linewidth=lw)
    p2, = par1.plot(x2,y2, "o-", color = colors[1],label=labels[1],linewidth=lw)
    p3, = par2.plot(x3, y3, "o-", color = colors[2],label=labels[2],linewidth=lw)

   
    if xticklabels is None:
        xticklabels = [None] * len(y1)
    host.set_xlim(-0.5,len(xticklabels)-0.5)
    host.set_xticks(np.linspace(0,len(xticklabels)-1,len(xticklabels)))
    host.set_xticklabels(xticklabels, rotation = 45)
    host.set_ylim(ylims[0])
    par1.set_ylim(ylims[1])
    par2.set_ylim(ylims[2])

    # host.set_xlabel("Models")
    host.set_ylabel(ylabels[0])
    par1.set_ylabel(ylabels[1])
    par2.set_ylabel(ylabels[2])
    host.set_title(title)
    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())
    par2.yaxis.label.set_color(p3.get_color())

    tkw = dict(size=4, width=1.5)
    host.tick_params(axis='y', colors=p1.get_color(), **tkw)
    par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    par2.tick_params(axis='y', colors=p3.get_color(), **tkw)
    host.tick_params(axis='x', **tkw)

    lines = [p1, p2, p3]

    host.legend(lines, [l.get_label() for l in lines], loc =legend_loc)
    if savepath is not None:
        create_folder(savepath)
        if figname is None:
            figname = 'model_performance_3axis.pdf'
        figfile =os.path.join(savepath,figname)
        fig.savefig(figfile,bbox_inches = 'tight')
    plt.show()


def pointplot_x(x,models, xlabel, color ='bo-',linewidth = 2,title = None, 
                figfile = None, ):

    """
    plot 1 objects x with one yaix
    models:  xticks
    xlabel: labels of x
    title:
    """
    fig,ax = plt.subplots(figsize = (10,8))
    plt.plot(x,color, label =xlabel, linewidth=linewidth)
    ax.set_xlim(-0.5,len(models)-0.5)
    ax.set_xticks(np.linspace(0,len(models)-1,len(models)))
    ax.set_xticklabels(models,fontsize=14, rotation = 60)
    ax.set_ylabel(xlabel,fontsize=16)
    ax.set_title(title, fontsize = 18)
    plt.legend()
    if figfile is not None:
        plt.savefig(figfile,bbox_inches = 'tight')
    plt.show()



def pointplot_2axis(y1,y2,model_names,labels,ylabels,colors=None,title=None,ylims=[None,None],loc='best',savepath = None,
                    figname=None,xlabel =None,figsize=None,ticksize = 4,xticks=None,
                    x1 = None, x2 = None):
    """
    plot 2 objects [y1,y2] with two yaixs
    models:  xticks
    labels: labels of [y1, y2]
    title:
    """

    fig, host = plt.subplots(figsize = figsize)
    # fig.subplots_adjust(right=0.9)
    par1 = host.twinx()
    if colors is None:
        colors =['b','k']

    if x1 is None:
        x1 = np.arange(0,len(y1))
    if x2 is None:
        x2 = np.arange(0,len(y2))
    p1, = host.plot(x1,y1, "o-",color=colors[0], label=labels[0])
    p2, = par1.plot(x2,y2, "o-", color=colors[1],label=labels[1])

    host.set_xlim(-0.5,len(model_names)-0.5)
    host.set_xticks(np.linspace(0,len(model_names)-1,len(model_names)))
    if xticks is None:
        xticks = model_names
    host.set_xticklabels(xticks,rotation = 45)
    host.set_ylim(ylims[0])
    par1.set_ylim(ylims[1])
    if xlabel is not None:
        host.set_xlabel(xlabel)
    host.set_ylabel(ylabels[0])
    par1.set_ylabel(ylabels[1])
    host.set_title(title)
    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())

    tkw = dict(labelsize=ticksize, width=1.5)
    host.tick_params(axis='y', colors=p1.get_color(), **tkw)
    par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    host.tick_params(axis='x', **tkw)



    lines = [p1, p2]

    host.legend(lines, [l.get_label() for l in lines],loc=loc)
    host.set_title(title)
    if savepath is not None:
        create_folder(savepath)
        if figname is None:
            figname = 'model_performance_twinx.pdf'
        figfile =os.path.join(savepath,figname)
        plt.savefig(figfile,bbox_inches = 'tight')
    plt.show()




def pointplot_2axis_(y1,y2,x1 = None, x2 = None, 
                    title='',figfile = None,
                    xticklabels =None,
                    ylims=[None]*2,ylabels = [None]*2,colors = [None]*2,labels = [None]*2,
                    figsize = (10,8), lw = 3,
                    savepath = None, figname = None,
                    legend_loc = 'best'):
    """
    plot 3 objects [y1,y2,y3] with two yaixs
    models:  xticks
    labels: labels of [y1, y2,y3]
    title:
    """

    fig, host = plt.subplots(figsize=figsize)
    # fig.subplots_adjust(right=0.9)
    par1 = host.twinx()
 
    if x1 is None:
        x1 = np.arange(0, len(y1))
    if x2 is None:
        x2 = np.arange(0, len(y2))

    p1, = host.plot(x1,y1, "o-", color = colors[0], label=labels[0],linewidth=lw)
    p2, = par1.plot(x2,y2, "o-", color = colors[1],label=labels[1],linewidth=lw)

   
    if xticklabels is None:
        xticklabels = [None] * len(y1)
    host.set_xlim(-0.5,len(xticklabels)-0.5)
    host.set_xticks(np.linspace(0,len(xticklabels)-1,len(xticklabels)))
    host.set_xticklabels(xticklabels, rotation = 45)
    host.set_ylim(ylims[0])
    par1.set_ylim(ylims[1])

    # host.set_xlabel("Models")
    host.set_ylabel(ylabels[0])
    par1.set_ylabel(ylabels[1])
    host.set_title(title)
    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())

    tkw = dict(size=4, width=1.5)
    host.tick_params(axis='y', colors=p1.get_color(), **tkw)
    par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    host.tick_params(axis='x', **tkw)

    lines = [p1, p2]

    host.legend(lines, [l.get_label() for l in lines], loc =legend_loc)
    if savepath is not None:
        create_folder(savepath)
        if figname is None:
            figname = 'model_performance_2axis.pdf'
        figfile =os.path.join(savepath,figname)
        fig.savefig(figfile,bbox_inches = 'tight')
    plt.show()




def pointplot_1axis(y1,x1 = None, 
                    title='',figfile = None,
                    xticklabels =None,
                    ylims=[None],ylabels = [None],colors = [None],labels = [None],
                    figsize = (10,8), lw = 3,
                    savepath = None, figname = None,
                    legend_loc = 'best'):
    """
    plot 3 objects [y1,y2,y3] with two yaixs
    models:  xticks
    labels: labels of [y1, y2,y3]
    title:
    """

    fig, ax = plt.subplots(figsize=figsize)
    # fig.subplots_adjust(right=0.9)
 
    if x1 is None:
        x1 = np.arange(0, len(y1))
   

    p1, = ax.plot(x1,y1, "o-", color = colors[0], label=labels[0],linewidth=lw)

   
    if xticklabels is None:
        xticklabels = [None] * len(y1)
    ax.set_xlim(-0.5,len(xticklabels)-0.5)
    ax.set_xticks(np.linspace(0,len(xticklabels)-1,len(xticklabels)))
    ax.set_xticklabels(xticklabels, rotation = 45)
    ax.set_ylim(ylims[0])

    # host.set_xlabel("Models")
    ax.set_ylabel(ylabels[0])
    ax.set_title(title)
    ax.yaxis.label.set_color(p1.get_color())

    tkw = dict(size=4, width=1.5)
    ax.tick_params(axis='y', colors=p1.get_color(), **tkw)
    ax.tick_params(axis='x', **tkw)


    plt.legend(loc =legend_loc)
    if savepath is not None:
        create_folder(savepath)
        if figname is None:
            figname = 'model_performance_1axis.pdf'
        figfile =os.path.join(savepath,figname)
        fig.savefig(figfile,bbox_inches = 'tight')
    plt.show()



def draw_brace(ax, xspan, yy, text, lw = 2, alpha = 0.5, textpos = 0.02,yscale = 1):
    """Draws an annotated brace on the axes."""
    xmin, xmax = xspan
    xspan = xmax - xmin
    ax_xmin, ax_xmax = ax.get_xlim()
    xax_span = ax_xmax - ax_xmin

    ymin, ymax = ax.get_ylim()
    yspan = ymax - ymin
    resolution = int(xspan/xax_span*100)*2+1 # guaranteed uneven
    beta = 300./xax_span # the higher this is, the smaller the radius

    x = np.linspace(xmin, xmax, resolution)
    x_half = x[:int(resolution/2)+1]
    y_half_brace = (1/(1.+np.exp(-beta*(x_half-x_half[0])))
                    + 1/(1.+np.exp(-beta*(x_half-x_half[-1]))))
    # y = np.concatenate((y_half_brace, y_half_brace[-2::-1]))
    # ax.text((xmax+xmin)/2., yy+.07*yspan, text, ha='center', va='bottom', alpha = alpha)
    y = np.concatenate((y_half_brace[::-1],y_half_brace[1:]))
   
    ax.text((xmax+xmin)/2., yy-textpos*yspan, text, ha='center', va='bottom', alpha = alpha)

    y = yy + (.05*y - .01)*yspan # adjust vertical position
    y = y*yscale
    ax.autoscale(False)
    ax.plot(x, y, color='black', lw=lw, alpha = alpha, clip_on=False)
    