#fun_artificial_data2_class.py
import pickle
import sys
sys.path.insert(1, '../../utils/')
sys.path.insert(1, './functions/')
import os
from importlib import reload
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from importlib import  reload

import pandas as pd
import seaborn as sns
import pickle
from utility import create_folder



class Model():
    def __init__(self,name = 'pca',update_order ='xw', 
                a_alpha = 1e-3, b_alpha = 1e-3, a_gamma = 1e-3, b_gamma = 1e-3):
        self.name = name
        self.update_order = update_order
        self.a_alpha = a_alpha
        self.b_alpha = b_alpha
        self.a_gamma = a_gamma
        self.b_gamma = b_gamma

    def estimate_yq(self,W,X):
        q = self.q
   
        W_q = W[:,0:q]
        X_q = X[0:q,:]
        yq = np.matmul(W_q, X_q)+self.mu.reshape((-1,1))
        
        self.yq = yq
        if q is not None:
            y_ = np.matmul(W, X)+self.mu.reshape((-1,1))
            self.y_ = y_
        return yq

    def sort_explained_variance(self,W,X):
        y = self.y

        D,_= X.shape
        y_var = np.var(y,axis =1).mean()
        evrs  =np.zeros(D)
        for i in range(D):
            wi = W[:,i]
            xi = X[i,:]
            yi = np.matmul(wi.reshape(-1,1),xi.reshape(1,-1))
            evrs[i] = np.var(yi,axis =1).mean()/y_var
        idx = np.argsort(-evrs)
        W_sort = W[:,idx]
        X_sort = X[idx,:]
        self.sort_idx = idx
        return W_sort,X_sort

    def ica_x(self,X):

        from sklearn.decomposition import FastICA
        ica = FastICA(n_components=self.D, random_state=0,whiten='unit-variance')
        S = ica.fit_transform(X.T)
        return S.T

    def compute_correlation(self,x,y):
        from scipy.stats import pearsonr
        q = self.q
        corr = np.zeros(q)

        for i in range(q):
            corr[i]=np.abs(pearsonr(x[i,:],y[i,:])[0])
        return np.mean(corr)

    def find_best_correlation(self,x,y,ica = False):
        from itertools import permutations
        q = self.q
        perm = list(permutations(range(q)))
        corrs = np.zeros((len(perm), len(perm)))
        if ica:
            x_ = self.ica_x(x)
        else:
            x_ = x
        for i in range(len(perm)):
            idx_x = perm[i]
            for j in range(len(perm)):
                idx_y = perm[j]
                corrs[i,j] = self.compute_correlation(x_[idx_x,:],y[idx_y,:])
        return np.max(corrs)


    def compare_correlation(self,x,y,ica = False):

        q = self.q
        x = x[0:q,:]
        y = y[0:q,:]
        corr = self.find_best_correlation(x,y,ica = ica)

        return corr

    def find_effective_nef(self,W,tol_ratio = 0.05):
        from scipy import stats

        D = W.shape[1]
        ttest = np.zeros(D)
        # W_norm = stats.zscore(W,axis = 1)
        W_norm=W/W.max()
        for i in range(D):
            # result = stats.ttest_1samp(a=np.abs(W_norm[:,i]), popmean=np.max(W_norm)*tol_ratio,alternative ='less')
            
            result = stats.ttest_1samp(a=np.abs(W_norm[:,i]), popmean=tol_ratio,alternative ='less')
            ttest[i]=(result.pvalue >1e-2)

        return np.sum(ttest)

    def pca(self):
        from sklearn.decomposition import PCA
        n_samples = self.y.shape[1]
        pca = PCA(n_components=self.D) #y.T = usvt =xt@wt
        X= pca.fit_transform(self.y.T).T #  U 
        W = pca.components_.T  * np.sqrt(pca.explained_variance_) # vt.T*s
        # v,s,ut = pca._fit(self.y.T)
        # W = pca.components_.T  * np.sqrt(s[:self.D]) # vt.T*s
        # eigen vector: components_

        yq=self.estimate_yq(W,X)
        self.log_like = pca.score(self.y.T)
        self.Q = []
        self.sigma2 = pca.noise_variance_
        return W, X,yq


    def pca_(self):
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=self.D)
        # pca.fit(self.y.T)
        v,s,ut = pca._fit(self.y.T)
        # X= pca.fit_transform(self.y.T).T #  (U*S).T, S: sqrt(eigen values)
        # _lambda =s**2/(self.y.shape[1]-1)
        _lambda = pca.explained_variance_
        W = ut.T
        # W = pca.components_.T* np.sqrt(_lambda) # Vt.T, eigen vectors*
        # X = u.T
        X = np.matmul(s*np.identity(len(s)),v.T)
        # eigen vector: components_

        yq=self.estimate_yq(W,X)
        self.log_like = pca.score(self.y.T)
        self.Q = []

        return W, X,yq


    def mlpca(self):
        """
        y: M*N
        D:
        -----------------
        return:
        W: M*D
        X: D*N

        ref: Tipping and Bishop, 1999. "Probabilistic principal component analysis."
        """
        y = self.y
        q = self.q
        D = self.D
    
        from sklearn.decomposition import PCA
        pca = PCA(n_components=D)
        pca.fit(y.T)  # pca contains y-mu 
        U = pca.components_.T # pca.componts 
        # Uq = U[:,:q]
        # _lambda_q = _lambda[:q]

        _lambda = pca.explained_variance_ # explained_variance, labmda, vector
        sigma2 = pca.noise_variance_ # contains mean :
                                     # explained_variance_[n_components:].mean()
        # W = U*np.sqrt(_lambda-sigma2)

        Lambda = _lambda*np.identity(D)
        W = np.matmul(U,np.sqrt(Lambda-sigma2*np.identity(D)))
        M = np.matmul(W.T,W) + sigma2
        mu = np.mean(y,axis = 1)
        x_post_ = np.matmul(np.linalg.inv(M), W.T)

        X = np.matmul(x_post_,y-mu.reshape(-1,1))
        yq = self.estimate_yq(W,X)

        self.log_like = pca.score(y.T)
        cov = pca.get_covariance()
        self.Q=[]
        self.sigma2 = sigma2
        return W,X,yq
    



    def mlpca_(self):
        """
        y: M*N
        D:
        -----------------
        return:
        W: M*D
        X: D*N

        ref: Tipping and Bishop, 1999. "Probabilistic principal component analysis."
        """
        y = self.y
        y = y-y.mean(axis=1).reshape((-1,1))
        q = self.q
        M,N = y.shape
        from sklearn.decomposition import PCA
        pca = PCA(n_components=self.D)
        pca.fit(y.T)
        from numpy.linalg import svd
        v,s,ut =svd(y.T)

        _lambda = s**2/(N-1)
        sigma2 = _lambda[q:].mean()

        mu = np.mean(y,axis = 1)

        W = ut[:q,:].T*np.sqrt(_lambda[:q]-sigma2)
        M = np.matmul(W.T,W) + sigma2

        x_mean_ = np.matmul(np.linalg.inv(M), W.T)
        X = np.matmul(x_mean_,y-mu.reshape(-1,1))
        yq = self.estimate_yq(W,X)

        self.log_like = pca.score(y.T)
        self.Q=[]
        self.sigma2 = sigma2

        return W,X,yq

    def fa(self):
        y = self.y
        
        
        from sklearn.decomposition import FactorAnalysis
        fa = FactorAnalysis(n_components=self.D, random_state=self.seed)
        X= fa.fit_transform(y.T).T

        W = fa.components_.T
        W_sort, X_sort =  self.sort_explained_variance(W,X)
        yq = self.estimate_yq(W_sort,X_sort)

        self.log_like = fa.score(y.T)
        self.sigma2 = fa.noise_variance_
        self.Q=[]
        return W_sort,X_sort,yq

    def bayespy_common_steps(self,Q, type ='bpca'):
        from bayespy.inference.vmp.transformations import RotateGaussianARD,RotationOptimizer
        X = Q['X']
        W = Q['W']
        D = self.D
        if type =='bpca':
            alpha =Q['alpha']
            rot_X = RotateGaussianARD(X)
            rot_W = RotateGaussianARD(W,alpha)
        elif type =='mbpca':
            alpha =Q['alpha']
            gamma = Q['gamma']
            rot_X = RotateGaussianARD(X,gamma)
            rot_W = RotateGaussianARD(W,alpha)
        elif type =='xbpca':
            gamma = Q['gamma']
            rot_X = RotateGaussianARD(X,gamma)
            rot_W = RotateGaussianARD(W)


        R = RotationOptimizer(rot_X, rot_W, D)
        Q.set_callback(R.rotate)

        Q.update(repeat =1000,verbose = self.verbose)
        self.lb = Q.compute_lowerbound()
        self.lb_y= Q['Y'].lower_bound_contribution()
        self.Q = Q
        self.sigma2 = np.squeeze(1/Q['tau'].get_moments()[0])
        W_ = np.squeeze(Q['W'].get_moments()[0])
        X_ = np.squeeze(Q['X'].get_moments()[0]).T
        W_sort, X_sort =  self.sort_explained_variance(W_, X_)
        yq = self.estimate_yq(W_sort,X_sort)
        return W_sort, X_sort,yq

    def bpca_common(self,a = 1e-3,b = 1e-3):
        y = self.y
        D = self.D
        M,N = y.shape
        from bayespy.nodes import GaussianARD, Gamma, SumMultiply
        from bayespy.inference import  VB

        X = GaussianARD(0,1,shape=(D,), plates=(1,N),name ='X')
        alpha = Gamma(a,b,plates=(D,),name='alpha')
        W = GaussianARD(0,alpha,shape =(D,),plates =(M,1),name ='W')
        tau = Gamma(1e-5,1e-5,name='tau')
        W.initialize_from_random()
        X.initialize_from_random()

        F = SumMultiply('d,d->',W,X)
        Y = GaussianARD(F,tau,name ='Y')
        mu_ = y.mean(axis = 1)
        y_ = y-mu_.reshape(-1,1)
        Y.observe(y_)
        
        if self.update_order =='xw':
            Q = VB(Y,X,W,alpha,tau)
        elif self.update_order =='wx':
            Q = VB(Y,W,X,alpha,tau)

        return self.bayespy_common_steps(Q,type ='bpca')





    def mbpca_common(self,a_gamma = 1e-3,b_gamma = 1e-3,a_alpha = 1e-3, b_alpha = 1e-3):
        y = self.y
        
        D = self.D
        M,N = y.shape
        from bayespy.nodes import GaussianARD, Gamma, SumMultiply
        from bayespy.inference import  VB

        gamma = Gamma(a_gamma,b_gamma, plates =(D,),name ='gamma')
        X = GaussianARD(0,gamma,shape=(D,), plates=(1,N),name ='X')
        alpha = Gamma(a_alpha,b_alpha,plates=(D,),name='alpha')
        W = GaussianARD(0,alpha,shape =(D,),plates =(M,1),name ='W')
        tau = Gamma(1e-5,1e-5,name='tau')
        W.initialize_from_random()
        X.initialize_from_random()

        F = SumMultiply('d,d->',W,X)
        Y = GaussianARD(F,tau,name='Y')
        mu_ = y.mean(axis = 1)
        y_ = y-mu_.reshape(-1,1)
        Y.observe(y_)
       
        if self.update_order =='xw':
            Q = VB(Y,X,W, gamma,alpha,tau)
        elif self.update_order =='wx':
             Q = VB(Y,W,X,alpha,gamma,tau)

        return self.bayespy_common_steps(Q,type='mbpca')



    def bpca_individual(self,a = 1e-3,b = 1e-3):
        """
        y: M*N
        D:
        -----------------
        return:
        W_sort: M*D
        X_sort: D*N

        ref:
        """
        y = self.y
        D = self.D
        M,N = y.shape
        from bayespy.nodes import GaussianARD, Gamma, SumMultiply
        from bayespy.inference import  VB
        X = GaussianARD(0,1,shape=(D,), plates=(1,N),name ='X')
        alpha = Gamma(a,b,plates=(D,),name='alpha')
        W = GaussianARD(0,alpha,shape =(D,),plates =(M,1),name ='W')
        tau = Gamma(1e-3,1e-3,plates =(M,1),name='tau')
        W.initialize_from_random()
        X.initialize_from_random()
        F = SumMultiply('d,d->',W,X)
        Y = GaussianARD(F,tau,name ='Y')
        mu_ = y.mean(axis = 1)
        y_ = y-mu_.reshape(-1,1)
        Y.observe(y_)
        
        if self.update_order =='xw':
            Q = VB(Y,X,W,alpha,tau)
            # Q = VB(Y,X,W,tau,alpha)
        elif self.update_order =='wx':
            Q = VB(Y,W,X,alpha,tau)
        
        return self.bayespy_common_steps(Q,type ='bpca')


    def mbpca_individual(self,a_gamma = 1e-3,b_gamma = 1e-3,a_alpha = 1e-3, b_alpha = 1e-3):
        y = self.y
        D = self.D
        from bayespy.nodes import GaussianARD, Gamma, SumMultiply
        from bayespy.inference import  VB
        M,N = y.shape
        gamma = Gamma(a_gamma,b_gamma, plates =(D,),name ='gamma')
        X = GaussianARD(0,gamma,shape=(D,), plates=(1,N),name ='X')
        alpha = Gamma(a_alpha,b_alpha,plates=(D,),name='alpha')
        W = GaussianARD(0,alpha,shape =(D,),plates =(M,1),name ='W')
        tau = Gamma(1e-3,1e-3,plates =(M,1),name='tau')

        F = SumMultiply('d,d->',W,X)
        Y = GaussianARD(F,tau,name='Y')

        mu_ = y.mean(axis = 1)
        y_ = y-mu_.reshape(-1,1)
        Y.observe(y_)
        # Q = VB(Y,X,W,tau,gamma,alpha)
        if self.update_order =='xw':
            Q = VB(Y,X,W,gamma,alpha,tau)
        elif self.update_order =='wx':
            Q = VB(Y,W,X,alpha,gamma,tau)
        W.initialize_from_random()
        X.initialize_from_random()
        return self.bayespy_common_steps(Q,type ='mbpca')


    def xbpca_common(self,a = 1e-3,b = 1e-3):
        y = self.y
        self.mu = y.mean(axis = 1)
        D = self.D
        M,N = y.shape
        from bayespy.nodes import GaussianARD, Gamma, SumMultiply
        from bayespy.inference import  VB

        gamma = Gamma(1e-3,1e-3, plates =(D,),name ='gamma')
        X = GaussianARD(0,gamma,shape=(D,), plates=(1,N),name ='X')
        W = GaussianARD(0,1,shape =(D,),plates =(M,1),name ='W')
        tau = Gamma(1e-3,1e-3,name='tau')
        W.initialize_from_random()
        X.initialize_from_random()

        F = SumMultiply('d,d->',W,X)
        Y = GaussianARD(F,tau,name='Y')
        mu_ = y.mean(axis = 1)
        y_ = y-mu_.reshape(-1,1)
        Y.observe(y_)
        
        if self.update_order =='xw':
            Q = VB(Y,X,W,gamma,tau)
        elif self.update_order =='wx':
            Q = VB(Y,W,X,gamma,tau)

        return self.bayespy_common_steps(Q,type='xbpca')

    def xbpca_individual(self,a = 1e-3,b = 1e-3):
        y = self.y
        self.mu = y.mean(axis = 1)
        D = self.D
        M,N = y.shape
        from bayespy.nodes import GaussianARD, Gamma, SumMultiply
        from bayespy.inference import  VB

        gamma = Gamma(1e-3,1e-3, plates =(D,),name ='gamma')
        X = GaussianARD(0,gamma,shape=(D,), plates=(1,N),name ='X')
        W = GaussianARD(0,1,shape =(D,),plates =(M,1),name ='W')
        tau = Gamma(1e-3,1e-3,plates =(M,1),name='tau')
        W.initialize_from_random()
        X.initialize_from_random()

        F = SumMultiply('d,d->',W,X)
        Y = GaussianARD(F,tau,name='Y')
        mu_ = y.mean(axis = 1)
        y_ = y-mu_.reshape(-1,1)
        Y.observe(y_)
        
        if self.update_order =='xw':
            Q = VB(Y,X,W,gamma, tau)
        elif self.update_order =='wx':
            Q = VB(Y,W,X,gamma,tau)
        return self.bayespy_common_steps(Q,type='xbpca')



   
    def get_mvds(self):
        """
        mvds: model variance difference score
        defined as:
        mvds = np.abs(1- [var(yq)+sigma2]/var(y))
        the small the better
        """
        y = self.y
        yq = self.yq
        sigma2 = self.sigma2

        numerator = np.var(yq, axis = 1) + sigma2
        denominator = np.var(y,axis = 1)
        mvds =1- np.mean(numerator/denominator)
        return np.abs(mvds)
    def get_score(self,z = None,  score_names = None, q = None,ica= False, nef_tol = 0.1):
        # y= self.y
        # y = self.c
        # y = self.y-self.mu.reshape((-1,1))
        if z is None:
            z = self.y
        if q is None:
            yq = self.yq
        else:
            W = self.W # sorted
            X = self.X
            yq = np.matmul(W[:,:q], X[:q,:])
        X_sort = self.X
        scores = dict()
        from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error,mean_absolute_error
        if 'r2' in score_names:
            r2 = r2_score(z.T, yq.T)
            scores['r2'] = r2
        if 'evs' in score_names:
            evs = explained_variance_score(z.T, yq.T)
            scores['evs'] = evs
        if 'evs_y' in score_names:
            evs = explained_variance_score(self.y.T, yq.T)
            scores['evs_y'] = evs
        if 'mse' in score_names:
            mse = mean_squared_error(z.T, yq.T)
            scores['mse'] = mse
        if 'corr' in score_names:
            corr = self.compare_correlation(X_sort,z,ica=ica)
            scores['corr'] = corr
        if 'lb' in score_names:
            if self.name =='pca' or self.name == 'mlpca' or self.name =='fa':
                scores['lb'] = 1e-5
            else:
                scores['lb'] =self.lb
        if 'log_like' in score_names:
            if self.name =='pca' or self.name == 'mlpca' or self.name =='fa':
                scores['log_like'] = self.log_like
            else:
                scores['log_like'] =self.lb_y/z.shape[1]
        if 'mvds' in score_names:
            mvds = self.get_mvds()
            scores['mvds']= mvds

        if 'mae' in score_names:
            mae = mean_absolute_error(z, yq)
            scores['mae']= mae

        if 'nef' in score_names:
            nef = self.find_effective_nef(self.X.T,tol_ratio = nef_tol)
            scores['nef'] = nef

       

        
        self.scores = scores
        return scores


    def fit(self,y, D = None, q = 3, seed=0,saveQ = False,verbose = True):
        """
        Inputs: 
        y: shape (N, T), N: dimentionality of observations(neurons); 
                         T: number of samples(frames)
        D: dimentionality of latent variables
        q: number of effective dimensions used for model comparison, 
            if q is None, q = D

        Returns: 
        W: shape: (N, D), sorted according to variance (W_sort)
        X: shape: (D, T), sorted according to variance (X_sort)
        yq: shape: (q, T), based on W_sort, X_sort
        """
        if q is None: 
            q = D
        self.q = q
        self.seed = seed
        a_alpha = self.a_alpha
        b_alpha = self.b_alpha
        a_gamma = self.a_gamma 
        b_gamma = self.b_gamma
        if D is None:
            D, N = y.shape
            self.D = D-1
        else: 
            self.D = D
        np.random.seed(seed)
        self.verbose = verbose
        self.y = y
        self.mu= y.mean(axis = 1)
        name = self.name
        if name =='pca':
            W,X,yq = self.pca()
        elif name =='mlpca':
            W,X,yq = self.mlpca()
        elif name =='fa':
            W,X,yq = self.fa()
        elif name =='bpca_common':
            W,X,yq = self.bpca_common(a=a_alpha, b = b_alpha)
        elif name =='bpca_individual':
            W,X,yq = self.bpca_individual(a = a_alpha, b = b_alpha)
        elif name =='mbpca_common':
            W,X,yq = self.mbpca_common(a_alpha = a_alpha, b_alpha = b_alpha, a_gamma=a_gamma, b_gamma = b_gamma)
        elif name =='mbpca_individual':
            W,X,yq = self.mbpca_individual(a_alpha = a_alpha, b_alpha = b_alpha, a_gamma=a_gamma, b_gamma = b_gamma)
        elif name =='xbpca_common':
            W,X,yq = self.xbpca_common(a= a_gamma, b = b_gamma)
        elif name =='xbpca_individual':
            W,X,yq = self.xbpca_individual(a = a_gamma, b = b_gamma)
        self.W = W
        self.X = X
        self.yq = yq
        
        return W,X,yq
    def find_best_seed(self, y,z = None,D =None, q = None, score_name = 'evs',seed_range = range(100),verbose = False):
        score_seeds = np.zeros(len(seed_range))
        if D is None: 
            D = y.shape[0]-1

        for i in range(len(seed_range)):
            print('seed:', i, ',Verbose:', verbose)
            seed = seed_range[i]
            model = Model(name =self.name)
            model.fit(y,D = D, q = q, seed = seed,verbose = verbose)
            scores = model.get_score(z, score_names=[score_name])
            score_seeds[i] = scores[score_name]


        seed_best = seed_range[np.argmax(score_seeds)]
        self.best_seed = seed_best
        return seed_best


    def imshow_wxyq(self,savepath = None,type ='yq',wmax = None, 
                    xmax = None, ymax = None, ymin = None,cmap ='coolwarm',
                    y_sort_idx = None):

        W = self.W
        X = self.X
        if type =='yq':
            yq = self.yq
            q = self.q
        elif type =='y_':
            yq = self.y_
            q = self.D-1

        if wmax is None:
            wmax = np.max(np.abs(W))
        if xmax is None:
            xmax = np.max(np.abs(X))
        if ymax is None:
            ymax = yq.max()
        if ymin is None:
            ymin = yq.min()
        if y_sort_idx is not None:
            yq = yq[y_sort_idx,:]
        fig,ax = plt.subplots(figsize=(15,5), ncols=3, nrows=1)
        ax0 = ax[0].imshow(W,aspect ='auto',cmap = cmap,vmax = wmax, vmin = -wmax);
        ax1 = ax[1].imshow(X,aspect ='auto',cmap = cmap,vmax = xmax, vmin = -xmax);
        ax2 = ax[2].imshow(yq,aspect ='auto',vmax = ymax, vmin = ymin);

        fig.colorbar(ax0,ax = ax[0])
        fig.colorbar(ax1,ax = ax[1])
        fig.colorbar(ax2,ax = ax[2])

        ax[0].set_title('W')
        ax[1].set_title('X')
        ax[2].set_title('y_reconstructed(q='+str(q)+')')
        fig.suptitle(self.name)
        fig.set_tight_layout(True)
        if savepath is not None:
            if type =='yq':
                filename = 'imshow_wxyq_'+self.name +'.pdf'
            elif type =='y_':
                filename = 'imshow_wxy_'+self.name +'.pdf'
            figfile = os.path.join(savepath,filename)
            fig.savefig(figfile)
        plt.show()

    def save(self,savepath, filename = None):
        del self.y
        if savepath is not None:
            import pickle
            if filename is None:
                filename = 'model_'+self.name+'.pickle'
            file = os.path.join(savepath,filename)
            with open(file,'wb') as f:
                pickle.dump(self, f)
class Models():
    def __init__(self, model_names=None, savepath = None,update_order = 'xw'):
       
        self.savepath = savepath
        if savepath is not None:
            create_folder(savepath)
        if model_names is None:
            model_names =['pca','mlpca','bpca_common','mbpca_common','bpca_individual','mbpca_individual']
        self.model_names = model_names
        self.update_order = update_order


    def fit(self,y,z=None,D = None,q = 3, seed = None, score_names = None,imshow_wxyq=True,cmap = 'seismic', 
            verbose = False,  nef_tol = 0.1, find_best_seed = False,score_name='lb',seed_range=range(100)):
        self.q = q
        # self.c = c

        model_names = self.model_names
        # scores_models = dict()
        if find_best_seed:
            seed_models = self.find_best_seed(y = y, z = z, 
                                            score_name = score_name, 
                                            seed_range = seed_range)
        else:
            if seed is None :
                seed_models = dict()
                for model_name in model_names:
                    seed_models[model_name] = 0
            else:
                seed_models = dict()
                for model_name in model_names:
                    seed_models[model_name] = seed
  
        results_models = dict()

        for model_name in model_names:
            model = Model(name =model_name,update_order = self.update_order)
            W,x,yq = model.fit(y, D = D, q=q,  seed = seed_models[model_name],verbose=verbose)
            if imshow_wxyq: 
                model.imshow_wxyq(savepath = self.savepath,cmap =cmap)
            if score_names is None:
                score_names =['evs','lb','r2']
            self.score_names = score_names
            scores = model.get_score(z = z,score_names=score_names, nef_tol = nef_tol)
            model.save(self.savepath)

            results_models[model_name]=dict()

            results_models[model_name]['W'] = W
            results_models[model_name]['X'] = x
            results_models[model_name]['yq'] = yq
            results_models[model_name]['sigma2'] = model.sigma2
            results_models[model_name]['scores'] = scores
            if find_best_seed:
                results_models[model_name]['best_seed']=seed_models[model_name]
        self.results_models = results_models

        return results_models

    def find_best_seed(self,y,z,score_name='evs',seed_range = range(10)):
        model_names = self.model_names
        best_seed_models = dict()
        for model_name in model_names:
            print(model_name, score_name)
            if model_name =='pca' or model_name =='mlpca' or model_name =='fa':
                best_seed_models[model_name]= 0
            else:
                model = Model(name =model_name)
                best_seed = model.find_best_seed(y,z, score_name=score_name,seed_range=seed_range)
                best_seed_models[model_name]= best_seed
        return best_seed_models

    def save(self, savepath = None, filename =None):

        import pickle
        results_models =self.results_models
        # savepath = self.savepath
        if savepath is not None:
            create_folder(savepath)
            if filename is None:
                filename ='result_models.pickle'
            savefile = os.path.join(savepath,filename)
            with open(savefile,'wb') as f:
                pickle.dump(results_models,f)



    def plot_score_comparsion(self,score_name='evs',model_names = None,savepath = None,ymin = 0, ymax = 1):
        savepath = self.savepath
        results_models = self.results_models
        if model_names is None:
            model_names = self.model_names

        # if score_name =='lb':
        #     if 'pca' in model_names:
        #         model_names.remove('pca')
        #     if 'mlpca' in model_names:
        #         model_names.remove('mlpca')
        #     if 'fa' in model_names:
        #         model_names.remove('fa')
        score_models = dict()
        for model_name in model_names:
            score_models[model_name] = results_models[model_name]['scores'][score_name]
        models,score =zip(*score_models.items())
        if ymax is None:
            ymax = np.max(score)
        if ymin is None:
            ymin = np.min(score)
        fig,ax = plt.subplots()


        ax.plot(score, 'o-',linewidth=3, color ='k')
        ax.set_ylim([ymin*0.9,ymax*1.1])
        ax.set_xticks(np.arange(len(models)))
        ax.set_xticklabels(models,rotation = 45,fontsize=14)
        # ax.set_xlabel('Methods')
        ax.set_title(score_name)

        savepath = self.savepath
        if savepath is not None:
            filename = score_name+'.pdf'
            figfile = os.path.join(savepath, filename)
            plt.savefig(figfile,bbox_inches='tight')
        plt.show()




def imshow_wxyq(W,X,yq,q = 3, figsize = (15,5),model_name ='pca',title = None,savepath = None,
                type ='yq',wmax = None, xmax = None, ymax = None, ymin = None,cmap ='coolwarm',
                y_sort_idx = None,xtick_s = False, framerate = None,xtickgap=200):
    if q is None:
        q = W.shape[1]
  

    if wmax is None:
        wmax = np.max(np.abs(W))
    if xmax is None:
        xmax = np.max(np.abs(X))
    if ymax is None:
        ymax = yq.max()
    if ymin is None:
        ymin = yq.min()
    if y_sort_idx is not None:
        yq = yq[y_sort_idx,:]
        W = W[y_sort_idx,:]
        print('sort yq')
    fig,ax = plt.subplots(figsize=figsize, ncols=3, nrows=1)

    ax0 = ax[0].imshow(W,aspect ='auto',cmap = cmap,vmax = wmax, vmin = -wmax);
    ax1 = ax[1].imshow(X,aspect ='auto',cmap = cmap,vmax = xmax, vmin = -xmax);
    ax2 = ax[2].imshow(yq,aspect ='auto',vmax = ymax, vmin = ymin);

    fig.colorbar(ax0,ax = ax[0])
    fig.colorbar(ax1,ax = ax[1])
    fig.colorbar(ax2,ax = ax[2])

    ax[0].set_title(r'$W$')
    ax[1].set_title(r'$X$')
    # ax[2].set_title('y_reconstructed(q='+str(q)+')')
    # ax[2].set_title(r'$y$_reconstructed')
    ax[2].set_title(r'$\widehat{y}$')
    ax[0].set_xlabel(r'Dimension of $X$')
    ax[0].set_ylabel(r'Dimension of $y$')
    ax[1].set_ylabel(r'Dimension of $X$')
    ax[2].set_ylabel(r'Dimension of $y$')
    
    if xtick_s: 
        if framerate is None:
            framerate = 3.86
           
        xticks = np.arange(0,yq.shape[1]+1,framerate*xtickgap)
        xticklabels = np.floor((xticks)/framerate).astype(int)
        for i in [1,2]:
            ax[i].set_xticks(xticks)
            ax[i].set_xticklabels(xticklabels)
            ax[i].set_xlabel('Time (s)')
    else:
        ax[1].set_xlabel('Samples')
        ax[2].set_xlabel('Samples')


    D = W.shape[1]
    if D >10:
        xtick = np.arange(0,D,10)
    else:
        xtick = np.arange(0,D,2)
    ax[0].set_xticks(xtick)

    if title is None:
        title = model_name
    fig.suptitle(title)
    # fig.set_tight_layout(True)
    fig.tight_layout(rect=[0, 0, 1, 1.09])

    if savepath is not None:
        if type =='yq':
            filename = 'imshow_wxyq_'+model_name +'.pdf'
        elif type =='y_':
            filename = 'imshow_wxy_'+model_name +'.pdf'
        figfile = os.path.join(savepath,filename)
        fig.savefig(figfile)
    plt.show()



def estimate_yq(y,W,X,q):
    W_q = W[:,0:q]
    X_q = X[0:q,:]
    mu = y.mean(axis = 1)
    yq = np.matmul(W_q, X_q)+mu.reshape((-1,1))
    return yq




def get_score(y,yq,score_names, z = None,ica = False):

    scores = dict()
    from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error
    if 'r2' in score_names:
        r2 = r2_score(y.T, yq.T)
        scores['r2'] = r2
    if 'evs' in score_names:
        evs = explained_variance_score(y.T, yq.T)
        scores['evs'] = evs
    if 'mse' in score_names:
        mse = mean_squared_error(y.T, yq.T)
        scores['mse'] = mse

    return scores



def plot_score_comparsion(score, model_names, score_name='evs',savepath = None,
                        ymin = None, ymax = None,
                        linewidth = 3, color ='k', ylabel='score', title =None):

    if ymax is None:
        ymax = np.max(score)*1.1
    if ymin is None:
        ymin = np.min(score)*0.9
    fig,ax = plt.subplots()
    ax.plot(score, 'o-',linewidth=linewidth, color =color)
    ax.set_ylim([ymin,ymax])
    ax.set_xticks(np.arange(len(model_names)))
    ax.set_xticklabels(model_names,rotation = 45,fontsize=12)
    # ax.set_xlabel('Methods')
    if title is None:
        title = score_name
    ax.set_title(title)
    ax.set_ylabel(ylabel)

    if savepath is not None:
        filename = score_name+'.pdf'
        figfile = os.path.join(savepath, filename)
        plt.savefig(figfile,bbox_inches='tight')
    plt.show()



def find_effective_nef( W,tol_ratio = 0.01):
    from scipy import stats

    D = W.shape[1]
    ttest = np.zeros(D)
    # W_norm = stats.zscore(W,axis = 1)
    W_norm=W/W.max()
    for i in range(D):
        # result = stats.ttest_1samp(a=np.abs(W_norm[:,i]), popmean=np.max(W_norm)*tol_ratio,alternative ='less')
        
        result = stats.ttest_1samp(a=np.abs(W_norm[:,i]), popmean=tol_ratio,alternative ='less')
        ttest[i]=(result.pvalue >1e-2)

    return np.sum(ttest)



def get_model_name_labels(model_names):
    xticks = [None]*len(model_names)
    for i in range(len(model_names)):
        if model_names[i] == 'pca':
            xticks[i] = 'PCA'
        elif model_names[i] =='mlpca':
            xticks[i] = 'Probabilistic PCA'
        elif model_names[i] =='fa':
            xticks[i] = 'Factor analysis'
        elif model_names[i] =='bpca_common':
            xticks[i] = 'Bayesian PCA'
        elif model_names[i] =='bpca_individual':
            # xticks[i] = 'Bayesian PCA (individual)'
            xticks[i] = 'variant Bayesian PCA'
        elif model_names[i] =='mbpca_common':
            # xticks[i] = 'dual ARD (common noise)'
            xticks[i] = 'variant dual ARD'
        elif model_names[i] =='mbpca_individual':
            # xticks[i] = 'dual ARD (individual)'
            xticks[i] = 'dual ARD'
        elif model_names[i] =='population':
            xticks[i] ='Full population neurons'
    return xticks

def plot_model_comparison_twinx(scores_models, loc = 'best',savepath = None,
                                ylims=[None,None],figsize = None,
                                ticksize = 4, xticks = None,
                                score_names = ['evs', 'mse'],
                                labels =['EVS','MSE'],
                                ylabels =['Explained variance score (EVS)','Mean squared error (MSE)']
                                ):
    import fun_plots;reload(fun_plots)
    from fun_plots import pointplot_2axis
    import pandas as pd
    model_names=scores_models.keys()
    df_scores_models = pd.DataFrame(scores_models).T
    
    colors =['darkblue','peru']
    title ='Model Performance'
    
    pointplot_2axis(df_scores_models[score_names[0]],df_scores_models[score_names[1]],
                    model_names=model_names,labels = labels,ylabels=ylabels,
                    colors =colors,title =title,loc = loc,savepath=savepath,ylims = ylims, 
                    figsize = figsize, ticksize = ticksize, 
                    xticks = xticks)




def plot_model_comparsion(score_models, score_name='evs',savepath = None,
                        ymin = None, ymax = None,
                         color ='k', ylabel='score', title =None,
                        figsize = (5,3), xticks = None):

    model_names = list(score_models.keys())
    score = np.zeros(len(model_names))
    for i in range(len(model_names)):
        model_name = model_names[i]
        score[i] = score_models[model_name]

    fig,ax = plt.subplots(figsize = figsize)
    ax.plot(score, 'o-', color =color)
    ax.set_ylim([ymin,ymax])
    ax.set_xticks(np.arange(len(model_names)))
    if xticks is None:
        xticks = model_names
    ax.set_xticklabels(xticks,rotation = 45)
    if title is None:
        title = score_name
    ax.set_title(title)
    ax.set_ylabel(ylabel)

    if savepath is not None:
        filename = score_name+'.pdf'
        figfile = os.path.join(savepath, filename)
        plt.savefig(figfile,bbox_inches='tight')
    plt.show()

# def imshow_model_wxyq(models_results, model_name = 'pca', cmap ='RdBu_r', 
#                       wmax = 1, xmax = 3, ymax = 8, ymin = 0, 
#                       figsize = (15,5), savepath = None, title='PCA', xtick_s = False, 
#                       framerate = None, xtickgap = 100):
#     model = models_results[model_name]

#     W = model['W']
#     x = model['X']
#     yq = model['yq']

#     imshow_wxyq(W,x, yq,  cmap =cmap, wmax = wmax, xmax = xmax, ymax = ymax, ymin = ymin,
#                 model_name =model_name,title = title, figsize = figsize, savepath = savepath,
#                 xtick_s= xtick_s, xtickgap= xtickgap, framerate=framerate)
    




def imshow_model_wxyq(models_results, model_name = 'pca', cmap ='RdBu_r', 
                      wmax = 1, xmax = 3, ymax = 8, ymin = 0, 
                      figsize = (15,5), savepath = None, title='PCA', 
                      y_sort_idx = None,xtick_s=False, framerate = None,xtickgap = 10):
    model = models_results[model_name]


    W = model['W']
    x = model['X']
    yq = model['yq']
    
    if y_sort_idx is not None:
        W = W[y_sort_idx,:]

    imshow_wxyq(W,x, yq,  cmap =cmap, wmax = wmax, xmax = xmax, ymax = ymax, ymin = ymin,
                model_name =model_name,title = title, figsize = figsize, savepath = savepath,
                y_sort_idx = y_sort_idx,xtick_s = xtick_s,framerate = framerate,xtickgap = xtickgap)
    