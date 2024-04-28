import pickle
import sys
sys.path.insert(1, '../../utils/')
sys.path.insert(1, './functions/')
import os
from importlib import reload
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from functions.read_data_old import read_X_depth, read_X
from importlib import  reload

import pandas as pd
import seaborn as sns
import pickle
from utility import create_folder
from scipy import stats

class Encoding_model():
    def __init__(self, spk, move, speed, spk_type, move_type, speed_type):
        """
        Inputs: 
        spk: (n_spk, T)
        move: (2,T)
        speed: (2,T)
        spk_type: basis functions type, 'stick_basis', 'gaussian_baiss', 'spline_basis'
        move_type: 'none'
        speed_type: 'speed_x', 'speed_y', 'speed_eu','speed_2d', 'none'
        
        """
        self.spk = spk
        self.move = move
        self.speed = speed
        self.spk_type = spk_type
        self.move_type = move_type 
        self.speed_type = speed_type

    def stick_basis_spk(self,spk, M=8, tw = 8, history= 0):
        """
        stick_basis: each onset last for M sticks, 
                    all onsets of the same sound cue share the same basis set
        Inputs:
        spk: n_spk* T
        M: number of basis
        tw: length of time window
        history: length of history in the time window

        Return: 
        spk_phi: D_spk*T, D_spk: number of predictors for spk, D_spk= n_spk*M
        """
        n_spk, T = spk.shape
        spk_phi = np.zeros((n_spk, M, T))
        for i in range(n_spk):
            onsets = np.where(spk[i,:]==1)[0]
            n_onsets = len(onsets)
            for j in range(M):
                for k in range(n_onsets):
                    onset = onsets[k]
                    # spk_phi[i,j,onset-history:onset+tw-history] = 1  
                    spk_phi[i,j,onset+j-history] = 1     
        return spk_phi.reshape(n_spk* M, T)


    def stick_basis_spk_ergodic(self,spk,M,tw, history = 0):
        """
        stick_basis: each onset last for M sticks, 
                    each onsets of the same sound cue has its own basis set

        D_spk_phi: n_spk*M*n_onsets
        Inputs: 
        spk: n_spk*T

        Return: 
        spk_phi: D_spk_phi*T, =n_spk*n_onsets*M
        """
        n_spk, T = spk.shape
        n_onsets = len(np.where(spk[0,:]==1)[0])
        spk_phi = np.zeros((n_spk, n_onsets,M, T))

        for i in range(n_spk):
            onsets = np.where(spk[i,:]==1)[0]
            n_onsets = len(onsets)
            for j in range(n_onsets):
                onset = onsets[j]
                for k in range(M): 
                    spk_phi[i,j,k,onset+tw-history] = 1
        return spk_phi.reshape(n_spk*n_onsets*M, -1)

        
    def gaussian_bases(self, M = 8, tw = 8,sigma= 0.5):
        """
        Create a gaussian basis set, with M basis, evenly distributed within tw long
        
        Return: 
        bases: (M, tw)
        """
        x= np.linspace(-1,1,M)
        centers = np.linspace(-1,1,tw)
        bases = np.zeros((M,tw))
        for i in range(tw):
            mean = centers[i]
            basis = np.exp(-0.5 * ((x - mean) / sigma) ** 2)
            bases[:,i] = basis
        return bases


    def gaussian_basis_spk(self, spk, M=8, tw =8, history = 0, sigma =0.5):
        """
        Create gaussian bsis of spk 
        Inputs: 
        spk: (n_spk,T)

        Return: 
        spk_phi: (n_spk*M, T)
        each spki_phi is a gaussian basis set for  n_onsets of a sound cue
        """
        n_spk, T = spk.shape
        n_onsets = len(np.where(spk[0,:]==1)[0])
        spk_phi = np.zeros((n_spk, M, T))
        bases = self.gaussian_bases(M = M, tw= tw, sigma = sigma)

        for i in range(n_spk):
            spki = spk[i,:]
            onsets = np.where(spki ==1)[0]
            n_onsets = len(onsets)
          
            for j in range(n_onsets):
                onset = onsets[j]
                spk_phi[i,:,onset-history: onset-history +tw] = bases
        return spk_phi.reshape(n_spk*M,-1)


    def gaussian_basis_spk_ergodic(self, spk, M=8, tw =8, history = 0, sigma =0.5):
        """
        Create gaussian basis of spk with ergodic repeating predictors
        Inputs: 
        spk: (n_spk, T)
        Return: 
        spk_phi: (n_spk*n_onest*M, T)

        each spki_phi is a gaussian basis for a oneset of a sound cue
        """
        n_spk, T = spk.shape
        n_onsets = len(np.where(spk[0,:]==1)[0])
        spk_phi = np.zeros((n_spk, n_onsets,M, T))
        bases = self.gaussian_bases(M, tw, sigma)

        for i in range(n_spk):
            spki = spk[i,:]
            onsets = np.where(spki ==1)[0]
            n_onsets = len(onsets)
            for j in range(n_onsets):
                onset = onsets[j,:]
                spk_phi[i,j,:,onset-history: onset-history +tw] = bases[i,:]
        return spk_phi.reshape(n_spk*n_onsets*M,-1)
    

    def spline_bases(self,M, tw, degree = 5):

        """
        Create a spline basis set using patsy
        Inputs: 
        M: number of basis
        degress: degree of spline basis, ususlly can be M-3

        Return: 
        bases: (M,tw)

        """
        import patsy
        x = np.linspace(0., 1., tw)

        bases = patsy.bs(x, df=M, degree=degree,include_intercept=True)
        return bases.T



    def spline_basis_spk(self, spk, M = 8, tw = 8, history = 2, degree = 5):
        """
        Create spline basis of spk 
        Input: 

        Outputs: 
        spk_phi: (n_spk*M, T)
        """
        n_spk, T = spk.shape
        n_onsets = len(np.where(spk[0,:]==1)[0])
        spk_phi = np.zeros((n_spk, M, T))
        bases = self.spline_bases(M = M, tw= tw, degree = degree)

        for i in range(n_spk):
            spki = spk[i,:]
            onsets = np.where(spki ==1)[0]
            n_onsets = len(onsets)
          
            for j in range(n_onsets):
                onset = onsets[j]
                spk_phi[i,:,onset-history: onset-history +tw] = bases
        return spk_phi.reshape(n_spk*M,-1)



    def construct_spk_phi(self, spk, spk_type= 'stick_basis', M= 8, tw = 8, history = 0, 
                            sigma = 0.5, degree = 0.5):
        """
        Construct spk_phi for predictor
        Input: 
        spk: (n_spk, T)
        spk_type: type of spk basis, 'none' means no spk, 

        Output: 
        spk_phi: (D_spk_phi, T)
        """
       
        T = spk.shape[1]
        if spk_type =='stick_basis':
            spk_phi = self.stick_basis_spk(spk, M, tw, history)
        elif spk_type =='gaussian_basis':
            spk_phi = self.gaussian_basis_spk(spk, M=M, tw=tw, history= history, sigma= sigma)
        elif spk_type =='stick_basis_ergodic':
            spk_phi = self.stick_basis_spk_ergodic(spk, M, tw, history)
        elif spk_type =='gaussian_basis_ergodic':
            spk_phi = self.gaussian_basis_spk_ergodic(spk, M, tw, history, sigma = sigma)

        elif spk_type =='spline_basis':
            spk_phi = self.spline_basis_spk(spk, M, tw, history, degree)
        elif spk_type =='none':
            spk_phi = np.array([]).reshape(-1,T)
        return spk_phi
        
    def construct_move_phi(self,move, move_type ='none'):
        """
        Construct predictor move_phi for move
        Input: 
        move: 2*T, 
        move_type: 'none' means no move
        """
        T = move.shape[1]
        if move_type =='none':
            move_phi = np.array([]).reshape(-1,T)
        return move_phi
    
    def construct_speed_phi(self, speed, speed_type ='eu'):
        """
        Construct predictor speed_phi for speed
        Inputs: 
        speed: 2*T 
        speed_type: 'none' means no speed, 
                    'speed_x': only use speed[0,:]
                    'speed_y': only use speed[1,:]
                    'speed_2d': use 2-d speed
                    'speed_eu': use 1-d speed_eu
        """
        T = speed.shape[1]
        speed_eu = np.linalg.norm(speed,axis = 0)
        # speed_zscore = stats.zscore(speed, axis = 1)
        # speed_eu_zscore = stats.zscore(speed_eu)
        speed_zscore = speed
        speed_eu_zscore = speed_eu

        if speed_type =='speed_x':
            speed_phi = speed_zscore[0,:]
        elif speed_type =='speed_y':
            speed_phi = speed_zscore[1,:]
        elif speed_type =='speed_eu':
            speed_phi = speed_eu_zscore
        elif speed_type =='speed_2d':
            speed_phi = speed_zscore
        elif speed_type =='none':
            speed_phi =  np.array([]).reshape(-1,T)
       
        return speed_phi

    def construct_phi(self,spk_M = 8, spk_tw = 8, spk_history = 0, 
                      spk_gaussian_basis_sigma = 0.5, 
                      spk_spline_basis_degree = 5):
        """
        Inputs: 
        spk_M: number of basis of sound cue
        spk_tw: time window length 
        spk_history: history of spk
        
        returns: 
        phi: (D,T), T: number of samples (frames), D: number of predictors
        """
        self.spk_tw = spk_tw
        self.spk_history = spk_history
        self.spk_M = spk_M
     
        spk_phi = self.construct_spk_phi(spk = self.spk, spk_type = self.spk_type,
                                         M = spk_M, tw = spk_tw, history = spk_history, 
                                         sigma = spk_gaussian_basis_sigma,
                                         degree = spk_spline_basis_degree)                            
        move_phi = self.construct_move_phi(move = self.move, move_type  = self.move_type)
        speed_phi = self.construct_speed_phi(speed = self.speed, speed_type= self.speed_type)
        phi = np.vstack((spk_phi,move_phi,speed_phi))
        return phi

    def clip_recording(self, x):
        """
        Clip recordings of latent varialbes(X) or neural recordings; and phi
        note: it also can be used for clip phi
        Inputs: 
        x: (D, T), D: dimensionality, T: number of samples(frames)

        Outputs: 
        x_clips: (D, T_clips): T_clips = tw * n_onsets_all
        """
        # D, T = x.shape
        if len(x.shape)==1:
            x=x.reshape(1,-1)
        D, T = x.shape
        spk = self.spk
        tw = self.spk_tw
        history = self.spk_history
        spks = spk.sum(axis = 0)
        onsets_all = np.where(spks ==1)[0]
        n_onsets_all = len(onsets_all)

        x_clips = np.array([]).reshape((x.shape[0],-1))

        x_clips = np.zeros((D, n_onsets_all*tw))
        
        
        for i in range(n_onsets_all):
            onset = onsets_all[i]
            x_clips[:,i*tw:(i+1)*tw] = x[:,onset-history:onset+tw-history]
            # x_tw = x[:,onset-history:onset+tw-history]# : (D, tw)
            # x_clips = np.hstack((x_clips, x_tw))
        return np.squeeze(x_clips)
    


    def MinMax_abs(self,x):
        """
        Scale using minmax_abs scaler

        Input: 
        x: (D,T), ( n_feature,n_samples)

        return: 
        x_scale: (D,T)
        """
        x_scale = np.zeros(x.shape)
        for i in range(x.shape[0]):
            xi = x[i,:]
            x_scale[i,:] = (xi-np.min(np.abs(xi)))/(np.max(xi)-np.min(np.abs(xi)))
        return x_scale

    def scale(self,x, d_start =None,method=None):
        '''
        Inputs:
         x: (D,T), ( n_feature,n_samples)
         d_start: for avoiding non, 

        return:
         x_scale: ( n_feature,n_samples)
        '''
        if x.ndim <2:
            x = x[np.newaxis, :]
        from scipy import stats
        if method is None:
            return x
        
        if d_start is not None:
            x_scale = np.zeros(x.shape)
            x_range = x[d_start:, :]
            x_scale[0:d_start,:] = x[0:d_start,:]
            
            if method =='zscore':
                x_scale[d_start:,:] =   stats.zscore(x_range,axis = 1)
            elif method =='minmax_abs':
                x_scale[d_start:,:]= self.MinMax_abs(x_range)
        else: 
            if method =='zscore':
                x_scale  =   stats.zscore(x,axis = 1)
            elif method =='minmax_abs':
                x_scale = self.MinMax_abs(x )
        return x_scale
            

    def clip_recording_scale(self,x, d_start = None, method ='zscore'):
        """
        Inputs: 
        x: (D, T), D: dimensionality, T: number of samples(frames)

        Outputs: 
        x_clips: (D, T_clips): T_clips = tw * n_onsets_all
        x_clips_scale: ( n_feature,n_samples), (D, T_clips)
        
        """
        x_clips = self.clip_recording(x)
        x_clips_scale = self.scale(x_clips, d_start = d_start, method = method)
        return x_clips_scale

    def fit(self,phi,y, method='LinearRegression', cv = 10, alphas = [0.1,0.2,0.5], 
            max_iter = 1000):
        
        """
        Inputs:
        phi: (n_samples, n_features)
        y: (n_samples)
        """
        

        if len(y.shape)==1:
            y=y.reshape(-1,1)
      
        if len(phi.shape) ==1: 
            phi = phi.reshape(-1,1)

        from sklearn.linear_model import RidgeCV,MultiTaskElasticNetCV,MultiTaskLassoCV,LinearRegression
        from sklearn.model_selection import cross_val_score
        if method =='ElasticNetCV':
            lrm = MultiTaskElasticNetCV(cv = cv,alphas = alphas,max_iter=max_iter)
        elif method == 'RidgeCV':
            lrm = RidgeCV(cv = cv, alphas = alphas)
        elif method =='LassoCV':
            lrm = MultiTaskLassoCV(cv = cv,alphas = alphas,max_iter=max_iter )
        elif method =='LinearRegressionCV':
            lrm = LinearRegression()
            score = cross_val_score(lrm, phi, y, cv=cv)
        lrm.fit(phi,y)
        yfit= lrm.predict(phi)
        score= lrm.score(phi,y)
        w = lrm.coef_
        
        self.lrm = lrm
        return yfit, score, w
    



    


