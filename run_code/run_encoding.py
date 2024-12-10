
import sys
sys.path.insert(1, 'functions/')
import pickle
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from importlib import  reload

from scipy.io import loadmat
import os
import fun_models_class; reload(fun_models_class)




def get_score(y_true, y_pred): 
    u =  ((y_true - y_pred)** 2).sum(axis =1 )
    v = ((y_true - y_true.mean()) ** 2).sum(axis = 1)
    score = 1-u/v
    return score

def get_partial_phi(phi, M = 8, del_cat ='spk0'):
    """
    partial_phi: generate partial predictors (partialphi) for getting relative contribution to the encoding (regression) results
    predictors disclude "del_cat"
    Inputs: 
    phi: (n_predictors, n_samples)
    M: number of kernels for sound cues, defaut is 8
    del_cat: predictor that not included for the partial encoding

    Return: 
    phi_del: (n_predictors_del, n_samples), predictors disclude "del_cat"
    """
    if del_cat[:3] =='spk':
        i = int(del_cat[3])
        phi_del = np.delete(phi, np.s_[i*M:(i+1)*M], axis = 0)
    elif del_cat =='speed':
        phi_del = np.delete(phi, np.s_[-2:], axis = 0)
    elif del_cat =='none':
        phi_del = phi

    return phi_del




def run_partial_encoding(model, X_effective,  M = 8, tw = 8, history = 2,n_spks = 12):
    """
    run encoding with partial predictors (not include certain predictor)

    Input: 
    model: save encoding model used for generate encoding results, also include encoding model 
    X_effective: the latent factors 
    M: number of kernals of predictors based on sound cue 
    tw: length of time window 
    history: length of history 
    n_spks: number of speakers (sources of sound cues)

    Return: 
    partical_scores： [n_stims, nef], fitting scores (r2) of partial encoding 
    partical_scores_xall: for the same purpose, just used matrix representation for encoding
    """
     
    nef = X_effective.shape[0]
    phi = model.construct_phi(spk_M = M, spk_tw = tw, spk_history = history) # phi: (n_predictors, n_samples)
    phi_clips_scale = model.clip_recording_scale(phi) # phi_clips_scale: (n_predictors, n_samples)
    # n_spks = spk_on.shape[0]
    n_stims = n_spks + 1 + 1
    partical_scores = np.zeros(( n_stims,nef))
    partical_scores_xall = np.zeros( (n_stims, 1))

    for i in range(n_stims):
        if i <n_spks:
            del_cat = 'spk'+str(i)
        elif i ==n_spks:
            del_cat ='speed'
        elif i ==n_spks + 1:
            del_cat ='none'
        
        phi_clips_scale_del = get_partial_phi(phi_clips_scale, M = M, del_cat = del_cat)
        for j in range(nef):
            x_clips_scale = model.clip_recording_scale(X_effective[j,:], method ='zscore')

            y_fit, score, w = model.fit(phi_clips_scale_del.T, x_clips_scale.T, 
                                    method ='RidgeCV', cv = 5, 
                                    alphas =[0.1])
        
            partical_scores[i,j] = score

        x_clips_scale = model.clip_recording_scale(X_effective, method ='zscore')

        y_fit, partical_scores_xall[i,:], w = model.fit(phi_clips_scale_del.T, x_clips_scale.T, 
                                    method ='RidgeCV', cv = 5, 
                                    alphas =[0.1])
    return partical_scores,partical_scores_xall


def get_relative_contribution_scores(partical_scores):
    """
    Get relative contribution scores based on partial scores
    
    """
    n_stims, nef = partical_scores.shape
    scores_relative = np.zeros((nef, n_stims-1))
    for i in range(nef):
        scorei = partical_scores[:,i]
        score_relative =1- scorei[:-1]/scorei[-1]
        score_relative = score_relative/score_relative.sum()
        scores_relative[i,:] = score_relative
    return scores_relative


def get_relative_weight(x,  beta):
    """
    x.shape: D*T, is phi 
    
    (based on:  https://www.listendata.com/2015/05/relative-importance-weight-analysis.html)


    (based on
      Tonidandel, S., LeBreton, J. M., & Johnson, J. W. (2009). 
    Determining the statistical significance of relative weights.
      Psychological methods, 14(4), 387.)
    """
    import scipy
    # Combine target and predictors
    D,T = x.shape
    corX = np.zeros((D,D))
    for i in range(D):
        for j in range(D):
            # print(scipy.stats.pearsonr(x[i,:],x[j,:]))
            corX[i,j],_ = scipy.stats.pearsonr(x[i,:],x[j,:])

    
    
    w,v= scipy.linalg.eig(corX)
    D = np.diag(w)
    delta = np.sqrt(D)
    l =  np.matmul(np.matmul(v,  delta), np.transpose(v))
    lambdasq = l**2
    # beta = np.matmul(np.linalg.inv(l),corY)
    
    rsquare = sum(np.power(beta,2))  
    rawWgt  = np.matmul(lambdasq, beta**2)
    importance = (rawWgt / rsquare) * 100
    

    return rawWgt, importance

from utility import lowpass_filter




if __name__ == "__main__":
    
    model_name = 'mbpca_individual'
    data_type ='data1'
    depth =int(sys.argv[1])
    savepath = os.path.join('./results_/results_update_order_xw/',data_type,model_name)
    filename =  model_name + '_'+data_type+'_depth'+str(depth)+'.pickle'
    savefile = os.path.join(savepath,filename)
    with open(savefile,'rb' ) as f:
        model = pickle.load(f)
    nef =int(model.scores['nef'])
    print(nef)
    X = model.X

    # read stimulus for encoding model 
    data = loadmat('./data/piezo_data1.mat')
    spk_on = data['all_spk_scan']
    spk_off = data['all_spk_scan_down']
    speed = data['all_speed_scan']
    move = data['all_move_scan']
    speed_filter = lowpass_filter(speed, cutoff=0.1, fs=3.86, order=2)
    move_filter = lowpass_filter(move, cutoff=0.1, fs=3.86, order=2)


    from fun_encoding import Encoding_model



    spk_type ='stick_basis'
    method ='RidgeCV'
    speed_type ='speed_2d'
    M =6
    tw = 6
    history = 1

    model = Encoding_model(spk=spk_on, move=move, speed=speed_filter, 
                                        spk_type =spk_type, 
                                        move_type = 'none', 
                                        speed_type = speed_type)
        
    phi = model.construct_phi(spk_M = M, spk_tw = tw, spk_history = history)
    phi_clips_scale = model.clip_recording_scale(phi)
    x_clips_scale = model.clip_recording_scale(X[0:nef,:], method ='zscore')

    y_fit, score, ws = model.fit(phi_clips_scale.T, x_clips_scale.T, 
                    method =method, cv = 5, 
                    alphas =[0.01])



    scores_ = get_score(x_clips_scale,y_fit.T)
    # relative weights
    relative_weights = np.zeros((nef, ws.shape[1]))
    importances = np.zeros((nef, ws.shape[1]))
    for i in range(nef):
        relative_weight, importance= get_relative_weight(phi_clips_scale,  ws[i,:])
        relative_weights[i,:] = relative_weight
        importances[i,:] = importance

    # relative contribution 
    n_spks = spk_on.shape[0]
    n_stims = n_spks + 1

    model = Encoding_model(spk_on, move, speed_filter, 
                                spk_type =spk_type, 
                                move_type = 'none', 
                                speed_type = speed_type)

    partial_socres,rc_all = run_partial_encoding(model,X[0:nef,:], M = M, tw = tw, history = history,
                                        n_spks = n_spks)
    relative_contribution_scores = get_relative_contribution_scores(partial_socres)
    rc_all = get_relative_contribution_scores(rc_all)


    results_encoding_analysis = dict()
    results_encoding_analysis['ws'] = ws
    results_encoding_analysis['y_fit'] = y_fit
    results_encoding_analysis['phi_clips_scale'] = phi_clips_scale
    results_encoding_analysis['x_clips_scale'] = x_clips_scale
    results_encoding_analysis['r2scores'] = scores_
    results_encoding_analysis['relative_weights'] = relative_weights
    results_encoding_analysis['importances'] = importances
    results_encoding_analysis['relative_contribution_scores'] = relative_contribution_scores
    results_encoding_analysis['rc_all'] = rc_all

    savefile = './results/results_encoding_analysis_depth'+str(depth)+'.pickle'
    with open(savefile,'wb') as f: 
        pickle.dump(results_encoding_analysis,f)