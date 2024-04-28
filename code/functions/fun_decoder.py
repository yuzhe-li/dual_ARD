import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys
from scipy import stats
from importlib import reload
sys.path.insert(1, './functions/')





from scipy.stats import zscore
def clip_recording(x, spki, tw =8, history = 2):
    """
    Input:
    x: (T,) or (n, T)

    """
    onsets = np.where(spki==1)[0]
    N = len(onsets)
    # clips = np.zeros((N,tw))
    if len(x.shape) ==1:
        x = x.reshape((1,-1))

    clips = np.array([]).reshape((x.shape[0],-1))
    for i in range(N):

        on = onsets[i]
        x_tw = x[:,on-history:on+tw-history]
        clips = np.hstack((clips, x_tw))

    return np.squeeze(clips)

class DecodingModel_tw():
    def __init__(self,spk, M = 8, tw = 8, history = 1,  type ='united',basis = 'gaussian'):
        self.M = M
        self.tw = tw
        self.history = history
        self.spk = spk
        self.type = type
        self.basis = basis

    def construct_x_tw(self, x, tw=None, history = None):
        """"
        Input:
        x: (T,)

        -----
        output:
        x_clips_all: (n_samples*n_speakers,tw)
        each sample has tw legth

        """
        spk = self.spk
        n_speakers = spk.shape[0]
        n_samples = spk[0,:].sum()

        if tw is None:
            tw = self.tw
        if history is None:
            history = self.history
        x_clips_all = np.zeros((int(n_samples*n_speakers), tw))
        for i in range(spk.shape[0]):

            spki = spk[i,:]
            idx_spki_onsets = np.where(spki ==1)[0]
            for j in range(n_samples):
                spkij = np.zeros(spk.shape[1])
                spkij[idx_spki_onsets[j]] = 1
                x_clips_ij = clip_recording(x,spkij, tw = tw, history = history)

                x_clips_all[int(i*n_samples + j),:] = x_clips_ij


        return x_clips_all
    def construct_X_tw(self,X):
        """
        X:(n,T)
        """
        spk = self.spk
        n_speakers = spk.shape[0]
        n_samples = spk[0,:].sum()
        X_clips_all = np.array([]).reshape((int(n_samples*n_speakers),-1))
        for i in range(X.shape[0]):
            x = X[i,:]
            x_clips = self.construct_x_tw(x)
            X_clips_all = np.hstack((X_clips_all,x_clips))
        return X_clips_all

    def assign_spk_cat(self, cat = 'LR'):
        """
        spk_L = [6,12,5,11]
        spk_R = [8,2,9,3]
        spk_FB = [7,1,4,10]

        spk_front = [1,7,6,8,12,2]
        spk_back = [4,10,11,3,5,9]
        spk_left = [7,6,12,5,11,4]
        spk_right =[1,8,2,9,3,10]
        """
        spk_L = [6,12,5,11]
        spk_R = [8,2,9,3]
        spk_FB = [7,1,4,10]

        spk_LF = [7,6,12]
        spk_RF = [1,8,2]
        spk_LB = [5,11,4]
        spk_RB = [10,3,9]

        spk_front = [1,7,6,8,12,2]
        spk_back = [4,10,11,3,5,9]
        spk_left = [7,6,12,5,11,4]
        spk_right =[1,8,2,9,3,10]
        spk_F = [7,1]
        spk_B = [4,10]

        spk_0 = [1,7]
        spk_1 = [6,12]
        spk_2 = [5,11]
        spk_3 = [4,10]
        spk_4 = [3,9]
        spk_5 = [2,8]

        spk_LL = [5,12]
        spk_RR = [2,9]

        spk_Ff = [7,1,6,8]
        spk_Bb = [4,10, 3, 11]

        spk = self.spk
        num_onsets = spk.sum()
        num_spks = 12
        num_samples = 10
        # tw = self.tw
        y = np.zeros(num_spks*num_samples)
        y_new = np.zeros(num_spks*num_samples)
        for i in range(num_spks):
            y[i*num_samples:(i+1)*num_samples] = i

        if cat == 'LR':
            for i in range(len(y)):
                if y[i]+1 in spk_left:
                    y_new[i] = 0
                else:
                    y_new[i] = 1

        elif cat =='FB':
            for i in range(len(y)):
                if y[i]+1 in spk_front:
                    y_new[i] = 0
                else:
                    y_new[i] = 1
        elif cat =='LRFb':
            for i in range(len(y)):
                if y[i]+1  in spk_L:
                    y_new[i] = 0
                elif y[i]+1  in spk_R:
                    y_new[i] = 1
                else:
                    y_new[i] =2

        elif cat == 'LRFB':
            for i in range(len(y)):
                if y[i]+1  in spk_L:
                    y_new[i] = 0
                elif y[i]+1  in spk_R:
                    y_new[i] = 1
                elif y[i]+1  in spk_F:
                    y_new[i] =2
                else:
                    y_new[i] =3
        elif cat =='quarter':
            for i in range(len(y)):
                if y[i]+1  in spk_LF:
                    y_new[i] = 0
                elif y[i]+1  in spk_LB:
                    y_new[i] = 1
                elif y[i]+1  in spk_RB:
                    y_new[i] =2
                else:
                    y_new[i] =3

        elif cat =='sixth':
            for i in range(len(y)):
                if y[i]+1  in spk_1:
                    y_new[i] = 0
                elif y[i]+1  in spk_1:
                    y_new[i] = 1
                elif y[i]+1  in spk_2:
                    y_new[i] =2
                elif y[i]+1  in spk_3:
                    y_new[i] =3
                elif y[i]+1  in spk_4:
                    y_new[i] =4
                else:
                    y_new[i] =5
        elif cat =='all':
            y_new = y

        elif cat =='LLRR':
            for i in range(len(y)):
                if y[i]+1  in spk_LL:
                    y_new[i] = 0
                elif y[i]+1  in spk_RR:
                    y_new[i]=1
                else:
                    y_new[i] =None

        elif cat =='FFBB':
            for i in range(len(y)):
                if y[i]+1  in spk_F:
                    y_new[i] = 0
                elif y[i]+1  in spk_B:
                    y_new[i]=1
                else:
                    y_new[i] =None
        elif cat =='LlRr':
            for i in range(len(y)):
                if y[i]+1  in spk_L:
                    y_new[i] = 0
                elif y[i]+1  in spk_R:
                    y_new[i]=1
                else:
                    y_new[i] =None
        elif cat =='FfBb':
            for i in range(len(y)):
                if y[i]+1  in spk_Ff:
                    y_new[i] = 0
                elif y[i]+1  in spk_Bb:
                    y_new[i]=1
                else:
                    y_new[i] =None
        return y_new






    def fit(self,x,cv = 10,cat ='LR',method ='ridge'):

        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.model_selection import cross_val_score
        if len(x.shape) ==1:
            x_clips = self.construct_x_tw(x)
        else:
            x_clips = self.construct_X_tw(x)
        self.x_clips = x_clips
        y = self.assign_spk_cat(cat =cat)
        # gpc = GaussianProcessClassifier()
        from sklearn.preprocessing import scale

        x_fit = scale(x_clips)

        if np.isnan(y).sum()>0:
            idx = np.where(np.isnan(y)==False)[0]
            y = y[idx]
            x_fit = x_fit[idx,:]
        from sklearn.linear_model import RidgeClassifierCV,LogisticRegressionCV
        from sklearn.linear_model import RidgeClassifier, LogisticRegression
        from sklearn import svm
        if method =='svm':
            clf = svm.LinearSVC()
            # scores = cross_val_score(clf, X, y, cv=5)
        elif method =='logistic':
            # clf =  LogisticRegressionCV(cv = cv,max_iter=1e4)
            clf =  LogisticRegression(max_iter=1e4)
        elif method =='ridge':
            # clf =  RidgeClassifierCV(cv = cv)
            clf =  RidgeClassifier()
        elif method =='bayes':
            #Import Gaussian Naive Bayes model
            from sklearn.naive_bayes import GaussianNB
            clf = GaussianNB()

        elif method =='ComplementNB':
            from sklearn.naive_bayes import ComplementNB
            clf = ComplementNB()
        elif method =='nn':
            from sklearn.neural_network import MLPClassifier
            clf = MLPClassifier()
        elif method =='qda':
            from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
            clf = QuadraticDiscriminantAnalysis()
        elif method =='gpc':
            from sklearn.gaussian_process import GaussianProcessClassifier
            from sklearn.gaussian_process.kernels import RBF
            clf =GaussianProcessClassifier(kernel=1.0 * RBF(1.0))




        from sklearn.model_selection import cross_val_predict,cross_val_score
        y_pred = cross_val_predict(clf, x_fit, y, cv=cv)
        score_cv = cross_val_score(clf, x_fit, y, cv=cv)
        score = np.mean(score_cv)
        # clf.fit(x_fit,y)

        # scores = cross_val_score(gpc, x_clips, y, cv=cv)
        # from sklearn.calibration import CalibratedClassifierCV
        # calibrated_clf = CalibratedClassifierCV(base_estimator=clf, cv=cv)
        # calibrated_clf.fit(x_clips,y)
        self.clf = clf
        self.score_cv = score_cv
        return clf, x_fit, y, y_pred, score


def shuffle_decoder(X_d, cat, spk, tw = 6, history = 1, N_shuffle = 1000):

    model = DecodingModel_tw(spk, history = history, tw = tw)
    # X_d_ = zscore(X_d, axis = 1)
    # x_fit = model.construct_X_tw(X_d_)
    x_fit = model.construct_X_tw(X_d)
    x_fit = zscore(x_fit, axis = 0)

    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    y_cat = model.assign_spk_cat(cat = cat)
    if np.isnan(y_cat).sum()>0:
        idx = np.where(np.isnan(y_cat)==False)[0]
        y_cat = y_cat[idx]
        x_fit = x_fit[idx,:]
    from sklearn.preprocessing import scale
    # x_fit = scale(x_fit)
    # x_fit = zscore(x_fit,axis = 1)
    clf.fit(x_fit, y_cat)
    score_fit =  clf.score(x_fit, y_cat)
    predict_proba = clf.predict_proba(x_fit)
    y_fit = y_cat
    score_shuffle = np.zeros(N_shuffle)
    for n in range(N_shuffle):
        clf = GaussianNB()
        np.random.shuffle(y_fit)
        clf.fit(x_fit, y_fit)
        score = clf.score(x_fit, y_fit)
        score_shuffle[n] = score


    # one tail (right-tail test, better than by chance)
    p = np.sum(score_shuffle>=score_fit)/N_shuffle
    result = dict()
    result['score_fit'] = score_fit
    result['predict_proba'] = predict_proba
    result['x_fit'] = x_fit
    result['y_cat'] = y_cat
    result['score_shuffle'] = score_shuffle
    result['p'] = p
    return result
def extract_score_models_cats(score_models_cats, model_names=None):
    if model_names is None:
        model_names = list(score_models_cats.keys())
    cats = list(score_models_cats[model_names[0]].keys())

    score_cats_models = dict()
    for cat in cats:
        score_cats_models[cat]=dict()
        for model_name in model_names:
            score_cats_models[cat][model_name] = score_models_cats[model_name][cat]


    return score_cats_models

# def shuffle_decoder_models(result_models, spk= None, speed=None, trace=None, tw = 6, history= 1, d= 'same',N_shuffle=1000, model_names = None,
#                            cats =['LR','FB']):

#     if model_names is None:
#         model_names = ['pca','mlpca','fa','bpca_common','bpca_individual','mbpca_common','mbpca_individual','population']

#     p_models_cats=dict()
#     score_models_cats= dict()
#     score_shuffle_models_cats = dict()
#     y_cat_models_cats = dict()
#     predict_prob_models_cats = dict()
#     nef_models = dict()
#     for model_name in model_names:
#         if model_name =='population':
#             x_var = np.var(trace,axis = 1)
#             idx = np.argsort(x_var)
#             X_sort = trace[idx,:]
#             if isinstance(d, int):
#                 X_d = X_sort[0:d, :]
#             elif d =='different':
#                 X_d = X_sort
#             elif d =='same':
#                 nef = result_models['mbpca_individual']['scores']['nef']
#                 X_d = X_sort[0:int(nef),:]
#         else:
#             if isinstance(d, int):
#                 nef_ = result_models[model_name]['scores']['nef']
#                 if d <= nef_:
#                     nef = d
#                 else:
#                     nef = nef_
#             elif d =='same':
#                 nef = result_models['mbpca_individual']['scores']['nef']
#             elif d =='different':
#                 nef = result_models[model_name]['scores']['nef']

#             X= result_models[model_name]['X']
#             X_d = X[0:int(nef),:]
#         nef_models[model_name] = X_d.shape[0]
#         # cats =['LR', 'FB']
#         # cats =['LLRR', 'FFBB']
#         p_models_cats[model_name] = dict()
#         score_models_cats[model_name] = dict()
#         score_shuffle_models_cats[model_name] = dict()
#         y_cat_models_cats[model_name] = dict()
#         predict_prob_models_cats[model_name] =dict()
#         for cat in cats:
#             shuffle_result =shuffle_decoder(X_d, cat=cat, spk=spk, tw = tw, history = history, N_shuffle = N_shuffle)
#             p_models_cats[model_name][cat] = shuffle_result['p']
#             score_models_cats[model_name][cat] = shuffle_result['score_fit']
#             score_shuffle_models_cats[model_name][cat] = shuffle_result['score_shuffle']
#             y_cat_models_cats[model_name][cat] = shuffle_result['y_cat']
#             predict_prob_models_cats[model_name][cat] = shuffle_result['predict_proba']
#     result = dict()
#     result['score_models_cats'] = score_models_cats
#     result['p_models_cats'] = p_models_cats
#     result['score_shuffle_models_cats'] = score_shuffle_models_cats
#     result['y_cat_models_cats'] = y_cat_models_cats
#     result['nef_models'] = nef_models
#     result['predict_prob_models_cats'] = predict_prob_models_cats

#     return result




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
            xticks[i] = 'Bayesian PCA (individual)'
        elif model_names[i] =='mbpca_common':
            xticks[i] = 'dual ARD (common)'
        elif model_names[i] =='mbpca_individual':
            xticks[i] = 'dual ARD (individual)'
        elif model_names[i] =='population':
            xticks[i] ='Full population neurons'
    return xticks




def shuffle_decoder_gmm(X_d, cat, spk, tw = 6, history = 1, N_shuffle = 1000,n_components = 3):
    """
    X_d: shape: (D, n_samples)
    x_clip: (n_samples, n_features)
    x_fit: (n_samples, n_features)

    """
    model = DecodingModel_tw(spk, history = 1, tw =4)
    x_clip = model.construct_X_tw(X_d) # clip: 
    x_fit = stats.zscore(x_clip, axis = 0)
    y_cat = model.assign_spk_cat(cat) 

    decoder = GaussianMixtureNB(n_components= n_components)
    
    predict_proba = decoder.fit(x_fit, y_cat.astype(float))
    p = decoder.shuffle_test(x_fit, y_cat.astype(float), N = N_shuffle)
    score_fit = decoder.score_fit   
    score_shuffle = decoder.score_shuffle
    result = dict()
    result['score_fit'] = score_fit
    result['predict_proba'] = predict_proba
    result['x_fit'] = x_fit
    result['y_cat'] = y_cat
    result['score_shuffle'] = score_shuffle
    result['p'] = p
    return result

from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture



class GaussianMixtureNB(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components=1, reg_covar=1e-06):
        self.n_components = n_components
        self.reg_covar = reg_covar
    def fit(self, X, y):
        """
        X: shape (n_samples, n_features)
        y: (n_samples, ) label of categories ... 
        
        """
        
        n_samples, n_features = X.shape
        values, counts_y = np.unique(y, return_counts=True)
        n_cats = len(values)

        # get prior based on counts
        y_prior=counts_y/n_samples # frequency of appearence of each type y
        # log_prior: (n_cats,)
        log_prior = np.log(y_prior.astype(float))

        # get log_pdf of likelihood based on gmm fitting
        gmm = GaussianMixture(n_components= self.n_components, reg_covar = self.reg_covar)
        log_likelihood_samples = np.zeros((n_cats,  n_features,n_samples))
        for cat_k in range(n_cats):
            y_catk  = values[cat_k]
            for feature_i in range(n_features):
                # gmm fit based on featurei with samples for catk
                x_featurei = X[:, feature_i].reshape(-1,1)# X: shape (n_samples, n_features)
                x_featurei_catk = x_featurei[y==y_catk].reshape(-1,1) 
                gmm.fit(x_featurei_catk)
                log_likelihood_samples[cat_k,feature_i,:] = gmm.score_samples(x_featurei)
                
        log_likelihood = log_likelihood_samples.sum(axis = 1)
        log_joint = log_prior.reshape(-1,1) + log_likelihood # (n_cats, n_samples)        
        log_posterior = log_joint - logsumexp(log_joint, axis=0, keepdims=True)
        prob_posterior = np.exp(log_posterior)

        y_predict = prob_posterior[1,:]>0.5
        score = (y==y_predict).sum()/n_samples # accuracy

        self.log_likelihood_samples = log_likelihood_samples
        self.log_prior = log_prior
        self.log_likelihood = log_likelihood
        self.log_posterior = log_posterior
        self.y_predict = y_predict
        self.score = score
        return prob_posterior
       
    
    def shuffle_test(self, X,y, N = 100):
        """
        Use shuffle test to evaluate the performance

        """
        prob = self.fit(X,y)
        score_fit = self.score.copy()
        self.score_fit = score_fit
        y_shuffle = y.copy()
        score_shuffle = np.zeros(N)
        for n in range(N): 
            np.random.shuffle(y_shuffle)
            prob = self.fit(X,y_shuffle)
            score_shuffle[n] = self.score
        p = np.sum(score_shuffle>=score_fit)/N
        self.score_shuffle = score_shuffle
        return p 
    


def shuffle_decoder_models(result_models, spk= None, speed=None, trace=None, tw = 6, history= 1, d= 'same',N_shuffle=1000, model_names = None,
                           cats =['LR','FB'],GMM_n_components = None ):

    if model_names is None:
        model_names = ['pca','mlpca','fa','bpca_common','bpca_individual','mbpca_common','mbpca_individual','population']

    p_models_cats=dict()
    score_models_cats= dict()
    score_shuffle_models_cats = dict()
    y_cat_models_cats = dict()
    predict_prob_models_cats = dict()
    nef_models = dict()
    for model_name in model_names:
        if model_name =='population':
            x_var = np.var(trace,axis = 1)
            idx = np.argsort(x_var)
            X_sort = trace[idx,:]
            if isinstance(d, int):
                X_d = X_sort[0:d, :]
            elif d =='different':
                X_d = X_sort
            elif d =='same':
                nef = result_models['mbpca_individual']['scores']['nef']
                X_d = X_sort[0:int(nef),:]
        else:
            if isinstance(d, int):
                nef_ = result_models[model_name]['scores']['nef']
                if d <= nef_:
                    nef = d
                else:
                    nef = nef_
            elif d =='same':
                nef = result_models['mbpca_individual']['scores']['nef']
            elif d =='different':
                nef = result_models[model_name]['scores']['nef']

            X= result_models[model_name]['X']
            X_d = X[0:int(nef),:]
        nef_models[model_name] = X_d.shape[0]
        # cats =['LR', 'FB']
        # cats =['LLRR', 'FFBB']
        p_models_cats[model_name] = dict()
        score_models_cats[model_name] = dict()
        score_shuffle_models_cats[model_name] = dict()
        y_cat_models_cats[model_name] = dict()
        predict_prob_models_cats[model_name] =dict()
        for cat in cats:
            if GMM_n_components is not None:
                shuffle_result =shuffle_decoder_gmm(X_d, cat=cat, spk=spk, tw = tw, history = history, N_shuffle = N_shuffle)
            else:
                shuffle_result =shuffle_decoder(X_d, cat=cat, spk=spk, tw = tw, history = history, N_shuffle = N_shuffle)
            p_models_cats[model_name][cat] = shuffle_result['p']
            score_models_cats[model_name][cat] = shuffle_result['score_fit']
            score_shuffle_models_cats[model_name][cat] = shuffle_result['score_shuffle']
            y_cat_models_cats[model_name][cat] = shuffle_result['y_cat']
            predict_prob_models_cats[model_name][cat] = shuffle_result['predict_proba']
    result = dict()
    result['score_models_cats'] = score_models_cats
    result['p_models_cats'] = p_models_cats
    result['score_shuffle_models_cats'] = score_shuffle_models_cats
    result['y_cat_models_cats'] = y_cat_models_cats
    result['nef_models'] = nef_models
    result['predict_prob_models_cats'] = predict_prob_models_cats

    return result
