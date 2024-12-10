
import pickle
import sys
sys.path.insert(1, './functions/')

from importlib import reload
import matplotlib.pyplot as plt
import numpy as np

import os
from utility import create_folder

import ast
argv = sys.argv

from fun_data2 import Data2
from fun_models_class import Models

model_names =['pca','mlpca','fa','bpca_common','xbpca_common','mbpca_common','bpca_individual','xbpca_individual','mbpca_individual']

score_names  = ['lb','log_like','evs','mse','r2','nef','corr']
models = Models(model_names = model_names, savepath = None)
T = 3000
gaussian_noise_type = 'kappa' # kappa or sn
y_b = 0
data_seed = 0  
method_seed = 12


find_best_score_name ='evs'
find_best_seed = True
method_seed_range = range(10)

rho_max = float(argv[4])
kappa_max = float(argv[5])
rhos = np.arange(0,rho_max,0.1)
kappas = np.arange(0,kappa_max,0.1)

models_results_rhos_kappas = dict()

scores_models = dict()
datas = dict()
datas['ys'] = dict()
datas['cs'] = dict()
datas['ss'] = dict()
for i in range(len(rhos)):
    rho = rhos[i]
    for j in range(len(kappas)):
        sn_kappa = kappas[j]
        if gaussian_noise_type =='kappa':
            print('rho:', rho, 'kappa:', sn_kappa)
        elif gaussian_noise_type =='sn':
            print('rho:', rho, 'sn:', sn_kappa)
        data2= Data2(n0=3, n1 = 7, T = 3000,firing_rate0=3,firing_rate1=0.15,frame_rate=30,
                    type =gaussian_noise_type,poisson_rho = rho,gaussian_sn_kappa = sn_kappa, 
                    y_b =y_b,seed = data_seed)
        y,c,s=data2.gen_data()
    
        models_results = models.fit(y,z = c,q = 3, seed=method_seed, score_names = score_names,
                                    imshow_wxyq = False,verbose = False,
                                    find_best_seed = find_best_seed, score_name = find_best_score_name,
                                    seed_range = method_seed_range,
                                    nef_tol = 0.2)
        models_results_rhos_kappas[i,j] = models_results
        datas['ys'][i,j] = y
        datas['cs'][i,j] = c
        datas['ss'][i,j] = s
       
   

datas['rhos'] = rhos
datas['kappas'] = kappas





# read scores
model_names = models_results.keys()
score_names = models_results[model_names[0]]['scores'].keys()

scores_models_rhos_kappas = dict()


for model_name in model_names:
    scores_models_rhos_kappas[model_name] = dict()        
    for score_name in score_names:
        scores_models_rhos_kappas[model_name][score_name] = np.zeros((len(rhos), len(kappas)))
        for i in range(len(rhos)):
            for j in range(len(kappas)):
                models_results =  models_results_rhos_kappas[i,j]
                scores_model =  models_results[model_name]['scores']
                scores_models_rhos_kappas[model_name][score_name][i,j] =scores_model['score_name']




savefile = './results/scores_models_rhos_kappas_0'+str(round(rho_max))+'1.pickle'
with open(savefile, 'wb') as f: 
    pickle.dump(scores_models_rhos_kappas, f)

savefile = './results/datas_rhos_kappas_021.pickle'
with open(savefile, 'wb') as f: 
    pickle.dump(datas, f)
