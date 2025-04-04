import sys
sys.path.insert(1, '../../utils/')
sys.path.insert(1, './functions/')


from read_data import read_X_depth
from fun_models_class import Models

if __name__ == "__main__":
    matfile = './data/piezo_data1.mat'
    depth = 0
    y = read_X_depth(matfile,depth,type ='df')
    # model_names =['bpca_common','bpca_individual']
    model_names =['bpca_common']

    score_names  = ['evs','r2','lb','log_like','nef','mvds']
    D = 390
    seed = 0
    savepath = './results'
    filename = 'piezo_model_D_'+model_names[0]+'.pickle'

    models = Models(model_names=model_names, savepath =None)
    models.fit(y,D=D,q = D,seed=seed, score_names = score_names,verbose = True, nef_tol = 0.1, imshow_wxyq= False)
    models.save(savepath =savepath,filename =filename)

    print('Finished. Files have been saved in: ' + savepath)
