# read data
import numpy as np
import pickle
from scipy.io import loadmat




def read_X_depth(matfile,depth, type='raw'):
    """
    Parameters:
    -------
    matfile: path of matfile,
            'Sig_piezo_All_TCMat_NewScan_Analog_S_20140129_2015-7-6-ray005_1.txt.mat_1.mat'
            'Sig_piezo_All_TCMat_NewScan_Analog_S_20140129_2015-7-6-ray005_1.txt.mat_1.mat'

            
    only for read from matfile 
    depth: int, [0,8]
    datatype: 'raw' or df

    Returns:
    -----------
    X_depth: num_neuroms * num_frames


    """

    from scipy.io import loadmat
    data = loadmat(matfile)
    trace_r =np.squeeze(data['Trace_r'])
    all_trace = np.squeeze(data['All_trace'])
    if type =='raw':
        trace = trace_r
    elif type =='df':
        trace = all_trace
    X_depth = trace[depth]['matrix']

    return X_depth

 

def read_n_neuron_depths(matfile,type='raw'):
    """
    Parameters:
    -------


    Returns:
    -----------
    """
    from scipy.io import loadmat
    data = loadmat(matfile)
    if type =='raw':
        record=np.squeeze(data['Trace_r'])
    elif type =='df':
        record = np.squeeze(data['All_trace'])
    n_depths = len(record)
    n_neuron_depths = np.zeros(n_depths)
    n_frame_depths = np.zeros(n_depths)
    for depth in range(n_depths):
        n_neuron_d,n_frame_d = np.squeeze(record[depth]['matrix']).shape
        n_frame_depths[depth] =n_frame_d 
        n_neuron_depths[depth] = n_neuron_d
    return n_neuron_depths, n_frame_depths

def read_coordinates_depth(matfile,depth,type = 'raw'):
    """
    Parameters:
    -------


    Returns:
    -----------
    """
    data = loadmat(matfile)


    X_plot = np.squeeze(data['X_plot']) # for trace_r, raw data
    Y_plot = np.squeeze(data['Y_plot'])
    X_plot_depth = np.squeeze(X_plot[depth]['matrix'])
    Y_plot_depth = np.squeeze(Y_plot[depth]['matrix'])

    x_circle_depth = np.squeeze(X_plot_depth['matrix'])
    y_circle_depth = np.squeeze(Y_plot_depth['matrix'])
    coordinates = np.zeros((len(x_circle_depth),2))
    for i in range(len(x_circle_depth)):
        xi_circle = np.squeeze(x_circle_depth[i])
        yi_circle = np.squeeze(y_circle_depth[i])
        xi_center = np.mean(xi_circle)
        yi_center = np.mean(yi_circle)
        coordinates[i]=[xi_center,yi_center]

    if type == 'raw':
        return coordinates
    elif type =='df':
        use_neuron = np.squeeze(data['use_neuron'])
        use_neuron_depth = np.array(np.squeeze(use_neuron[depth]['matrix']))
        idx = np.where(use_neuron_depth==1)[0]
        coordinates_use = coordinates[idx]

        return coordinates_use
