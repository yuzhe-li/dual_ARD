import matplotlib.pyplot as plt
scale = 5
cm2inch = 0.39
# fontsize 
ssmall_SIZE = 3*scale
SMALL_SIZE = 4*scale
MEDIUM_SIZE = 6*scale
BIGGER_SIZE = 8*scale
LINE_WIDTH = 0.8*scale
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('xtick', labelsize=ssmall_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=ssmall_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=ssmall_SIZE)    # legend fontsize

plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('lines', linewidth=LINE_WIDTH)



def create_folder(my_folder):
    from pathlib import Path
    Path(my_folder).mkdir(parents=True, exist_ok=True)




def lowpass_filter(data, cutoff, fs, order=5):
    """
    Input: data.shape: (d,N)
    return: y.shape: (d,N)
    """
    from scipy.signal import butter, filtfilt

    b, a = butter(order, cutoff, fs=fs, btype='lowpass', analog=False)

    # y = lfilter(b, a, data)
    y = filtfilt(b,a,data, method ='gust', axis = -1)
    return y

# def lowpass_filter(data, cutoff, fs, order=5):
#     """
#     Input: data.shape: (N, d)
#     return: y.shape: (N, d)
#     """
#     import numpy as np
#     from scipy.signal import butter, filtfilt
#     b, a = butter(order, cutoff, fs=fs, btype='lowpass', analog=False)
#     if len(data.shape) ==1: 
#         data = data[:,np.newaxis]
    
#     y = np.zeros(data.shape)
#     for i in range(data.shape[1]):
#         y[:,i] = filtfilt(b,a,data[:,i], method ='gust')
#     return np.squeeze(y)


def Figsize_scale(figsize_x,figsize_y, scale=5,cm2inch= 0.39 ):
    return (figsize_x*scale*cm2inch, figsize_y*scale*cm2inch)
def get_cmap_ci(cmap, i):
    import matplotlib
    return matplotlib.colors.rgb2hex(cmap(i)[:3])


