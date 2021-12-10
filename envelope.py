import matplotlib.pyplot as plt
import numpy as np
import wave
import sys
import librosa
from sklearn import preprocessing
from scipy.interpolate import interp1d
import scipy.io.wavfile
from tqdm import tqdm
from scipy.signal import savgol_filter


def find_env_curve(env, wav):
    e = savgol_filter(env, 3001, 4)
    ri = []
    rj = []
    ri.append(1)
    rj.append(e[1])
    for i, j in enumerate(e):
        if i>1:
            if (e[i]-e[i-1])*(e[i-1]-e[i-2]) <= 0:
                ri.append(i)
                rj.append(j)
        if i==len(e)-1:
            ri.append(i)
            rj.append(j)
        pass

    return np.array([ri, rj])

    pass


def normalize(arr, t_min, t_max):
    arr_min = min(arr)
    arr_max = max(arr)
    norm_arr = []
    diff = t_max - t_min
    diff_arr = arr_max - arr_min
    for i in tqdm(arr):
        temp = (((i - arr_min)*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr
def hl_envelopes_idx(s,dmin=1,dmax=1):

    # locals min      
    lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1 
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1 
    # the following might help in some case by cutting the signal in "half"
    s_mid = np.mean(s)
    # pre-sort of locals min based on sign 
    lmin = lmin[np.array(s)[lmin]<np.array(s_mid)]
    # pre-sort of local max based on sign 
    lmax = lmax[np.array(s)[lmax]>np.array(s_mid)]
    # global max of dmax-chunks of locals max 
    lmin = lmin[[i+np.argmin(np.array(s)[lmin[i:i+dmin]]) for i in range(0,len(lmin),dmin)]]
    # global min of dmin-chunks of locals min 
    lmax = lmax[[i+np.argmax(np.array(s)[lmax[i:i+dmax]]) for i in range(0,len(lmax),dmax)]]
    
    return lmax

def env_write(num, post_inter_y):
    scipy.io.wavfile.write("envelope_out/envelope%s.wav" %str(num), 44100, np.array(post_inter_y, dtype = "float64"))


def envelope(data, num, fs, min, max):
    t = np.linspace(0, len(data) / fs, num=len(data))
    # plt.subplot(1, 2, 1)
    # plt.plot(data)

    low_idx = hl_envelopes_idx(data, dmin=min,dmax=max)

    high_data = []
    for i in range(len(data)):
        if(i in low_idx):
            if(data[i] < 0):
                high_data.append(0)
            else:
                high_data.append(data[i])
        else:
            high_data.append(0)
    pre_inter_x = []
    pre_inter_y = []
    pre_inter_x.append(0)
    pre_inter_y.append(data[low_idx[0]])
    for i in low_idx:
        pre_inter_x.append(2*i)
        pre_inter_y.append(data[i])
        last = data[i]
    pre_inter_x.append(len(data)*2)
    pre_inter_y.append(last)
    f = interp1d(pre_inter_x, pre_inter_y)
    post_inter_x = []
    post_inter_y = []
    for i in range(len(data)*2):
        if float(f(i))<0:
            post_inter_y.append(0)
        else:
            post_inter_y.append(float(f(i)))
        post_inter_x.append(i/44100)
    # plt.subplot(1, 2, 2)
    # new_start = env_update(post_inter_y, start, end)
    # plt.plot(post_inter_x, post_inter_y)
    # plt.show()
    return post_inter_y
    
    # return new_start