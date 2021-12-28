import matplotlib.pyplot as plt
import numpy as np
import librosa
from sklearn import preprocessing
from scipy.interpolate import interp1d
import scipy.io.wavfile
from tqdm import tqdm
from scipy.signal import savgol_filter

def normalize(arr, t_min, t_max):
    arr_min = min(arr)
    arr_max = max(arr)
    norm_arr = []
    diff = t_max - t_min
    diff_arr = arr_max - arr_min
    for i in arr:
        temp = (((i - arr_min)*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr

def envelopes_idx(s,dmax=1):
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1 
    # the following might help in some case by cutting the signal in "half"
    s_mid = np.mean(s)
    # pre-sort of local max based on sign 
    lmax = lmax[np.array(s)[lmax]>np.array(s_mid)]
    # global min of dmin-chunks of locals min 
    lmax = lmax[[i+np.argmax(np.array(s)[lmax[i:i+dmax]]) for i in range(0,len(lmax),dmax)]]
    return lmax

def env_write(num, env, fs):
    scipy.io.wavfile.write("envelope_out/envelope%s.wav" %str(num), fs, np.array(env, dtype = "float64"))
    return "envelope_out/envelope"+ str(num) +".wav"
