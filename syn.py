"""
Synthesis

under 4096 sample size, use WIN_SIZE=1024
C#7+ < 10 harmonics
"""
import enum
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import random
import pickle
import threading
import time
import multiprocessing as mp
from multiprocessing import Manager
import random
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
######
SAMPLE_RATE = 44100
WINDOW_SIZE = 4096
HOP_SIZE = 64
PITCH_WIN_TIME = 0.01

######

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

def get_envelope(data, L = 7):
    idx = envelopes_idx(data, 7)
    high_data = []
    for i in range(len(data)):
        if(i in idx):
            if(data[i] < 0):
                high_data.append(0)
            else:
                high_data.append(data[i])
        else:
            high_data.append(0)
    pre_inter_x = []
    pre_inter_y = []
    pre_inter_x.append(0)
    pre_inter_y.append(data[idx[0]])
    for i in idx:
        pre_inter_x.append(i)
        pre_inter_y.append(data[i])
        last = data[i]
    pre_inter_x.append(len(data))
    pre_inter_y.append(last)
    f = interp1d(pre_inter_x, pre_inter_y)
    post_inter_x = []
    post_inter_y = []
    for i in range(len(data)):
        if float(f(i))<0:
            post_inter_y.append(0)
        else:
            post_inter_y.append(float(f(i)))
        post_inter_x.append(i/44100)

    return post_inter_y

def synthesis(note, time_series):

    dur = note["end"] - note["start"]
    samples = int(dur*SAMPLE_RATE)

    print(note["start"])

    WINDOW_SIZE = 4096
    HOP_SIZE = 64

    harmonics_f = note["harmonics"]
    harmonics_t = np.transpose(harmonics_f).tolist()
    pitch_contour = note["pitch"]

    normalized_max_value = np.max(harmonics_f)
    normalized_min_value = np.min(harmonics_f)
    normalized_range = normalized_max_value - normalized_min_value
    # timescale (pitch contour) > timescale (harmonics)
    # interpolate pitch contour
    pitch_contour = signal.resample(pitch_contour, samples)

    
    windows = np.array([])
    window_sum = np.array([0.0 for _ in range(samples)])
    deltas = 0.0


    for harmonic_num , harmonic in enumerate(harmonics_f):
        window = np.array([0.0 for _ in range(samples)])
        phase = 0.0
        
        harmonic = signal.resample(harmonic, samples)
        for t , pitch in enumerate(pitch_contour):
            if (harmonic_num+1)*pitch>2000:
                pitch = note["frequency"]
            delta = (pitch * (harmonic_num+1)) / SAMPLE_RATE * (2 * math.pi)
            phase += delta
            h = -pow(10, harmonic[t] / 20)
            window[t] += h * math.sin(phase)
        

        window_sum += window
    

    window_sum = (window_sum - window_sum.mean()) / abs(window_sum).max()
    # wavfile.write("out/before_"+str(note["num"])+".wav", 44100, window_sum)
    divide = []
    window_sum = np.array(window_sum)
    window_env = get_envelope(window_sum)
    window_env = savgol_filter(window_env, 701, 2)
    perf_env = np.array(note["envelope"])

    env_len = min(len(perf_env), len(window_env))
    window_env = window_env[:env_len]
    perf_env = perf_env[:env_len]

    for w, p in zip(window_env, perf_env):
        if w==0:
            divide.append(0)
        else:
            divide.append(p/w)
    # plt.plot(divide)
    # plt.plot(savgol_filter(divide, 1501, 2))
    # plt.show()

    divide = savgol_filter(divide, 1501, 2)
    for i, j in enumerate(divide):
        window_sum[i] *= j

    window_length = 1000
    fadein_window = np.linspace(0.0, 1.0, window_length)
    fadeout_window = np.linspace(1.0, 0.0, window_length)
    window_sum = np.array(window_sum)
    window_sum[0: window_length] = window_sum[0: window_length] * fadein_window
    window_sum[-window_length:] = window_sum[-window_length:] * fadeout_window

    # wavfile.write("out/after_"+str(note["num"])+".wav", 44100, window_sum)

    
    time_series[int(note["start"] * SAMPLE_RATE):int(note["start"] * SAMPLE_RATE) + len(window_sum)] += window_sum






def syn(name = "presto"):
    path = "output/pickle/" + name + ".pickle"
    with open(path, "rb") as f:
        song = pickle.load(f)
    length = len(song.keys())
    time_series = np.array([0. for _ in range(int((song[length - 1]["end"]+5) * SAMPLE_RATE))])


    now = time.time()

    thread_list = []


    for note_name in song.keys():
        note = song[note_name]
        synthesis(note, time_series)

    final_output = (time_series - time_series.mean()) / abs(time_series).max()
    wavfile.write("output/audio/" + name + ".wav", SAMPLE_RATE, final_output)
    print(time.time()-now, "s")

if __name__ == "__main__":
    syn()
