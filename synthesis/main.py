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
######
SAMPLE_RATE = 44100
WINDOW_SIZE = 4096
HOP_SIZE = 1024
PITCH_WIN_TIME = 0.01
path = "./Bach_sonata_no1.pickle"
######
import pickle

with open(path, "rb") as f:
    song = pickle.load(f)


length = len(song.keys())

time_series = np.array([0. for _ in range(int((song[length - 1]["end"]+5) * SAMPLE_RATE))])
num = 0
for note_name in song.keys():
    note = song[note_name]
    dur = note["end"] - note["start"]

    if dur < 4096 / SAMPLE_RATE:
        WINDOW_SIZE = 4096
        HOP_SIZE = 1024
    else:
        WINDOW_SIZE = 4096
        HOP_SIZE = 1024

    harmonics_f = note["harmonics"]
    harmonics_t = np.transpose(harmonics_f).tolist()
    pitch_contour = note["pitch"]

    normalized_max_value = np.max(harmonics_f)
    normalized_min_value = np.min(harmonics_f)
    normalized_range = normalized_max_value - normalized_min_value
    # timescale (pitch contour) > timescale (harmonics)
    # interpolate pitch contour
    pitch_contour = signal.resample(pitch_contour, len(harmonics_t))
    
    # normalize harmonics
    phases = [0 for _ in harmonics_f]
    deltas = [0 for _ in harmonics_f]
    windows = np.array([])
    window = np.array([0 for _ in range(WINDOW_SIZE)])
    p = []
    for t, pitch in enumerate(pitch_contour):
        # initial delta value
        # supposed phases are all start from 0
        for f, _ in enumerate(harmonics_f):
            deltas[f] = (pitch * (f+1)) / SAMPLE_RATE * (2 * math.pi)

        # single window manipulate
        window = [0 for _ in range(WINDOW_SIZE)]
        for i in range(WINDOW_SIZE):
            for f, harmonic in enumerate(harmonics_t[t]):
                harmonic = -pow(10, harmonic / 20)
                phases[f] += deltas[f]
                window[i] += harmonic * math.sin(phases[f]) # + (random.random() - 1) / 2 * 0.0005

        # append to windows
        ## multiply by triangle window
        tri = signal.triang(WINDOW_SIZE)
        # overlap and add
        window *= tri
        if len(windows) == 0:
            windows = window
        else:
            windows[-(WINDOW_SIZE - HOP_SIZE):] += window[ :(WINDOW_SIZE - HOP_SIZE)]
            windows = np.concatenate((windows, window[(WINDOW_SIZE - HOP_SIZE):]))
            for i in range(WINDOW_SIZE-HOP_SIZE):
                for f, harmonic in enumerate(harmonics_t[t]):
                    phases[f] -= deltas[f]
        
    #     if num == 31:
    #         p.append(harmonics_t[t][0])
    # if num == 31:
    #     plt.plot(p)
    #     plt.show()
    # fade in/out
    half = int(len(windows)/2)
    windows[:half] *= [i/half for i in range(half)]
    windows[-half:] *= [1-i/half for i in range(half)]

    # envelope
    # note["envelope"] = signal.resample(note["envelope"], len(windows))
    # windows *= note["envelope"]

    windows = np.array(windows, dtype=np.float32)
    # w_windows = (windows - windows.mean())/ abs(windows).max()
    wavfile.write("./cut/whole-faded_" + str(num) + ".wav", SAMPLE_RATE, windows)
    print(note["start"], note["end"])
    time_series[int(note["start"] * SAMPLE_RATE):int(note["start"] * SAMPLE_RATE) + len(windows)] += windows
    num += 1
 
# time_series = (time_series - time_series.min()) / (time_series.max() - time_series.min())
time_series = (time_series - time_series.mean()) / abs(time_series).max()
wavfile.write("whole-faded.wav", SAMPLE_RATE, time_series)