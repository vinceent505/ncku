import syn
import librosa
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

perf = "input/audio/perf/Hilary/Hilary_P2_4_perf.wav"
score = "input/audio/score/Hilary/Hilary_P2_4_score.wav"

def main():
    
    perf_wave, fs = librosa.load(perf, sr=44100)
    score_wave, fs = librosa.load(score, sr=44100)

    
    perf_wave = np.array(perf_wave)
    perf_env = syn.get_envelope(perf_wave)
    perf_env = savgol_filter(perf_env, 701, 2)
    score_wave = np.array(score_wave)
    score_env = syn.get_envelope(score_wave)
    score_env = savgol_filter(score_env, 701, 2)


if __name__ == "__main__":
    main()
