import librosa
import numpy as np

def upsample(data, fs, n):
    return librosa.resample(np.array(data), fs, fs*n)

def downsample(data, fs, n):
    return librosa.resample(np.array(data), fs, fs/n)
