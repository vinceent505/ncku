from scipy.signal.signaltools import resample
import parselmouth
import image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import endtime
import librosa
import librosa.display
import numpy as np
import pandas as pd
import envelope
import filter
import resample
import sklearn
import scipy.io.wavfile


def pitch_dec(frag, num, fs, freq):

    up = resample.upsample(frag, fs, 2)
    snd = parselmouth.Sound(up, sampling_frequency = fs)



    pitch = snd.to_pitch()
    pitch_values = pitch.selected_array['frequency']*2
    pitch_values[pitch_values==0] = np.nan
    for j, i in enumerate(pitch_values):
        if i>2*freq or i<freq/2:
            pitch_values[pitch_values==i] = np.nan
        if np.isnan(i):
            for k in range(j):
                pitch_values[k] = np.nan
    print(pitch_values)

    return pitch_values




