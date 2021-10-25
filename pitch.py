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
    scipy.io.wavfile.write("out/filted_%s.wav" %str(num) , fs, frag)
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

if __name__ == "__main__":
    x_1, fs = librosa.load('Bach/bach_hil.wav')
    file = pd.read_csv("start_end.csv")

    start_file = pd.read_csv("start_time.csv")
    start_time = []
    for i in start_file['start']:
        start_time.append(i)
    endtime.find_endtime()
    end_file = pd.read_csv("end_time.csv")
    end_time = []
    for i in end_file['end']:
        end_time.append(i)
    p = []
    for count in range(20):
        frag = x_1[int(start_time[count]*fs):int(end_time[count]*fs)]
        frag = filter.filt(frag, count, fs)
        a = pitch_dec(frag, fs)
        for i in a:
            p.append(i)
    plt.plot(p)
    plt.show()



