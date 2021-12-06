import numpy as np
import wave
import struct
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import librosa
import librosa.display
import pandas as pd
import scipy.io.wavfile
from scipy.fftpack import fft, ifft, rfft, irfft, fftfreq
from scipy.signal import butter, lfilter
import resample
from scipy.signal import hilbert, chirp

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
    
def butter_bandstop(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandstop')
    return b, a

def butter_bandstop_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandstop(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def fft_filter(frag, base_freq, num, fs):
    x = np.linspace(0.0, len(frag)/fs, len(frag), endpoint=False)
    f = fft(frag)[:len(frag)//2]
    x = fftfreq(len(frag), 1/fs)[:len(frag)//2]
    # print(x)
    for i, j in enumerate(x):
        if j>base_freq*1.06 or j<base_freq/1.06:
            f[i] = 0
    # plt.plot(x, f[:len(frag)//2])
    # plt.show()
    # print(base_freq)
    f = np.fft.irfft(f)

    return f

def filt(frag, num, fs, base_freq, note, partial, shift_octave):
    n = 2**shift_octave
    if partial==0:
        pass

    shifted = librosa.effects.pitch_shift(np.array(frag, dtype='float32'), fs, n_steps = shift_octave*12)


    y = butter_bandpass_filter(shifted, base_freq/1.03*n, base_freq*1.03*n, fs, order=4)
    for i in range(partial):
        if base_freq*(i+2)*1.06*n > fs/2:
            break
        tmp = butter_bandpass_filter(shifted, base_freq*(i+2)/1.03*n, base_freq*(i+2)*1.03*n, fs, order=4)
        tmp = butter_bandpass_filter(tmp, base_freq*(i+2)/1.03*n, base_freq*(i+2)*1.03*n, fs, order=4)
        tmp = butter_bandpass_filter(tmp, base_freq*(i+2)/1.03*n, base_freq*(i+2)*1.03*n, fs, order=4)
        scipy.io.wavfile.write("out/filted_%s_part.wav" %str(i) , fs, tmp)
        y = y + tmp

    y = librosa.effects.pitch_shift(y, fs, n_steps = shift_octave*(-12))
    for i in range(partial, int(20000/base_freq)-1):
        tmp = butter_bandpass_filter(frag, base_freq*(i)/1.03, base_freq*(i)*1.03, fs, order=4)
        tmp = butter_bandpass_filter(tmp, base_freq*(i)/1.03, base_freq*(i)*1.03, fs, order=4)
        scipy.io.wavfile.write("out/filted_%s_part.wav" %str(i) , fs, tmp)
        y = y + tmp
        pass
    scipy.io.wavfile.write("out/filted_%s_all.wav" %str(num) , fs, y)
    scipy.io.wavfile.write("out/original_%s___.wav" %str(num) , fs, np.array(frag, dtype='float32'))

    return shifted

def filtt(frag, num, fs, base_freq, note):
    n = 2

    print("num:", num, "base freq:", base_freq, "note:", note)

    down = resample.downsample(frag, fs, n)

    y = butter_bandpass_filter(down, base_freq/1.06*n, base_freq*1.06*n, fs, order=2)
    
    up = resample.upsample(y, fs, n)
    return up

def harmonics_filter(freq, up, fs, num):
    n = 2
    # scipy.io.wavfile.write("out/filted_%s_3.wav" %str(num) , fs, up)
    for i in range(1, 10):
        if i*freq>15500:
            break
        shifted = librosa.effects.pitch_shift(np.array(up, dtype='float32'), fs, n_steps = 12)
        y = butter_bandstop_filter(shifted, freq/1.06*n*i, freq*1.06*n*i, fs, order=2)
        shifted = librosa.effects.pitch_shift(y, fs, n_steps = -12)    
    return shifted



if __name__ == "__main__":
    filename = input()
    num = input()
    data, fs = librosa.load(filename)
    scipy.io.wavfile.write("out/filted_f_1.wav" , fs, filt(data, int(num), fs))
    