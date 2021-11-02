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
    # scipy.io.wavfile.write("out/filted_%s_1.wav" %str(num) , fs, f)

    return f

def filt(frag, num, fs, base_freq, note):
    n = 2

    # up = frag
    # t = np.arange(len(up)) / fs
    # analytic_signal = hilbert(up)
    # plt.plot(analytic_signal)
    # amplitude_envelope = np.abs(analytic_signal)
    # instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    # instantaneous_frequency = (np.diff(instantaneous_phase) /
    #                         (2.0*np.pi) * fs)
    # plt.figure()
    # plt.plot(up)
    # fig, (ax0, ax1) = plt.subplots(nrows=2)
    # ax0.plot(t, up, label='signal')
    # ax0.plot(t, amplitude_envelope, label='envelope')
    # ax0.set_xlabel("time in seconds")
    # ax0.legend()
    # ax1.plot(t[1:], instantaneous_frequency)
    # ax1.set_xlabel("time in seconds")
    # ax1.set_ylim(0.0, 120.0)
    # fig.tight_layout()
    # plt.show()

    print("num:", num, "base freq:", base_freq, "note:", note)


    down = resample.downsample(frag, fs, n)
    #scipy.io.wavfile.write("out/filted_%s_1.wav" %str(num) , fs, down)



    y = butter_bandpass_filter(down, base_freq/1.06*n, base_freq*1.06*n, fs, order=2)
    for i in range(2):
        tmp = butter_bandpass_filter(down, base_freq*(i+2)/1.06*n, base_freq*(i+2)*1.06*n, fs, order=2)
        y = y + tmp
    up = resample.upsample(y, fs, n)
    scipy.io.wavfile.write("out/filted_%s_3.wav" %str(num) , fs, up)
    return up

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
        down = resample.downsample(up, fs, n)
        y = butter_bandstop_filter(down, freq/1.06*n*i, freq*1.06*n*i, fs, order=2)
        up = resample.upsample(y, fs, n)
    
    return up



if __name__ == "__main__":
    filename = input()
    num = input()
    data, fs = librosa.load(filename)
    scipy.io.wavfile.write("out/filted_f_1.wav" , fs, filt(data, int(num), fs))
    