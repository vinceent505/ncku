import numpy as np
import wave
import struct
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import librosa
import librosa.display
import pandas as pd
import scipy.io.wavfile
from scipy import signal
from scipy.fftpack import fft, ifft, rfft, irfft, fftfreq
from scipy.signal import butter, lfilter, cheby1
import resample
from scipy.signal import hilbert, chirp

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = cheby1(order, 5, [low, high], btype='band')
    return b, a

    
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
def butter_highpass_filter(data, cutoff, fs, order=3):
    sos = signal.butter(order, cutoff, 'hp', fs=fs, output='sos')
    filtered = signal.sosfilt(sos, data)
    return filtered
def butter_lowpass_filter(data, cutoff, fs, order=3):
    sos = signal.butter(order, cutoff, 'lp', fs=fs, output='sos')
    filtered = signal.sosfilt(sos, data)
    return filtered
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y





def fft_filter(frag, base_freq, num, fs):
    x = np.linspace(0.0, len(frag)/fs, len(frag), endpoint=False)
    f = fft(frag)[:len(frag)//2]
    x = fftfreq(len(frag), 1/fs)[:len(frag)//2]
    for i, j in enumerate(x):
        if j>base_freq*1.06 or j<base_freq/1.06:
            f[i] = 0
    f = np.fft.irfft(f)
    return f

def stft_filter(frag, num, fs, base_freq, note, shift_octave):


    return





def filt(frag, num, fs, base_freq, note, partial, shift_octave):
    n = 2**shift_octave
    if partial==0:
        pass

    scipy.io.wavfile.write("out/filted_part.wav", fs, np.array(frag, dtype='float32'))
    shifted = librosa.effects.pitch_shift(np.array(frag, dtype='float32'), fs, n_steps = shift_octave*12)

    g = 1.04
    part = base_freq * (g-1) * n
    final = np.zeros(len(frag))

    if partial == -1:
        # y = butter_highpass_filter(shifted, base_freq/g*n, fs, order=7)
        for i in range(1, 13):
            if i*base_freq*n+part > fs/2:
                break
            else:
                print(" ")
                print("low: ", i*base_freq*n-part)
                print("high: ", i*base_freq*n+part)
                y = butter_bandpass_filter(shifted, i*base_freq*n/g, i*base_freq*n*g, fs, 4)
                final += y
                scipy.io.wavfile.write("out/filted_%s_part.wav" %str(i) , fs, y)
                pass
        
        final = librosa.effects.pitch_shift(final, fs, n_steps = shift_octave*(-12))
        scipy.io.wavfile.write("out/filted_final_part.wav", fs, final)
        plt.plot(final)
        plt.show()
        pass
    else:
        pass


    for o in range(1, 9):
    
        b, a = butter_bandpass(base_freq*n-part, base_freq*n+part, fs, order=o)
        w, h = signal.freqz(b, a, worN=2000)
        plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % o)


    plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
             '--', label='sqrt(0.5)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.grid(True)
    plt.legend(loc='best')

    plt.show()
    return final

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
    