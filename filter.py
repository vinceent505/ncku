from scipy import signal
from scipy.fftpack import fft, ifft, rfft, irfft, fftfreq
from scipy.signal import butter, lfilter, cheby1
from scipy.signal import hilbert, chirp

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

    
def butter_bandstop(lowcut, highcut, fs, order=4):
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


if __name__ == "__main__":
    pass
    