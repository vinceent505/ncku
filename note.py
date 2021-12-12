import librosa
import matplotlib.pyplot as plt
import numpy as np
import resampy
import crepe
from scipy import signal
from scipy.interpolate import interp1d
import envelope
import scipy.io.wavfile
from scipy.signal import savgol_filter
import filter


class note:
    def __init__(self, num, name, data, fs, base_freq, start, end):
        self.num = num
        self.name = name
        self.data = np.array(data)
        self.fs = fs
        self.base_freq = base_freq
        self.start = start
        self.end = end

        self.g = 1.04
        self.stft = self.stft_filt()
        self.filtered = self.filt(-1)
        self.fundamental = self.filt(1)

        self.envelope = self.get_envelope()
        self.adsr = self.find_adsr()

        self.pitch = self.pitch_dec()
        
        self.harmonics = []
        self.noise = {"means":[], "covariance":[], "weight":[]}

    def stft_filt(self):
        f, t, Zxx = signal.stft(self.data, self.fs, nperseg=2048)
        for i, ii in enumerate(Zxx):#f
            for k in range(100):
                if (k+1)*self.base_freq/1.04 > self.fs/2:
                    break
                if f[i]==0 or(f[i]>k*self.base_freq*self.g and f[i]<(k+1)*self.base_freq/self.g):
                    for j, jj in enumerate(ii):#t
                        Zxx[i][j]=0
                    break
                if f[i]<k*self.base_freq*1.04:
                    break

        _, s = signal.istft(Zxx, self.fs, nperseg=2048)
        return s

    def filt(self, partial):
        if partial == -1:
            filtered = np.zeros(len(self.data))
            for i in range(1, 100):
                if i*self.base_freq*self.g > self.fs/2:
                    break
                else:
                    y = filter.butter_bandpass_filter(self.data, i*self.base_freq/self.g, i*self.base_freq*self.g, self.fs, 4)
                    filtered += y

            # plt.plot(self.data)
            # plt.plot(filtered)
            # plt.plot(self.stft)
            # plt.show()
            return filtered
        else:        
            y = filter.butter_bandpass_filter(self.data, self.base_freq/self.g, self.base_freq*self.g, self.fs, 4)
            
            scipy.io.wavfile.write("out/fundamental"+str(self.num)+".wav", self.fs, y)
            return y
            pass

    def pitch_dec(self):
        d = resampy.resample(self.fundamental, self.fs, 16000)
        _, p, _, _ = crepe.predict(d, 16000, viterbi=True)
        return p



    def get_envelope(self):
        idx = envelope.envelopes_idx(self.data, 7)
        high_data = []
        for i in range(len(self.data)):
            if(i in idx):
                if(self.data[i] < 0):
                    high_data.append(0)
                else:
                    high_data.append(self.data[i])
            else:
                high_data.append(0)
        pre_inter_x = []
        pre_inter_y = []
        pre_inter_x.append(0)
        pre_inter_y.append(self.data[idx[0]])
        for i in idx:
            pre_inter_x.append(2*i)
            pre_inter_y.append(self.data[i])
            last = self.data[i]
        pre_inter_x.append(len(self.data)*2)
        pre_inter_y.append(last)
        f = interp1d(pre_inter_x, pre_inter_y)
        post_inter_x = []
        post_inter_y = []
        for i in range(len(self.data)*2):
            if float(f(i))<0:
                post_inter_y.append(0)
            else:
                post_inter_y.append(float(f(i)))
            post_inter_x.append(i/self.fs)

        envelope.env_write(self.num, post_inter_y, self.fs)
        return post_inter_y

    def find_adsr(self):
        
        e = savgol_filter(self.envelope, 3001, 4)
        ri = []
        rj = []
        ri.append(1)
        rj.append(e[1])
        for i, j in enumerate(e):
            if i>1:
                if (e[i]-e[i-1])*(e[i-1]-e[i-2]) <= 0:
                    ri.append(i)
                    rj.append(j)
            if i==len(e)-1:
                ri.append(i)
                rj.append(j)
            pass

        return np.array([ri, rj])




