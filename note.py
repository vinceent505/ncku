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
import math
import filter

class note:
    def __init__(self, num, name, data, fs, base_freq, start, end):
        self.num = num
        self.name = name
        self.fs = fs
        self.data = np.array(data)            
        # scipy.io.wavfile.write("out/original"+str(self.num)+".wav", self.fs, self.data)

        self.base_freq = base_freq
        self.start = start
        self.end = end

        self.g = 1.04
        self.filtered = self.stft_filt()            
        # scipy.io.wavfile.write("out/filtered"+str(self.num)+".wav", self.fs, self.filtered)


        self.envname = ""
        self.envelope = self.get_envelope(self.filtered)
        self.adsr = self.find_adsr()

        self.pitch = self.pitch_dec()
        
        self.harmonics = self.harmonics_poly()
        # print(len(self.harmonics))
        # for i in range(len(self.harmonics)):
        #     plt.plot(np.linspace(0, self.end-self.start, len(self.harmonics[i])), self.harmonics[i])
        # plt.legend(range(1, len(self.harmonics)+1))
        # plt.show()
        self.noise = []#self.noise_poly()

    def stft_filt(self):
        if len(self.data)<4096:
            window_size = 2048
            overlap = window_size/2
        else:
            window_size = 4096
            overlap = window_size*3/4


        f, t, Zxx = signal.stft(self.data, self.fs, nperseg=window_size, noverlap=overlap)
        for i, ii in enumerate(Zxx):#f
            for k in range(100):
                if (k+1)*self.base_freq/self.g > self.fs/2:
                    break
                if f[i]==0 or(f[i]>k*self.base_freq*self.g and f[i]<(k+1)*self.base_freq/self.g):
                    for j, jj in enumerate(ii):#t
                        Zxx[i][j]=0
                    break
                if f[i]<k*self.base_freq*self.g:
                    break
        _, s = signal.istft(Zxx, self.fs, nperseg=window_size, noverlap=overlap)
        # print("stft length: ", len(s))
        return s[:len(self.data)]

    def pitch_dec(self):
        d = resampy.resample(self.filtered, self.fs, 16000)
        _, p, _, _ = crepe.predict(d, 16000, viterbi=True)
        return p



    def get_envelope(self, data, wr = True, L = 7):
        idx = envelope.envelopes_idx(data, L)
        high_data = []
        for i in range(len(data)):
            if(i in idx):
                if(data[i] < 0):
                    high_data.append(0)
                else:
                    high_data.append(data[i])
            else:
                high_data.append(0)
        pre_inter_x = []
        pre_inter_y = []
        pre_inter_x.append(0)
        pre_inter_y.append(data[idx[0]])
        for i in idx:
            pre_inter_x.append(i)
            pre_inter_y.append(data[i])
            last = data[i]
        pre_inter_x.append(len(data))
        pre_inter_y.append(last)
        f = interp1d(pre_inter_x, pre_inter_y)
        post_inter_x = []
        post_inter_y = []
        for i in range(len(data)):
            if float(f(i))<0:
                post_inter_y.append(0)
            else:
                post_inter_y.append(float(f(i)))
            post_inter_x.append(i/self.fs)
        if wr:
            self.envname = envelope.env_write(self.num, post_inter_y, self.fs)
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

    def harmonics_poly(self):
        f_harmonics = []
        if len(self.data)<4096:
            window_size = 2048
            overlap = window_size/2
        else:
            window_size = 4096
            overlap = window_size*3/4
        f, t, Zxx = signal.stft(self.filtered, self.fs, nperseg=window_size, noverlap=overlap)
        for i, ii in enumerate(Zxx):#f
            pass
        for k in range(10):        
            harmonics = []
            if (k+1)*self.base_freq/self.g > self.fs/2:
                break
            
            if abs(f[int((k+1)*self.base_freq/(self.fs/window_size))]-self.base_freq) < abs(f[int((k+1)*self.base_freq/(self.fs/window_size))+1]-self.base_freq):
                index = int((k+1)*self.base_freq/(self.fs/window_size))
            else:
                index = int((k+1)*self.base_freq/(self.fs/window_size))+1
            for j, jj in enumerate(Zxx[index]):#t
                harmonics.append(20*math.log(abs(jj), 10))
            f_harmonics.append(harmonics)
        return str(f_harmonics)


    def noise_poly(self):
        noise = self.data-self.filtered
        heatmap = librosa.amplitude_to_db(np.abs(librosa.stft(noise, n_fft=window_size, hop_length = window_size//4)))
        f_means = []
        f_covariances = []
        f_weights = []
        for i in range(len(heatmap[0])): #time
            poly_x = []
            poly_y = []
            for j, k in enumerate(heatmap): #freq
                if (j*self.fs/window_size)>self.fs/2:
                    continue

                poly_x.append(self.fs/window_size*j)
                poly_y.append(heatmap[j][i])

            env = self.get_envelope(savgol_filter(poly_y, 9, 3), False, 3)

            nor_env = envelope.normalize(env, 0, 1)
            test_data = np.array(nor_env)
            # Find peak
            peaks, _ = find_peaks(test_data, height=-15)
            # Fit GMM
            gmm = GaussianMixture(n_components=len(peaks)-3, covariance_type="full", tol=0.001)
            random_data = [np.random.uniform(i, i+1, int(1000*aa)) for i, aa in enumerate(test_data)]
            data = []
            for i in random_data:
                for ii in i:
                    data.append(ii)

            gmm = gmm.fit(X=np.expand_dims(np.array(data), 1)) #要2D
            # # Evaluate GMM
            weights = []
            for w in gmm.weights_.ravel():
                weights.append(w)

            f_weights.append(np.array(weights).max())
        return np.array(f_weights)



