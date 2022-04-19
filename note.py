import librosa
import librosa.effects
import matplotlib.pyplot as plt
import numpy as np
import resampy
import crepe
from scipy import signal
from scipy.interpolate import interp1d
from scipy import fft
import envelope
import scipy.io.wavfile
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from sklearn.mixture import GaussianMixture
import math
import filter


check = -1
OVERLAP = 4096-512


class note:
    def __init__(self, num, name, data, fs, base_freq, start, end):
        self.num = num
        self.name = name
        self.fs = fs
        self.data = np.array(data)          
        # scipy.io.wavfile.write("test/original"+str(self.num)+".wav", self.fs, self.data)

        self.base_freq = base_freq
        self.start = start
        self.end = end

        self.g = 1.04
        self.filtered = self.stft_filt()   
        self.fundamental = self.stft_fundamental()  
        self.pitch = self.pitch_dec()  
        # plt.plot(self.pitch)
        # plt.show()  
        # scipy.io.wavfile.write("out/filtered"+str(self.num)+".wav", self.fs, self.filtered)
        #scipy.io.wavfile.write("out/fund"+str(self.num)+".wav", self.fs, self.fundamental)


        self.envname = ""
        self.envelope = self.get_envelope(self.filtered)
        # plt.plot(self.envelope)
        # plt.show()
        self.adsr = []#self.find_adsr()

        self.noise = self.noise_poly()
        self.harmonics = self.harmonics_poly()     
        # for i in self.harmonics:
        #     plt.plot(i)
        # plt.show()
        
        # print(len(self.harmonics))
        # for i in range(len(self.harmonics)):
        #     plt.plot(np.linspace(0, self.end-self.start, len(self.harmonics[i])), self.harmonics[i])
        # plt.legend(range(1, len(self.harmonics)+1))
        # plt.show()

    def stft_filt(self):
        if len(self.data)<4096:
            data_padding = np.zeros(4096)
            data_padding[:len(self.data)] += self.data
            input_data = data_padding
        else:
            input_data = self.data
        window_size = 4096
        overlap = OVERLAP
        f, t, Zxx = signal.stft(input_data, self.fs, nperseg=window_size, noverlap=overlap)
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
        
        return s[:len(self.data)]



        
    def stft_fundamental(self):
        l = False
        if len(self.data)<4096:
            data_padding = np.zeros(4096)
            data_padding[:len(self.data)] += self.data
            input_data = data_padding
            l = True
        else:
            input_data = self.data
        window_size = 4096
        overlap = OVERLAP
        f, t, Zxx = signal.stft(input_data, self.fs, nperseg=window_size, noverlap=overlap)
        for i, ii in enumerate(Zxx):#f
            for k in range(1):
                if (k+1)*self.base_freq/self.g > self.fs/2:
                    break
                if f[i]==0 or(f[i]>(k+1)*self.base_freq*self.g or f[i]<(k+1)*self.base_freq/self.g):
                    for j, jj in enumerate(ii):#t
                        Zxx[i][j]=0
                    break
                if f[i]<k*self.base_freq*self.g:
                    break

        _, s = signal.istft(Zxx, self.fs, nperseg=window_size, noverlap=overlap)
        # print("stft length: ", len(s))
        return s[:len(self.data)]

    def pitch_dec(self):
        length = int(len(self.data)/2)
        fadein_window = np.linspace(0.0, 1.0, length)
        fadeout_window = np.linspace(1.0, 0.0, length)
        faded = np.array(self.fundamental)
        faded[0: length] = faded[0: length] * fadein_window
        faded[-length:] = faded[-length:] * fadeout_window

        # scipy.io.wavfile.write("out/faded"+str(self.num)+".wav", self.fs, faded)

        if self.base_freq > 1500:
            dest_freq = 500
            ratio = self.base_freq/dest_freq
            d_upsample = resampy.resample(faded, self.fs, self.fs*ratio)
            d = resampy.resample(d_upsample, self.fs, 16000)
            # scipy.io.wavfile.write("out/shift"+str(self.num)+".wav", 16000, d)
            _, p, _, _ = crepe.predict(d, 16000, viterbi=True, verbose=0, step_size = 10*ratio)
            p = p*ratio
            if len(p)//2*2-1 > 3:
                p = savgol_filter(p, len(p)//2*2-1, 3)
        else:
            d = resampy.resample(faded, self.fs, 16000)
            _, p, _, _ = crepe.predict(d, 16000, viterbi=True, verbose=0)
            if len(p)//2*2-1 > 3:
                p = savgol_filter(p, len(p)//2*2-1, 3)
        for num, i in enumerate(p):
            if p[num] < self.base_freq/1.06 or p[num] > self.base_freq*1.06:
                p[num] = self.base_freq
        if self.num == check:
            print(p)
        return p



    def get_envelope(self, data, wr = False, L = 7):
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
        window_size = 4096
        overlap = 4096-64
        l = False
        t_len = math.ceil(len(self.filtered)/1024)
        if len(self.filtered)<4096:
            data_padding = np.zeros(4096)
            data_padding[:len(self.filtered)] += self.filtered
            input_data = data_padding
            l = True
        else:
            input_data = self.filtered




        
        f, t, Zxx = signal.stft(input_data, self.fs, boundary = None, nperseg=window_size, noverlap=overlap)
    
        if self.num == check:
            plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
            plt.title('STFT Magnitude')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.figure()
        for k in range(1, 100):
            if k*self.base_freq > 20000:
                break    
            harmonics = []
            check_harmonics = [[], [], [], [], [], []]
            for time in range(len(t)):
                time_index = int(t[time]/0.01)
                if time_index > len(self.pitch):
                    break
                elif time_index == len(self.pitch):
                    freq_index = int(self.base_freq*k/(self.fs/window_size))
                else:
                    freq_index = int(self.pitch[time_index]*k/(self.fs/window_size))
                candidate = []
                for count_check, c in enumerate(range(-3, 3)):
                    candidate.append(abs(Zxx[freq_index+c][time]))


                    if self.num == check:
                        if k == 1:
                            check_harmonics[count_check].append(abs(Zxx[freq_index+c][time]))

                harmonics.append(20*(math.log(max(candidate), 10)))

            if self.num == check:
                # for i in check_harmonics:
                #     plt.plot(i)
                # print("___________")
                # plt.legend(range(-3, 3))
                # plt.show()
                # plt.figure()
                if k <= 10:
                    plt.plot(harmonics)

            f_harmonics.append(harmonics)

        if self.num == check:
            plt.legend(range(10))
            plt.show()
            
        return f_harmonics



    def noise_poly(self):
        noise = self.data-self.filtered

        window_size = 4096
        overlap = OVERLAP
        if len(self.data)<4096:
            data_padding = np.zeros(4096)
            data_padding[:len(noise)] += noise
            input_data = data_padding
            l = True
        else:
            input_data = self.data
        f, t, Zxx = signal.stft(input_data, self.fs, nperseg=window_size, noverlap=overlap)

        heatmap = librosa.amplitude_to_db(np.abs(Zxx))

        
        avg = []
        for i in range(len(heatmap[0])): #time
            noise_weights = []
            poly_x = []
            poly_y = []
            for j, k in enumerate(heatmap): #freq
                if (j*self.fs/window_size)>self.fs/2:
                    continue

                poly_x.append(self.fs/window_size*j)
                if self.fs/window_size*j < 20:
                    poly_y.append(-75)
                else:
                    poly_y.append(heatmap[j][i])

            peaks, _ = find_peaks(poly_y, distance=15)
            for i in peaks:
                noise_weights.append(poly_y[i])
            avg.append(np.array(noise_weights).mean())
        return avg