import librosa
import numpy as np
import matplotlib.pyplot as plt
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
import envelope


window_size = 256
noise_window = 2048


def noise_poly(frag, freq, fs):
    
    heatmap = librosa.amplitude_to_db(np.abs(librosa.stft(frag,n_fft=noise_window)))
    f_means = []
    f_covariances = []
    f_weights = []
    for i in range(len(heatmap[0])): #time
        poly_x = []
        poly_y = []
        for j, k in enumerate(heatmap): #freq
            if (j*fs/noise_window)>15500:
                continue

            poly_x.append(fs/noise_window*j)
            poly_y.append(heatmap[j][i])

        env = envelope.envelope(savgol_filter(poly_y, 9, 3), 0, 44100, 1, 1)
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
        means = []
        covariances = []
        weights = []
        for m, c, w in zip(gmm.means_.ravel(), gmm.covariances_.ravel(), gmm.weights_.ravel()):
            means.append(m)
            covariances.append(c)
            weights.append(w)

        f_means.append(means)
        f_covariances.append(covariances)
        f_weights.append(weights)
    return np.array(f_means), np.array(f_covariances), np.array(f_weights)




def harmonics_poly(frag, freq, fs):
    heatmap = librosa.amplitude_to_db(np.abs(librosa.stft(frag,n_fft=noise_window)))
    freq_index = np.linspace(1, int(15500//freq), int(15500//freq))*int(freq/fs*noise_window)
    
    f_harmonics = []
    for t in range(len(heatmap[0])):
        harmonics = []
        for i in freq_index:
            print(i)
            harmonics.append(heatmap[int(i)][t])
            pass
        f_harmonics.append(harmonics)
    return f_harmonics

if __name__ == "__main__":
    pass


