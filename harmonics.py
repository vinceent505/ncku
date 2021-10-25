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

def gaussian(x, height, center, width, offset):
    return height*np.exp(-(x - center)**2/(2*width**2)) + offset
def three_gaussians(x, h1, c1, w1, h2, c2, w2, h3, c3, w3, offset):
    return (gaussian(x, h1, c1, w1, offset=0) +
        gaussian(x, h2, c2, w2, offset=0) +
        gaussian(x, h3, c3, w3, offset=0) + offset)

errfunc3 = lambda p, x, y: (three_gaussians(x, *p) - y)**2

# Define simple gaussian
def gauss_function(x, amp, x0, sigma):
    return amp * np.exp(-(x - x0) ** 2. / (2. * sigma ** 2.))


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
        # print(test_data)
        # print(test_data.shape)
        # plt.plot(test_data)
        # plt.title('after smoothing, window size = 9, polyorder = 3')
        # plt.show()

        # Find peak
        peaks, _ = find_peaks(test_data, height=-15)
        # plt.plot(test_data)
        # plt.plot(peaks, test_data[peaks], "x")
        # plt.plot(np.zeros_like(test_data), "--", color="gray")
        # plt.title('Find Peaks, number of peaks = %d'%(len(peaks)))
        # plt.show()

        # Fit GMM
        gmm = GaussianMixture(n_components=len(peaks)-3, covariance_type="full", tol=0.001)
        random_data = [np.random.uniform(i, i+1, int(1000*aa)) for i, aa in enumerate(test_data)]
        data = []
        for i in random_data:
            for ii in i:
                data.append(ii)

        gmm = gmm.fit(X=np.expand_dims(np.array(data), 1)) #要2D
        # print(gmm)

        # # Evaluate GMM
        # # gmm_x = np.linspace(-(envelope_value[10].max()), envelope_value[10].max(), 337)
        # gmm_x = np.linspace(0, len(test_data), len(test_data)+1)
        # gmm_y = np.exp(gmm.score_samples(gmm_x.reshape(-1, 1)))
        # # Construct function manually as sum of gaussians
        # gmm_y_sum = np.full_like(gmm_x, fill_value=0, dtype=np.float32)
        means = []
        covariances = []
        weights = []
        for m, c, w in zip(gmm.means_.ravel(), gmm.covariances_.ravel(), gmm.weights_.ravel()):
            # print("Means: ", m, "Covariances: ", c, "Weights: ", w)
            means.append(m)
            covariances.append(c)
            weights.append(w)
            # gauss = gauss_function(x=gmm_x, amp=1, x0=m, sigma=np.sqrt(c))
            # gmm_y_sum += gauss / np.trapz(gauss, gmm_x) * w

        f_means.append(means)
        f_covariances.append(covariances)
        f_weights.append(weights)
    return np.array(f_means), np.array(f_covariances), np.array(f_weights)



        # '''One has to normalise the individual components, not the sums''' 
        # # Make regular histogram
        # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[8, 5])

        # ax.plot(gmm_x*21.5, gmm_y, color="crimson", lw=4, label="GMM")
        # ax.plot(gmm_x*21.5, gmm_y_sum, color="black", lw=4, label="Gauss_sum", linestyle="dashed")

        # # Annotate diagram
        # ax.set_ylabel("Probability density")
        # ax.set_xlabel("Arbitrary units")

        # # Make legend
        # plt.legend()

        # plt.show()



        # poly_z = np.poly1d(np.polyfit(poly_x, poly_y, 20))
        # plt.plot(poly_x, poly_y, '-', t, poly_z(t), '-')

        # z = savgol_filter(poly_y, 21, 2)
        # t = np.linspace(0, 15500, len(z))
        # for j in range(10):
        #     z_part = z[int(j*len(z)/10):int((j+1)*len(z)/10)]
        #     t_part = t[int(j*len(t)/10):int((j+1)*len(t)/10)]
        #     guess3 = [50, 180*j, 1, 50, 360*j, 1, 50, 540*j, 1, 1]
        #     optim3, success = optimize.leastsq(errfunc3, guess3[:], args=(t_part, z_part))


        #     plt.plot(poly_x, poly_y, '-', t_part, z_part, '-')
        #     plt.plot(t_part, three_gaussians(t_part, *optim3),lw=3, c='b', label='fit of 3 Gaussians')
        #     plt.show()
        #     pass
        # pass

    pass



def harmonics_poly(frag, freq, fs):
    heatmap = librosa.amplitude_to_db(np.abs(librosa.stft(frag,n_fft=noise_window)))
    freq_index = np.linspace(1, int(15500//freq), int(15500//freq))*int(freq/fs*noise_window)
    # print(freq_index)
    x = np.linspace(1, len(heatmap)*fs/noise_window, len(heatmap))
    
    a = []
    for i in range(len(heatmap)):
        a.append(heatmap[i][0])
    # plt.plot(x, a)
    f_harmonics = []
    for t in range(len(heatmap[0])):
        harmonics = []
        for i in freq_index:
            harmonics.append(heatmap[int(i)][t])
            pass
        f_harmonics.append(harmonics)
        # plt.plot(freq_index*fs/noise_window, harmonics)

        # plt.show()
    return f_harmonics

if __name__ == "__main__":
    pass


