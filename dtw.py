from __future__ import print_function
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import librosa
import librosa.display
import pandas as pd
import csv
from scipy import signal
import scipy.io.wavfile

import resampy
import crepe
from scipy.signal import savgol_filter


HOP_SIZE = 512
OVERLAP = 4096-512
window_size = 4096
overlap = OVERLAP
pitch_list = ['C0', 'D-0', 'D0', 'E-0', 'E0', 'F0', 'G-0', 'G0', 'A-0', 'A0', 'B-0', 'B0'
        ,'C1', 'D-1', 'D1', 'E-1', 'E1', 'F1', 'G-1', 'G1', 'A-1', 'A1', 'B-1', 'B1'
        ,'C2', 'D-2', 'D2', 'E-2', 'E2', 'F2', 'G-2', 'G2', 'A-2', 'A2', 'B-2', 'B2'
        ,'C3', 'D-3', 'D3', 'E-3', 'E3', 'F3', 'G-3', 'G3', 'A-3', 'A3', 'B-3', 'B3'
        ,'C4', 'D-4', 'D4', 'E-4', 'E4', 'F4', 'G-4', 'G4', 'A-4', 'A4', 'B-4', 'B4'
        ,'C5', 'D-5', 'D5', 'E-5', 'E5', 'F5', 'G-5', 'G5', 'A-5', 'A5', 'B-5', 'B5'
        ,'C6', 'D-6', 'D6', 'E-6', 'E6', 'F6', 'G-6', 'G6', 'A-6', 'A6', 'B-6', 'B6'
        ,'C7', 'D-7', 'D7', 'E-7', 'E7', 'F7', 'G-7', 'G7', 'A-7', 'A7', 'B-7', 'B7']
frequency_list = np.array([12.35, 17.32, 18.35, 19.45, 20.6, 21.83, 23.12, 24.5, 25.96, 17.5, 29.14, 30.87
                 , 32.7, 34.65, 36.71, 38.89, 41.2, 43.65, 46.25, 49.0, 51.91, 55.0, 58.27, 61.74
                 , 65.40, 69.29, 73.41, 77.78, 82.4, 87.3, 92.49, 98.0, 103.83, 110.0, 116.54, 123.47
                 , 130.81, 138.59, 146.83, 155.56, 164.81, 174.61, 185.0, 196.0, 207.65, 220.0, 233.08, 246.94
                 , 261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99, 392.0, 415.30, 440.0, 466.16, 493.88
                 , 523.25, 554.37, 587.33, 622.25, 659.26, 698.46, 739.99, 783.99, 830.61, 880.0, 923.33, 987.77
                 , 1046.5, 1108.73, 1174.66, 1244.51, 1319.51, 1396.91, 1479.98, 1567.99, 1661.22, 1760.0, 1864.66, 1975.53
                 ,2093.0, 2217.46, 2349.32, 2489.02, 2637.02, 2793.83, 2959.96, 3135.96, 3322.44, 3520.0, 3729.31, 3951.07])


        
def stft_fundamental(data, base_freq):
    l = False
    if len(data)<4096:
        data_padding = np.zeros(4096)
        data_padding[:len(data)] += data
        input_data = data_padding
        l = True
    else:
        input_data = data
    f, t, Zxx = signal.stft(input_data, 44100, nperseg=window_size, noverlap=overlap)
    for i, ii in enumerate(Zxx):#f
        for k in range(1):
            if (k+1)*base_freq/1.04 > 44100/2:
                break
            if f[i]==0 or(f[i]>(k+1)*base_freq*1.04 or f[i]<(k+1)*base_freq/1.04):
                for j, jj in enumerate(ii):#t
                    Zxx[i][j]=0
                break
            if f[i]<k*base_freq*1.04:
                break

    _, s = signal.istft(Zxx, 44100, nperseg=window_size, noverlap=overlap)
    
    return s[:len(data)]


def check_start_time(start, end, data, f0, num):
    data = stft_fundamental(data, f0)
    o_env = librosa.onset.onset_strength(data, 44100)
    data = np.array(data)
    if start==0:
        return 0

    onset = librosa.onset.onset_detect(y=data, sr=44100)
    times = librosa.times_like(o_env, sr=44100)
    if len(onset) == 0:
        return start

    if len(onset)>1:
        if o_env[onset[0]]*2.5<o_env[onset[1]] and onset[1] != len(o_env)-1 and onset[1] != len(o_env)-2:
    #         print(num, "_____________")
    #         print(onset)
    #         print(start+times[onset])
    #         print("start: ", start+times[onset[1]])
    #         plt.plot(o_env)
    #         plt.show()
            return start + times[onset[1]]
        else:
    #         print(num, "_____________")
    #         print(onset)
    #         print(start+times[onset])
    #         print("start: ", start+times[onset[0]])
    #         plt.plot(o_env)     
    #         plt.show()
            return start + times[onset[0]]

    # # for i in onset:
    # #     print(i)
    # #     print(start+times[i])
    # #     print(o_env[i])
        
    # print(num, "_____________")
    # print("start: ", start+times[onset[0]])
    # plt.plot(o_env)
    # plt.show()
    return start+times[onset[0]]


def start_time_order(time):
    order = []
    o = 0
    for i, j in enumerate(time):
        if i==0:
            order.append(o)
        else:
            if time[i] == time[i-1]:
                order.append(o)
            else:
                o += 1
                order.append(o)
    return order









def dtw(musician_filename, score_filename, score):
    note = []
    start_time = []
    for i in score:
        note.append(score[i]["name"])
        start_time.append(score[i]["start"])

    print("length:", len(start_time))

    x_1, fs = librosa.load(score_filename, sr=16000)

    x_2, fs = librosa.load(musician_filename, sr=16000)

    n_fft = 4096
    hop_size = 64

    start_idx = []
    for i in start_time:
        start_idx.append(int(i/hop_size*fs)*3)
    start_idx_np = np.array(start_idx)
    print(len(start_idx_np))

    x_1_chroma = librosa.feature.chroma_cqt(y=x_1, sr=fs, tuning=0, norm=2,
                                            hop_length=hop_size, n_chroma = 36, bins_per_octave = 72)
    x_2_chroma = librosa.feature.chroma_cqt(y=x_2, sr=fs, tuning=0, norm=2,
                                            hop_length=hop_size, n_chroma = 36, bins_per_octave = 72)

    D, wp = librosa.sequence.dtw(X=x_1_chroma, Y=x_2_chroma, metric='cosine')

    wp_s = np.asarray(wp) * hop_size / fs

    fig = plt.figure(figsize=(16, 8))

    # Plot x_1
    plt.subplot(2, 1, 1)
    librosa.display.waveplot(x_1, sr=fs)
    plt.title('Synthesis Version $X_1$')
    ax1 = plt.gca()

    # Plot x_2
    plt.subplot(2, 1, 2)
    librosa.display.waveplot(x_2, sr=fs)
    plt.title('Hil Version $X_2$')
    ax2 = plt.gca()

    plt.tight_layout()

    trans_figure = fig.transFigure.inverted()
    lines = []
    arrows = 30
    points_idx = np.int16(np.round(np.linspace(0, wp.shape[0] - 1, arrows)))


    count = 0
    first = False
    dtw_start_time = []
    # for tp1, tp2 in zip((wp[points_idx, 0]) * hop_size, (wp[points_idx, 1]) * hop_size):
    for i in wp * hop_size / fs:
        tp1 = 0
        tp2 = 0
        for j in start_idx_np:
            if round(i[0], 2) == round(j* hop_size / fs / 3, 2) and i[1] > 0:
                if(first):
                    break
                if(i[0] == 0):
                    first = True
                tp1 = i[0]
                tp2 = i[1]
                dtw_start_time.append(tp2)
                count += 1
                break

        # get position on axis for a given index-pair
        coord1 = trans_figure.transform(ax1.transData.transform([tp1, 0]))
        coord2 = trans_figure.transform(ax2.transData.transform([tp2, 0]))
        # draw a line
        line = matplotlib.lines.Line2D((coord1[0], coord2[0]),
                                    (coord1[1], coord2[1]),
                                    transform=fig.transFigure,
                                    color='r')
        lines.append(line)
    fig.lines = lines
    plt.tight_layout()
    dtw_start_time.reverse()
    plt.show()
    tmp = []
    final_time = []
    for i in range(len(dtw_start_time)):
        if(i==len(dtw_start_time)-1):
            final_time.append(min(tmp))
            break
        else:
            tmp.append(dtw_start_time[i])
            if((dtw_start_time[i+1]-dtw_start_time[i])>0.01):
                final_time.append(min(tmp))
                tmp = []
    print(len(final_time))
    final_csv = []

    start_count = 0
    first = True
    for i, j in enumerate(final_time):
        if first:
            final_csv.append([0])
            first = False
        else:
            final_csv.append([final_time[i]])
        for count in range(start_count+1, len(start_time)):
            if start_time[start_count] == start_time[count]:
                final_csv.append([final_time[i]])
                start_count += 1
                continue
            else:
                break
        start_count += 1
        pass
    # for i in final_time:
    #     final_csv.append([i])

    name = ["start"]
    start_time_csv = "dtw_output_csvs/no1_start_" +  time.strftime("%Y%m%d-%H%M%S") + ".csv"
    with open(start_time_csv, "w") as f:
        writer = csv.writer(f)
        writer.writerow(name)
        writer.writerows(final_csv)
    return start_time_csv


if __name__ == "__main__":
    dtw() 



