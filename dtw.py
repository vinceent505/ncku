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
import pickle
import resampy
import crepe
from scipy.signal import savgol_filter
import aubio



show = False
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


       

def check_start_time(start, data, f0, num, order, prev_start):
    onset_env = librosa.onset.onset_strength(y=np.array(data), sr=44100)

    if order[num]==0:
        return 0
    onset = librosa.onset.onset_detect(onset_envelope=onset_env, sr=44100)
    times = librosa.times_like(onset_env, sr=44100)

    dest_freq = 500
    ratio = f0/dest_freq
    d_upsample = resampy.resample(np.array(data), 44100, 44100*ratio)
    d = resampy.resample(d_upsample, 44100, 16000)
    # scipy.io.wavfile.write("out/shift"+str(self.num)+".wav", 16000, d)
    _, p, _, _ = crepe.predict(d, 16000, viterbi=True, verbose=0, step_size = 10*ratio)
    p = p*ratio

    pp = []
    for k in range(len(p)-1):
        if k==len(p)-5:
            break
        pp.append(abs(p[k+1]-p[k]))

    m = np.argmax(np.array(pp))


    pitch_idx = []
    for n, i in enumerate(times[onset]):
        if len(onset)>2 and n==len(onset)-1:
            break
        pitch_idx.append(round(i/0.01))
    pitch_idx = np.absolute(np.array(pitch_idx)-m)

    return start+times[onset[np.argmin(pitch_idx)]]

    



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









def dtw(musician_filename, score_filename, score, music_name, musician_name):
    note = []
    start_time = []
    for i in score:
        note.append(score[i]["name"])
        start_time.append(score[i]["start"])




    x_1, fs = librosa.load(score_filename, sr=16000)

    x_2, fs = librosa.load(musician_filename, sr=16000)

    for i in start_time:
        if i>len(x_1)/fs:
            start_time.remove(i)
    print("length:", len(start_time))

    n_fft = 4096
    hop_size = 64

    start_idx = []
    for i in start_time:
        start_idx.append(int(i/hop_size*fs)*3)
    start_idx_np = np.array(start_idx)

    x_1_chroma = librosa.feature.chroma_cqt(y=x_1, sr=fs, tuning=0, norm=2,
                                            hop_length=hop_size, n_chroma = 36, bins_per_octave = 72)
    x_2_chroma = librosa.feature.chroma_cqt(y=x_2, sr=fs, tuning=0, norm=2,
                                            hop_length=hop_size, n_chroma = 36, bins_per_octave = 72)

    _, wp = librosa.sequence.dtw(X=x_1_chroma, Y=x_2_chroma, metric='cosine')
    wp_s = np.array(wp)[::-1] * hop_size / fs



    first = False
    final_time = []
    for j in start_time:
        for i in wp_s:
            if round(i[0], 2) == round(j, 2) and i[1]>0:
                final_time.append([i[1]])
                break


    # for i in final_time:
    #     final_csv.append([i])

    if show:
        D, wp = librosa.sequence.dtw(X=x_1_chroma, Y=x_2_chroma, metric='cosine')
        wp_s = np.asarray(wp) * hop_size / fs

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        librosa.display.specshow(D, x_axis='time', y_axis='time',
                                cmap='gray_r', hop_length=hop_size)
        imax = ax.imshow(D, cmap=plt.get_cmap('gray_r'),
                        origin='lower', interpolation='nearest', aspect='auto')
        ax.plot(wp_s[:, 1], wp_s[:, 0], marker='o', color='r')
        ax.set(xlabel="MIDI Synthesized Audio Time[second]", ylabel="Violinist\'s Recording Time[second]")
        plt.title('Warping Path on Acc. Cost Matrix $D$')
        plt.colorbar()
        plt.show()


        fig = plt.figure(figsize=(16, 8))

        # Plot x_1
        plt.subplot(2, 1, 1)
        librosa.display.waveplot(x_1, sr=fs, x_axis="s")
        plt.title('Midi Synthesized')
        plt.ylabel("Amplitude")
        plt.xlabel("Time[second]")
        ax1 = plt.gca()

        # Plot x_2
        plt.subplot(2, 1, 2)
        librosa.display.waveplot(x_2, sr=fs)
        plt.title('Musician Performance')
        plt.ylabel("Amplitude")
        plt.xlabel("Time[second]")
        ax2 = plt.gca()

        plt.tight_layout()

        trans_figure = fig.transFigure.inverted()
        lines = []

        count = 0
        first = False
        for j in start_time:
            for i in wp_s:
                if round(i[0], 2) == round(j, 2) and i[1]>0:
                    tp1 = i[0]
                    tp2 = i[1]
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
        plt.show()
        pass





    name = ["start"]
    start_time_csv = "dtw_output_csvs/" + musician_name + "/" + musician_name + "_" + music_name + ".csv"
    with open(start_time_csv, "w") as f:
        writer = csv.writer(f)
        writer.writerow(name)
        writer.writerows(final_time)
    return start_time_csv


if __name__ == "__main__":
    music_name = "S1_4_2nd"
    musician_name = "Henryk" 

    perf_filepath = "input/audio/perf/" + musician_name + "/"
    score_filepath = "input/audio/score/" + musician_name + "/"
    dtw_csvdir = "dtw_output_csvs/" + musician_name + "/"
    musician_filename = perf_filepath + musician_name + "_" + music_name + "_perf.wav" # musician original audio
    score_filename = score_filepath + musician_name + "_" + music_name + "_score.wav" # score synthesis audio
    score_data = "input/data/"+musician_name+"/"+musician_name+"_"+music_name+".pickle" # score midi data    
    with open(score_data, "rb") as f:
        score = pickle.load(f)

    dtw(musician_filename, score_filename, score, music_name, musician_name) 



