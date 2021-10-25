# Code source: Stefan Balke
# License: ISC
# sphinx_gallery_thumbnail_number = 4

from __future__ import print_function
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import librosa
import librosa.display
import pandas as pd
import csv

csv_path = "csvs/"
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
                 , 130.81, 138.59, 146.83, 155.56, 164.81, 174.61, 185.0, 196.0, 207.65, 220.0, 233.08, 269.94
                 , 261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99, 392.0, 415.30, 440.0, 466.16, 493.88
                 , 523.25, 554.37, 587.33, 622.25, 659.26, 698.46, 739.99, 783.99, 830.61, 880.0, 923.33, 987.77
                 , 1046.5, 1108.73, 1174.66, 1244.51, 1319.51, 1396.91, 1479.98, 1567.99, 1661.22, 1760.0, 1864.66, 1975.53
                 ,2093.0, 2217.46, 2349.32, 2489.02, 2637.02, 2793.83, 2959.96, 3135.96, 3322.44, 3520.0, 3729.31, 3951.07])

def dtw():
    file = pd.read_csv("midi-output-csvs/double.csv")
    note = []
    start_time = []
    count = 0
    for i in file['note']:
        note.append(i)
        count += 1


    for i in file['start_absolute_time']:
        start_time.append(i)
    print("length:", len(start_time))
    start_time = np.unique(start_time).tolist()

    x_1, fs = librosa.load('Bach/double_syn.wav', sr=44100)

    x_2, fs = librosa.load('Bach/double_hil.wav', sr=44100)

    n_fft = 4410
    hop_size = 64

    x_1_chroma = librosa.feature.chroma_cqt(y=x_1, sr=fs, tuning=0, norm=2,
                                            hop_length=hop_size, n_chroma = 36, bins_per_octave = 72)
    x_2_chroma = librosa.feature.chroma_cqt(y=x_2, sr=fs, tuning=0, norm=2,
                                            hop_length=hop_size, n_chroma = 36, bins_per_octave = 72)

    D, wp = librosa.sequence.dtw(X=x_1_chroma, Y=x_2_chroma, metric='cosine')
    start_idx = []
    for i in start_time:
        start_idx.append(int(i/hop_size*fs)*3)

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
    start_idx_np = np.array(start_idx)

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
            final_time.append(sum(tmp)/len(tmp))
            break
        else:
            tmp.append(dtw_start_time[i])
            if((dtw_start_time[i+1]-dtw_start_time[i])>0.01):
                final_time.append(sum(tmp)/len(tmp))
                tmp = []

    final_csv = []
    for i in final_time:
        final_csv.append([i])

    name = ["start"]
    with open(csv_path + "start_time.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(name)
        writer.writerows(final_csv)


if __name__ == "__main__":
    dtw()