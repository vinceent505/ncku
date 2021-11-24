import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pandas as pd
import envelope
import filter
import endtime
import pitch
import time
import csv
import harmonics
import dtw
import tqdm

output_filename = "final_output_csvs/Bach_sonata_no1.csv"

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

                 
if __name__ == "__main__":
    # dtw.dtw()

    note = []
    start_time = []
    end_time = []
    pitch_contour = []


    file = pd.read_csv("Bach_Sonata_No1.csv")
    for i in file['note']:
        note.append(i)

    x_1, fs = librosa.load('Bach/bach_Hil.wav', sr=44100)
    print("Normalize Start!!")
    x_1 = envelope.normalize(x_1, -1, 1)
    print("Normalize Done!!")

    file = pd.read_csv("start_end.csv")
    start_file = pd.read_csv("start_time.csv")
    for i in start_file['start']:
        start_time.append(i)
    endtime.find_endtime(start_time)
    end_file = pd.read_csv("end_time.csv")
    for i in end_file['end']:
        end_time.append(i)

    n_start_time = []
    t = []
    time_1 = time.time()    
    note_num = len(note)

    for count in tqdm.trange(note_num):
        time_start = time.time()
        if(note[count][1] == '#'):
            f0 = frequency_list[pitch_list.index(note[count][0]+note[count][2])+1]
            pass
        else:
            f0 = frequency_list[pitch_list.index(note[count])]
        frag = x_1[int(start_time[count]*fs):int(end_time[count]*fs)]

        k = filter.harmonics_filter(f0, frag, fs, count)

        # harmonics_ = harmonics.harmonics_poly(frag, f0, fs) #正式執行再解除註解
        # means, covariances, weight = harmonics.noise_poly(k, f0, fs)


        a = filter.fft_filter(frag, f0, count, fs)
        frag_filt_1 = filter.filt(frag, count, fs, f0, note[count])

        envelope.find_env_curve(envelope.envelope(frag_filt_1, count, fs, 10, 7), frag_filt_1)

        p = pitch.pitch_dec(a, count, fs, f0)
        
        pitch_onetone = []
        for i, j in enumerate(p):
            pitch_onetone.append(j)
        envelope.env_write(count, envelope.envelope(frag_filt_1, count, fs, 10, 7))
        pitch_contour.append(pitch_onetone)
        t.append(time.time() - time_start)
    for j, i in enumerate(pitch_contour):
        p = np.linspace(start_time[j], end_time[j], len(i))
        plt.plot(p, i)


    col_names = ["num", "note", "start", "end", "pitch"]

    # d = {"note": note,
    #      "start": start_time,
    #      "end": end_time,
    #      "pitch": pitch_contour

    # }

    w_file = open(output_filename, 'w')
    fieldnames = col_names
    writer = csv.DictWriter(w_file, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(note_num):
        d = {"num": i, "note": note[i], "start": start_time[i], "end": end_time[i], "pitch": pitch_contour[i]}
        writer.writerow(d)
    w_file.close()


    plt.show()

    # print("Total Time: ", time.time()-time_1)
    # plt.figure()
    # plt.plot(t)
    # plt.show()