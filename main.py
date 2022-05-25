import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import scipy.io.wavfile
import pandas as pd
import endtime
import time
import csv
import harmonics
import dtw
from tqdm import tqdm
import note
import envelope
import pickle
import multiprocessing as mp
import sys
import syn
import paper_fig



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

note_list = []

def get_feature(data, fs, startfile, endfile, notefile):
    
    for num in tqdm(range(len(startfile))):
        s = startfile[num]
        e = endfile[num]
        n = notefile[num]
        if(n[1] == '#'):
            f0 = frequency_list[pitch_list.index(n[0]+n[2])+1]
            pass
        else:
            f0 = frequency_list[pitch_list.index(n)]
        frag = data[int(s*fs):int(e*fs)]
        # print(s)
        # print(e)
        note_list.append(note.note(num, n, frag, fs, f0, s, e))
    




def main(do_dtw = True, do_end = True):
    music_name = "presto_cut"
    # musician_name = "Milstein" 
    # musician_name = "Henryk" 
    musician_name = "Hilary" 

    perf_filepath = "input/audio/perf/" + musician_name + "/"
    score_filepath = "input/audio/score/" + musician_name + "/"
    dtw_csvdir = "dtw_output_csvs/" + musician_name + "/"


    musician_filename = perf_filepath + musician_name + "_" + music_name + "_perf.wav" # musician original audio
    score_filename = score_filepath + musician_name + "_" + music_name + "_score.wav" # score synthesis audio
    score_data = "input/data/"+music_name+".pickle" # score midi data


    
    with open(score_data, "rb") as f:
        score = pickle.load(f)
    


    start_csv = "dtw_output_csvs/manual/" + musician_name + "/" + musician_name + "_" + music_name + "_manual.csv" 
    ########################
    if do_dtw:
        start_csv =  "dtw_output_csvs/" + musician_name + "/" + musician_name + "_" + music_name + ".csv"
    ########################
    # for i in score:
    #     print(score[i]["start"])
    start_time = []
    end_time = []
    pitch_contour = []

    note_name = []

    x_1, fs = librosa.load(musician_filename, sr=44100)
    start_file = pd.read_csv(start_csv)




    order_time = []
    for i in score:
        order_time.append(score[i]["start"])
    order = dtw.start_time_order(order_time)[:len(start_file["start"])]


    print("Normalize Start!!")
    x_1 = envelope.normalize(x_1, -1, 1)
    print("Normalize Done!!")


    





    prev_start = 0.0
    for n, i in enumerate(start_file["start"]):
        no = score[n]["name"]
        note_name.append(no)

        if(no[1] == '#'):
            f0 = frequency_list[pitch_list.index(no[0]+no[2])+1]
            pass
        else:
            f0 = frequency_list[pitch_list.index(no)]


        for o_n, o in enumerate(order):
            if order[n] == o-1:
                nxt_start = start_file["start"][o_n]
                break
            nxt_start = len(x_1)/fs
        if n>0:
            print(i-0.02)
            paper_fig.onset_fig(np.array(x_1[int((i-0.02)*44100):int((i+0.11)*44100)]), 44100)
            if i>0.05:
                i -= 0.05
            s = dtw.check_start_time(i, x_1[int((i)*fs):int((nxt_start)*fs)], f0, n, order, prev_start)
        else:
            s = 0.0
        start_time.append(s)
        if order[n] == order[-1]:
            print(s)
            print(n)
            continue
        elif order[n+1] != order[n]:
            prev_start = s
        print(s)
        print(n)


    final_time = []
    for i in start_time:
        final_time.append([i])
    col_name = ["start"]
    start_time_csv = "dtw_output_csvs/manual/" + musician_name + "_" + music_name + "_manual" +  time.strftime("%Y%m%d-%H%M%S") + ".csv"
    with open(start_time_csv, "w") as f:
        writer = csv.writer(f)
        writer.writerow(col_name)
        writer.writerows(final_time)


    # print(note_name)
    end_csv = "dtw_output_csvs/" + musician_name + "/"+ musician_name + "_" + music_name + "end.csv"
    if do_end:
        end_csv = endtime.find_endtime(musician_filename, note_name, order, start_time, musician_name, music_name)

    end_file = pd.read_csv(end_csv)


    get_feature(x_1, fs, start_time, end_file["end"], note_name)






    #output    
    # for i in note_list:
    #     plt.plot(np.linspace(i.start, i.end, len(i.pitch)), i.pitch)
    # plt.show()  

    output_list = []
    for i in note_list:
        output_list.append({"num": i.num,"name": i.name,"frequency": i.base_freq, "start": i.start,"end": i.end,"pitch": i.pitch,"envelope": i.envelope, "adsr": i.adsr, "harmonics": i.harmonics, "noise":i.noise})


    with open("output/pickle/" + musician_name + "_" + music_name + ".pickle", "wb") as f:
        d = dict(enumerate(output_list))
        pickle.dump(d, f)

    syn.syn(musician_name, music_name)
    
                 
if __name__ == "__main__":
    if len(sys.argv)>1:
        if sys.argv[1] == "1":
            start = True
        else:
            start = False
        
        if sys.argv[2] == "1":
            end = True
        else:
            end = False
        print("NO DTW")
        main(start, end)
    else:
        print("DTW")
        main()

