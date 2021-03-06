import pandas as pd
import numpy as np

start_filename = "start_time.csv"
end_filename = "end_time.csv"
note_filename = "Bach_Sonata_No1.csv"


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


def find_channel():
    start_file = pd.read_csv(start_filename)
    end_file = pd.read_csv(end_filename)
    note_file = pd.read_csv(note_filename)

    start_list, end_list, note_list = [], [], []

    for i in start_file["start"]:
        start_list.append(i)

    for i in end_file["end"]:
        end_list.append(i)

    for i in note_file["note"]:
        note_list.append(i)
    
    print(len(start_list))
    print(len(end_list))

    max = 0
    for i in range(len(start_list)):
        channel_num = 0
        start = start_list[i]
        end = end_list[i]
        idx = []
        overlap_time = []
        for j in start_list:
            if j>start and j<end:
                channel_num += 1
                idx.append(start_list.index(j))
                overlap_time.append("start")
                overlap_time.append(j)
                overlap_time.append(start_list.index(j))
        for j in end_list:
            if j>start and j<end:
                overlap_time.append("end")
                overlap_time.append(j)
                overlap_time.append(end_list.index(j))

                append_bool = True
                for k in idx:
                    if end_list[k]==j:
                        append_bool = False
                if(append_bool):
                    channel_num += 1
                    idx.append(end_list.index(j))
        if max <= channel_num:
            max = channel_num

    return max

if __name__ == "__main__":
    m = find_channel()