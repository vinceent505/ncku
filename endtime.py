import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import time

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

window_size = 2048

def find_endtime(musician_filename, compare_csv, start_list):
	data, fs = librosa.load(musician_filename)
	note_file = pd.read_csv(compare_csv)

	note_list = []
	count = 0
	for i in note_file['note']:
		if i[1] == '#':
			note_list.append(pitch_list[pitch_list.index(i[0]+i[2])+1])
		else:
			note_list.append(i)
		count += 1


	end_list = []
	for i, t in enumerate(start_list):
		start = t
		note = note_list[i]
		freq = frequency_list[pitch_list.index(note)]
		next_index = -1
		stop = False
		for j, next_name in enumerate(note_list[i+1:i+3]):
			if next_name == note:
				if i==len(start_list):
					next_start = len(data)/fs
				next_index = j+i+1
				end_list.append(start_list[next_index])
				stop = True
				break
		if stop:
			continue

		if(len(data)/fs - start)>5:
			next_start = start+5
		else:
			next_start = len(data)/fs

		frag = data[int((start+0.06)*fs) : int(next_start*fs)]#time bin = 4 * ( file_length / Ts * windowsTime)
		heatmap = librosa.amplitude_to_db(np.abs(librosa.stft(frag,n_fft=window_size))) #frequency bins = Frames frequency gap fs/n_fft.

		freq_index = int(freq/fs*window_size)
		end_time = next_start
		for j in range(len(heatmap[0])):
			f = True
			if heatmap[freq_index][j] < 20:
				
				x = start+(next_start-start-0.06)/len(heatmap[0])*(j+1)+0.06
				if i<len(start_list)-1:
					if x > start_list[i+1]:
						end_time = x
						f = False
						break
					else:
						continue
		# print("start:", start)
		# print("end:", end_time)
		# print("___________")
		
		if i<len(start_list)-1:
			if f:
				end_time = start_list[i+1]
		else:
			end_time = len(data)/fs
		end_list.append(end_time)
	

	final_csv = []
	for i in end_list:
		final_csv.append([i])


	end_time_csv = "dtw_output_csvs/no1_end" +  time.strftime("%Y%m%d-%H%M%S") + ".csv"
	with open(end_time_csv, "w") as f:
		writer = csv.writer(f)
		writer.writerow(["end"])
		writer.writerows(final_csv)

	return end_time_csv
		



if __name__ == "__main__":

	find_endtime()
			

	# # plt.plot(librosa.amplitude_to_db(np.array(ll)))
	# # plt.show()