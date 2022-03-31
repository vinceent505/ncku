import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import time
import scipy.signal as signal
from tqdm import tqdm
from scipy.signal import savgol_filter

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

window_size = 4096
bias = 0.1
hop_size = 64
overlap = window_size-hop_size



def find_endtime(musician_filename, score, order, start_list):
	data, fs = librosa.load(musician_filename, sr=44100)

	# Create a list of the track.
	note_list = []
	count = 0
	for i in score:
		if i[1] == '#':
			note_list.append(pitch_list[pitch_list.index(i[0]+i[2])+1])
		else:
			note_list.append(i)
		count += 1


	# Find the end time of each note.
	end_list = []
	print("Finding End Time!!")
	# for i, start_time in enumerate(start_list):
	for i, start_time in enumerate(tqdm(start_list)):
		if order[i]==order[-1]:
			end_time = len(data)/fs
			end_list.append(end_time)
			continue


		# If the note after next is the same, set it for the temporal cut time for parting whole data into pieces.
		# Else set the start time + bias(s) for the temporal cut time.
		note = note_list[i]
		freq = frequency_list[pitch_list.index(note)]
		find = False
		for j, next_name in enumerate(note_list[i+1:i+5]):
			if next_name == note:
				cut_end_time = start_list[j+i+1]
				find = True
				break
		if not find:
			if(len(data)/fs - start_time)>5:
				cut_end_time = start_time + 5
			else:
				cut_end_time = len(data)/fs


		next_find = False
		for c in range(1, 6):
			if i+c>=len(start_list):
				break
			if start_list[i] == start_list[i+c]:
				continue
			else:
				cut_start_time = start_list[i+c]
				next_find = True
				break
		if not next_find:
			end_list.append(len(data)/fs)
			continue

		# print(cut_start_time, "   ", cut_end_time)

		start_index = int(cut_start_time*fs/hop_size)
		end_index = int(cut_end_time*fs/hop_size)

		freq_index = int(freq/fs*window_size)
		end_time = cut_end_time



		_, _, Zxx = signal.stft(data, fs, boundary=None, nperseg=window_size, noverlap=overlap)
		heatmap = librosa.amplitude_to_db(np.abs(Zxx)) #frequency bins = Frames frequency gap fs/n_fft.
		fundamental = np.array(heatmap[freq_index][start_index:end_index])



		# Find local minimum of energy curve.
		append = False
		max_end = -1.0
		for harmonic in range(1, 3):
			fundamental_contour = np.array(heatmap[freq_index*harmonic][start_index:end_index])
			# plt.plot(fundamental_contour)
			# plt.show()
			if np.min(fundamental_contour) > -40:
				max_end = end_time
				append = True
				break
			for j in range(1, len(fundamental_contour)):
				if fundamental_contour[j] <= -52:
					end_time = j*hop_size/fs+cut_start_time
					local_min = fundamental_contour[j]
					for k in range(j+1, len(fundamental_contour)):
						if fundamental_contour[k] < -55:
							end_time = k*hop_size/fs+cut_start_time
							break
						if fundamental_contour[k]<local_min:
							local_min = fundamental_contour[k]
							end_time = k*hop_size/fs+cut_start_time
							continue
						elif k+1==len(fundamental_contour):
							break
						elif fundamental_contour[k+1]<local_min:
							continue
						else:
							break
					if end_time > max_end:
						max_end = end_time
					# end_list.append(end_time)

					append = True
					break

		if append:
			end_list.append(max_end)
		elif not append and find:
			end_time = cut_end_time
			end_list.append(end_time)
			append = True
		if not append:
			print("ERROR!!!!!!!")
			print(i)
			print(fundamental_contour)
			plt.plot(fundamental_contour)
			plt.show()
			quit()
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