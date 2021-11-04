import mido 
import sys
import csv

def main():
	filename = str(sys.argv[1])
	mid = mido.MidiFile(filename)
	time = []
	note = []
	for i in mid:
		if i.type == "note_on":
			note.append([i.note])

		if i.time!=0:
			time.append(i.time)
	t = []
	print(note)
	for i in range(len(time)):
		if i>0:
			time[i] += time[i-1]
	for i in time:
		t.append([i])
	# print(len(time))

	name = ["start_absolute_time"]
	with open("midi-output-csvs/double.csv", "w") as f:
		writer = csv.writer(f)
		writer.writerow(name)
		writer.writerows(t)

	pass

if __name__ == "__main__":
	main()