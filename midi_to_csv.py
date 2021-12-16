import mido 
import sys
import csv
import audiolazy

def main():
	filename = str(sys.argv[1])
	mid = mido.MidiFile(filename)
	start = []
	end = []
	note = []
	first = True
	noteon_midi = []
	for i in mid:
		if i.type == "note_on":
			noteon_midi.append(i)

	for i in range(0, len(noteon_midi), 2):
		note.append(noteon_midi.note)
		if first:
			start.append(noteon_midi[i].time)
			end.append(noteon_midi[i+1].time + start[i/2])
			first = False
			pass
		else:
			start.append(end[i/2-1])
			end.append(noteon_midi[i+1].time + start[i/2])
			pass
		pass
	t = []
	print(len(start))
	print(len(note))
	for i in range(len(time)):
		if i>0:
			time[i] += time[i-1]
	for i in time:
		t.append([i])
	tt = []
	for i in range(len(t)):
		tt.append([t[i][0] - t[0][0], audiolazy.midi2str(note[i])])
	
	name = ["start", "note"]
	with open("midi_output_csvs/" + filename[:-4] + ".csv", "w") as f:
		writer = csv.writer(f)
		writer.writerow(name)
		writer.writerows(tt)

	pass

if __name__ == "__main__":
	main()