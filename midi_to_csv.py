import mido 
import sys
import csv
import audiolazy

def main():
	filename = str(sys.argv[1])
	print(filename)
	mid = mido.MidiFile(filename)
	start = []
	end = []
	note = []
	first = True
	noteon_midi = []
	for i in mid:
		if i.type == "note_on" or i.type == "note_off":
			noteon_midi.append(i)

	for i in range(0, len(noteon_midi), 2):
		note.append(noteon_midi[i].note)
		if first:
			start.append(noteon_midi[i].time)
			end.append(noteon_midi[i+1].time + start[i//2])
			first = False
		else:
			start.append(end[i//2-1])
			end.append(noteon_midi[i+1].time + start[i//2])

	out = []
	for i in range(len(start)):
		out.append([i, start[i], audiolazy.midi2str(note[i])])
	filename_output = "midi_output_csvs/" + filename[:-4] + ".csv"
	print(filename_output)
	name = ["num","start", "note"]
	with open(filename, "w") as f:
		# writer = csv.writer(f)
		# writer.writerow(name)
		# writer.writerows(out)
		pass

	

if __name__ == "__main__":
	main()