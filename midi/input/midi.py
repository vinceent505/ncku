import mido

filename = "double.mid"
mid = mido.MidiFile(filename)
for i in mid:
    if i.type!="note_on":
        print(i)