from scipy.io import wavfile
import audiolazy
import pandas as pd
import mido
from mido import Message, MidiFile, MidiTrack
import channel_num


csv_path = "csvs/"

def mapping(sourceValue, sourceRangeMin, sourceRangeMax, targetRangeMin, targetRangeMax):
    return targetRangeMin + ((targetRangeMax - targetRangeMin) * (sourceValue - sourceRangeMin)) / (sourceRangeMax - sourceRangeMin)


if __name__ == "__main__":


    num_input = channel_num.find_channel()

    file_path = "envelope_out/"
    envelope=[]
    for i in range(20):
        note_path = file_path + "envelope" + str(i) + ".wav"
        fs, envelope_data = wavfile.read(note_path)
        envelope.append(envelope_data)

    csv_path = csv_path + "start_end.csv"
    csv = pd.read_csv(csv_path)
    csv = csv[0:-1]



    start_file = pd.read_csv(csv_path + "tart_time.csv")
    end_file = pd.read_csv(csv_path + "end_time.csv")


    midinote=[]
    for note in csv["note"]:
        if note[1] == "-":
            temp= note[0] + "b" + note[2]
            midinote.append(audiolazy.str2midi(temp))
        else:
            midinote.append(audiolazy.str2midi(note))

    tempo = mido.bpm2tempo(120)
    mido.second2tick(1, 480, tempo)
    start=[]
    gap=[]
    end = []
    first=True
    for time in zip(start_file["start"], end_file["end"]):
        start.append(mido.second2tick(time[0], 480, tempo))
        end.append(mido.second2tick(time[1], 480, tempo))
        tick = mido.second2tick(time[1] - time[0], 480, tempo)
        gap.append(tick)


    outfile = MidiFile()
    track = MidiTrack()
    outfile.tracks.append(track)
    track.append(Message('control_change', channel = 0, control = 15, value = 0, time = 0 ))
    track.append(Message('control_change', channel = 0, control = 55, value = 10, time = 0 ))
    track.append(Message('control_change', channel = 0, control = 56, value = 127, time = 0 ))
    track.append(Message('control_change', channel = 0, control = 57, value = 127, time = 0 ))
    track.append(Message('control_change', channel = 0, control = 58, value = 10, time = 0 ))
    first=True

    for num_channel in range(num_input):
        prev_value = 0
        outfile = MidiFile()
        track = MidiTrack()
        outfile.tracks.append(track)
        track.append(Message('control_change', channel = 0, control = 15, value = 0, time = 0 ))
        track.append(Message('control_change', channel = 0, control = 55, value = 10, time = 0 ))
        track.append(Message('control_change', channel = 0, control = 56, value = 127, time = 0 ))
        track.append(Message('control_change', channel = 0, control = 57, value = 127, time = 0 ))
        track.append(Message('control_change', channel = 0, control = 58, value = 10, time = 0 ))
        count = 0
        pre_note = midinote[num_channel]
        for j in range(len(midinote)):
            if(j%num_input)==num_channel:
                if(j==num_channel):
                    track.append(Message('note_on', channel = 0, note=midinote[j], velocity=127, time = int(start[j])))
                else:
                    if(pre_note == midinote[j]):
                        track.append(Message('note_on', channel = 0, note=midinote[j], velocity=127, time = int(start[j])-int(end[j-num_input])))
                    else:
                        track.append(Message('note_off', note=midinote[j-num_input], velocity=127, time = int(start[j])-int(end[j-num_input])))
                        track.append(Message('note_on', channel = 0, note=midinote[j], velocity=127, time = 0))
                count = 0

            temp = len(envelope[j]) / int(gap[j])
            expression = 0
            for i in range(0, len(envelope[j]), int(temp)):
                expression = int(mapping(envelope[j][i], 0, 1, 0, 127))
                if(j%num_input)==num_channel:
                    track.append(Message('control_change', channel = 0, control = 11, value = expression, time = 1))
                else:
                    count += 1
            pre_note = midinote[j]
        outfile.save(filename = "bach_test_%d.mid" %num_channel)