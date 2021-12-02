from scipy.io import wavfile
import audiolazy
import pandas as pd
import mido
from mido import Message, MidiFile, MidiTrack
import channel_num
import math
import matplotlib.pyplot as plt
from scipy.ndimage import interpolation

def mapping(sourceValue, sourceRangeMin, sourceRangeMax, targetRangeMin, targetRangeMax):
    return targetRangeMin + ((targetRangeMax - targetRangeMin) * (sourceValue - sourceRangeMin)) / (sourceRangeMax - sourceRangeMin)




if __name__ == "__main__":


    total_channel_num = channel_num.find_channel()


    csv_path = "final_output_csvs/Bach_sonata_no1.csv"
    csv = pd.read_csv(csv_path)





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
    gap_2 = []
    release = []
    first=True
    for time in zip(csv["start"], csv["end"]):
        start.append(mido.second2tick(time[0], 480, tempo))
        end.append(mido.second2tick(time[1], 480, tempo))
        tick = mido.second2tick(time[1] - time[0], 480, tempo)
        gap.append(int(math.ceil(tick)))
        if first:
            first = False
            previous_start = time[0]
            previous_end = time[1]
        else:
            gap_2.append(int(mido.second2tick(time[0]-previous_start, 480, tempo)))
            release.append(previous_end - time[0])
            previous_start = time[0]
            previous_end = time[1]
    gap_2.append(gap[-1])
    release.append(0.001)


    csv = pd.read_csv(csv_path)
    pitch_tmplist=list(csv["pitch"])
    pitch_list = {}
    for i, item in enumerate(pitch_tmplist):
        # pitch list 
        t = item[1:-1].split(',')
        for j, k in enumerate(t):
            if k=='nan' or k==' nan':
                t[j] = audiolazy.midi2freq(midinote[i])
        t = [float(j) for j in t]

        out = interpolation.zoom(t, gap[i]/len(t))
        pitch_list[i] = out

    outfile = MidiFile()
    track = MidiTrack()
    outfile.tracks.append(track)
    track.append(Message('control_change', channel = 0, control = 15, value = 0, time = 0 ))
    track.append(Message('control_change', channel = 0, control = 55, value = 10, time = 0 ))
    track.append(Message('control_change', channel = 0, control = 56, value = 127, time = 0 ))
    track.append(Message('control_change', channel = 0, control = 57, value = 127, time = 0 ))
    track.append(Message('control_change', channel = 0, control = 58, value = 10, time = 0 ))
    first=True
    p = []
    prev_value = 0
    outfile = MidiFile()
    track = MidiTrack()
    outfile.tracks.append(track)
    track.append(Message('control_change', channel = 0, control = 15, value = 0, time = 0 ))
    track.append(Message('control_change', channel = 0, control = 55, value = 10, time = 0 ))
    track.append(Message('control_change', channel = 0, control = 56, value = 127, time = 0 ))
    track.append(Message('control_change', channel = 0, control = 57, value = 127, time = 0 ))
    track.append(Message('control_change', channel = 0, control = 58, value = 10, time = 0 ))

    for note_num in range(len(midinote)):
        if(note_num==0):#每個track第一個音
            track.append(Message('note_on', channel = 0, note=midinote[note_num], velocity=127, time = int(start[note_num])))
            track.append(Message('control_change', channel = 0, control = 58, value = int(math.ceil(release[note_num]*127)), time = 0))
        else:#後面剩下的音
            track.append(Message('note_on', channel = 0, note=midinote[note_num], velocity=127, time = 0))
            track.append(Message('control_change', channel = 0, control = 58, value = int(math.ceil(release[note_num]*127)), time = 0))
        
        
        for tick_num in range(gap_2[note_num]):
            f = pitch_list[note_num][tick_num]/audiolazy.midi2freq(midinote[note_num])
            if f==0:
                pitch_bend = 0
            else:
                pitch_bend = 12*math.log(f, 2)*8192
                if pitch_bend>8191:
                    pitch_bend = 8191
                elif pitch_bend<-8191:
                    pitch_bend = -8191
            track.append(Message('pitchwheel', pitch = int(pitch_bend) ,time = 1, channel = 0))
        track.append(Message('note_off', note=midinote[note_num], velocity=127, time = 0))
    outfile.save(filename = "bach_test.mid")