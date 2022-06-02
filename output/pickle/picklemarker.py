import pickle
import csv
import pandas as pd




if __name__ == "__main__":
    musician_name = "Hilary"
    music_name = "P1_6"
    file = musician_name + "_" + music_name + ".pickle"

    with open (file,'rb') as f:
        new_dict = pickle.load(f)

    start = []
    num = []

    for i in new_dict:
        num.append('Marker'+' '+str(new_dict[i]["num"]))
        s = new_dict[i]["start"]
        m = int(s/60)
        s = round(s%60, 3)
        start.append(str(m)+":"+str(s))
    duration = ['0:00.000']*len(start)
    time_format = ['decimal']*len(start)
    types = ['Cue']*len(start)
    descriptions = ['']*len(start)
    print(len(num),len(start),len(duration),len(time_format),len(types),len(descriptions))
    df = {
        'Name':num,
        'Start':start,
        'Duration':duration,
        'Time Format':time_format,
        'Type':types,
        'Description':descriptions
    }
    df = pd.DataFrame(df)
    df.columns = ["Name","Start","Duration","Time Format","Type","Description"]
    df.to_csv("markercsv/"+ musician_name + "_" + music_name + "markers.csv",encoding='utf-8',index = True)