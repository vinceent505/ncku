import pickle

class midi_note:
    def __init__(self):
        self.name = 0
        self.num = 0
        self.start = 0
        self.end = 0
    
    def update(self,start, end, num, name):
        self.name = name
        self.num = num
        self.start = start
        self.end = end

musician_name = "Hilary"
music_name = "S1_4"


def main():
    f = open("output/pickle/"+musician_name+"/"+musician_name+"_"+music_name+".pickle", "rb")
    d = pickle.load(f)
    note_list = []
    for i in d:
        a = midi_note()
        a.update(d[i]["start"], 0, d[i]["num"], d[i]["name"])
        note_list.append(a)

    output_list = []
    for j, i in enumerate(note_list):
        output_list.append({"num": i.num,"name": i.name, "start": i.start,"end": i.end})


    with open("input/data/" + musician_name+"/"+musician_name+"_"+music_name + "_2nd.pickle", "wb") as f:
        d = dict(enumerate(output_list))
        pickle.dump(d, f)


if __name__ == "__main__":
    main()
