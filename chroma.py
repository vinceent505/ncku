import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

if __name__ == "__main__":
    x, sr = librosa.load("Bach/bach_Hil_cut_1_10.wav")
    chromagram = librosa.feature.chroma_cqt(x, sr=sr, hop_length=512, n_chroma = 12)
    plt.figure(figsize=(15, 5))
    librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=512, cmap='coolwarm')
    plt.show()