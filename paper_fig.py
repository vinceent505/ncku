# Code source: Stefan Balke
# License: ISC
# sphinx_gallery_thumbnail_number = 4

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import librosa
import librosa.display
def spec(y, sr):
    o_env = librosa.onset.onset_strength(y=y, sr=sr)
    times = librosa.times_like(o_env, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)
    D = np.abs(librosa.stft(y))
    times = librosa.times_like(D, sr=sr)
    fig, ax = plt.subplots(nrows=2)
    librosa.display.waveplot(y, sr=sr, ax=ax[0])
    ax[0].set(title='Time Domain', ylabel="Amplitude")
    ax[0].label_outer()

    
    librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
                            y_axis='log', x_axis='time', ax=ax[1], sr=sr)
    ax[1].set(title='Power spectrogram', xlabel="Time[second]")
    ax[1].label_outer()
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)

    plt.show()
def spec_pitch(y, sr, pitch):
    o_env = librosa.onset.onset_strength(y=y, sr=sr)
    times = librosa.times_like(o_env, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)
    D = np.abs(librosa.stft(y))
    times = librosa.times_like(D, sr=sr)
    fig, ax = plt.subplots(nrows=3, sharex=True)
    ax[0].plot(np.arange(len(y))/44100, y)
    ax[0].set(title='Time Domain', ylabel="Amplitude")
    ax[0].label_outer()

    
    librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
                            y_axis='log', x_axis='time', ax=ax[1], sr=sr)
    ax[1].set(title='Power spectrogram', xlabel="Time[second]")
    ax[1].label_outer()

    ax[2].plot(np.arange(len(pitch))/100,pitch)
    ax[2].set(xlabel="Time[second]", ylabel="Hz")

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)

    plt.show()

def onset_fig(y, sr):
    print(len(y))
    D = np.abs(librosa.stft(y))
    o_env = librosa.onset.onset_strength(y=y, sr=sr)
    times = librosa.times_like(o_env, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr, hop_length=64)
    times = librosa.times_like(D, sr=sr)
    fig, ax = plt.subplots(nrows=2, sharex=True)
    librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
                            y_axis='log', x_axis='time', ax=ax[0], sr=sr)
    ax[0].set(title='Power spectrogram')
    ax[0].label_outer()
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    ax[1].plot(times, onset_env / onset_env.max(), alpha=0.8,
            label='Mean (mel)')

    ax[0].set(ylabel="Hz")
    ax[1].set(xlabel="Time[second]", ylabel="SF")
    
    # ax[1].vlines(times[onset_frames-1], 0, o_env.max(), color='r', alpha=0.9,
    #        linestyle='--', label='Onsets')
    plt.show()


def offset_fig(y, curve):
    fig, ax = plt.subplots(nrows=2, sharex=True)
    ax[0].plot(np.arange(len(y))/44100, y)
    ax[0].set(xlabel="Time[second]", ylabel="Amplitude")


    ax[1].plot(np.arange(len(curve))*64/44100, curve)
    ax[1].set(xlabel="Time[second]", ylabel="dB")
    plt.show()


if __name__ == "__main__":
    # y, sr = librosa.load("input/audio/perf/Hilary/k.wav", sr=44100)
    y, sr = librosa.load("output/audio/k.wav", sr=44100)
    spec(y, sr)