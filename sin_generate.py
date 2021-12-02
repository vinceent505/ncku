import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.io.wavfile

dt = 1./44100
time = np.arange(0., 6., dt)
frequency = 660. - 10*np.sin(2*math.pi*time*1.)  # a 1Hz oscillation
phase_correction = np.add.accumulate(time*np.concatenate((np.zeros(1), 2*np.pi*(frequency[:-1]-frequency[1:]))))
waveform = np.sin(2*math.pi*time*frequency + phase_correction)

scipy.io.wavfile.write("sine_vibrato_1.wav", 44100, waveform)