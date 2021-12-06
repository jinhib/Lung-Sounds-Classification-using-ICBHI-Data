from scipy.io import wavfile
from scipy import fftpack
import librosa
import os
import csv
import numpy as np
import matplotlib.pyplot as plt


base_dir = 'F:/Experiment/2021.09.LungSound_2017ICBHI/0.data/norm_bandpass_filter_down_sampling'

f = open('fft_norm_bandpass_filter_down_sampling.csv', 'a', encoding='utf-8', newline='')
wr = csv.writer(f)

for label, category in enumerate(['normal', 'crackle', 'wheeze', 'mixed']):
    category_dir = base_dir + '/' + category
    file_list = os.listdir(category_dir)

    print(category + " start")

    for file_name in file_list:
        if 'Litt3200' in file_name:
            sr = 4000
        else:
            sr = 44100
        audio, sr = librosa.load(category_dir + '/' + file_name, sr=sr)

        freqs = fftpack.fftfreq(len(audio))
        mask = freqs > 0
        n_waves = freqs * len(audio)

        fft_vals = fftpack.fft(audio)
        fft_norm = fft_vals * (1.0 / len(audio))
        fft_theo = 2.0 * abs(fft_norm)

        # plt.figure()
        # plt.bar(freqs[mask] * len(audio), fft_theo[mask])
        # plt.show()

        fft_data = list(fft_theo[mask])
        fft_data = fft_data[120:2201]
        fft_data.insert(0, file_name)
        fft_data.append(label)
        wr.writerow(fft_data)

f.close()


'''
base_dir = 'F:/Experiment/2021.09.LungSound_2017ICBHI/0.data/'
data_dir = base_dir + 'down_sampling/normal/'

sr, audio = wavfile.read(data_dir + '217_1b1_Tc_sc_Meditron_11.wav')

freqs = fftpack.fftfreq(len(audio))
mask = freqs > 0
n_waves = freqs * len(audio)

fft_vals = fftpack.fft(audio)
fft_norm = fft_vals * (1.0 / len(audio))
fft_theo = 2.0 * abs(fft_norm)

plt.figure()
plt.bar(freqs[mask] * len(audio), fft_theo[mask])
plt.show()
'''