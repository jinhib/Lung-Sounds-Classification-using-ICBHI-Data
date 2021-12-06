from scipy.io import wavfile
import os
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np

def Mel_S(y_list, sr, opt):
    # mel-spectrogram
    # wav_length = len(y)/sr
    sr = sr // 2
    frame_length = 0.025
    frame_stride = 0.010

    input_nfft = int(round(sr * frame_length))
    input_stride = int(round(sr * frame_stride))
    y = np.array(y_list)
    S = librosa.feature.melspectrogram(y=y, n_mels=40, n_fft=input_nfft, hop_length=input_stride)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), sr=sr, hop_length=input_stride)
    # plt.colorbar(format='%+2.0f dB')
    if opt == 1:
        plt.savefig(org_dir + 'haar_to_mel_mod_sr_avg/' + label + '/' + str(num + 1) + '/' + fname[:-4] + '.png')
        plt.close()
    elif opt == 2:
        plt.savefig(org_dir + 'haar_to_mel_mod_sr_dif/' + label + '/' + str(num + 1) + '/' + fname[:-4] + '.png')
        plt.close()
    return S


for label in ['crackle', 'wheeze', 'normal', 'mixed']:
    org_dir = 'F:/Experiment/2021.09.LungSound_2017ICBHI/0.data/'
    path_dir = org_dir + 'wav_labeling/' + label
    file_list = os.listdir(path_dir)

    for fname in file_list:
        fs, data = wavfile.read(path_dir + '/' + fname)
        data_c = data.tolist()
        avg, dif = [], []

        for num in range(6):
            avg_level, dif_level = [], []
            if len(data_c) % 2 == 1:
                data_c = data_c[:-1]
            i = 0
            while i < len(data_c):
                avg_value = (data_c[i] + data_c[i + 1]) // 2 ** (1 / 2)
                dif_value = (data_c[i] - data_c[i + 1]) // 2 ** (1 / 2)
                avg_level.append(avg_value)
                dif_level.append(dif_value)
                i += 2
            avg.append(avg_level)
            dif.append(dif_level)
            data_c = avg_level

            # plt.subplot(2, 1, 1)
            # plt.plot(avg_level)
            # plt.subplot(2, 1, 2)
            # plt.plot(dif_level)
            # # plt.show()
            # plt.savefig(org_dir + '/wav_to_haar/' + label + '/' + str(num+1) + '/' + fname + '.png')
            # plt.close()

            Mel_S(avg_level, fs, 1)
            Mel_S(dif_level, fs, 2)
