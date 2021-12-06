from scipy.io import wavfile
import os
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

frame_length = 0.025
frame_stride = 0.010

def Mel_S(file, sr):
    # mel-spectrogram
    y, fs = librosa.load(file, sr=sr)
    input_nfft = int(round(fs * frame_length))
    input_stride = int(round(fs * frame_stride))

    S = librosa.feature.melspectrogram(y=y, n_mels=60, n_fft=input_nfft, hop_length=input_stride)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), sr=fs, hop_length=input_stride, y_axis='mel', x_axis='time')

    plt.axis('off'), plt.xticks([]), plt.yticks([])

    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
    plt.tight_layout()

    plt.savefig(org_dir + 'Experiment_original_mel_60/' + label + '/' + fname[:-4] + '.jpg', bbox_inches='tight', pad_inches=0, dpi=100)
    # plt.savefig(org_dir + 'Experiment_original_mel/' + label + '/' + fname[:-4] + '.jpg')

    # plt.show()
    plt.close('all')

local_device_protos = device_lib.list_local_devices()
print([x.name for x in local_device_protos if x.device_type == 'GPU'])

with tf.device("/gpu:0"):
    label = 'wheeze'
    org_dir = 'F:/Experiment/2021.09.LungSound_2017ICBHI/0.data/'
    path_dir = org_dir + 'wav_labeling/' + label
    file_list = os.listdir(path_dir + '/')
    # file_list = file_list[0:369]
    # file_list = file_list[369:738]
    file_list = file_list[738:1107]
    # file_list = file_list[1107:1476]
    # file_list = file_list[1476:1845]
    # file_list = file_list[1845:2214]
    # file_list = file_list[2214:2583]
    # file_list = file_list[2583:2952]
    # file_list = file_list[2952:3321]
    # file_list = file_list[3321:]

    for fname in file_list:
        fs1, data = wavfile.read(path_dir + '/' + fname)
        data_c = data.tolist()
        Mel_S(path_dir + '/' + fname, fs1)
        # Mel_S('103_2b2_Ar_mc_LittC2SE_1.wav', fs1)
