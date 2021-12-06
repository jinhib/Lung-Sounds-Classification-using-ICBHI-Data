import librosa
import os
import soundfile as sf
from scipy.io import wavfile

def down_sample(input_wav, origin_sr, resample_sr):
    y, sr = librosa.load(input_wav, sr=origin_sr)
    resample = librosa.resample(y, sr, resample_sr)
    sf.write(org_dir + 'norm_bandpass_filter_down_sampling/' + label + '/' + fname, resample, resample_sr, format='WAV', endian='LITTLE', subtype='PCM_16')

for label in ['crackle', 'wheeze', 'normal', 'mixed']:
    org_dir = 'F:/Experiment/2021.09.LungSound_2017ICBHI/0.data/'
    path_dir = org_dir + 'norm_bandpass_filter/' + label
    file_list = os.listdir(path_dir)

    for fname in file_list:
        fs, data = wavfile.read(path_dir + '/' + fname)
        down_sample(path_dir + '/' + fname, fs, 4000)

    print('label ' + label + ' end')