from scipy.signal import butter, sosfilt
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import simpleaudio as sa
import librosa
import soundfile as sf
import os


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut/nyq
    high = highcut/nyq
    sos = butter(order, [low, high], analog=False, btype='bandpass', output='sos')
    return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order)
    y = sosfilt(sos, data)
    return y


def playSineWaveSound():
    frequency = 400  # Our played note will be 440 Hz
    fs = 4000  # 4000 samples per second
    seconds = 3  # Note duration of 3 seconds
    t = np.linspace(0, seconds, seconds * fs, False)
    note = np.sin(frequency * t * 2 * np.pi)
    audio = note * (2 ** 15 - 1) / np.max(np.abs(note))
    audio = audio.astype(np.int16)
    play_obj = sa.play_buffer(audio, 1, 2, fs)
    play_obj.wait_done()
    plt.title('Sine signal')
    plt.plot(range(0,audio.shape[0]), audio, label='Sine signal')
    plt.show()


def callBandPassSample():

    for label in ['crackle', 'wheeze', 'normal', 'mixed']:
        org_dir = 'F:/Experiment/2021.09.LungSound_2017ICBHI/0.data/'
        path_dir = org_dir + 'norm/' + label
        file_list = os.listdir(path_dir)

        for fname in file_list:
            real_file = path_dir + '/' + fname
            if 'Litt3200' in fname:
                sr = 4000
            else:
                sr = 44100
            audio_wave_data, sr = librosa.load(real_file, sr=sr)  # returns numpy.ndarray
            audio = audio_wave_data * (2 ** 15 - 1) / np.max(np.abs(audio_wave_data))
            y1 = butter_bandpass_filter(audio, 120, 1800, sr, 5)

            #y2= butter_bandpass_filter(audio, 500,700, sr, 3)
            #y3= butter_bandpass_filter(audio, 1000,1700, sr, 3)
            xx = range(0,audio.shape[0])

            # plt.title('Original signal')
            # plt.plot(xx, audio, label='')
            # plt.show()
            #
            # plt.title('Bandpass signal')
            # plt.plot(xx, y1, label='')
            # plt.show()

            audio2 = y1.astype(np.int16)
            sf.write(org_dir + 'norm_bandpass_filter/' + label + '/' + fname, audio2, sr, format='WAV', endian='LITTLE', subtype='PCM_16')
            # play_obj = sa.play_buffer(audio2, 1, 2, sr)
            # play_obj.wait_done()
            pass


if __name__ == "__main__":
    callBandPassSample()
    pass