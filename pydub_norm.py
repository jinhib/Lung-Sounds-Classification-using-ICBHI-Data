from pydub import AudioSegment
import os

def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

for label in ['crackle', 'wheeze', 'normal', 'mixed']:
    org_dir = 'F:/Experiment/2021.09.LungSound_2017ICBHI/0.data/'
    path_dir = org_dir + 'wav_labeling/' + label
    file_list = os.listdir(path_dir)

    for file_name in file_list:
        sound = AudioSegment.from_file(path_dir + '/' + file_name, "wav")
        normalized_sound = match_target_amplitude(sound, -20.0)

        normalized_sound.export(org_dir + 'norm/' + label + '/' + file_name, format="wav")
