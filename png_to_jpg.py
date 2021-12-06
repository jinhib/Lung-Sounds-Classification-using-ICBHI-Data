from PIL import Image
import os

for label in ['crackle', 'wheeze', 'normal', 'mixed']:
    org_dir = 'F:/Experiment/2021.09.LungSound_2017ICBHI/0.data/'
    path_dir = org_dir + 'down_sampling_mel_15_png/' + label
    file_list = os.listdir(path_dir)

    for fname in file_list:
        im = Image.open(path_dir + '/' + fname).convert('RGB')
        im.save(org_dir + 'down_sampling_mel_15/' + label + '/' + fname[:-4] + '.jpg', 'jpeg')