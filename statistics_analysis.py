import csv
import numpy as np

base_dir = 'F:/Experiment/2021.09.LungSound_2017ICBHI/0.data/norm_fft_csv/norm_bandpass_filter_down_sampling/'

f = open(base_dir + 'fft_norm_bandpass_filter_down_sampling.csv', 'r', encoding='utf-8')
data = csv.reader(f)

new_dataset = []
for d in data:
    if len(d) == 2083:
        new_dataset.append(d)
f.close()

f = open(base_dir + 'fft_norm_bandpass_filter_down_sampling_features.csv', 'a', encoding='utf-8', newline='')
features = csv.writer(f)
features.writerow(['file_name', 'Total_Avg', 'Total_Std',
                   '119to200_Avg', '201to300_Avg', '301to400_Avg', '401to500_Avg', '501to600_Avg', '601to700_Avg', '701to800_Avg', '801to900_Avg', '901to1000_Avg', '1001to1100_Avg', '1101to1200_Avg', '1201to1300_Avg', '1301to1400_Avg', '1401to1500_Avg', '1501to1600_Avg', '1601to1700_Avg', '1701to1800_Avg', '1801to1900_Avg', '1901to2000_Avg', '2001to2100_Avg', '2101to2199_Avg',
                   '119to200_Std', '201to300_Std', '301to400_Std', '401to500_Std', '501to600_Std', '601to700_Std', '701to800_Std', '801to900_Std', '901to1000_Std', '1001to1100_Std', '1101to1200_Std', '1201to1300_Std', '1301to1400_Std', '1401to1500_Std', '1501to1600_Std', '1601to1700_Std', '1701to1800_Std', '1801to1900_Std', '1901to2000_Std', '2001to2100_Std', '2101to2199_Std',
                   '119to200_Max', '201to300_Max', '301to400_Max', '401to500_Max', '501to600_Max', '601to700_Max', '701to800_Max', '801to900_Max', '901to1000_Max', '1001to1100_Max', '1101to1200_Max', '1201to1300_Max', '1301to1400_Max', '1401to1500_Max', '1501to1600_Max', '1601to1700_Max', '1701to1800_Max', '1801to1900_Max', '1901to2000_Max', '2001to2100_Max', '2101to2199_Max',
                   '119to300_Avg', '310to500_Avg', '501to700_Avg', '701to900_Avg', '901to1100_Avg', '1100to1300_Avg', '1300to1500_Avg',  '1501to1700_Avg', '1701to1900_Avg', '1901to2199_Avg',
                   '119to300_Std', '310to500_Std', '501to700_Std', '701to900_Std', '901to1100_Std', '1100to1300_Std', '1300to1500_Std',  '1501to1700_Std', '1701to1900_Std', '1901to2199_Std',
                   '119to300_Max', '310to500_Max', '501to700_Max', '701to900_Max', '901to1100_Max', '1100to1300_Max', '1300to1500__Max',  '1501to1700_Max', '1701to1900_Max', '1901to2199_Max',
                   '119to400_Avg', '401to700_Avg', '701to1000_Avg', '1001to1300_Avg', '1301to1600_Avg', '1601to1900_Avg', '1900to2199_Avg',
                   '119to400_Std', '401to700_Std', '701to1000_Std', '1001to1300_Std', '1301to1600_Std', '1601to1900_Std', '1900to2199_Std',
                   '119to400_Max', '401to700_Max', '701to1000_Max', '1001to1300_Max', '1301to1600_Max', '1601to1900_Max', '1900to2199_Max',
                   '119to500_Avg', '501to900_Avg', '901to1300_Avg', '1301to1700_Avg', '1701to2199_Avg',
                   '119to500_Std', '501to900_Std', '901to1300_Std', '1301to1700_Std', '1701to2199_Std',
                   '119to500_Max', '501to900_Max', '901to1300_Max', '1301to1700_Max', '1701to2199_Max',
                   '119to600_Avg', '601to1100_Avg', '1101to1600_Avg', '1601to2199_Avg',
                   '119to600_Std', '601to1100_Std', '1101to1600_Std', '1601to2199_Std',
                   '119to600_Max', '601to1100_Max', '1101to1600_Max', '1601to2199_Max',
                   'label'])

for data in new_dataset:
    f_name = data[0]
    label = data[-1]
    data = list(map(float, data[1:-1]))
    feature_set = [f_name, np.average(data), np.std(data),
                   np.average(data[:82]), np.average(data[82:182]), np.average(data[182:282]), np.average(data[282:382]), np.average(data[382:482]), np.average(data[482:582]), np.average(data[582:682]), np.average(data[682:782]), np.average(data[782:882]), np.average(data[882:982]), np.average(data[982:1082]), np.average(data[1082:1182]), np.average(data[1182:1282]), np.average(data[1282:1382]), np.average(data[1382:1482]), np.average(data[1482:1582]), np.average(data[1582:1682]), np.average(data[1682:1782]), np.average(data[1782:1882]), np.average(data[1882:1982]), np.average(data[1982:]),
                   np.std(data[:82]), np.std(data[82:182]), np.std(data[182:282]), np.std(data[282:382]), np.std(data[382:482]), np.std(data[482:582]), np.std(data[582:682]), np.std(data[682:782]), np.std(data[782:882]), np.std(data[882:982]), np.std(data[982:1082]), np.std(data[1082:1182]), np.std(data[1182:1282]), np.std(data[1282:1382]), np.std(data[1382:1482]), np.std(data[1482:1582]), np.std(data[1582:1682]), np.std(data[1682:1782]), np.std(data[1782:1882]), np.std(data[1882:1982]), np.std(data[1982:]),
                   np.max(data[:82]), np.max(data[82:182]), np.max(data[182:282]), np.max(data[282:382]), np.max(data[382:482]), np.max(data[482:582]), np.max(data[582:682]), np.max(data[682:782]), np.max(data[782:882]), np.max(data[882:982]), np.max(data[982:1082]), np.max(data[1082:1182]), np.max(data[1182:1282]), np.max(data[1282:1382]), np.max(data[1382:1482]), np.max(data[1482:1582]), np.max(data[1582:1682]), np.max(data[1682:1782]), np.max(data[1782:1882]), np.max(data[1882:1982]), np.max(data[1982:]),
                   np.average(data[:182]), np.average(data[182:382]), np.average(data[382:582]), np.average(data[582:782]), np.average(data[782:982]), np.average(data[982:1182]), np.average(data[1182:1382]), np.average(data[1382:1582]), np.average(data[1582:1782]), np.average(data[1782:]),
                   np.std(data[:182]), np.std(data[182:382]), np.std(data[382:582]), np.std(data[582:782]), np.std(data[782:982]), np.std(data[982:1182]), np.std(data[1182:1382]), np.std(data[1382:1582]), np.std(data[1582:1782]), np.std(data[1782:]),
                   np.max(data[:182]), np.max(data[182:382]), np.max(data[382:582]), np.max(data[582:782]), np.max(data[782:982]), np.max(data[982:1182]), np.max(data[1182:1382]), np.max(data[1382:1582]), np.max(data[1582:1782]), np.max(data[1782:]),
                   np.average(data[:282]), np.average(data[282:582]), np.average(data[582:882]), np.average(data[882:1182]), np.average(data[1182:1482]), np.average(data[1482:1782]), np.average(data[1782:]),
                   np.std(data[:282]), np.std(data[282:582]), np.std(data[582:882]), np.std(data[882:1182]), np.std(data[1182:1482]), np.std(data[1482:1782]), np.std(data[1782:]),
                   np.max(data[:282]), np.max(data[282:582]), np.max(data[582:882]), np.max(data[882:1182]), np.max(data[1182:1482]), np.max(data[1482:1782]), np.max(data[1782:]),
                   np.average(data[:382]), np.average(data[382:782]), np.average(data[782:1182]), np.average(data[1182:1582]), np.average(data[1582:]),
                   np.std(data[:382]), np.std(data[382:782]), np.std(data[782:1182]), np.std(data[1182:1582]), np.std(data[1582:]),
                   np.max(data[:382]), np.max(data[382:782]), np.max(data[782:1182]), np.max(data[1182:1582]), np.max(data[1582:]),
                   np.average(data[:482]), np.average(data[482:982]), np.average(data[982:1482]), np.average(data[1482:]),
                   np.std(data[:482]), np.std(data[482:982]), np.std(data[982:1482]), np.std(data[1482:]),
                   np.max(data[:482]), np.max(data[482:982]), np.max(data[982:1482]), np.max(data[1482:]),
                   label]

    features.writerow(feature_set)

f.close()
