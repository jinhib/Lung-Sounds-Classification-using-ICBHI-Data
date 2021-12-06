import csv

dir = 'norm_bandpass_filter_down_sampling'

base_dir = 'F:/Experiment/2021.09.LungSound_2017ICBHI/0.data/norm_fft_csv/' + dir

f = open(base_dir + '/' + 'fft_' + dir + '.csv', 'r', encoding='utf-8')
r = csv.reader(f)

AKGC417L = []
Litt3200 = []
LittC2SE = []
Meditron = []

for data in r:
    if 'AKGC417L' in data[0]:
        AKGC417L.append(data)
    elif 'Litt3200' in data[0]:
        Litt3200.append(data)
    elif 'LittC2SE' in data[0]:
        LittC2SE.append(data)
    elif 'Meditron' in data[0]:
        Meditron.append(data)

f.close()

for scope in ['AKGC417L', 'Litt3200', 'LittC2SE', 'Meditron']:
    f = open(base_dir + '/' + scope + '/fft_' + dir + '_' + scope + '.csv', 'a', encoding='utf-8', newline='')
    wr = csv.writer(f)
    if scope == 'AKGC417L':
        for data in AKGC417L:
            wr.writerow(data)
    elif scope == 'Litt3200':
        for data in Litt3200:
            wr.writerow(data)
    elif scope == 'LittC2SE':
        for data in LittC2SE:
            wr.writerow(data)
    elif scope == 'Meditron':
        for data in Meditron:
            wr.writerow(data)

    f.close()

for scope in ['AKGC417L', 'Litt3200', 'LittC2SE', 'Meditron']:
    fAl = open(base_dir + '/' + scope + '/Al/fft_' + dir + '_' + scope + '_Al.csv', 'a', encoding='utf-8', newline='')
    fAr = open(base_dir + '/' + scope + '/Ar/fft_' + dir + '_' + scope + '_Ar.csv', 'a', encoding='utf-8', newline='')
    fLl = open(base_dir + '/' + scope + '/Ll/fft_' + dir + '_' + scope + '_Ll.csv', 'a', encoding='utf-8', newline='')
    fLr = open(base_dir + '/' + scope + '/Lr/fft_' + dir + '_' + scope + '_Lr.csv', 'a', encoding='utf-8', newline='')
    fPl = open(base_dir + '/' + scope + '/Pl/fft_' + dir + '_' + scope + '_Pl.csv', 'a', encoding='utf-8', newline='')
    fPr = open(base_dir + '/' + scope + '/Pr/fft_' + dir + '_' + scope + '_Pr.csv', 'a', encoding='utf-8', newline='')
    fTc = open(base_dir + '/' + scope + '/Tc/fft_' + dir + '_' + scope + '_Tc.csv', 'a', encoding='utf-8', newline='')

    wr_Al = csv.writer(fAl)
    wr_Ar = csv.writer(fAr)
    wr_Ll = csv.writer(fLl)
    wr_Lr = csv.writer(fLr)
    wr_Pl = csv.writer(fPl)
    wr_Pr = csv.writer(fPr)
    wr_Tc = csv.writer(fTc)

    if scope == 'AKGC417L':
        for data in AKGC417L:
            if 'Al' in data[0]:
                wr_Al.writerow(data)
            elif 'Ar' in data[0]:
                wr_Ar.writerow(data)
            elif 'Ll' in data[0]:
                wr_Ll.writerow(data)
            elif 'Lr' in data[0]:
                wr_Lr.writerow(data)
            elif 'Pl' in data[0]:
                wr_Pl.writerow(data)
            elif 'Pr' in data[0]:
                wr_Pr.writerow(data)
            elif 'Tc' in data[0]:
                wr_Tc.writerow(data)
    elif scope == 'Litt3200':
        for data in Litt3200:
            if 'Al' in data[0]:
                wr_Al.writerow(data)
            elif 'Ar' in data[0]:
                wr_Ar.writerow(data)
            elif 'Ll' in data[0]:
                wr_Ll.writerow(data)
            elif 'Lr' in data[0]:
                wr_Lr.writerow(data)
            elif 'Pl' in data[0]:
                wr_Pl.writerow(data)
            elif 'Pr' in data[0]:
                wr_Pr.writerow(data)
            elif 'Tc' in data[0]:
                wr_Tc.writerow(data)
    elif scope == 'LittC2SE':
        for data in LittC2SE:
            if 'Al' in data[0]:
                wr_Al.writerow(data)
            elif 'Ar' in data[0]:
                wr_Ar.writerow(data)
            elif 'Ll' in data[0]:
                wr_Ll.writerow(data)
            elif 'Lr' in data[0]:
                wr_Lr.writerow(data)
            elif 'Pl' in data[0]:
                wr_Pl.writerow(data)
            elif 'Pr' in data[0]:
                wr_Pr.writerow(data)
            elif 'Tc' in data[0]:
                wr_Tc.writerow(data)
    elif scope == 'Meditron':
        for data in Meditron:
            if 'Al' in data[0]:
                wr_Al.writerow(data)
            elif 'Ar' in data[0]:
                wr_Ar.writerow(data)
            elif 'Ll' in data[0]:
                wr_Ll.writerow(data)
            elif 'Lr' in data[0]:
                wr_Lr.writerow(data)
            elif 'Pl' in data[0]:
                wr_Pl.writerow(data)
            elif 'Pr' in data[0]:
                wr_Pr.writerow(data)
            elif 'Tc' in data[0]:
                wr_Tc.writerow(data)

    fAr.close()
    fAl.close()
    fLl.close()
    fLr.close()
    fPl.close()
    fPr.close()
    fTc.close()
