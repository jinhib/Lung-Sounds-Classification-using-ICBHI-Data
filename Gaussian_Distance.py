import numpy as np
from itertools import combinations
import csv


def Gaussian_Distance(features, labels, num_classes):

    features = features.astype('float')
    labels = labels.astype('int')

    temp = []
    for idx in range(num_classes):
        temp.append([])

    for feature, label in zip(features, labels):
        temp[label].append(feature)

    mean_std_set = []
    for idx in range(num_classes):
        mean_std_set.append((np.mean(temp[idx]), np.std(temp[idx])))

    combination_set = list(combinations(mean_std_set, 2))

    score = 0
    for combination in combination_set:
        # Gaussian Distance
        gd = abs(combination[0][0] - combination[1][0]) / (2 * combination[0][1] + 2 * combination[1][1])
        # Additional Point
        if gd >= 1:
            score += 1
        score += gd / 100

    if np.isnan(score):
        return 0
    return score


file_name = 'img_features_2class_76.0.csv'

original_dataset = np.genfromtxt(file_name, delimiter=',', encoding='UTF8', dtype='float64')
original_dataset = original_dataset[1:, :]
dataset_name = np.genfromtxt(file_name, delimiter=',', encoding='UTF8', dtype=str)
dataset_name = dataset_name[0]
max_value = np.max(original_dataset[:, -1])

print('len_of_original_features : ', len(original_dataset[0]) - 1)

for f_num in range(100, 1001, 100):

    dataset = np.array(original_dataset)

    scores = []
    for index in range(len(dataset[0]) - 1):
        gd_score = Gaussian_Distance(dataset[:, index], dataset[:, -1], int(max_value + 1))
        temp = [gd_score, index]
        scores.append(temp)

    sorted_list = sorted(scores, key=lambda x: x[0], reverse=True)

    feature_name = dataset_name

    labels = dataset[:, -1]
    labels = np.expand_dims(labels, axis=1)
    dataset = dataset[:, :-1]
    dataset = np.transpose(dataset)
    feature_name = np.transpose(feature_name)

    subset = []
    f_name = []
    # Gaussian_list = []

    for score, index in sorted_list[:f_num]:
        subset.append(dataset[int(index)])
        f_name.append(feature_name[int(index)])
        # Gaussian_list.append(score)

    f_name.append('label')
    # Gaussian_list.append('None')
    print('len_of_GD_features : ', len(subset))

    subset = np.transpose(subset)
    f_name = np.transpose(f_name)
    f_name = np.expand_dims(f_name, axis=0)

    # Gaussian_list = np.transpose(Gaussian_list)
    # Gaussian_list = np.expand_dims(Gaussian_list, axis=0)

    dataset = np.concatenate((subset, labels), axis=1)
    # f_name = np.concatenate((f_name, Gaussian_list), axis=0)
    Gaussian_Distance_dataset = np.concatenate((f_name, dataset), axis=0)

    file_path = ''
    f = open('C:/Users/HP/Desktop/배진희/thesis/mel40/' + file_name[:-4] + '_Gaussian_Distance_' + str(f_num) + '.csv', 'w', encoding='utf-8',
             newline='')
    wr = csv.writer(f)
    for line in Gaussian_Distance_dataset:
        wr.writerow(line)
    f.close()
