from keras import models
from keras import layers
from keras.optimizers import Adam
import numpy
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical
import random
import tensorflow as tf
from tensorflow.python.client import device_lib

from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings(action='ignore')

local_device_protos = device_lib.list_local_devices()
print([x.name for x in local_device_protos if x.device_type == 'GPU'])

f_num = 500
classes_num = 2

max_acc = 0
max_score = 0
icbhi_sens = 0
icbhi_spec = 0
icbhi_har = 0

with tf.device("/gpu:0"):
    while True:

        dataset = numpy.loadtxt('C:/Users/HP/Desktop/배진희/thesis/mel40/img_features_' + str(classes_num) +
                                'class_58.29_Gaussian_Distance_' + str(f_num) + '.csv', delimiter=",", skiprows=1)

        random.shuffle(dataset)

        train_index = len(dataset) * 0.8
        train_index = int(train_index)

        # val_index = len(dataset) * 0.8
        # val_index = int(val_index)

        train_data = dataset[:train_index, :-1]
        train_labels = dataset[:train_index, -1]

        # val_data = dataset[train_index:val_index, :-1]
        # val_labels = dataset[train_index:val_index, -1]

        test_data = dataset[train_index:, :-1]
        test_label = dataset[train_index:, -1]

        train_labels = to_categorical(train_labels, classes_num)
        # val_labels = to_categorical(val_labels, classes_num)
        test_labels = to_categorical(test_label, classes_num)

        network = models.Sequential()
        network.add(layers.Dense(50, activation='relu', input_dim=f_num))
        network.add(layers.Dense(10, activation='relu'))
        network.add(layers.Dense(10, activation='relu'))
        network.add(layers.Dense(classes_num, activation='softmax'))

        network.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, mode='auto')
        network.fit(train_data, train_labels, epochs=100, verbose=0)

        test_loss, test_acc = network.evaluate(test_data, test_labels)
        test_acc = round(test_acc, 3)
        acc = test_acc

        print("===============================================================================================")
        print(str(f_num) + ' test_acc :', test_acc)

        if acc > max_acc:
            max_acc = acc

        print(str(f_num) + ' max_acc: ' + str(max_acc))
        print("---------------------------------------------------------------------")

        test_pred = network.predict_classes(test_data, verbose=0)
        cm = confusion_matrix(test_label, test_pred)

        print('Confusion Matrix')
        print(cm)
        print("---------------------------------------------------------------------")

        four_target_names = ['normal: 0', 'crackle: 1', 'wheeze: 2', 'mixed: 3']
        two_target_names = ['normal: 0', 'abnormal: 1']

        if classes_num == 4:
            '''
            TN = cm[0][0]
            TP = cm[1][1] + cm[2][2] + cm[3][3]
            FP = cm[0][1] + cm[0][2] + cm[0][3]
            FN = sum(sum(cm)) - (TN + TP + FP)
            print('TN: ' + str(TN) + ', TP: ' + str(TP) + ', FN: ' + str(FN) + ', FP: ' + str(FP))
            print("---------------------------------------------------------------------")

            # Accuracy
            acc = round((TP + TN)/(TP + TN + FP + FN), 3)
            # Sensitivity
            sens = round(TP/(TP + FN), 3)
            # Precision
            pre = round(TP/(TP + FP), 3)
            # Specificity
            spec = round(TN/(FP + TN), 3)
            # F1 Score
            f1 = round((2*sens*pre)/(sens + pre), 3)

            print('Accuracy: ' + str(acc))
            print('Sensitivity: ' + str(sens))
            print('Specificity: ' + str(spec))
            print('Precision: ' + str(pre))
            print('F1 Score: ' + str(f1))
            print("---------------------------------------------------------------------")
            '''
            print(classification_report(test_label, test_pred, target_names=four_target_names))
            print("---------------------------------------------------------------------")

            # ICBHI Sensitivity
            icbhi_se = (cm[1][1] + cm[2][2] + cm[3][3]) / (sum(cm[1]) + sum(cm[2]) + sum(cm[3]))
            # ICBHI Specificity
            icbhi_sp = cm[0][0] / sum(cm[0])
            # ICBHI Average Score
            icbhi_as = (icbhi_se + icbhi_sp) / 2
            # ICBHI Harmonic Score
            icbhi_hs = (2 * icbhi_se * icbhi_sp) / (icbhi_se + icbhi_sp)

            print("---------------------------------------------------------------------")
            print('ICBHI Sensitivity: ' + str(icbhi_se))
            print('ICBHI Specificity: ' + str(icbhi_sp))
            print('ICBHI Average Score: ' + str(icbhi_as))
            print('ICBHI Harmonic Score: ' + str(icbhi_hs))

            score = icbhi_as

            if score > max_score:
                max_score = score

            print("---------------------------------------------------------------------")
            print(str(f_num) + ' Max ICBHI Average Score: ' + str(max_score))
            print("===============================================================================================")
        else:
            '''
            TN = cm[0][0]
            TP = cm[1][1]
            FP = cm[0][1]
            FN = cm[1][0]
            print('TN: ' + str(TN) + ', TP: ' + str(TP) + ', FN: ' + str(FN) + ', FP: ' + str(FP))
            print("---------------------------------------------------------------------")

            # Accuracy
            acc = round((TP + TN) / (TP + TN + FP + FN), 3)
            # Sensitivity
            sens = round(TP / (TP + FN), 3)
            # Precision
            pre = round(TP / (TP + FP), 3)
            # Specificity
            spec = round(TN / (FP + TN), 3)
            # F1 Score
            f1 = round((2 * sens * pre) / (sens + pre), 3)
            # ICBHI Score
            icbhi = round((sens + spec) / 2, 3)

            print('Accuracy: ' + str(acc))
            print('Sensitivity: ' + str(sens))
            print('Specificity: ' + str(spec))
            print('Precision: ' + str(pre))
            print('F1 Score: ' + str(f1))
            print("---------------------------------------------------------------------")
            '''
            print(classification_report(test_label, test_pred, target_names=two_target_names))
            print("---------------------------------------------------------------------")

            # ICBHI Sensitivity
            icbhi_se = cm[1][1] / sum(cm[1])
            # ICBHI Specificity
            icbhi_sp = cm[0][0] / sum(cm[0])
            # ICBHI Average Score
            icbhi_as = (icbhi_se + icbhi_sp) / 2
            # ICBHI Harmonic Score
            icbhi_hs = (2 * icbhi_se * icbhi_sp) / (icbhi_se + icbhi_sp)

            print('ICBHI Sensitivity: ' + str(icbhi_se))
            print('ICBHI Specificity: ' + str(icbhi_sp))
            print('ICBHI Average Score: ' + str(icbhi_as))
            print('ICBHI Harmonic Score: ' + str(icbhi_hs))

            score = icbhi_as

            if score > max_score:
                max_score = score
                icbhi_sens = icbhi_se
                icbhi_spec = icbhi_sp
                icbhi_har = icbhi_hs

            print("---------------------------------------------------------------------")
            print(str(f_num) + ' ICBHI Sensitivity: ' + str(icbhi_sens))
            print(str(f_num) + ' ICBHI Specificity: ' + str(icbhi_spec))
            print(str(f_num) + ' Max ICBHI Average Score: ' + str(max_score))
            print(str(f_num) + ' ICBHI Harmonic Score: ' + str(icbhi_har))
            print("===============================================================================================")

        K.clear_session()