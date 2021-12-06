from keras.models import load_model, Model
from keras.preprocessing import image
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.python.client import device_lib
from keras import backend as K


def feature_extract(data_path, model_name, save_path, model_path, classes):
    model = load_model(model_path + model_name)

    flatten = model.get_layer('flatten')
    test_model = Model(inputs=model.input, outputs=flatten.output)

    crackle_img_dir = data_path + '/' + 'crackle'
    wheeze_img_dir = data_path + '/' + 'wheeze'
    normal_img_dir = data_path + '/' + 'normal'
    mixed_img_dir = data_path + '/' + 'mixed'

    crackle_img = []
    for img_name in os.listdir(crackle_img_dir):
        img_path = os.path.join(crackle_img_dir, img_name)
        img = image.load_img(img_path, target_size=(150, 150))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.
        crackle_img.append(img_tensor)

    wheeze_img = []
    for img_name in os.listdir(wheeze_img_dir):
        img_path = os.path.join(wheeze_img_dir, img_name)
        img = image.load_img(img_path, target_size=(150, 150))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.
        wheeze_img.append(img_tensor)

    normal_img = []
    for img_name in os.listdir(normal_img_dir):
        img_path = os.path.join(normal_img_dir, img_name)
        img = image.load_img(img_path, target_size=(150, 150))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.
        normal_img.append(img_tensor)

    mixed_img = []
    for img_name in os.listdir(mixed_img_dir):
        img_path = os.path.join(mixed_img_dir, img_name)
        img = image.load_img(img_path, target_size=(150, 150))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.
        mixed_img.append(img_tensor)

    if classes == 4:
        crackle_labels = np.array([1 for i in range(len(crackle_img))])
        wheeze_labels = np.array([2 for i in range(len(wheeze_img))])
        normal_labels = np.array([0 for i in range(len(normal_img))])
        mixed_labels = np.array([3 for i in range(len(mixed_img))])
    else:
        crackle_labels = np.array([1 for i in range(len(crackle_img))])
        wheeze_labels = np.array([1 for i in range(len(wheeze_img))])
        normal_labels = np.array([0 for i in range(len(normal_img))])
        mixed_labels = np.array([1 for i in range(len(mixed_img))])

    total_labels = np.concatenate((crackle_labels, wheeze_labels), axis=0)
    total_labels = np.concatenate((total_labels, normal_labels), axis=0)
    total_labels = np.concatenate((total_labels, mixed_labels), axis=0)

    img_features = []
    for img in crackle_img:
        img_features.append(test_model.predict(img)[0])

    for img in wheeze_img:
        img_features.append(test_model.predict(img)[0])

    for img in normal_img:
        img_features.append(test_model.predict(img)[0])

    for img in mixed_img:
        img_features.append(test_model.predict(img)[0])

    # print(len(img_features))

    # length of image features
    # print(len(img_features[0]))

    # Dataset + Label
    feature_set = []
    for i in range(len(total_labels)):
        temp = np.append(img_features[i], total_labels[i])
        feature_set.append(temp)

    # Feature column name
    features_column = []
    for i in range(len(img_features[0])):
        features_column.append('img_f' + str(i))

    features_column.append('label')

    df = pd.DataFrame(feature_set, columns=features_column)
    if classes == 4:
        df.to_csv(save_path + 'img_features_4class_' + model_name[-8:-3] + '.csv', index=False)
    else:
        df.to_csv(save_path + 'img_features_2class_' + model_name[-8:-3] + '.csv', index=False)


save_path = ''
model_path = 'C:/Users/HP/Desktop/제비/'
data_path = 'F:/Experiment/2021.09.LungSound_2017ICBHI/0.data/Experiment_original_mel_40'

local_device_protos = device_lib.list_local_devices()
print([x.name for x in local_device_protos if x.device_type == 'GPU'])

classes_num = 2
with tf.device("/gpu:0"):
    feature_extract(data_path, 'save_model_2class_70.67.h5', save_path, model_path, classes_num)
    K.clear_session()
