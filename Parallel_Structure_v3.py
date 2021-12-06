from keras.preprocessing import image
import os
import numpy as np
from keras.applications import *
from keras import layers, optimizers
from keras.callbacks import EarlyStopping
import keras.backend as K
from keras.models import Model
import tensorflow as tf
from tensorflow.python.client import device_lib
from keras.utils.np_utils import to_categorical
from keras.layers import TimeDistributed

base_dir = 'F:/Experiment/2021.09.LungSound_2017ICBHI/0.data/Experiment_original_mel_40'
train_dir = base_dir + '/' + 'train'
test_dir = base_dir + '/' + 'test'
val_dir = base_dir + '/' + 'validation'

local_device_protos = device_lib.list_local_devices()
print([x.name for x in local_device_protos if x.device_type == 'GPU'])

train_data = []
train_label = []
for label, category in enumerate(["crackle", "wheeze", "normal", "mixed"]):
    data_list = os.listdir(train_dir + '/' + category)
    for data_name in data_list:
        train_sep = []
        img = image.load_img(train_dir + '/' + category + '/' + data_name, target_size=(150, 150))
        img_tensor = image.img_to_array(img)
        img_tensor /= 255.

        for i in range(10):
            train_sep.append(img_tensor[:, (i*150):((i+1)*150), :])

        train_data.append(train_sep)
        train_label.append(label)

val_data = []
val_label = []
for label, category in enumerate(["crackle", "wheeze", "normal", "mixed"]):
    data_list = os.listdir(val_dir + '/' + category)
    for data_name in data_list:
        val_sep = []
        img = image.load_img(val_dir + '/' + category + '/' + data_name, target_size=(150, 1500))
        img_tensor = image.img_to_array(img)
        img_tensor /= 255.

        for i in range(10):
            val_sep.append(img_tensor[:, (i*150):((i+1)*150), :])

        val_data.append(val_sep)
        val_label.append(label)

test_data = []
test_label = []
for label, category in enumerate(["crackle", "wheeze", "normal", "mixed"]):
    data_list = os.listdir(test_dir + '/' + category)
    for data_name in data_list:
        test_sep = []
        img = image.load_img(test_dir + '/' + category + '/' + data_name, target_size=(150, 1500))
        img_tensor = image.img_to_array(img)
        img_tensor /= 255.

        for i in range(10):
            test_sep.append(img_tensor[:, (i*150):((i+1)*150), :])

        test_data.append(test_sep)
        test_label.append(label)

train_data = [train_data]
test_data = [test_data]
val_data = [val_data]

train_label = to_categorical(train_label, 4)
test_label = to_categorical(test_label, 4)
val_label = to_categorical(val_label, 4)

for i in range(100):
    with tf.device("/gpu:0"):

        input_common = layers.Input(shape=(10, 150, 150, 3))

        conv_base_VGG16 = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

        # fine-tuning
        '''
        conv_base_VGG16.trainable = True
        set_trainable = False
    
        for layer in conv_base_VGG16.layers:
            if layer.name == 'block5_conv1':
                set_trainable = True
            if set_trainable:
                layer.trainable = True
            else:
                layer.trainable = False
        '''

        cnn = models.Sequential()
        cnn.add(conv_base_VGG16)
        cnn.add(layers.convolutional.Conv2D(64, (3, 3), activation='relu', padding='same'))
        cnn.add(layers.convolutional.Conv2D(64, (3, 3), activation='relu', padding='same'))
        cnn.add(layers.convolutional.MaxPooling2D(2, 2))
        cnn.add(layers.Flatten())

        rnn = models.Sequential()
        rnn.add(layers.GRU(64, return_sequences=False))

        ann = models.Sequential()
        ann.add(layers.Dense(64, activation='relu'))
        ann.add(layers.Dense(4, activation='softmax'))

        model = TimeDistributed(cnn)(input_common)
        model = rnn(model)
        model = ann(model)

        final_model = Model(inputs=input_common, outputs=model)

        # model.summary()

        final_model.compile(optimizer=optimizers.RMSprop(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, mode='auto')
        history = final_model.fit(train_data, train_label, validation_data=(val_data, val_label), epochs=60,
                            batch_size=20, callbacks=[early_stopping])

        test_loss, test_acc = final_model.evaluate(test_data, test_label)

        acc = round(test_acc * 100, 2)
        print(acc)

        f = open('fine_tunning_X_result.txt', 'a')
        f.write(str(acc) + '\n')
        f.close()

        K.clear_session()
