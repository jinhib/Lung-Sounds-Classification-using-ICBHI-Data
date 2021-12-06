from keras.preprocessing import image
import os
from keras.applications import *
from keras import layers, optimizers
from keras.callbacks import EarlyStopping
import keras.backend as K
from keras.models import Model
import tensorflow as tf
from tensorflow.python.client import device_lib
from keras.utils.np_utils import to_categorical
from keras.layers import TimeDistributed
import numpy as np

base_dir = 'F:/Experiment/2021.09.LungSound_2017ICBHI/0.data/Experiment_down_sampling_mel_15_rotate'
train_dir = base_dir + '/' + 'train'
test_dir = base_dir + '/' + 'test'
val_dir = base_dir + '/' + 'validation'

local_device_protos = device_lib.list_local_devices()
print([x.name for x in local_device_protos if x.device_type == 'GPU'])

train_data = []
train_label = []
for label, category in enumerate(["crackle", "wheeze", "normal", "mixed"]):
    data_list = os.listdir(train_dir + '/' + category)
    if category == 'normal':
        data_list = data_list[:1117]
    for data_name in data_list:
        train_sep = []
        img = image.load_img(train_dir + '/' + category + '/' + data_name, target_size=(1500, 150))
        img_tensor = image.img_to_array(img)
        img_tensor /= 255.

        for i in range(10):
            train_sep.append(img_tensor[(i*150):((i+1)*150), :, :])

        train_data.append(train_sep)
        train_label.append(label)

val_data = []
val_label = []
for label, category in enumerate(["crackle", "wheeze", "normal", "mixed"]):
    data_list = os.listdir(val_dir + '/' + category)
    for data_name in data_list:
        val_sep = []
        img = image.load_img(val_dir + '/' + category + '/' + data_name, target_size=(1500, 150))
        img_tensor = image.img_to_array(img)
        img_tensor /= 255.

        for i in range(10):
            val_sep.append(img_tensor[(i*150):((i+1)*150), :, :])

        val_data.append(val_sep)
        val_label.append(label)

test_data = []
test_label = []
for label, category in enumerate(["crackle", "wheeze", "normal", "mixed"]):
    data_list = os.listdir(test_dir + '/' + category)
    for data_name in data_list:
        test_sep = []
        img = image.load_img(test_dir + '/' + category + '/' + data_name, target_size=(1500, 150))
        img_tensor = image.img_to_array(img)
        img_tensor /= 255.

        for i in range(10):
            test_sep.append(img_tensor[(i*150):((i+1)*150), :, :])

        test_data.append(test_sep)
        test_label.append(label)

train_data = [train_data]
test_data = [test_data]
val_data = [val_data]

train_label = to_categorical(train_label, 4)
test_label = to_categorical(test_label, 4)
val_label = to_categorical(val_label, 4)

for i in range(1):
    with tf.device("/gpu:0"):
        input_common = layers.Input(shape=(10, 150, 150, 3))
        input_cnn = layers.Input(shape=(150, 150, 3))

        conv_base_VGG16 = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
        conv_base_VGG19 = VGG19(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
        # conv_base_Xception = Xception(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
        # conv_base_ResNet50 = ResNet50(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

        VGG16_output = conv_base_VGG16(input_cnn)
        VGG19_output = conv_base_VGG19(input_cnn)
        # Xception_output = conv_base_Xception(input_cnn)
        # ResNet50_output = conv_base_ResNet50(input_cnn)

        # Xception_output = layers.convolutional.MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(Xception_output)
        # ResNet50_output = layers.convolutional.MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(ResNet50_output)

        # fine-tuning
        conv_base_VGG16.trainable = True
        conv_base_VGG19.trainable = True
        # conv_base_Xception.trainable = True
        # conv_base_ResNet50.trainable = True

        set_trainable = False

        conv_base_name = [conv_base_VGG16.layers, conv_base_VGG19.layers] #conv_base_Xception.layers, conv_base_ResNet50.layers
        layer_name = ['block5_conv1', 'block5_conv1'] #, 'block13_sepconv1_act', 'conv5_block1_1_conv'

        for i in range(2):
            for layer in conv_base_name[i]:
                if layer.name == layer_name[i]:
                    set_trainable = True
                if set_trainable:
                    layer.trainable = True
                else:
                    layer.trainable = False

        c = layers.concatenate([VGG16_output, VGG19_output]) # Xception_output, ResNet50_output
        c = layers.convolutional.Conv2D(64, (3, 3), activation='relu', padding='same')(c)
        c = layers.convolutional.Conv2D(64, (3, 3), activation='relu', padding='same')(c)
        c = layers.convolutional.MaxPooling2D(2, 2)(c)
        output_c = layers.Flatten(name='flatten')(c)

        cnn = Model(input=input_cnn, output=output_c)

        td = TimeDistributed(cnn)(input_common)

        r = layers.LSTM(64, return_sequences=True)(td)
        r = layers.LSTM(64, return_sequences=True)(r)
        r = layers.LSTM(64, return_sequences=False)(r)

        a = layers.Dense(64, activation='relu')(r)
        a = layers.Dense(32, activation='relu')(a)
        output = layers.Dense(4, activation='softmax')(a)

        model = Model(inputs=input_common, outputs=output)

        # model.summary()

        model.compile(optimizer=optimizers.RMSprop(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, mode='auto')
        history = model.fit(train_data, train_label, validation_data=(val_data, val_label), epochs=60, batch_size=10,
                            callbacks=[early_stopping])

        test_loss, test_acc = model.evaluate(test_data, test_label, batch_size=10)
        pred_lst = model.predict(test_data, batch_size=10)

        '''
        for pred in pred_lst:
            print(np.argmax(pred))
        '''

        acc = round(test_acc * 100, 2)
        print(acc)

        f = open('crnn_result.txt', 'a')
        f.write(str(acc) + '\n')
        f.close()

        K.clear_session()
