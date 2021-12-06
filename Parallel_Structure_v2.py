from keras.preprocessing.image import ImageDataGenerator
from keras.applications import *
from keras import models, layers, optimizers
from keras.callbacks import EarlyStopping
import keras.backend as K
from keras.models import Model
import tensorflow as tf
from tensorflow.python.client import device_lib
from keras.layers.convolutional import Conv2D, MaxPooling2D
from sklearn.metrics import classification_report
import numpy as np
from sklearn import metrics

base_dir = 'F:/Experiment/2021.09.LungSound_2017ICBHI/0.data/Experiment_down_sampling_mel_15'
train_dir = base_dir + '/' + 'train'
test_dir = base_dir + '/' + 'test'
val_dir = base_dir + '/' + 'validation'

local_device_protos = device_lib.list_local_devices()
print([x.name for x in local_device_protos if x.device_type == 'GPU'])


for iterations in range(100):
    with tf.device("/gpu:0"):

        train_datagen = ImageDataGenerator(rescale=1. / 255)
        train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=10, class_mode='categorical')
        val_datagen = ImageDataGenerator(rescale=1. / 255)
        val_generator = val_datagen.flow_from_directory(val_dir, target_size=(150, 150), batch_size=10, class_mode='categorical')
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        test_generator = test_datagen.flow_from_directory(test_dir, target_size=(150, 150), batch_size=10, class_mode='categorical')

        conv_base_VGG16 = VGG16(weights='imagenet', include_top=False)
        conv_base_VGG19 = VGG19(weights='imagenet', include_top=False)
        conv_base_Xception = Xception(weights='imagenet', include_top=False)
        conv_base_MobileNet = MobileNetV2(weights='imagenet', include_top=False)
        conv_base_ResNet50 = ResNet50(weights='imagenet', include_top=False)

        # fine-tuning
        """conv_base_VGG16.trainable = True
        conv_base_VGG19.trainable = True
        conv_base_VGG16.conv_base_Xception = True
        conv_base_VGG16.ResNet50 = True
    
        set_trainable = False
    
        conv_base_name = [conv_base_VGG16.layers, conv_base_VGG19.layers, conv_base_Xception.layers, conv_base_ResNet50.layers]
        layer_name = ['block5_conv1', 'block5_conv1', 'block13_sepconv1_act', 'conv5_block1_1_conv']
    
        for i in range(4):
            for layer in conv_base_name[i]:
                if layer.name == layer_name[i]:
                    set_trainable = True
                if set_trainable:
                    layer.trainable = True
                else:
                    layer.trainable = False"""

        input_common = layers.Input(shape=(150, 150, 3))

        VGG16_output = conv_base_VGG16(input_common)
        VGG19_output = conv_base_VGG19(input_common)
        Xception_output = conv_base_Xception(input_common)
        MobileNet_output = conv_base_MobileNet(input_common)
        ResNet50_output = conv_base_ResNet50(input_common)

        Xception_output = layers.convolutional.MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(Xception_output)
        ResNet50_output = layers.convolutional.MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(ResNet50_output)

        x = layers.concatenate([VGG16_output, VGG19_output, Xception_output, MobileNet_output, ResNet50_output])
        x = layers.convolutional.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.convolutional.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.convolutional.MaxPooling2D(2, 2)(x)
        x = layers.Flatten(name='flatten')(x)

        x = layers.Dense(256, activation='relu')(x)
        output = layers.Dense(4, activation='softmax')(x)

        model = Model(inputs=input_common, outputs=output)

        # model.summary()

        model.compile(optimizer=optimizers.RMSprop(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, mode='auto')
        history = model.fit_generator(train_generator, steps_per_epoch=50, epochs=60, callbacks=[early_stopping], validation_data=val_generator, validation_steps=5)

        test_loss, test_acc = model.evaluate_generator(test_generator, steps=10)
        acc = round(test_acc * 100, 2)
        print(acc)
        f = open('fine_tunning_X_result.txt', 'a')

        f.write(str(acc) + '\n')
        f.close()

        K.clear_session()