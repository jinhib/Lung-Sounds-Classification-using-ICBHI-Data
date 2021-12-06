from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import InceptionV3

from keras import models, layers, optimizers
from keras.callbacks import EarlyStopping
import keras.backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from sklearn.metrics import classification_report
import numpy as np
from sklearn import metrics

for iterations in range(1):
    base_dir = 'F:/Experiment/2021.09.LungSound_2017ICBHI/0.data/Experiment_down_sampling_mel_15'
    train_dir = base_dir + '/' + 'train'
    test_dir = base_dir + '/' + 'test'

    train_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=10, class_mode='categorical')
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(test_dir, target_size=(150, 150), batch_size=10, class_mode='categorical')

    conv_base_VGG16 = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    conv_base_VGG19 = VGG19(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

    # fine-tuning
    # conv_base_VGG16.trainable = True
    # set_trainable = False
    # for layer in conv_base_VGG16.layers:
    #     if layer.name == 'block5_conv1':
    #         set_trainable = True
    #     if set_trainable:
    #         layer.trainable = True
    #     else:
    #         layer.trainable = False

    model_VGG16 = models.Sequential()
    model_VGG16.add(conv_base_VGG16)
    model_VGG16.add(layers.MaxPooling2D(2, 2))

    model_VGG19 = models.Sequential()
    model_VGG19.add(conv_base_VGG19)
    model_VGG19.add(layers.MaxPooling2D(2, 2))

    # merge_models = layers.concatenate([model_VGG16, model_VGG19])

    total_model = models.Sequential()
    total_model.add(layers.merge.concatenate([model_VGG16, model_VGG19]))
    total_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    total_model.add(layers.MaxPooling2D(2, 2))
    total_model.add(layers.Flatten())
    total_model.add(layers.Dense(256, activation='relu'))
    total_model.add(layers.Dense(4, activation='softmax'))

    total_model.compile(optimizer=optimizers.RMSprop(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_acc', patience=5, mode='auto')
    history = total_model.fit_generator(train_generator, steps_per_epoch=80, epochs=100, callbacks=[early_stopping])

    # predictions = total_model.predict_generator(test_generator, steps=np.math.ceil(test_generator.samples / test_generator.batch_size),)
    # predicted_classes = np.argmax(predictions, axis=1)
    # true_classes = test_generator.classes
    # class_labels = list(test_generator.class_indices.keys())
    # report = classification_report(true_classes, predicted_classes, target_names=class_labels)
    # accuracy = metrics.accuracy_score(true_classes, predicted_classes)

    test_loss, test_acc = total_model.evaluate_generator(test_generator, steps=10)
    # print(test_acc)
    acc = round(test_acc * 100, 2)
    f = open("result.txt", 'w')
    f.write("acc: " + acc + '\n')
    f.close()

    K.clear_session()