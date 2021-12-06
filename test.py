from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras import models, layers, optimizers
from keras.callbacks import EarlyStopping
import keras.backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from sklearn.metrics import classification_report
import numpy as np
from sklearn import metrics

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())

def recall(y_true, y_pre째다d):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

def f_beta(y_true, y_pred):
    beta = 1
    precision_value = precision(y_true, y_pred)
    recall_value = recall(y_true, y_pred)
    numer = (1 + beta ** 2) * (precision_value * recall_value)
    denom = ((beta ** 2) * precision_value) + recall_value + K.epsilon()
    return numer / denom

for iterations in range(1):
    base_dir = 'F:/Experiment/2021.09.LungSound_2017ICBHI/0.data/Experiment_down_sampling_mel_15'
    train_dir = base_dir + '/' + 'train'
    test_dir = base_dir + '/' + 'test'

    train_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=10, class_mode='categorical')
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(test_dir, target_size=(150, 150), batch_size=10, class_mode='categorical')

    conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

    # print(conv_base.summary())

    conv_base.trainable = True

    set_trainable = False
    for layer in conv_base.layers:
        if layer.name == 'block5_conv1':
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten(name='flatten'))
    # print(model.summary())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))

    model.compile(optimizer=optimizers.RMSprop(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_acc', patience=5, mode='auto')
    history = model.fit_generator(train_generator, steps_per_epoch=5, epochs=10, callbacks=[early_stopping])

    predictions = model.predict_generator(test_generator, steps=np.math.ceil(test_generator.samples / test_generator.batch_size),)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())
    report = classification_report(true_classes, predicted_classes, target_names=class_labels)
    accuracy = metrics.accuracy_score(true_classes, predicted_classes)

    test_loss, test_acc = model.evaluate_generator(test_generator, steps=10)
    # print(test_acc)
    acc = round(test_acc * 100, 2)
    print(report)
    # f = open("VGG16_result.txt", 'w')
    # # f.write("acc: " + acc + )
    # f.close()

    y_pred = model.predict_generator(test_generator, steps=10)
    print(precision(true_classes, predicted_classes))
    print(recall(true_classes, predicted_classes))
    print(f_beta(true_classes, predicted_classes))

    K.clear_session()