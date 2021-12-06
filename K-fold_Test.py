from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy
import tensorflow as tf
from tensorflow.python.client import device_lib

# local_device_protos = device_lib.list_local_devices()
# print([x.name for x in local_device_protos if x.device_type == 'GPU'])

f_num = 200

with tf.device("/cpu:0"):
    while (1):

        dataset = numpy.loadtxt('C:/Users/HP/Desktop/배진희/thesis/mel40/Gaussian_Distance_' + str(f_num) +
                                '_img_features_mel40_47.07.csv', delimiter=",", skiprows=1)
        features_num = len(dataset[0, :-1])
        print(features_num)

        def create_model():
            # create model
            model = Sequential()
            model.add(Dense(50, input_dim=features_num, activation='relu'))
            model.add(Dense(10, activation='relu'))
            model.add(Dense(4, activation='softmax'))
            # Compile model
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            return model


        X = dataset[:, 0:features_num]
        Y = dataset[:, features_num]

        model = KerasClassifier(build_fn=create_model, epochs=100, verbose=0)

        kfold = StratifiedKFold(n_splits=5, shuffle=True)
        results = cross_val_score(model, X, Y, cv=kfold)

        print("5-fold Accuracy : " + str(numpy.average(results)))

        K.clear_session()