from keras import models
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

base_dir = 'F:/Experiment/2021.09.LungSound_2017ICBHI/0.data/Experiment_down_sampling_mel_15'
train_dir = base_dir + '/' + 'train'
test_dir = base_dir + '/' + 'test'

train_generator = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_generator = ImageDataGenerator(rescale=1./255)
train_set = train_generator.flow_from_directory(train_dir, target_size=(64, 64), batch_size=32, class_mode='categorical')
test_set = test_generator.flow_from_directory(test_dir, target_size=(64, 64), batch_size=32, class_mode='categorical')

# automl: 파라미터가 알고리즘에 의해 구조 생성
model = models.Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(train_set, steps_per_epoch=80, epochs=100, validation_data=test_set, validation_steps=5)
scores = model.evaluate_generator(test_set, steps=5)
acc = round(scores[1]*100, 1)
print(acc)