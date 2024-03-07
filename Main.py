from genericpath import isdir
import os
from re import T
import numpy as np
import PictureOrganisation as po
import ConfusionMatrixPlot as pcm
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dropout, BatchNormalization, Dense, Flatten
from keras.optimizers import Adam, SGD, Adagrad
from keras.utils import np_utils


if __name__ == '__main__':
    class_list = ['brzina20', 'brzina50', 'brzina80', 'brzina100', 'kruzniTok', 'opasnost', 'pesacki', 'prvenstvo', 'semafor', 'stop']
    if os.path.isdir('data/train') is False:
        po.organiseFolders()
    train_dir = 'data/train'
    valid_dir = 'data/validate'
    test_dir = 'data/test'

    train_batches = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input). \
        flow_from_directory(directory=train_dir, target_size=(32, 32), classes=class_list, batch_size=32)

    valid_batches = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input). \
        flow_from_directory(directory=valid_dir, target_size=(32, 32), classes=class_list, batch_size=32)

    test_batches = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input). \
        flow_from_directory(directory=test_dir, target_size=(32, 32), classes=class_list, batch_size=32, shuffle=False)
    

    model = Sequential()

    # Input blok

    model.add(Conv2D(32, kernel_size=(3,3), input_shape = (32, 32, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(2))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    # Drugi blok
    model.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D(2))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    # Treci blok
    model.add(Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dropout(0.2))

    # Cetvrti blok

    # model.add(Dense(32, activation='relu'))
    # model.add(Dropout(0.3))
    # model.add(BatchNormalization())   

    # Output blok
    
    model.add(Dense(10, activation='softmax'))

    model.summary()
    
    sum = []
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
    for i in range(10):
        model.fit(x=train_batches, validation_data=valid_batches, epochs=20, verbose=2, batch_size=16)

        test_labels = test_batches.classes
        predictions = model.evaluate(x=test_batches, verbose=0)
        print(predictions)
        sum.append(predictions[1])
        model.reset_metrics
        model.reset_states
    avg =  0.0
    print(sum)
    for i in sum:
        avg = avg+i
    avg = avg / 10.0
    
    # cm = confusion_matrix(y_true=test_labels, y_pred=predictions.argmax(axis=1))

    # cm_plot_labels = class_list
    # pcm.plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')