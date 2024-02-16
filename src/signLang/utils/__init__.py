import os
import sys
import numpy as np
import pandas as pd
import yaml

from tensorflow.keras.preprocessing.image import array_to_img, ImageDataGenerator
from tensorflow.keras.preprocessing.image import array_to_img, ImageDataGenerator
from keras import layers, Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from src.signLang.logging import logger
from src.signLang.exception import CustomException


def read_yaml(PARAMS_PATH):
    try:
        with open(PARAMS_PATH, 'r') as stream:
            params = yaml.safe_load(stream)
            return params

    except Exception as e:
        raise CustomException(e, sys)


def process_img(dataframe):
    try:
        # Extract labels and pixel values
        labels = dataframe['label'].values.astype(np.int32)
        images = dataframe.drop('label', axis=1).values.astype(np.float32)

        # Reshape images to (num_samples, 28, 28)
        images = images.reshape(-1, 28, 28)
        return images, labels

    except Exception as e:
        raise CustomException(e, sys)


def augmentation(train_images, train_labels, val_images, val_labels):
    try:
        train_datagen = ImageDataGenerator(rescale=1./255,
                                           rotation_range=20,
                                           width_shift_range=0.2,
                                           height_shift_range=0.2,
                                           shear_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip=True,
                                           fill_mode='nearest')

        val_datagen = ImageDataGenerator(rescale=1./255)
        return train_datagen, val_datagen

    except Exception as e:
        raise CustomException(e, sys)


def create_deep_network(input_shape, output_shape):
    try:
        model = Sequential([
            layers.Conv2D(64, (5, 5), activation='relu',
                          padding='same', input_shape=input_shape),
            layers.Conv2D(64, (5, 5), activation='relu', padding='same'),
            layers.MaxPool2D(2),

            layers.Dropout(0.2),

            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPool2D(2),

            layers.Dropout(0.2),
            layers.Flatten(),

            layers.Dense(256, activation='relu'),

            layers.Dropout(0.2),

            layers.Dense(output_shape, activation='softmax')
        ])
        return model

    except Exception as e:
        raise CustomException(e, sys)
