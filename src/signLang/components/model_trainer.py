import os
import sys
import numpy as np
import pandas as pd
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import dagshub
import yaml
import tensorflow as tf

from tensorflow.keras.preprocessing.image import array_to_img, ImageDataGenerator
from keras import layers, Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from src.signLang.logging import logger
from src.signLang.exception import CustomException
from src.signLang.utils import read_yaml, process_img, create_deep_network, augmentation
from dataclasses import dataclass

PARAMS_PATH = 'params.yaml'


@dataclass
class ModelTrainer():
    def initiate_model_training(self, train_data_path, test_data_path):
        try:
            with mlflow.start_run():
                logger.info("Starting model training.")

                params = read_yaml(PARAMS_PATH)
                IMAGE_SIZE = params['IMAGE_SIZE']
                BATCH_SIZE = params['BATCH_SIZE']
                EPOCHS = params['EPOCHS']
                CLASSES = params['CLASSES']

                mlflow.log_params(params)

                train_data = pd.read_csv(train_data_path)
                test_data = pd.read_csv(test_data_path)

                print(f"train data size: {train_data.shape}")
                print(f"test data size: {test_data.shape}")
                print(f"train data size: {train_data.head()}")

                logger.info("Creating the neural network")
                model = create_deep_network(IMAGE_SIZE, CLASSES)

                model.compile(optimizer='adam',
                              loss='sparse_categorical_crossentropy',
                              metrics=['accuracy'])

                earlyStopping = EarlyStopping(
                    monitor='val_accuracy', min_delta=1e-4, patience=5, restore_best_weights=True)
                reduceLR = ReduceLROnPlateau(
                    monitor='val_accuracy', patience=3, factor=0.5, min_lr=1e-5)

                logger.info('Neural network established! Augmenting the data.')
                train_img, train_labels = process_img(train_data)
                val_img, val_labels = process_img(test_data)

                train_datagen, val_datagen = augmentation(
                    train_img, train_labels, val_img, val_labels)

                train_data = train_datagen.flow(
                    train_img, train_labels, batch_size=BATCH_SIZE)
                val_data = val_datagen.flow(
                    val_img, val_labels, batch_size=BATCH_SIZE)

                logger.info("Initiating model training...")
                history = model.fit(train_data, validation_data=test_data, epochs=EPOCHS, callbacks=[
                    earlyStopping, reduceLR], workers=4, use_multiprocessing=True)

                for metric_name, metric_values in history.history.items():
                    for epoch, value in enumerate(metric_values, 1):
                        mlflow.log_metric(f"{metric_name}_epoch{epoch}", value)

                mlflow.keras.log_model(model, "models")
                return history.history['accuracy']

        except Exception as e:
            raise CustomException(e, sys)
