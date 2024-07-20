# File: code/model/model_manager.py
import os
import numpy as np
import pickle
from tensorflow.keras.models import load_model, save_model
from custom_metrics import CustomMeanSquaredError
import tensorflow as tf

# Registrar explícitamente la función mse
@tf.keras.utils.register_keras_serializable()
def mse(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

class ModelManager:
    @staticmethod
    def save_model_and_hyperparameters(model, hyperparameters, model_path, hyperparameters_path):
        save_model(model, model_path)
        with open(hyperparameters_path, 'wb') as f:
            pickle.dump(hyperparameters, f)
        print(f"Model and hyperparameters saved at {model_path} and {hyperparameters_path}")

    @staticmethod
    def load_model_and_hyperparameters(model_path, hyperparameters_path):
        custom_objects = {'mean_squared_error': CustomMeanSquaredError, 'mse': mse}
        print(f"Loading model from {model_path} with custom objects: {custom_objects}")
        model = load_model(model_path, custom_objects=custom_objects)
        with open(hyperparameters_path, 'rb') as f:
            hyperparameters = pickle.load(f)
        print("Model and hyperparameters loaded from disk")
        return model, hyperparameters

    @staticmethod
    def model_exists(model_path, hyperparameters_path):
        return os.path.exists(model_path) and os.path.exists(hyperparameters_path)
