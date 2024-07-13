import os
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model, save_model

class ModelManager:
    @staticmethod
    def save_model_and_hyperparameters(model, hyperparameters, model_path, hyperparameters_path):
        save_model(model, model_path)
        with open(hyperparameters_path, 'wb') as f:
            pickle.dump(hyperparameters, f)
        print(f"Model and hyperparameters saved at {model_path} and {hyperparameters_path}")

    @staticmethod
    def load_model_and_hyperparameters(model_path, hyperparameters_path):
        model = load_model(model_path)
        with open(hyperparameters_path, 'rb') as f:
            hyperparameters = pickle.load(f)
        print("Model and hyperparameters loaded from disk")
        return model, hyperparameters

    @staticmethod
    def model_exists(model_path, hyperparameters_path):
        return os.path.exists(model_path) and os.path.exists(hyperparameters_path)
