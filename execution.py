# File: execution.py
import sys
import os
import numpy as np
import pandas as pd
import tensorflow as tf

def add_subdirectories_to_syspath(directory):
    for dirpath, dirnames, filenames in os.walk(directory):
        sys.path.append(dirpath)

code_directory = os.path.join(os.path.dirname(__file__), 'code')
add_subdirectories_to_syspath(code_directory)

from tensorflow.keras.models import load_model
from keras_tuner import Hyperband
from tensorflow.keras.optimizers import Adam
from model.model_manager import ModelManager
from model.model_evaluator import ModelEvaluator
from model.model_tuner import ModelTuner
from utils.data_plotter import DataPlotter
from keras.callbacks import EarlyStopping
from data.data_splitter import DataSplitter
from data.data_inspector import DataInspector
from utils.next_sequence_predictor import NextSequencePredictor

time_step = 10
filter_months = 300 
max_epochs = 100  # Máximo número de épocas por trial
hyperband_iterations = 5  # Número de iteraciones de Hyperband

try:
    drive_dir = f'/Users/jflorezgaleano/Documents/JulianFlorez/PredictBal/models_ia/'
    model_path_five = f'{drive_dir}/model_five.h5'
    model_path_sixth = f'{drive_dir}/model_sixth.h5'
    hyperparameters_path_five = f'{drive_dir}/hyperparameters_five.pkl'
    hyperparameters_path_sixth = f'{drive_dir}/hyperparameters_sixth.pkl'

    print("Init DataSplitter")
    data_splitter = DataSplitter(time_step)
    data_splits = data_splitter.split_data(train_ratio=0.8, val_ratio=0.1)
    print("Init DataInspector")
    data_inspector = DataInspector(data_splits)
    data_inspector.inspect_data()

    X_all_data_five, y_all_data_five, X_all_data_sixth, y_all_data_sixth = data_splits['all_data']
    X_train_five, y_train_five = data_splits['train_five']
    X_val_five, y_val_five = data_splits['val_five']
    X_test_five, y_test_five = data_splits['test_five']
    X_train_sixth, y_train_sixth = data_splits['train_sixth']
    X_val_sixth, y_val_sixth = data_splits['val_sixth']
    X_test_sixth, y_test_sixth = data_splits['test_sixth']
    scalers = data_splits['scalers']

    print("Init Search for model")
    if ModelManager.model_exists(model_path_five, hyperparameters_path_five):
        print("Cargando el mejor modelo guardado desde el disco local.")
        model_five, best_hps_values_five = ModelManager.load_model_and_hyperparameters(model_path_five, hyperparameters_path_five)

        print("Hiperparámetros cargados desde el disco local:")
        print(best_hps_values_five)
    else:
        print("Init Model Tuner")
        tuner = ModelTuner(X_train_five, y_train_five, X_val_five, y_val_five, X_train_sixth, y_train_sixth, X_val_sixth, y_val_sixth, time_step)
        model_five, model_sixth = tuner.tune_model(filter_months, max_epochs=max_epochs, hyperband_iterations=hyperband_iterations)

        # Guardar ambos modelos y sus hiperparámetros
        best_trial_five = tuner.oracle.get_best_trials(num_trials=1)[0]
        best_hps_values_five = best_trial_five.hyperparameters.values

        best_trial_sixth = tuner.oracle.get_best_trials(num_trials=1)[1]
        best_hps_values_sixth = best_trial_sixth.hyperparameters.values

        print("--------SAVE MODEL---------")
        ModelManager.save_model_and_hyperparameters(model_five, best_hps_values_five, model_path_five, hyperparameters_path_five)
        ModelManager.save_model_and_hyperparameters(model_sixth, best_hps_values_sixth, model_path_sixth, hyperparameters_path_sixth)

    print("---------------GRAFICAR toDOS LOS DATOS-----------------")
    DataPlotter.plot_data(X_all_data_five, y_all_data_five, "Original Data (First Five)")
    DataPlotter.plot_data(X_all_data_sixth, y_all_data_sixth, "Original Data (Sixth)")

    print("---------------GRAFICAR DATOS TEST-----------------")
    DataPlotter.plot_data(X_test_five, y_test_five, "Test Data (First Five)")
    DataPlotter.plot_data(X_test_sixth, y_test_sixth, "Test Data (Sixth)")

    data_inspector.inspect_data()

    evaluator_five = ModelEvaluator(model_five, X_test_five, y_test_five, scalers[:5], market_data)
    evaluator_five.evaluate()

    evaluator_sixth = ModelEvaluator(model_sixth, X_test_sixth, y_test_sixth, scalers[5:], market_data)
    evaluator_sixth.evaluate()

    print("--------------PREDICTION-----------------")
    # Obtener los últimos 10 elementos de X_test_five y X_test_sixth
    last_sequence_five = X_test_five[-1]
    last_sequence_sixth = X_test_sixth[-1]
    
    # Usar NextSequencePredictor para predecir la siguiente secuencia
    predictor_five = NextSequencePredictor(model_five, scalers[:5], time_step)
    next_five = predictor_five.predict_next_sequence(last_sequence_five)
    
    predictor_sixth = NextSequencePredictor(model_sixth, scalers[5:], time_step)
    next_sixth = predictor_sixth.predict_next_sequence(last_sequence_sixth)
    
    # Combinar los resultados y mostrarlos
    next_prediction = next_five + [next_sixth]
    print("Predicción para los siguientes 5 números y el sexto número:", next_prediction)

    print("--------------FINISH ALL-----------------")
except ValueError as e:
    print(f"Error: {e}")
except KeyError as e:
    print(f"KeyError: {e}")
