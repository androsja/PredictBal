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
from keras_tuner.oracles import HyperbandOracle
from tensorflow.keras.optimizers import Adam
from model.model_manager import ModelManager
from model.model_evaluator import ModelEvaluator
from model.model_tuner import ModelTuner
from utils.data_plotter import DataPlotter
from keras.callbacks import EarlyStopping
from data.data_splitter import DataSplitter
from data.data_inspector import DataInspector
from utils.next_sequence_predictor import NextSequencePredictor

time_step = 100
max_epochs = 100  # Máximo número de épocas por trial
hyperband_iterations = 3  # Número de iteraciones de Hyperband

try:
    drive_dir = f'/Users/jflorezgaleano/Documents/JulianFlorez/PredictBalGit/models_ia/'
    model_path_five = f'{drive_dir}/five/model_five.h5'
    model_path_sixth = f'{drive_dir}/sixth/model_sixth.h5'
    hyperparameters_path_five = f'{drive_dir}/five/hyperparameters_five.pkl'
    hyperparameters_path_sixth = f'{drive_dir}/sixth/hyperparameters_sixth.pkl'

    print("Init DataSplitter")
    data_splitter = DataSplitter(time_step)
    data_splits = data_splitter.split_data(train_ratio=0.9, val_ratio=0.05)
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
    if ModelManager.model_exists(model_path_five, hyperparameters_path_five) and ModelManager.model_exists(model_path_sixth, hyperparameters_path_sixth):
        print("Cargando el mejor modelo guardado desde el disco local.")
        model_five, best_hps_values_five = ModelManager.load_model_and_hyperparameters(model_path_five, hyperparameters_path_five)
        model_sixth, best_hps_values_sixth = ModelManager.load_model_and_hyperparameters(model_path_sixth, hyperparameters_path_sixth)

        print("Hiperparámetros cargados desde el disco local:")
        print(best_hps_values_five)
        print(best_hps_values_sixth)
    else:
        if not ModelManager.model_exists(model_path_five, hyperparameters_path_five):
            print("Init Model Tuner for 'five'")
            tuner_five = ModelTuner(X_train_five, y_train_five, X_val_five, y_val_five, time_step, 5, 5, drive_dir, "five", "five")
            oracle_five = HyperbandOracle(
                objective='val_loss',
                max_epochs=max_epochs,
                factor=2,
                hyperband_iterations=hyperband_iterations
            )
            model_five, best_hps_five, _, _ = tuner_five.tune_model(oracle_five, max_epochs=max_epochs)
            ModelManager.save_model_and_hyperparameters(model_five, best_hps_five, model_path_five, hyperparameters_path_five)

        if not ModelManager.model_exists(model_path_sixth, hyperparameters_path_sixth):
            print("Init Model Tuner for 'sixth'")
            tuner_sixth = ModelTuner(X_train_sixth, y_train_sixth, X_val_sixth, y_val_sixth, time_step, 6, 1, drive_dir, "sixth", "sixth")
            oracle_sixth = HyperbandOracle(
                objective='val_loss',
                max_epochs=max_epochs,
                factor=2,
                hyperband_iterations=hyperband_iterations
            )
            model_sixth, best_hps_sixth, _, _ = tuner_sixth.tune_model(oracle_sixth, max_epochs=max_epochs)
            ModelManager.save_model_and_hyperparameters(model_sixth, best_hps_sixth, model_path_sixth, hyperparameters_path_sixth)

    print("---------------GRAFICAR toDOS LOS DATOS-----------------")
    DataPlotter.plot_data(X_all_data_five, y_all_data_five, "Original Data (First Five)")
    DataPlotter.plot_data(X_all_data_sixth, y_all_data_sixth, "Original Data (Sixth)")

    print("---------------GRAFICAR DATOS TEST-----------------")
    DataPlotter.plot_data(X_test_five, y_test_five, "Test Data (First Five)")
    DataPlotter.plot_data(X_test_sixth, y_test_sixth, "Test Data (Sixth)")

    data_inspector.inspect_data()

    evaluator_five = ModelEvaluator(model_five, X_test_five, y_test_five, scalers[:5], X_all_data_five)
    evaluator_five.evaluate()

    evaluator_sixth = ModelEvaluator(model_sixth, X_test_sixth, y_test_sixth, scalers[5:], X_all_data_sixth)
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
