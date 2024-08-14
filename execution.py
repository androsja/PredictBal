import sys
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error

def add_subdirectories_to_syspath(directory):
    for dirpath, dirnames, filenames in os.walk(directory):
        sys.path.append(dirpath)

code_directory = os.path.join(os.path.dirname(__file__), 'code')
add_subdirectories_to_syspath(code_directory)

from tensorflow.keras.models import load_model
from keras_tuner.tuners import Hyperband
from tensorflow.keras.optimizers import Adam
from model.model_manager import ModelManager
from utils.data_plotter import DataPlotter
from keras.callbacks import EarlyStopping
from data.data_splitter import DataSplitter
from data.data_inspector import DataInspector
from utils.next_sequence_predictor import NextSequencePredictor

# Importar y registrar la métrica personalizada
from custom_metrics import CustomMeanSquaredError

time_step = 1
max_epochs = 500  # Máximo número de épocas por trial
hyperband_iterations = 3  # Número de iteraciones de Hyperband

def build_model(hp):
    model = tf.keras.Sequential()
    num_layers = hp.Int('num_layers', 4, 5)
    for i in range(num_layers):
        model.add(tf.keras.layers.LSTM(
            units=hp.Int('units', min_value=32, max_value=64, step=32), 
            return_sequences=True if i < num_layers - 1 else False,
            input_shape=(time_step, 5)  # Ajusta los valores de time_step y 5 según corresponda
        ))
        model.add(tf.keras.layers.Dropout(rate=hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)))
    model.add(tf.keras.layers.Dense(units=5, activation='linear'))  # Ajusta el valor de 5 según corresponda
    model.compile(
        optimizer=Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
        loss='mean_squared_error'
    )
    return model

try:
    drive_dir = f'models_ia/'
    model_path_five = f'{drive_dir}/five/model_five.h5'
    model_path_sixth = f'{drive_dir}/sixth/model_sixth.h5'
    hyperparameters_path_five = f'{drive_dir}/five/hyperparameters_five.pkl'
    hyperparameters_path_sixth = f'{drive_dir}/sixth/hyperparameters_sixth.pkl'

    print("Init DataSplitter")
    data_splitter = DataSplitter(time_step)
    data_splits = data_splitter.split_data(train_ratio=1.0, val_ratio=0.0)
    
    print("Data split completed")
    print(f"Data splits keys: {data_splits.keys()}")
    for key, value in data_splits.items():
        if isinstance(value, tuple):
            for sub_value in value:
                print(f"{key} shape: {sub_value.shape}")
        else:
            print(f"{key} value: {value}")

    print("Init DataInspector")
    data_inspector = DataInspector(data_splits)
    data_inspector.inspect_data()
    print("Data inspection completed")

    X_all_data_five, y_all_data_five = data_splits['all_data']
    X_train_five, y_train_five = data_splits['train_five']
    X_test_five, y_test_five = data_splits['test_five']
    scalers = data_splits['scalers']

    print("Init Search for model")
    if ModelManager.model_exists(model_path_five, hyperparameters_path_five):
        print("Cargando el mejor modelo guardado desde el disco local.")
        try:
            print(f"Loading model from {model_path_five} with custom objects: {{'mean_squared_error': CustomMeanSquaredError}}")
            model_five, best_hps_values_five = ModelManager.load_model_and_hyperparameters(model_path_five, hyperparameters_path_five)
            print("Hiperparámetros cargados desde el disco local:")
            print(best_hps_values_five)
        except Exception as e:
            print(f"Error loading model and hyperparameters: {e}")
            raise
    else:
        print("Init Model Tuner for 'five'")
        tuner_five = Hyperband(
            hypermodel=build_model,
            objective='loss',
            max_epochs=max_epochs,
            factor=2,
            hyperband_iterations=hyperband_iterations,
            directory=drive_dir,
            project_name="five"
        )

        tuner_five.search(X_train_five, y_train_five, epochs=max_epochs, validation_split=0.2, callbacks=[EarlyStopping('val_loss', patience=5)])
        best_hps_five = tuner_five.get_best_hyperparameters(num_trials=1)[0]
        model_five = build_model(best_hps_five)
        ModelManager.save_model_and_hyperparameters(model_five, best_hps_five, model_path_five, hyperparameters_path_five)

    print("---------------GRAFICAR toDOS LOS DATOS-----------------")
    DataPlotter.plot_data(X_all_data_five, y_all_data_five, "Original Data (First Five)")

    print("---------------GRAFICAR DATOS TEST-----------------")
    if X_test_five.size > 0:
        DataPlotter.plot_data(X_test_five, y_test_five, "Test Data (First Five)")

    data_inspector.inspect_data()
    print("Data inspection completed again")


    print("--------------PREDICTION-----------------")
    # Obtener los últimos elementos de X_train_five
    last_sequence_five = X_train_five[-1]

    # Usar NextSequencePredictor para predecir la siguiente secuencia
    predictor_five = NextSequencePredictor(model_five, scalers, time_step)
    next_five = predictor_five.predict_next_sequence(last_sequence_five)

    # Mostrar la predicción
    print("Predicción para los siguientes 5 números:", next_five)

    print("--------------FINISH ALL-----------------")
except ValueError as e:
    print(f"Error: {e}")
except KeyError as e:
    print(f"KeyError: {e}")
except FileNotFoundError as e:
    print(f"FileNotFoundError: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
