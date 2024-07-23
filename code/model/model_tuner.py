# File: code/model/model_tuner.py
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import pickle
import tensorflow as tf
from keras.callbacks import EarlyStopping
from custom_tuner import CustomTuner
from model_manager import ModelManager
from model_builder_factory import ModelBuilderFactory
from tensorflow.keras import backend as K
from utils.plot_saver import PlotSaver
from keras_tuner import RandomSearch
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

class ModelTuner:
    def __init__(self, X_train, y_train, time_step, output_size, max_units, directory, project_name):
        self.X_train = X_train
        self.y_train = y_train
        self.time_step = time_step
        self.output_size = output_size
        self.max_units = max_units
        self.directory = directory
        self.project_name = project_name

    def build_model(self, hp):
        model = Sequential()
        num_layers = hp.Int('num_layers', 1, 5)
        for i in range(num_layers):
            model.add(LSTM(units=hp.Int('units', min_value=32, max_value=self.max_units, step=32), 
                           return_sequences=True if i < num_layers - 1 else False, 
                           input_shape=(self.time_step, self.X_train.shape[2])))
            model.add(Dropout(rate=hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)))
        model.add(Dense(units=self.output_size, activation='linear'))
        model.compile(optimizer=Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')), loss='mean_squared_error')
        return model

    def tune_model(self, tuner, max_epochs):
        print(f"Tuner summary:\n{tuner}")
        tuner.search(self.X_train, self.y_train, epochs=max_epochs, validation_split=0.2, callbacks=[EarlyStopping('val_loss', patience=5)])
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        model = tuner.hypermodel.build(best_hps)
        return model, best_hps

    def print_details(self):
        print(f"ModelTuner initialized with X_train shape: {self.X_train.shape}, y_train shape: {self.y_train.shape}, time_step: {self.time_step}, output_size: {self.output_size}, max_units: {self.max_units}, directory: {self.directory}, project_name: {self.project_name}")


    def print_model_summary(self, model, best_hps):
        print("Model summary:")
        model.summary()
        print(f"Optimal hyperparameters: Units: {best_hps.get('units')}, Layers: {best_hps.get('num_layers')}, Learning rate: {best_hps.get('learning_rate')}, Batch size: {best_hps.get('batch_size')}")

    def print_evaluation_metrics(self, history, X_train, y_train, model):
        y_pred_train = model.predict(X_train)
        mse_train = mean_squared_error(y_train, y_pred_train)
        mae_train = mean_absolute_error(y_train, y_pred_train)
        r2_train = tf.keras.metrics.R2Score()
        r2_train.update_state(y_train.reshape(-1, 1), y_pred_train.reshape(-1, 1))
        r2_train_result = r2_train.result().numpy()

        print(f"Evaluación en datos de entrenamiento: MSE = {mse_train:.4f}, MAE = {mae_train:.4f}, R² = {r2_train_result:.4f}")
