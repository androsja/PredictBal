# File: code/model/model_tuner.py
import os
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras_tuner.oracles import HyperbandOracle
from custom_tuner import CustomTuner
from model_manager import ModelManager
from model.model_builder_factory_five import ModelBuilderFactoryFive
from model.model_builder_factory_sixth import ModelBuilderFactorySixth
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras import backend as K
from tune_report_callback import TuneReportCallback
from utils.plot_saver import PlotSaver

class ModelTuner:
    def __init__(self, X_train_five, y_train_five, X_val_five, y_val_five, X_train_sixth, y_train_sixth, X_val_sixth, y_val_sixth, time_step):
        self.X_train_five = X_train_five
        self.y_train_five = y_train_five
        self.X_val_five = X_val_five
        self.y_val_five = y_val_five
        self.X_train_sixth = X_train_sixth
        self.y_train_sixth = y_train_sixth
        self.X_val_sixth = X_val_sixth
        self.y_val_sixth = y_val_sixth
        self.time_step = time_step
        self.tuner_five = None
        self.tuner_sixth = None
        self.oracle = None
        self.drive_dir = f'/Users/jflorezgaleano/Documents/JulianFlorez/PredictBal/models_ia/'

        if not os.path.exists(self.drive_dir):
            os.makedirs(self.drive_dir)

        self.model_path_five = os.path.join(self.drive_dir, f'model_five.h5')
        self.model_path_sixth = os.path.join(self.drive_dir, f'model_sixth.h5')
        self.hyperparameters_path_five = os.path.join(self.drive_dir, f'hyperparameters_five.pkl')
        self.hyperparameters_path_sixth = os.path.join(self.drive_dir, f'hyperparameters_sixth.pkl')

    def tune_model(self, filter_months, max_epochs=100, hyperband_iterations=1):

        self.oracle = HyperbandOracle(
            objective='val_loss',
            max_epochs=max_epochs,
            factor=2,
            hyperband_iterations=hyperband_iterations,
        )
        project_name_dir_in = 'lottery_prediction'

        # Tuning for first five numbers
        model_builder_five = ModelBuilderFactoryFive.create(self.time_step, 5)

        self.tuner_five = CustomTuner(
            oracle=self.oracle,
            X_val=self.X_val_five,
            y_val=self.y_val_five,
            plot_saver = PlotSaver(self.drive_dir, project_name_dir_in),
            hypermodel=model_builder_five.build_model,
            directory=self.drive_dir,
            project_name=project_name_dir_in + '_five',
            overwrite=False
        )

        tune_report_callback = TuneReportCallback(self.X_val_five, self.y_val_five)

        self.tuner_five.search(
            x=self.X_train_five, y=self.y_train_five,
            epochs=max_epochs,
            callbacks=[EarlyStopping(monitor='val_loss', patience=3), tune_report_callback]
        )

        best_trial_five = self.oracle.get_best_trials(num_trials=1)[0]
        best_hps_five = best_trial_five.hyperparameters

        with open(self.hyperparameters_path_five, 'wb') as f:
            pickle.dump(best_hps_five.values, f)

        model_five = self.tuner_five.hypermodel.build(best_hps_five)

        es = EarlyStopping(monitor='loss', patience=3)
        history = model_five.fit(self.X_train_five, self.y_train_five, epochs=max_epochs, batch_size=best_hps_five.get('batch_size'), validation_data=(self.X_val_five, self.y_val_five), callbacks=[es])

        self.print_model_summary(model_five, best_hps_five)
        self.print_evaluation_metrics(history, self.X_train_five, self.y_train_five, model_five)

        ModelManager.save_model_and_hyperparameters(model_five, best_hps_five, self.model_path_five, self.hyperparameters_path_five)

        # Tuning for sixth number
        model_builder_sixth = ModelBuilderFactorySixth.create(self.time_step, 6)

        self.tuner_sixth = CustomTuner(
            oracle=self.oracle,
            X_val=self.X_val_sixth,
            y_val=self.y_val_sixth,
            plot_saver = PlotSaver(self.drive_dir, project_name_dir_in),
            hypermodel=model_builder_sixth.build_model,
            directory=self.drive_dir,
            project_name=project_name_dir_in + '_sixth',
            overwrite=False
        )

        tune_report_callback = TuneReportCallback(self.X_val_sixth, self.y_val_sixth)

        self.tuner_sixth.search(
            x=self.X_train_sixth, y=self.y_train_sixth,
            epochs=max_epochs,
            callbacks=[EarlyStopping(monitor='val_loss', patience=3), tune_report_callback]
        )

        best_trial_sixth = self.oracle.get_best_trials(num_trials=1)[0]
        best_hps_sixth = best_trial_sixth.hyperparameters

        with open(self.hyperparameters_path_sixth, 'wb') as f:
            pickle.dump(best_hps_sixth.values, f)

        model_sixth = self.tuner_sixth.hypermodel.build(best_hps_sixth)

        es = EarlyStopping(monitor='loss', patience=3)
        history = model_sixth.fit(self.X_train_sixth, self.y_train_sixth, epochs=max_epochs, batch_size=best_hps_sixth.get('batch_size'), validation_data=(self.X_val_sixth, self.y_val_sixth), callbacks=[es])

        self.print_model_summary(model_sixth, best_hps_sixth)
        self.print_evaluation_metrics(history, self.X_train_sixth, self.y_train_sixth, model_sixth)

        ModelManager.save_model_and_hyperparameters(model_sixth, best_hps_sixth, self.model_path_sixth, self.hyperparameters_path_sixth)

        K.clear_session()

        return model_five, model_sixth

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

        val_loss = history.history['val_loss'][-1]
        print(f"Pérdida en datos de validación: {val_loss:.4f}")
