import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import numpy as np
import pandas as pd



class TuneReportCallback(tf.keras.callbacks.Callback):
    def __init__(self, X_val, y_val):
        super(TuneReportCallback, self).__init__()
        self.X_val = X_val
        self.y_val = y_val

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        y_pred = self.model.predict(self.X_val)
        mse = mean_squared_error(self.y_val, y_pred)
        mae = mean_absolute_error(self.y_val, y_pred)
        r2 = tf.keras.metrics.R2Score()
        r2.update_state(self.y_val.reshape(-1, 1), y_pred.reshape(-1, 1))
        r2_result = r2.result().numpy()

        print(f"Epoch {epoch + 1}/{self.params['epochs']}")
        print(f"val_loss: {logs.get('val_loss')}")
        print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2_result:.4f}")