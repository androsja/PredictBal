# File: code/model/model_evaluator.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

class ModelEvaluator:
    def __init__(self, model, X_test, y_test, scaler, data):
        self.X_test = X_test
        self.y_test = y_test
        self.scaler = scaler
        self.data = data
        self.model = model

    def evaluate(self):
        print("---------------EVALUATE 1-----------------")

        y_pred = self.model.predict(self.X_test)

        y_test_inversed = self.scaler.inverse_transform(self.y_test)  # Ajustar forma
        y_pred_inversed = self.scaler.inverse_transform(y_pred)  # Ajustar forma
        print("---------------EVALUATE 2-----------------")
        # Gráfica 1: Datos completos
        plt.figure(figsize=(14, 7))
        plt.plot(self.data.index[-len(self.y_test):], y_test_inversed, label='Original')
        plt.plot(self.data.index[-len(self.y_test):], y_pred_inversed, label='Predicted', linestyle='--', color='yellow', marker='o', markersize=2, linewidth=2)
        print("---------------EVALUATE 3-----------------")
        plt.title('Final Model Prediction')
        plt.xlabel('Sample')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()
