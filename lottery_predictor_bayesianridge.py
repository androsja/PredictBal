import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split

class LotteryPredictorBayesianRidge:
    def __init__(self, historical_data):
        self.historical_data = historical_data
        self.regular_scaler = MinMaxScaler()
        self.complementary_scaler = MinMaxScaler()
        self.regular_models = []
        self.complementary_model = None
        self.X_train = None
        self.y_train_regular = None
        self.y_train_complementary = None

    def prepare_data(self):
        # Preparar datos para números regulares (1-5)
        regular_data = self.historical_data.iloc[:, :5]
        complementary_data = self.historical_data.iloc[:, 5]

        # Escalar los datos
        self.X_train = self.regular_scaler.fit_transform(regular_data)
        
        # Preparar datos de entrenamiento para números regulares
        self.y_train_regular = regular_data.values
        
        # Preparar datos de entrenamiento para número complementario
        self.y_train_complementary = complementary_data.values.reshape(-1, 1)
        self.y_train_complementary = self.complementary_scaler.fit_transform(self.y_train_complementary)

    def train(self):
        self.prepare_data()
        
        # Entrenar modelos para números regulares
        for i in range(5):
            model = BayesianRidge(
                max_iter=300,  # Cambiado de n_iter_ a max_iter
                tol=1e-6,
                alpha_1=1e-6,
                alpha_2=1e-6,
                lambda_1=1e-6,
                lambda_2=1e-6
            )
            model.fit(self.X_train, self.y_train_regular[:, i])
            self.regular_models.append(model)
            print(f"Score for regular number {i+1}: {model.score(self.X_train, self.y_train_regular[:, i]):.4f}")

        # Entrenar modelo para número complementario
        self.complementary_model = BayesianRidge(
            max_iter=300,  # Cambiado de n_iter_ a max_iter
            tol=1e-6,
            alpha_1=1e-6,
            alpha_2=1e-6,
            lambda_1=1e-6,
            lambda_2=1e-6
        )
        self.complementary_model.fit(self.X_train, self.y_train_complementary.ravel())
        print(f"Score for complementary number: {self.complementary_model.score(self.X_train, self.y_train_complementary.ravel()):.4f}")

    def predict(self):
        if not self.regular_models or not self.complementary_model:
            raise Exception("Models not trained. Call train() first.")

        # Preparar datos de entrada
        last_numbers = self.historical_data.iloc[-1:, :5]
        X_pred = self.regular_scaler.transform(last_numbers)

        # Predecir números regulares
        regular_predictions = []
        for model in self.regular_models:
            pred = round(float(model.predict(X_pred)))
            # Asegurar que está en el rango 1-43
            pred = max(1, min(43, pred))
            regular_predictions.append(pred)

        # Asegurar que no hay duplicados
        regular_predictions = self._ensure_unique_numbers(regular_predictions, 1, 43)
        regular_predictions.sort()

        # Predecir número complementario
        complementary_pred = self.complementary_model.predict(X_pred)
        complementary_pred = self.complementary_scaler.inverse_transform(complementary_pred.reshape(-1, 1))
        complementary_number = round(float(complementary_pred[0]))
        
        # Asegurar que está en el rango 1-16
        complementary_number = max(1, min(16, complementary_number))
        
        # Asegurar que el complementario no está en los regulares
        while complementary_number in regular_predictions:
            complementary_number = (complementary_number % 16) + 1

        return regular_predictions + [complementary_number]

    def _ensure_unique_numbers(self, numbers, min_val, max_val):
        unique_numbers = []
        used_numbers = set()
        
        for num in numbers:
            while num in used_numbers:
                num = (num % max_val) + min_val
            used_numbers.add(num)
            unique_numbers.append(num)
            
        return unique_numbers 