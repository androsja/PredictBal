import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

class LotteryPredictorXGBoost:
    def __init__(self, data):
        self.data = data
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.models = []  # Para los primeros 5 números
        self.complementary_model = None  # Para el número complementario

    def prepare_data(self):
        # Escalar los datos
        scaled_data = self.scaler.fit_transform(self.data)
        X = scaled_data[:-1]
        y = scaled_data[1:]
        return X, y

    def train(self):
        X, y = self.prepare_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Entrenar un modelo XGBoost para cada uno de los primeros 5 números
        for i in range(5):  # Solo los primeros 5 números
            model = XGBRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train[:, i])
            self.models.append(model)
            score = model.score(X_test, y_test[:, i])
            print(f"Score for regular number {i+1}: {score:.2f}")

        # Entrenar un modelo específico para el complementario (última columna)
        self.complementary_model = XGBRegressor(n_estimators=100, random_state=42)
        self.complementary_model.fit(X_train, y_train[:, 5])  # Índice 5 para el complementario
        score = self.complementary_model.score(X_test, y_test[:, 5])
        print(f"Score for complementary number: {score:.2f}")

    def predict(self):
        # Usar el último conjunto de datos para predecir el siguiente
        last_data = self.scaler.transform(self.data)[-1].reshape(1, -1)
        
        # Predecir los primeros 5 números
        regular_predictions = []
        for i in range(5):
            pred = self.models[i].predict(last_data)[0]
            regular_predictions.append(pred)
        
        # Convertir predicciones a números reales
        regular_numbers = self.scaler.inverse_transform([regular_predictions + [0]])[0][:5]
        regular_numbers = np.round(regular_numbers).astype(int)
        
        # Asegurar que los números regulares están en el rango correcto y son únicos
        regular_numbers = np.clip(regular_numbers, 1, 43)
        regular_numbers = list(set(regular_numbers))
        while len(regular_numbers) < 5:
            new_num = np.random.randint(1, 44)
            if new_num not in regular_numbers:
                regular_numbers.append(new_num)
        regular_numbers = sorted(regular_numbers[:5])
        
        # Predecir el complementario
        complementary = self.complementary_model.predict(last_data)[0]
        complementary = self.scaler.inverse_transform([[0, 0, 0, 0, 0, complementary]])[0][5]
        complementary = int(np.clip(round(complementary), 1, 16))
        
        # Asegurar que el complementario no está en los números regulares
        while complementary in regular_numbers:
            complementary = (complementary % 16) + 1

        return regular_numbers + [complementary] 