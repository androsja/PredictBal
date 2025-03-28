import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

class LotteryPredictorCatBoost:
    def __init__(self, data):
        self.data = data
        self.scaler_regular = MinMaxScaler(feature_range=(0, 1))
        self.scaler_complementary = MinMaxScaler(feature_range=(0, 1))
        self.regular_models = []
        self.complementary_model = None
        self.last_predictions = []

    def prepare_data(self):
        # Separar datos regulares y complementario
        regular_data = self.data.iloc[:, :5].values
        complementary_data = self.data.iloc[:, 5].values
        
        print("\nPreparando datos:")
        print(f"Últimos 5 números regulares: {regular_data[-1]}")
        print(f"Último complementario: {complementary_data[-1]}")
        
        # Preparar datos regulares
        X_regular = regular_data[:-1]
        y_regular = regular_data[1:]
        
        # Preparar datos complementario de manera más simple
        X_complementary = complementary_data[:-1].reshape(-1, 1)
        y_complementary = complementary_data[1:].reshape(-1, 1)
        
        # Escalar datos
        X_regular_scaled = self.scaler_regular.fit_transform(X_regular)
        y_regular_scaled = self.scaler_regular.transform(y_regular)
        
        X_complementary_scaled = self.scaler_complementary.fit_transform(X_complementary)
        y_complementary_scaled = self.scaler_complementary.transform(y_complementary)
        
        print(f"Rango de complementarios originales: [{min(complementary_data)}, {max(complementary_data)}]")
        print(f"Rango de complementarios escalados: [{min(X_complementary_scaled)}, {max(X_complementary_scaled)}]")
        
        return (X_regular_scaled, y_regular_scaled), (X_complementary_scaled, y_complementary_scaled)

    def train(self):
        # Obtener datos preparados
        (X_regular, y_regular), (X_complementary, y_complementary) = self.prepare_data()
        
        # Entrenar modelos para números regulares
        for i in range(5):
            model = CatBoostRegressor(
                iterations=500,
                depth=6,
                learning_rate=0.03,
                random_seed=42 + i,
                verbose=0
            )
            model.fit(X_regular, y_regular[:, i])
            self.regular_models.append(model)
        
        # Entrenar modelo para el complementario
        self.complementary_model = CatBoostRegressor(
            iterations=300,
            depth=4,
            learning_rate=0.05,
            random_seed=42,
            verbose=0
        )
        self.complementary_model.fit(X_complementary, y_complementary.ravel())

    def predict(self):
        if not self.regular_models or self.complementary_model is None:
            return [1, 2, 3, 4, 5, 1]
        
        # Predecir números regulares
        last_regular = self.data.iloc[-1:, :5].values
        last_regular_scaled = self.scaler_regular.transform(last_regular)
        
        regular_predictions = []
        for model in self.regular_models:
            pred = model.predict(last_regular_scaled)[0]
            regular_predictions.append(pred)
        
        regular_numbers = self.scaler_regular.inverse_transform([regular_predictions])[0]
        regular_numbers = np.round(regular_numbers).astype(int)
        regular_numbers = np.clip(regular_numbers, 1, 43)
        regular_numbers = list(set(regular_numbers))
        
        while len(regular_numbers) < 5:
            new_num = np.random.randint(1, 44)
            if new_num not in regular_numbers:
                regular_numbers.append(new_num)
        regular_numbers = sorted(regular_numbers[:5])
        
        # Predecir complementario con mayor variabilidad
        last_complementary = self.data.iloc[-1:, 5].values.reshape(-1, 1)
        last_complementary_scaled = self.scaler_complementary.transform(last_complementary)
        
        # Realizar predicción base
        base_pred = self.complementary_model.predict(last_complementary_scaled)[0]
        base_unscaled = self.scaler_complementary.inverse_transform([[base_pred]])[0][0]
        base_complementary = int(round(np.clip(base_unscaled, 1, 16)))
        
        # Generar número complementario con más variabilidad
        if np.random.random() < 0.3:  # 30% de las veces usar un número aleatorio
            complementary = np.random.randint(1, 17)
        else:  # 70% de las veces usar una variación del número predicho
            variation = np.random.randint(-2, 3)  # Variación de -2 a +2
            complementary = np.clip(base_complementary + variation, 1, 16)
        
        # Asegurar que el complementario no está en los números regulares
        original_complementary = complementary
        while complementary in regular_numbers:
            complementary = (complementary % 16) + 1
        
        prediction = regular_numbers + [complementary]
        self.last_predictions.append(prediction)
        
        print("\nDetalles de la predicción:")
        print(f"Números regulares finales: {regular_numbers}")
        print(f"Predicción base del complementario: {base_complementary}")
        print(f"Complementario final seleccionado: {complementary}")
        
        return prediction 