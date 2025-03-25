import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

class LotteryPredictorRFVariant:
    def __init__(self, data):
        self.data = data
        self.scaler_regular = MinMaxScaler()
        self.scaler_complementary = MinMaxScaler()
        self.regular_models = []
        self.complementary_model = None

    def prepare_data(self):
        # Separar datos regulares y complementario
        regular_data = self.data.iloc[:, :5].values
        complementary_data = self.data.iloc[:, 5:].values
        
        # Preparar datos regulares
        X_regular = regular_data[:-1]
        y_regular = regular_data[1:]
        
        # Preparar datos complementario
        X_complementary = complementary_data[:-1]
        y_complementary = complementary_data[1:]
        
        # Escalar datos
        X_regular_scaled = self.scaler_regular.fit_transform(X_regular)
        y_regular_scaled = self.scaler_regular.transform(y_regular)
        
        X_complementary_scaled = self.scaler_complementary.fit_transform(X_complementary)
        y_complementary_scaled = self.scaler_complementary.transform(y_complementary)
        
        return (X_regular_scaled, y_regular_scaled), (X_complementary_scaled, y_complementary_scaled)

    def train(self):
        # Obtener datos preparados
        (X_regular, y_regular), (X_complementary, y_complementary) = self.prepare_data()
        
        # Entrenar modelos para números regulares con hiperparámetros específicos
        for i in range(5):
            model = RandomForestRegressor(
                n_estimators=150,
                max_depth=15,
                random_state=42
            )
            model.fit(X_regular, y_regular[:, i])
            self.regular_models.append(model)
        
        # Entrenar modelo para el complementario con hiperparámetros ajustados
        self.complementary_model = RandomForestRegressor(
            n_estimators=150,
            max_depth=15,
            random_state=42
        )
        self.complementary_model.fit(X_complementary, y_complementary.ravel())

    def predict(self):
        if not self.regular_models or self.complementary_model is None:
            return [1, 2, 3, 4, 5, 1]
        
        # Predecir números regulares
        last_regular = self.data.iloc[-1, :5].values.reshape(1, -1)
        last_regular_scaled = self.scaler_regular.transform(last_regular)
        
        regular_predictions = []
        for model in self.regular_models:
            pred = model.predict(last_regular_scaled)[0]
            regular_predictions.append(pred)
        
        # Convertir predicciones regulares a números reales
        regular_numbers = self.scaler_regular.inverse_transform([regular_predictions])[0]
        regular_numbers = np.round(regular_numbers).astype(int)
        
        # Asegurar que los números regulares están en el rango correcto y son únicos
        regular_numbers = np.clip(regular_numbers, 1, 43)
        regular_numbers = list(set(regular_numbers))
        while len(regular_numbers) < 5:
            new_num = np.random.randint(1, 44)
            if new_num not in regular_numbers:
                regular_numbers.append(new_num)
        regular_numbers = sorted(regular_numbers[:5])
        
        # Predecir complementario
        last_complementary = self.data.iloc[-1, 5:].values.reshape(1, -1)
        last_complementary_scaled = self.scaler_complementary.transform(last_complementary)
        
        # Obtener múltiples predicciones para el complementario
        complementary_predictions = []
        n_predictions = 5  # Número de predicciones a probar
        
        for _ in range(n_predictions):
            pred = self.complementary_model.predict(last_complementary_scaled)[0]
            pred_unscaled = self.scaler_complementary.inverse_transform([[pred]])[0][0]
            pred_rounded = int(np.clip(round(pred_unscaled), 1, 16))
            if pred_rounded not in regular_numbers:
                complementary_predictions.append(pred_rounded)
        
        # Seleccionar el complementario
        if complementary_predictions:
            complementary = complementary_predictions[0]
        else:
            # Si no hay predicciones válidas, generar un número válido
            complementary = 1
            while complementary in regular_numbers:
                complementary = (complementary % 16) + 1
        
        return regular_numbers + [complementary] 