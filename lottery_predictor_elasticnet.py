import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split

class LotteryPredictorElasticNet:
    def __init__(self, data, alpha=1.0, l1_ratio=0.5):
        self.data = data
        self.scaler_regular = MinMaxScaler(feature_range=(0, 1))
        self.scaler_complementary = MinMaxScaler(feature_range=(0, 1))
        self.regular_models = []
        self.complementary_model = None
        self.alpha = alpha
        self.l1_ratio = l1_ratio

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
        
        # Split de datos para números regulares
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
            X_regular, y_regular, test_size=0.2, random_state=42
        )
        
        # Entrenar modelos ElasticNet para números regulares
        for i in range(5):
            model = ElasticNet(
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                random_state=42
            )
            model.fit(X_train_reg, y_train_reg[:, i])
            self.regular_models.append(model)
            score = model.score(X_test_reg, y_test_reg[:, i])
            print(f"Score for regular number {i+1}: {score:.4f}")
        
        # Split de datos para complementario
        X_train_comp, X_test_comp, y_train_comp, y_test_comp = train_test_split(
            X_complementary, y_complementary, test_size=0.2, random_state=42
        )
        
        # Entrenar modelo ElasticNet para el complementario
        self.complementary_model = ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            random_state=42
        )
        self.complementary_model.fit(X_train_comp, y_train_comp.ravel())
        score = self.complementary_model.score(X_test_comp, y_test_comp.ravel())
        print(f"Score for complementary number: {score:.4f}")

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
        
        # Realizar predicción del complementario
        complementary_pred = self.complementary_model.predict(last_complementary_scaled)[0]
        complementary = self.scaler_complementary.inverse_transform([[complementary_pred]])[0][0]
        complementary = int(np.clip(round(complementary), 1, 16))
        
        # Asegurar que el complementario no está en los números regulares
        while complementary in regular_numbers:
            complementary = (complementary % 16) + 1
        
        return regular_numbers + [complementary] 