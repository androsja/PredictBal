import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

class LotteryPredictorAdaBoost:
    def __init__(self, data, n_estimators=50):
        self.data = data
        self.scaler_regular = MinMaxScaler(feature_range=(0, 1))
        self.scaler_complementary = MinMaxScaler(feature_range=(0, 1))
        self.regular_models = []
        self.complementary_model = None
        self.n_estimators = n_estimators

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
        
        # Entrenar modelos AdaBoost para números regulares
        for i in range(5):
            model = AdaBoostRegressor(
                estimator=DecisionTreeRegressor(max_depth=4),
                n_estimators=self.n_estimators,
                learning_rate=0.1,
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
        
        # Entrenar modelo AdaBoost para el complementario
        self.complementary_model = AdaBoostRegressor(
            estimator=DecisionTreeRegressor(max_depth=3),  # Menor profundidad para el complementario
            n_estimators=self.n_estimators,
            learning_rate=0.1,
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
        
        # Predecir complementario con múltiples intentos
        last_complementary = self.data.iloc[-1, 5:].values.reshape(1, -1)
        last_complementary_scaled = self.scaler_complementary.transform(last_complementary)
        
        # Realizar múltiples predicciones del complementario
        complementary_predictions = []
        for _ in range(5):  # Hacer 5 predicciones
            complementary_pred = self.complementary_model.predict(last_complementary_scaled)[0]
            complementary = self.scaler_complementary.inverse_transform([[complementary_pred]])[0][0]
            complementary = int(np.clip(round(complementary), 1, 16))
            complementary_predictions.append(complementary)
        
        # Analizar la distribución de predicciones
        unique_predictions, counts = np.unique(complementary_predictions, return_counts=True)
        
        # Si hay predicciones únicas, usar la más frecuente
        if len(unique_predictions) > 1:
            most_frequent = unique_predictions[np.argmax(counts)]
            complementary = most_frequent
        else:
            # Si todas las predicciones son iguales, usar la predicción original
            complementary = complementary_predictions[0]
        
        # Asegurar que el complementario no está en los números regulares
        while complementary in regular_numbers:
            # Si el complementario está en los números regulares, elegir el siguiente número disponible
            available_numbers = [n for n in range(1, 17) if n not in regular_numbers]
            if available_numbers:
                complementary = available_numbers[0]
            else:
                # Si no hay números disponibles, usar el siguiente número después del último
                complementary = (complementary % 16) + 1
        
        # Imprimir información de depuración
        print(f"\nPredicción del complementario:")
        print(f"Predicciones múltiples: {complementary_predictions}")
        print(f"Números únicos predichos: {unique_predictions}")
        print(f"Frecuencia de cada número: {counts}")
        print(f"Complementario final seleccionado: {complementary}")
        
        return regular_numbers + [complementary] 