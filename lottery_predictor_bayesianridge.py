import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split

class LotteryPredictorBayesianRidge:
    def __init__(self, historical_data):
        self.historical_data = historical_data
        self.regular_scaler = MinMaxScaler(feature_range=(1, 43))  # Escalar directamente al rango
        self.complementary_scaler = MinMaxScaler(feature_range=(1, 16))  # Escalar directamente al rango
        self.regular_models = []
        self.complementary_model = None
        self.X_train = None
        self.y_train_regular = None
        self.y_train_complementary = None

    def prepare_data(self):
        # Preparar datos para números regulares (1-5)
        regular_data = self.historical_data.iloc[:, :5]
        complementary_data = self.historical_data.iloc[:, 5]

        # Preparar datos de entrenamiento
        self.X_train = regular_data.values
        self.y_train_regular = regular_data.values
        self.y_train_complementary = complementary_data.values.reshape(-1, 1)

    def train(self):
        self.prepare_data()
        
        # Entrenar modelos para números regulares con diferentes configuraciones
        for i in range(5):
            model = BayesianRidge(
                max_iter=500,  # Cambiado de n_iter a max_iter
                tol=1e-6,
                alpha_1=1e-6,
                alpha_2=1e-6,
                lambda_1=1e-6,
                lambda_2=1e-6,
                compute_score=True,
                fit_intercept=True
            )
            model.fit(self.X_train, self.y_train_regular[:, i])
            self.regular_models.append(model)
            print(f"Score for regular number {i+1}: {model.score(self.X_train, self.y_train_regular[:, i]):.4f}")

        # Entrenar modelo para número complementario con configuración específica
        self.complementary_model = BayesianRidge(
            max_iter=500,  # Cambiado de n_iter a max_iter
            tol=1e-6,
            alpha_1=1e-6,
            alpha_2=1e-6,
            lambda_1=1e-6,
            lambda_2=1e-6,
            compute_score=True,
            fit_intercept=True
        )
        self.complementary_model.fit(self.X_train, self.y_train_complementary.ravel())
        print(f"Score for complementary number: {self.complementary_model.score(self.X_train, self.y_train_complementary.ravel()):.4f}")

    def predict(self):
        if not self.regular_models or not self.complementary_model:
            raise Exception("Models not trained. Call train() first.")

        # Preparar datos de entrada
        last_numbers = self.historical_data.iloc[-1:, :5].values
        
        # Predecir números regulares con variabilidad
        regular_predictions = []
        used_numbers = set()
        
        for model in self.regular_models:
            # Realizar múltiples predicciones con ruido
            predictions = []
            for _ in range(15):  # 15 predicciones por número
                # Añadir ruido aleatorio a la entrada
                noise = np.random.normal(0, 0.15, last_numbers.shape)
                noisy_input = last_numbers + noise
                
                # Obtener predicción y sigma
                pred, std = model.predict(noisy_input, return_std=True)
                
                # Añadir variabilidad basada en la incertidumbre
                pred = pred[0] + np.random.normal(0, std[0])
                predictions.append(int(round(pred)))
            
            # Tomar la predicción más frecuente
            unique_preds, counts = np.unique(predictions, return_counts=True)
            pred = unique_preds[np.argmax(counts)]
            
            # Asegurar que el número es válido y único
            while pred in used_numbers or pred < 1 or pred > 43:
                if pred < 1:
                    pred = 1
                elif pred > 43:
                    pred = 43
                else:
                    # Usar números cercanos a la predicción original
                    window = 5
                    lower = max(1, pred - window)
                    upper = min(43, pred + window)
                    pred = np.random.randint(lower, upper + 1)
                    if pred in used_numbers:
                        pred = np.random.randint(1, 44)
            
            regular_predictions.append(pred)
            used_numbers.add(pred)
        
        regular_numbers = sorted(regular_predictions)
        
        # Predecir complementario con más variabilidad
        complementary_predictions = []
        
        # Realizar múltiples predicciones
        for _ in range(20):  # 20 predicciones para el complementario
            # Añadir ruido aleatorio
            noise = np.random.normal(0, 0.2, last_numbers.shape)
            noisy_input = last_numbers + noise
            
            # Obtener predicción y sigma
            pred, std = self.complementary_model.predict(noisy_input, return_std=True)
            
            # Añadir variabilidad basada en la incertidumbre
            pred = pred[0] + np.random.normal(0, std[0])
            complementary_predictions.append(int(round(pred)))
        
        # Filtrar predicciones válidas
        valid_predictions = [p for p in complementary_predictions if 1 <= p <= 16]
        if not valid_predictions:
            valid_predictions = list(range(1, 17))
        
        # Calcular la moda de las predicciones válidas
        unique_predictions, counts = np.unique(valid_predictions, return_counts=True)
        complementary = unique_predictions[np.argmax(counts)]
        
        # Si el complementario está en los números regulares, elegir otro
        while complementary in regular_numbers:
            valid_predictions = [p for p in valid_predictions if p != complementary]
            if not valid_predictions:
                # Si no hay más opciones, elegir un número cercano no usado
                available = [n for n in range(1, 17) if n not in regular_numbers]
                if available:
                    complementary = np.random.choice(available)
                else:
                    complementary = (complementary % 16) + 1
            else:
                unique_predictions, counts = np.unique(valid_predictions, return_counts=True)
                complementary = unique_predictions[np.argmax(counts)]
        
        # Imprimir información de depuración
        print("\nPredicción del complementario (BayesianRidge):")
        print(f"Último número complementario: {int(self.historical_data.iloc[-1, 5])}")
        print(f"Predicciones múltiples: {complementary_predictions}")
        print(f"Predicciones válidas: {valid_predictions}")
        print(f"Números únicos predichos: {unique_predictions}")
        print(f"Frecuencia de cada predicción: {dict(zip(unique_predictions, counts))}")
        print(f"Predicción final: {complementary}")
        
        return regular_numbers + [complementary]

    def _ensure_unique_numbers(self, numbers, min_val, max_val):
        unique_numbers = []
        used_numbers = set()
        
        for num in numbers:
            while num in used_numbers:
                # Intentar números cercanos primero
                window = 5
                lower = max(min_val, num - window)
                upper = min(max_val, num + window)
                num = np.random.randint(lower, upper + 1)
                if num in used_numbers:
                    num = np.random.randint(min_val, max_val + 1)
            used_numbers.add(num)
            unique_numbers.append(num)
            
        return unique_numbers 