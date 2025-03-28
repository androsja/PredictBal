import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

class LotteryPredictorBagging:
    def __init__(self, data, n_estimators=100):
        self.data = data
        self.regular_models = []
        self.complementary_model = None
        self.n_estimators = n_estimators
        self.history = []
        self.window_size = 3

    def prepare_data(self):
        # Separar datos regulares y complementario
        regular_data = self.data.iloc[:, :5].values
        complementary_data = self.data.iloc[:, 5:].values
        
        # Crear ventanas deslizantes para números regulares
        X_regular, y_regular = [], []
        for i in range(len(regular_data) - self.window_size):
            window = regular_data[i:i+self.window_size]
            target = regular_data[i+self.window_size]
            X_regular.append(window.flatten())
            y_regular.append(target)
        
        X_regular = np.array(X_regular)
        y_regular = np.array(y_regular)
        
        # Para el complementario, usar ventana más corta
        X_complementary, y_complementary = [], []
        for i in range(len(complementary_data) - 1):
            # Usar últimos números regulares y complementario como features
            features = np.concatenate([regular_data[i], complementary_data[i]])
            X_complementary.append(features)
            y_complementary.append(complementary_data[i+1])
        
        X_complementary = np.array(X_complementary)
        y_complementary = np.array(y_complementary)
        
        return (X_regular, y_regular), (X_complementary, y_complementary)

    def train(self):
        (X_regular, y_regular), (X_complementary, y_complementary) = self.prepare_data()
        
        # Split de datos para números regulares
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
            X_regular, y_regular, test_size=0.2, random_state=42
        )
        
        # Entrenar modelos para números regulares
        for i in range(5):
            max_depth = np.random.randint(4, 10)
            min_samples = np.random.randint(2, 6)
            
            model = BaggingRegressor(
                estimator=DecisionTreeRegressor(
                    max_depth=max_depth,
                    min_samples_split=min_samples,
                    min_samples_leaf=min_samples // 2
                ),
                n_estimators=self.n_estimators,
                max_samples=0.7 + np.random.rand() * 0.3,
                max_features=0.7 + np.random.rand() * 0.3,
                bootstrap=True,
                bootstrap_features=True,
                random_state=None
            )
            model.fit(X_train_reg, y_train_reg[:, i])
            self.regular_models.append(model)
            score = model.score(X_test_reg, y_test_reg[:, i])
            print(f"Score for regular number {i+1}: {score:.4f}")
        
        # Split de datos para complementario
        X_train_comp, X_test_comp, y_train_comp, y_test_comp = train_test_split(
            X_complementary, y_complementary, test_size=0.2, random_state=42
        )
        
        # Modelo para el complementario
        self.complementary_model = BaggingRegressor(
            estimator=DecisionTreeRegressor(
                max_depth=8,
                min_samples_split=2,
                min_samples_leaf=1
            ),
            n_estimators=self.n_estimators * 2,
            max_samples=0.8,
            max_features=0.9,
            bootstrap=True,
            bootstrap_features=True,
            random_state=None
        )
        self.complementary_model.fit(X_train_comp, y_train_comp.ravel())
        score = self.complementary_model.score(X_test_comp, y_test_comp.ravel())
        print(f"Score for complementary number: {score:.4f}")

    def predict(self):
        if not self.regular_models or self.complementary_model is None:
            return [1, 2, 3, 4, 5, 1]
        
        # Preparar datos para predicción de números regulares
        last_regular_data = self.data.iloc[-self.window_size:, :5].values
        last_regular_window = last_regular_data.flatten().reshape(1, -1)
        
        # Realizar múltiples predicciones para cada número regular
        regular_predictions = []
        used_numbers = set()
        
        for model in self.regular_models:
            predictions = []
            for _ in range(7):  # Aumentado a 7 predicciones por modelo
                # Agregar ruido aleatorio para obtener diferentes predicciones
                noise = np.random.normal(0, 0.05, last_regular_window.shape)
                noisy_input = last_regular_window + noise
                pred = model.predict(noisy_input)[0]
                pred = int(round(pred))
                if 1 <= pred <= 43:
                    predictions.append(pred)
            
            # Seleccionar predicción final para este número
            if not predictions:
                available = list(set(range(1, 44)) - used_numbers)
                pred = np.random.choice(available) if available else np.random.randint(1, 44)
            else:
                # Usar una combinación de moda y aleatoriedad
                unique_preds, counts = np.unique(predictions, return_counts=True)
                weights = counts / np.sum(counts)
                pred = np.random.choice(unique_preds, p=weights)
            
            # Asegurar número único
            attempts = 0
            while pred in used_numbers and attempts < 20:
                pred = np.random.randint(1, 44)
                attempts += 1
            
            regular_predictions.append(pred)
            used_numbers.add(pred)
        
        regular_numbers = sorted(regular_predictions)
        
        # Preparar datos para predicción del complementario
        last_data = np.concatenate([
            self.data.iloc[-1, :5].values,
            self.data.iloc[-1, 5:].values
        ]).reshape(1, -1)
        
        # Sistema híbrido mejorado para predicción del complementario
        complementary_predictions = []
        
        # 1. Predicciones basadas en modelo con mayor variabilidad
        for _ in range(8):  # Reducido de 10 a 8 para dar más peso a otras fuentes
            noise_scale = 0.2 + np.random.rand() * 0.1  # Ruido variable entre 0.2 y 0.3
            noise = np.random.normal(0, noise_scale, last_data.shape)
            noisy_input = last_data + noise
            pred = self.complementary_model.predict(noisy_input)[0]
            pred = int(round(pred))
            if 1 <= pred <= 16:
                complementary_predictions.append(pred)
        
        # 2. Añadir números aleatorios estratégicos
        for _ in range(6):  # Aumentado de 5 a 6
            # Generar número evitando los que ya aparecen mucho
            counts = {}
            for p in complementary_predictions:
                counts[p] = counts.get(p, 0) + 1
            
            # Evitar números que ya aparecen más de 2 veces
            available = [n for n in range(1, 17) if counts.get(n, 0) <= 2]
            if available:
                new_num = np.random.choice(available)
            else:
                new_num = np.random.randint(1, 17)
            
            complementary_predictions.append(new_num)
        
        # 3. Considerar histórico reciente de forma balanceada
        recent_history = self.data.iloc[-4:, 5].values.astype(int)  # Últimos 4 números
        # Añadir cada número histórico solo una vez
        for num in np.unique(recent_history):
            if 1 <= num <= 16:
                complementary_predictions.append(num)
        
        # Balancear las predicciones usando pesos suavizados
        valid_predictions = [p for p in complementary_predictions if 1 <= p <= 16]
        if not valid_predictions:
            complementary = np.random.randint(1, 17)
        else:
            unique_predictions, counts = np.unique(valid_predictions, return_counts=True)
            # Usar raíz cuadrada para suavizar los pesos
            weights = np.sqrt(counts) / np.sum(np.sqrt(counts))
            complementary = np.random.choice(unique_predictions, p=weights)
        
        # Evitar que el complementario esté en los números regulares
        attempts = 0
        max_attempts = 25  # Aumentado para más intentos
        while complementary in regular_numbers and attempts < max_attempts:
            if attempts < 15:
                # Primero intentar con números disponibles no usados
                available = list(set(range(1, 17)) - set(regular_numbers))
                if available:
                    complementary = np.random.choice(available)
                else:
                    # Si no hay disponibles, usar predicciones válidas con peso reducido
                    weights = np.sqrt(counts) / np.sum(np.sqrt(counts))
                    complementary = np.random.choice(unique_predictions, p=weights)
            else:
                # Como último recurso, número aleatorio
                complementary = np.random.randint(1, 17)
            attempts += 1
        
        # Imprimir información detallada de depuración
        print("\nPredicción del complementario (Bagging):")
        print(f"Último número complementario: {int(self.data.iloc[-1, 5])}")
        print(f"Predicciones múltiples: {complementary_predictions}")
        print(f"Predicciones válidas: {valid_predictions}")
        print(f"Números únicos predichos: {np.unique(valid_predictions)}")
        print(f"Distribución de predicciones: {dict(zip(unique_predictions, weights))}")
        print(f"Intentos para evitar duplicados: {attempts}")
        print(f"Predicción final: {complementary}")
        
        prediction = regular_numbers + [complementary]
        self.history.append(prediction)
        return prediction 