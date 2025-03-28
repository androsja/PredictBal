import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler

class LotteryPredictorSVR:
    def __init__(self, data):
        """
        Inicializa el predictor SVR para lotería.
        
        Args:
            data (pandas.DataFrame): Datos históricos de la lotería
        """
        self.data = data
        self.regular_models = []  # Modelos SVR para números regulares
        self.complementary_model = None  # Modelo SVR para complementario
        self.scaler_regular = MinMaxScaler(feature_range=(1, 43))
        self.scaler_complementary = MinMaxScaler(feature_range=(1, 16))
        self.sequence_length = 10
        
    def prepare_data(self):
        """
        Prepara los datos separando números regulares y complementario.
        
        Returns:
            tuple: ((X_regular, y_regular), (X_complementary, y_complementary))
        """
        numbers = []
        for row in self.data.values:
            row_str = str(row)
            row_str = row_str.replace('[', '').replace(']', '').replace('Name:', '').replace('dtype:', '').replace('object', '').strip()
            nums = [int(n.strip()) for n in row_str.split() if n.strip()]
            if nums:
                numbers.append(nums)
        
        numbers = np.array(numbers)
        
        # Separar datos regulares y complementario
        regular_data = numbers[:, :5]
        complementary_data = numbers[:, 5:]
        
        # Preparar datos regulares
        X_regular = regular_data[:-1]
        y_regular = regular_data[1:]
        
        # Preparar datos complementario
        X_complementary = complementary_data[:-1]
        y_complementary = complementary_data[1:]
        
        return (X_regular, y_regular), (X_complementary, y_complementary)
    
    def train(self):
        """
        Entrena modelos SVR separados para números regulares y complementario.
        """
        (X_regular, y_regular), (X_complementary, y_complementary) = self.prepare_data()
        
        # Entrenar modelos para números regulares con parámetros diferentes para cada modelo
        for i in range(5):
            svr = SVR(
                kernel='rbf',
                C=5 + i,  # Variar C para cada modelo
                epsilon=0.3,  # Aumentado para más variabilidad
                gamma='scale',  # Cambiado a scale para mejor adaptación
                cache_size=1000
            )
            svr.fit(X_regular, y_regular[:, i])
            self.regular_models.append(svr)
        
        # Entrenar modelo para complementario con parámetros más flexibles
        self.complementary_model = SVR(
            kernel='rbf',
            C=3,  # Reducido para evitar sobreajuste
            epsilon=0.4,  # Aumentado significativamente para más variabilidad
            gamma='scale',
            cache_size=1000
        )
        self.complementary_model.fit(X_complementary, y_complementary.ravel())
    
    def predict(self):
        """
        Realiza predicciones separadas para números regulares y complementario.
        
        Returns:
            list: Lista con los 6 números predichos (5 regulares + 1 complementario)
        """
        if not self.regular_models or not self.complementary_model:
            return [1, 2, 3, 4, 5, 1]
        
        # Obtener últimos datos para predicción
        last_regular = self.data.values[-1, :5].reshape(1, -1)
        last_complementary = self.data.values[-1, 5:].reshape(1, -1)
        
        # Predecir números regulares con variabilidad mejorada
        regular_predictions = []
        used_numbers = set()
        
        for i, model in enumerate(self.regular_models):
            # Realizar múltiples predicciones con ruido variable
            predictions = []
            for _ in range(7):  # Aumentado a 7 predicciones por modelo
                noise_scale = 0.15 + (i * 0.05)  # Ruido diferente para cada modelo
                noise = np.random.normal(0, noise_scale, last_regular.shape)
                noisy_input = last_regular + noise
                pred = model.predict(noisy_input)[0]
                
                # Añadir variabilidad adicional
                pred += np.random.normal(0, 1)
                pred = int(round(pred))
                
                if 1 <= pred <= 43:
                    predictions.append(pred)
            
            # Si no hay predicciones válidas, generar número aleatorio
            if not predictions:
                available = list(set(range(1, 44)) - used_numbers)
                pred = np.random.choice(available) if available else np.random.randint(1, 44)
            else:
                # Usar una combinación de moda y aleatoriedad
                unique_preds, counts = np.unique(predictions, return_counts=True)
                top_preds = unique_preds[counts >= np.max(counts) - 1]
                pred = np.random.choice(top_preds)
            
            # Asegurar número único
            attempts = 0
            while pred in used_numbers and attempts < 20:
                pred = np.random.randint(1, 44)
                attempts += 1
            
            regular_predictions.append(pred)
            used_numbers.add(pred)
        
        regular_numbers = sorted(regular_predictions)
        
        # Sistema híbrido para predicción del complementario
        complementary_predictions = []
        
        # 1. Predicciones basadas en modelo
        for _ in range(5):
            noise = np.random.normal(0, 0.3, last_complementary.shape)
            noisy_input = last_complementary + noise
            pred = self.complementary_model.predict(noisy_input)[0]
            pred = int(round(pred))
            if 1 <= pred <= 16:
                complementary_predictions.append(pred)
        
        # 2. Añadir algunos números aleatorios
        for _ in range(3):
            complementary_predictions.append(np.random.randint(1, 17))
        
        # 3. Considerar el histórico reciente
        recent_complementary = self.data.values[-5:, 5].astype(int)
        complementary_predictions.extend(recent_complementary)
        
        # Filtrar predicciones válidas y seleccionar
        valid_predictions = [p for p in complementary_predictions if 1 <= p <= 16]
        if not valid_predictions:
            complementary = np.random.randint(1, 17)
        else:
            # Usar una combinación de frecuencia y aleatoriedad
            unique_predictions, counts = np.unique(valid_predictions, return_counts=True)
            weights = counts / np.sum(counts)
            complementary = np.random.choice(unique_predictions, p=weights)
        
        # Evitar que el complementario esté en los números regulares
        attempts = 0
        while complementary in regular_numbers and attempts < 16:
            if attempts < 8:
                # Intentar con otro número del conjunto de predicciones
                complementary = np.random.choice(valid_predictions)
            else:
                # Si no funciona, generar uno nuevo
                complementary = np.random.randint(1, 17)
            attempts += 1
        
        # Imprimir información de depuración
        print("\nPredicción del complementario (SVR):")
        print(f"Último número complementario: {int(self.data.values[-1, 5])}")
        print(f"Predicciones múltiples: {complementary_predictions}")
        print(f"Predicciones válidas: {valid_predictions}")
        print(f"Números únicos predichos: {np.unique(valid_predictions)}")
        print(f"Predicción final: {complementary}")
        
        return regular_numbers + [complementary] 