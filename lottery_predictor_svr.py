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
        self.scaler_regular = MinMaxScaler(feature_range=(0, 1))
        self.scaler_complementary = MinMaxScaler(feature_range=(0, 1))
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
        
        # Escalar datos
        X_regular_scaled = self.scaler_regular.fit_transform(X_regular)
        y_regular_scaled = self.scaler_regular.transform(y_regular)
        
        X_complementary_scaled = self.scaler_complementary.fit_transform(X_complementary)
        y_complementary_scaled = self.scaler_complementary.transform(y_complementary)
        
        return (X_regular_scaled, y_regular_scaled), (X_complementary_scaled, y_complementary_scaled)
    
    def train(self):
        """
        Entrena modelos SVR separados para números regulares y complementario.
        """
        (X_regular, y_regular), (X_complementary, y_complementary) = self.prepare_data()
        
        # Entrenar modelos para números regulares
        for i in range(5):
            svr = SVR(
                kernel='rbf',
                C=100,
                epsilon=0.1,
                gamma='scale',
                cache_size=1000
            )
            svr.fit(X_regular, y_regular[:, i])
            self.regular_models.append(svr)
        
        # Entrenar modelo para complementario
        self.complementary_model = SVR(
            kernel='rbf',
            C=100,
            epsilon=0.1,
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
        
        # Obtener último dato para predicción
        last_regular = self.data.values[-1, :5].reshape(1, -1)
        last_complementary = self.data.values[-1, 5:].reshape(1, -1)
        
        # Escalar datos
        last_regular_scaled = self.scaler_regular.transform(last_regular)
        last_complementary_scaled = self.scaler_complementary.transform(last_complementary)
        
        # Predecir números regulares
        regular_predictions = []
        for model in self.regular_models:
            pred = model.predict(last_regular_scaled)[0]
            regular_predictions.append(pred)
        
        # Convertir predicciones a números reales
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
        complementary = self.complementary_model.predict(last_complementary_scaled)[0]
        complementary = self.scaler_complementary.inverse_transform([[complementary]])[0][0]
        complementary = int(np.clip(round(complementary), 1, 16))
        
        # Asegurar que el complementario no está en los números regulares
        while complementary in regular_numbers:
            complementary = (complementary % 16) + 1
        
        return regular_numbers + [complementary] 