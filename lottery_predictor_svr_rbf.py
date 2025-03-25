import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

class LotteryPredictorSVRRBF:
    def __init__(self, data):
        self.data = data
        self.scaler_regular = MinMaxScaler()
        self.scaler_complementary = MinMaxScaler()
        self.regular_models = []  # Modelos SVR para números regulares
        self.complementary_model = None  # Modelo XGBoost para complementario

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
        
        # Entrenar modelos SVR para números regulares
        for i in range(5):
            model = SVR(kernel='rbf', C=100, epsilon=0.1, gamma='scale')
            model.fit(X_regular, y_regular[:, i])
            self.regular_models.append(model)
        
        # Entrenar modelo XGBoost para el complementario
        self.complementary_model = XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        self.complementary_model.fit(
            X_complementary,
            y_complementary.ravel()
        )

    def predict(self):
        if not self.regular_models or self.complementary_model is None:
            return [1, 2, 3, 4, 5, 1]
        
        # Predecir números regulares con SVR
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
        
        # Predecir complementario con XGBoost
        last_complementary = self.data.iloc[-1, 5:].values.reshape(1, -1)
        last_complementary_scaled = self.scaler_complementary.transform(last_complementary)
        complementary = self.complementary_model.predict(last_complementary_scaled)[0]
        complementary = self.scaler_complementary.inverse_transform([[complementary]])[0][0]
        complementary = int(np.clip(round(complementary), 1, 16))
        
        # Asegurar que el complementario no está en los números regulares
        while complementary in regular_numbers:
            complementary = (complementary % 16) + 1
        
        return regular_numbers + [complementary] 