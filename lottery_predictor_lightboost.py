from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb
import numpy as np

class LotteryPredictorLightBoost:
    def __init__(self, data):
        self.data = data
        # Crear modelos separados para números regulares y complementario
        self.regular_models = [lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            num_leaves=31,
            min_child_samples=20
        ) for _ in range(5)]
        self.complementary_model = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            num_leaves=15,
            min_child_samples=20
        )
        self.regular_scaler = MinMaxScaler()
        self.complementary_scaler = MinMaxScaler()

    def prepare_data(self):
        X = self.data.iloc[:-1].values
        y_regular = self.data.iloc[1:, :5].values
        y_complementary = self.data.iloc[1:, 5].values

        # Escalar los datos
        X_scaled = self.regular_scaler.fit_transform(X)
        return X_scaled, y_regular, y_complementary

    def train(self):
        X, y_regular, y_complementary = self.prepare_data()
        
        # Entrenar modelos para números regulares
        for i, model in enumerate(self.regular_models):
            score = model.fit(X, y_regular[:, i]).score(X, y_regular[:, i])
            print(f"LightBoost - Score para número regular {i+1}: {score:.4f}")

        # Entrenar modelo para número complementario
        score = self.complementary_model.fit(X, y_complementary).score(X, y_complementary)
        print(f"LightBoost - Score para número complementario: {score:.4f}")

    def predict(self):
        # Preparar datos para predicción
        X = self.data.values
        X_scaled = self.regular_scaler.transform(X)
        
        # Predecir números regulares
        regular_predictions = []
        for model in self.regular_models:
            pred = model.predict(X_scaled[-1].reshape(1, -1))[0]
            regular_predictions.append(int(round(pred)))

        # Predecir número complementario
        complementary = int(round(self.complementary_model.predict(X_scaled[-1].reshape(1, -1))[0]))
        
        # Asegurar que los números regulares estén en el rango 1-43
        regular_predictions = [max(1, min(43, n)) for n in regular_predictions]
        # Asegurar que el complementario esté en el rango 1-16
        complementary = max(1, min(16, complementary))
        
        # Asegurar que no hay duplicados en números regulares
        regular_predictions = list(set(regular_predictions))
        while len(regular_predictions) < 5:
            new_num = np.random.randint(1, 44)
            if new_num not in regular_predictions:
                regular_predictions.append(new_num)
        
        # Asegurar que el complementario no está en los números regulares
        while complementary in regular_predictions:
            complementary = np.random.randint(1, 17)
        
        return sorted(regular_predictions[:5]) + [complementary] 