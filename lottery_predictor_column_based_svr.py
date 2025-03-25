import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class LotteryPredictorColumnBasedSVR:
    def __init__(self, data):
        self.data = data
        # Separar escaladores para números regulares y complementario
        self.regular_scalers = [MinMaxScaler() for _ in range(5)]  # 5 números regulares
        self.complementary_scaler = MinMaxScaler()  # Escalador para complementario
        
        # Modelos SVR para números regulares y complementario
        self.regular_models = [SVR(kernel='rbf', C=100, epsilon=0.1) for _ in range(5)]
        self.complementary_model = SVR(kernel='rbf', C=100, epsilon=0.1)

    def prepare_regular_data(self, column_index):
        if isinstance(self.data, pd.DataFrame):
            column_data = self.data.iloc[:, column_index].values.reshape(-1, 1)
        else:
            column_data = self.data[:, column_index].reshape(-1, 1)
            
        # Escalar datos para la columna específica
        scaled_data = self.regular_scalers[column_index].fit_transform(column_data)
        X = scaled_data[:-1]
        y = scaled_data[1:]
        return X, y

    def prepare_complementary_data(self):
        if isinstance(self.data, pd.DataFrame):
            complementary_data = self.data.iloc[:, 5].values.reshape(-1, 1)
        else:
            complementary_data = self.data[:, 5].reshape(-1, 1)
            
        # Escalar datos del complementario
        scaled_data = self.complementary_scaler.fit_transform(complementary_data)
        X = scaled_data[:-1]
        y = scaled_data[1:]
        return X, y

    def train(self):
        # Entrenar modelos para números regulares
        for i in range(5):
            X, y = self.prepare_regular_data(i)
            # Split de datos para validación
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            # Entrenar modelo
            self.regular_models[i].fit(X_train, y_train.ravel())
            # Calcular score
            score = self.regular_models[i].score(X_test, y_test)
            print(f"Score for regular number {i+1}: {score:.4f}")
        
        # Entrenar modelo para el complementario
        X_comp, y_comp = self.prepare_complementary_data()
        # Split de datos para validación
        X_train_comp, X_test_comp, y_train_comp, y_test_comp = train_test_split(
            X_comp, y_comp, test_size=0.2, random_state=42
        )
        # Entrenar modelo
        self.complementary_model.fit(X_train_comp, y_train_comp.ravel())
        # Calcular score
        score = self.complementary_model.score(X_test_comp, y_test_comp)
        print(f"Score for complementary number: {score:.4f}")

    def predict(self):
        regular_predictions = []
        
        # Predecir números regulares
        for i in range(5):
            if isinstance(self.data, pd.DataFrame):
                last_data = self.regular_scalers[i].transform(
                    self.data.iloc[-1, i].reshape(1, -1)
                )
            else:
                last_data = self.regular_scalers[i].transform(
                    self.data[-1, i].reshape(1, -1)
                )
                
            prediction = self.regular_models[i].predict(last_data)
            predicted_value = self.regular_scalers[i].inverse_transform(
                prediction.reshape(-1, 1)
            )[0, 0]
            regular_predictions.append(predicted_value)
        
        # Redondear y ajustar números regulares
        regular_numbers = np.round(regular_predictions).astype(int)
        regular_numbers = np.clip(regular_numbers, 1, 43)
        regular_numbers = list(set(regular_numbers))
        
        # Asegurar que tenemos 5 números únicos
        while len(regular_numbers) < 5:
            new_num = np.random.randint(1, 44)
            if new_num not in regular_numbers:
                regular_numbers.append(new_num)
        regular_numbers = sorted(regular_numbers[:5])
        
        # Predecir complementario
        if isinstance(self.data, pd.DataFrame):
            last_complementary = self.complementary_scaler.transform(
                self.data.iloc[-1, 5].reshape(1, -1)
            )
        else:
            last_complementary = self.complementary_scaler.transform(
                self.data[-1, 5].reshape(1, -1)
            )
            
        complementary_pred = self.complementary_model.predict(last_complementary)
        complementary = self.complementary_scaler.inverse_transform(
            complementary_pred.reshape(-1, 1)
        )[0, 0]
        complementary = int(np.clip(round(complementary), 1, 16))
        
        # Asegurar que el complementario no está en los números regulares
        while complementary in regular_numbers:
            complementary = (complementary % 16) + 1
        
        return regular_numbers + [complementary] 