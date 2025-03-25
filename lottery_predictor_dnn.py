import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

class LotteryPredictorDNN:
    def __init__(self, data):
        self.data = data
        self.scaler_regular = MinMaxScaler()
        self.scaler_complementary = MinMaxScaler()
        self.regular_model = None
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

    def create_regular_model(self, input_shape):
        model = Sequential([
            Dense(64, activation='relu', input_shape=(input_shape,)),
            BatchNormalization(),
            Dropout(0.2),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(5)  # 5 números regulares
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def create_complementary_model(self, input_shape):
        model = Sequential([
            Dense(32, activation='relu', input_shape=(input_shape,)),
            BatchNormalization(),
            Dropout(0.2),
            Dense(16, activation='relu'),
            BatchNormalization(),
            Dense(1)  # 1 número complementario
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def train(self):
        # Obtener datos preparados
        (X_regular, y_regular), (X_complementary, y_complementary) = self.prepare_data()
        
        # Configurar early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Entrenar modelo para números regulares
        self.regular_model = self.create_regular_model(X_regular.shape[1])
        self.regular_model.fit(
            X_regular, y_regular,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Entrenar modelo para el complementario
        self.complementary_model = self.create_complementary_model(X_complementary.shape[1])
        self.complementary_model.fit(
            X_complementary, y_complementary,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )

    def predict(self):
        if self.regular_model is None or self.complementary_model is None:
            return [1, 2, 3, 4, 5, 1]
        
        # Predecir números regulares
        last_regular = self.data.iloc[-1, :5].values.reshape(1, -1)
        last_regular_scaled = self.scaler_regular.transform(last_regular)
        regular_predictions = self.regular_model.predict(last_regular_scaled, verbose=0)
        regular_numbers = self.scaler_regular.inverse_transform(regular_predictions)[0]
        
        # Redondear y ajustar números regulares
        regular_numbers = np.round(regular_numbers).astype(int)
        regular_numbers = np.clip(regular_numbers, 1, 43)
        regular_numbers = list(set(regular_numbers))
        
        # Asegurar que tenemos 5 números únicos
        while len(regular_numbers) < 5:
            new_num = np.random.randint(1, 44)
            if new_num not in regular_numbers:
                regular_numbers.append(new_num)
        regular_numbers = sorted(regular_numbers[:5])
        
        # Predecir complementario
        last_complementary = self.data.iloc[-1, 5:].values.reshape(1, -1)
        last_complementary_scaled = self.scaler_complementary.transform(last_complementary)
        complementary_prediction = self.complementary_model.predict(last_complementary_scaled, verbose=0)
        complementary = self.scaler_complementary.inverse_transform(complementary_prediction)[0][0]
        complementary = int(np.clip(round(complementary), 1, 16))
        
        # Asegurar que el complementario no está en los números regulares
        while complementary in regular_numbers:
            complementary = (complementary % 16) + 1
        
        return regular_numbers + [complementary] 