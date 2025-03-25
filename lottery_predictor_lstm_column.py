import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from sklearn.preprocessing import MinMaxScaler

class LotteryPredictorLSTMColumn:
    def __init__(self, data):
        self.data = data
        self.sequence_length = 10
        # Separar escaladores para números regulares y complementario
        self.regular_scalers = [MinMaxScaler(feature_range=(0, 1)) for _ in range(5)]
        self.complementary_scaler = MinMaxScaler(feature_range=(0, 1))
        self.regular_models = [None] * 5  # Modelos para números regulares
        self.complementary_model = None  # Modelo para complementario
        self.processed_data = None
        
    def prepare_data(self):
        # Convertir datos a matriz numérica
        numbers = []
        for row in self.data.values:
            row_str = str(row)
            row_str = row_str.replace('[', '').replace(']', '').replace('Name:', '').replace('dtype:', '').replace('object', '').strip()
            nums = [int(n.strip()) for n in row_str.split() if n.strip()]
            if nums:
                numbers.append(nums)
        
        self.processed_data = np.array(numbers)
        
        # Preparar datos por columna
        X_regular = []
        y_regular = []
        
        # Preparar datos para números regulares (5 columnas)
        for col in range(5):
            X_col, y_col = [], []
            column_data = self.processed_data[:, col]
            
            for i in range(len(column_data) - self.sequence_length):
                sequence = column_data[i:i + self.sequence_length]
                target = column_data[i + self.sequence_length]
                
                X_col.append(sequence)
                y_col.append(target)
            
            X_col = np.array(X_col)
            y_col = np.array(y_col)
            
            # Reshape y escalar datos
            X_col = X_col.reshape(-1, self.sequence_length, 1)
            y_col = y_col.reshape(-1, 1)
            
            # Escalar datos usando el scaler específico de la columna
            X_col_scaled = self.regular_scalers[col].fit_transform(X_col.reshape(-1, 1)).reshape(-1, self.sequence_length, 1)
            y_col_scaled = self.regular_scalers[col].transform(y_col)
            
            X_regular.append(X_col_scaled)
            y_regular.append(y_col_scaled)
        
        # Preparar datos para el complementario
        X_comp, y_comp = [], []
        complementary_data = self.processed_data[:, 5]
        
        for i in range(len(complementary_data) - self.sequence_length):
            sequence = complementary_data[i:i + self.sequence_length]
            target = complementary_data[i + self.sequence_length]
            
            X_comp.append(sequence)
            y_comp.append(target)
        
        X_comp = np.array(X_comp)
        y_comp = np.array(y_comp)
        
        # Reshape y escalar datos del complementario
        X_comp = X_comp.reshape(-1, self.sequence_length, 1)
        y_comp = y_comp.reshape(-1, 1)
        
        X_comp_scaled = self.complementary_scaler.fit_transform(X_comp.reshape(-1, 1)).reshape(-1, self.sequence_length, 1)
        y_comp_scaled = self.complementary_scaler.transform(y_comp)
        
        return X_regular, y_regular, X_comp_scaled, y_comp_scaled

    def create_regular_model(self):
        model = Sequential([
            LSTM(64, input_shape=(self.sequence_length, 1), 
                 return_sequences=True,
                 kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            BatchNormalization(),
            Dropout(0.3),
            LSTM(32, return_sequences=False),
            BatchNormalization(),
            Dropout(0.2),
            Dense(16, activation='relu'),
            BatchNormalization(),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model

    def create_complementary_model(self):
        model = Sequential([
            LSTM(64, input_shape=(self.sequence_length, 1), 
                 return_sequences=True,
                 kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            BatchNormalization(),
            Dropout(0.3),
            LSTM(32, return_sequences=False),
            BatchNormalization(),
            Dropout(0.2),
            Dense(16, activation='relu'),
            BatchNormalization(),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model

    def train(self):
        # Preparar datos
        X_regular, y_regular, X_comp, y_comp = self.prepare_data()
        
        # Early stopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Entrenar modelos para números regulares
        for col in range(5):
            print(f"Entrenando modelo para número regular {col + 1}")
            
            # Crear y entrenar modelo para la columna regular
            self.regular_models[col] = self.create_regular_model()
            
            # Dividir datos
            split = int(len(X_regular[col]) * 0.8)
            X_train = X_regular[col][:split]
            X_val = X_regular[col][split:]
            y_train = y_regular[col][:split]
            y_val = y_regular[col][split:]
            
            # Entrenar modelo
            self.regular_models[col].fit(
                X_train, y_train,
                epochs=100,
                batch_size=32,
                validation_data=(X_val, y_val),
                callbacks=[early_stopping],
                verbose=0
            )
        
        # Entrenar modelo para el complementario
        print("Entrenando modelo para el complementario")
        self.complementary_model = self.create_complementary_model()
        
        # Dividir datos del complementario
        split_comp = int(len(X_comp) * 0.8)
        X_train_comp = X_comp[:split_comp]
        X_val_comp = X_comp[split_comp:]
        y_train_comp = y_comp[:split_comp]
        y_val_comp = y_comp[split_comp:]
        
        # Entrenar modelo del complementario
        self.complementary_model.fit(
            X_train_comp, y_train_comp,
            epochs=100,
            batch_size=32,
            validation_data=(X_val_comp, y_val_comp),
            callbacks=[early_stopping],
            verbose=0
        )

    def predict(self):
        if any(model is None for model in self.regular_models) or self.complementary_model is None:
            return [1, 2, 3, 4, 5, 1]
        
        regular_predictions = []
        
        # Predecir números regulares
        for col in range(5):
            # Obtener la última secuencia para la columna
            last_sequence = self.processed_data[-self.sequence_length:, col]
            
            # Preparar datos para predicción
            X_pred = last_sequence.reshape(1, self.sequence_length, 1)
            X_pred_scaled = self.regular_scalers[col].transform(X_pred.reshape(-1, 1)).reshape(1, self.sequence_length, 1)
            
            # Realizar predicción
            pred_scaled = self.regular_models[col].predict(X_pred_scaled, verbose=0)
            pred = self.regular_scalers[col].inverse_transform(pred_scaled)
            
            # Redondear y ajustar al rango
            num = max(1, min(43, round(float(pred[0][0]))))
            regular_predictions.append(num)
        
        # Asegurar números regulares únicos
        regular_predictions = list(set(regular_predictions))
        while len(regular_predictions) < 5:
            new_num = np.random.randint(1, 44)
            if new_num not in regular_predictions:
                regular_predictions.append(new_num)
        regular_predictions = sorted(regular_predictions[:5])
        
        # Predecir complementario
        last_sequence_comp = self.processed_data[-self.sequence_length:, 5]
        X_pred_comp = last_sequence_comp.reshape(1, self.sequence_length, 1)
        X_pred_comp_scaled = self.complementary_scaler.transform(X_pred_comp.reshape(-1, 1)).reshape(1, self.sequence_length, 1)
        
        # Realizar predicción del complementario
        pred_comp_scaled = self.complementary_model.predict(X_pred_comp_scaled, verbose=0)
        pred_comp = self.complementary_scaler.inverse_transform(pred_comp_scaled)
        complementary = max(1, min(16, round(float(pred_comp[0][0]))))
        
        # Asegurar que el complementario no está en los números regulares
        while complementary in regular_predictions:
            complementary = (complementary % 16) + 1
        
        return regular_predictions + [complementary] 