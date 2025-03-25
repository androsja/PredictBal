import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from sklearn.preprocessing import MinMaxScaler

class LotteryPredictorLSTM:
    def __init__(self, data):
        self.data = data
        self.sequence_length = 10
        self.scaler_regular = MinMaxScaler(feature_range=(0, 1))
        self.scaler_complementary = MinMaxScaler(feature_range=(0, 1))
        self.regular_model = None
        self.complementary_model = None
        self.processed_regular = None
        self.processed_complementary = None

    def prepare_data(self):
        # Separar números regulares y complementario
        regular_numbers = []
        complementary_numbers = []
        
        for row in self.data.values:
            row_str = str(row)
            row_str = row_str.replace('[', '').replace(']', '').replace('Name:', '').replace('dtype:', '').replace('object', '').strip()
            nums = [int(n.strip()) for n in row_str.split() if n.strip()]
            if nums:
                regular_numbers.append(nums[:5])
                complementary_numbers.append([nums[5]])
        
        self.processed_regular = np.array(regular_numbers)
        self.processed_complementary = np.array(complementary_numbers)
        
        # Preparar secuencias para números regulares
        X_reg, y_reg = [], []
        for i in range(len(self.processed_regular) - self.sequence_length):
            sequence = self.processed_regular[i:i + self.sequence_length]
            target = self.processed_regular[i + self.sequence_length]
            X_reg.append(sequence)
            y_reg.append(target)
        
        X_reg = np.array(X_reg)
        y_reg = np.array(y_reg)
        
        # Preparar secuencias para complementario
        X_comp, y_comp = [], []
        for i in range(len(self.processed_complementary) - self.sequence_length):
            sequence = self.processed_complementary[i:i + self.sequence_length]
            target = self.processed_complementary[i + self.sequence_length]
            X_comp.append(sequence)
            y_comp.append(target)
        
        X_comp = np.array(X_comp)
        y_comp = np.array(y_comp)
        
        # Normalizar datos regulares
        X_reg_reshaped = X_reg.reshape(-1, X_reg.shape[-1])
        y_reg_reshaped = y_reg.reshape(-1, y_reg.shape[-1])
        X_reg_scaled = self.scaler_regular.fit_transform(X_reg_reshaped)
        y_reg_scaled = self.scaler_regular.transform(y_reg_reshaped)
        X_reg_final = X_reg_scaled.reshape(X_reg.shape)
        y_reg_final = y_reg_scaled.reshape(y_reg.shape)
        
        # Normalizar datos complementario
        X_comp_reshaped = X_comp.reshape(-1, X_comp.shape[-1])
        y_comp_reshaped = y_comp.reshape(-1, y_comp.shape[-1])
        X_comp_scaled = self.scaler_complementary.fit_transform(X_comp_reshaped)
        y_comp_scaled = self.scaler_complementary.transform(y_comp_reshaped)
        X_comp_final = X_comp_scaled.reshape(X_comp.shape)
        y_comp_final = y_comp_scaled.reshape(y_comp.shape)
        
        return (X_reg_final, y_reg_final), (X_comp_final, y_comp_final)

    def create_regular_model(self):
        model = Sequential([
            LSTM(128, input_shape=(self.sequence_length, 5), 
                 return_sequences=True,
                 kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            BatchNormalization(),
            Dropout(0.3),
            LSTM(64, return_sequences=False),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(16, activation='relu'),
            BatchNormalization(),
            Dense(5, activation='sigmoid')
        ])
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
            Dropout(0.3),
            Dense(16, activation='relu'),
            BatchNormalization(),
            Dense(1, activation='sigmoid')
        ])
        return model

    def train(self):
        # Preparar datos
        (X_reg, y_reg), (X_comp, y_comp) = self.prepare_data()
        
        # Crear y entrenar modelo regular
        self.regular_model = self.create_regular_model()
        self.regular_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Crear y entrenar modelo complementario
        self.complementary_model = self.create_complementary_model()
        self.complementary_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Dividir datos y entrenar modelos
        split_reg = int(len(X_reg) * 0.8)
        split_comp = int(len(X_comp) * 0.8)
        
        # Entrenar modelo regular
        self.regular_model.fit(
            X_reg[:split_reg], y_reg[:split_reg],
            epochs=100,
            batch_size=32,
            validation_data=(X_reg[split_reg:], y_reg[split_reg:]),
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Entrenar modelo complementario
        self.complementary_model.fit(
            X_comp[:split_comp], y_comp[:split_comp],
            epochs=100,
            batch_size=32,
            validation_data=(X_comp[split_comp:], y_comp[split_comp:]),
            callbacks=[early_stopping],
            verbose=0
        )

    def predict(self):
        if self.regular_model is None or self.complementary_model is None:
            return [1, 2, 3, 4, 5, 1]
        
        # Predecir números regulares
        last_regular_sequence = self.processed_regular[-self.sequence_length:]
        X_reg_pred = last_regular_sequence.reshape(1, self.sequence_length, 5)
        X_reg_pred_reshaped = X_reg_pred.reshape(-1, X_reg_pred.shape[-1])
        X_reg_pred_scaled = self.scaler_regular.transform(X_reg_pred_reshaped)
        X_reg_pred_final = X_reg_pred_scaled.reshape(X_reg_pred.shape)
        
        reg_prediction_scaled = self.regular_model.predict(X_reg_pred_final, verbose=0)
        reg_prediction_reshaped = reg_prediction_scaled.reshape(-1, 5)
        reg_prediction = self.scaler_regular.inverse_transform(reg_prediction_reshaped)
        
        # Procesar números regulares
        regular_numbers = [max(1, min(43, round(x))) for x in reg_prediction[0]]
        regular_numbers = list(set(regular_numbers))
        while len(regular_numbers) < 5:
            new_num = np.random.randint(1, 44)
            if new_num not in regular_numbers:
                regular_numbers.append(new_num)
        regular_numbers = sorted(regular_numbers[:5])
        
        # Predecir complementario
        last_comp_sequence = self.processed_complementary[-self.sequence_length:]
        X_comp_pred = last_comp_sequence.reshape(1, self.sequence_length, 1)
        X_comp_pred_reshaped = X_comp_pred.reshape(-1, X_comp_pred.shape[-1])
        X_comp_pred_scaled = self.scaler_complementary.transform(X_comp_pred_reshaped)
        X_comp_pred_final = X_comp_pred_scaled.reshape(X_comp_pred.shape)
        
        comp_prediction_scaled = self.complementary_model.predict(X_comp_pred_final, verbose=0)
        comp_prediction_reshaped = comp_prediction_scaled.reshape(-1, 1)
        comp_prediction = self.scaler_complementary.inverse_transform(comp_prediction_reshaped)
        
        # Procesar complementario
        complementary = int(max(1, min(16, round(comp_prediction[0][0]))))
        while complementary in regular_numbers:
            complementary = (complementary % 16) + 1
        
        return regular_numbers + [complementary] 