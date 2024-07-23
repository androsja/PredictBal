import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Leer los datos del archivo
data = []
with open('historic_data/bal_results_i.txt', 'r') as file:
    for line in file:
        numbers = [int(num.strip()) for num in line.split('-')]
        data.append(numbers)

# Convertir a DataFrame
df = pd.DataFrame(data, columns=[f'Num{i+1}' for i in range(6)])

# Crear características adicionales
df['Sum'] = df.sum(axis=1)
df['Mean'] = df.mean(axis=1)
df['Std'] = df.std(axis=1)
df['Min'] = df.min(axis=1)
df['Max'] = df.max(axis=1)

# Normalizar los datos
scaler_1_43 = MinMaxScaler(feature_range=(0, 1))
scaled_data_1_43 = scaler_1_43.fit_transform(df.iloc[:, :5])

scaler_1_16 = MinMaxScaler(feature_range=(0, 1))
scaled_data_1_16 = scaler_1_16.fit_transform(df.iloc[:, 5:6])

scaled_data = np.hstack([scaled_data_1_43, scaled_data_1_16])

# Crear secuencias para el entrenamiento
def create_sequences(data, seq_length, column_index):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, column_index])  # Predecir solo la columna especificada
    return np.array(X), np.array(y)

seq_length = 3  # Longitud de secuencia

models = []
for i in range(6):
    # Crear secuencias para cada columna
    X, y = create_sequences(scaled_data, seq_length, i)
    
    # Definir y entrenar el modelo para cada columna
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, X.shape[2])))
    model.add(LSTM(100))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    
    # Entrenar el modelo usando todos los datos
    model.fit(X, y, epochs=200, batch_size=32, validation_split=0.2)
    
    models.append(model)

# Evaluar los modelos en los mismos datos de entrenamiento
for i, model in enumerate(models):
    X, y = create_sequences(scaled_data, seq_length, i)
    loss = model.evaluate(X, y)
    print(f'Loss para la columna {i+1}: {loss}')

# Predecir el siguiente conjunto de números
next_prediction = []
for i, model in enumerate(models):
    last_sequence = scaled_data[-seq_length:].reshape(1, seq_length, scaled_data.shape[1])
    next_prediction_scaled = model.predict(last_sequence)
    if i < 5:
        # Crear una matriz de ceros y reemplazar la columna correspondiente
        inverse_scaled = np.zeros((1, scaled_data_1_43.shape[1]))
        inverse_scaled[:, i] = next_prediction_scaled
        next_prediction.append(scaler_1_43.inverse_transform(inverse_scaled)[0][i])
    else:
        # Para la sexta columna
        next_prediction.append(scaler_1_16.inverse_transform(next_prediction_scaled)[0][0])

next_prediction = np.round(next_prediction).astype(int)

print(f'Predicción del siguiente conjunto de números: {next_prediction}')
