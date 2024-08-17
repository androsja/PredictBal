import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import os
import matplotlib.pyplot as plt
import glob

# Configurar los argumentos de línea de comandos
parser = argparse.ArgumentParser(description="Entrenar un modelo LSTM usando datos de un archivo específico.")
parser.add_argument('data_file', type=str, help="Nombre del archivo de datos (e.g., 'rev_six.txt')")
args = parser.parse_args()

# Leer el nombre del archivo desde los argumentos de línea de comandos
data_file = os.path.join('historic_data', args.data_file)

# Leer los datos del fichero proporcionado
data = pd.read_csv(data_file, sep=" - ", header=None, engine='python')
data = data.values

# Almacenar una copia de los datos originales
data_original = data.copy()

# Escalar los datos entre 0 y 1 para la única columna
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Crear los conjuntos de entrenamiento
X = []
y = []

time_steps = 15 
for i in range(time_steps, len(data_scaled)):
    X.append(data_scaled[i-time_steps:i])
    y.append(data_scaled[i])

X, y = np.array(X), np.array(y)

# Crear un directorio para guardar los modelos
checkpoint_dir = 'models_ai/checkpoints_six'
os.makedirs(checkpoint_dir, exist_ok=True)

# Buscar el último checkpoint
checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, 'model_epoch_*.keras')))
latest_checkpoint = None

# Intentar cargar el último checkpoint válido
for checkpoint in reversed(checkpoint_files):
    try:
        print(f"Intentando cargar el modelo desde: {checkpoint}")
        model = load_model(checkpoint)
        latest_checkpoint = checkpoint
        initial_epoch = int(checkpoint.split('_')[-1].split('.')[0])
        break
    except OSError as e:
        print(f"Error al cargar {checkpoint}: {e}")
        print("Checkpoint corrupto. Intentando con un checkpoint anterior...")

if latest_checkpoint is None:
    print("No se pudo cargar ningún checkpoint válido. Iniciando entrenamiento desde el principio.")
    # Definir el modelo LSTM
    model = Sequential()

    # Añadir capas LSTM con Dropout
    model.add(LSTM(units=512, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.3))
    model.add(LSTM(units=512))
    model.add(Dropout(0.3))

    # Capa de salida con 1 neurona para predecir el siguiente valor de la columna
    model.add(Dense(units=1))

    # Compilar el modelo
    model.compile(optimizer='adam', loss='mean_squared_error')

    initial_epoch = 0

# Definir el callback para guardar el modelo en cada epoch
checkpoint_callback = ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, 'model_epoch_{epoch:02d}.keras'),
    save_weights_only=False,
    save_best_only=False,
    save_freq='epoch'
)

# Entrenar el modelo desde el último epoch válido
model.fit(X, y, epochs=2000, batch_size=64, callbacks=[checkpoint_callback], initial_epoch=initial_epoch)

# Borrar checkpoints antiguos, manteniendo solo los más recientes
num_checkpoints_to_keep = 5
checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, 'model_epoch_*.keras')))
if len(checkpoint_files) > num_checkpoints_to_keep:
    for file_to_remove in checkpoint_files[:-num_checkpoints_to_keep]:
        os.remove(file_to_remove)
        print(f"Checkpoint eliminado: {file_to_remove}")

# Predecir los valores
y_pred_scaled = model.predict(X)
y_pred = scaler.inverse_transform(y_pred_scaled)
y_real = scaler.inverse_transform(y)

# Predicción del último conjunto de X que no tiene Y
last_X = data_scaled[-time_steps:]  # Toma los últimos 'time_steps' elementos
last_X = np.expand_dims(last_X, axis=0)  # Ajustar la dimensión para la predicción
predicted_last_y_scaled = model.predict(last_X)
predicted_last_y = scaler.inverse_transform(predicted_last_y_scaled)

# Guardar los datos en un archivo con el formato solicitado
output_file = os.path.join('models_ai', 'rev_six_predict_detailed.txt')

with open(output_file, 'w') as f:
    for i in range(len(X)):
        f.write(f"X[{i}]:\n")
        seq_str = "\n".join(map(str, data_original[i:i+time_steps].flatten()))  # Imprimir los valores originales de X
        f.write(f"{seq_str}\n")
        
        real_str = str(y_real[i][0])
        pred_str = str(np.round(y_pred[i][0]).astype(int))
        f.write(f"Y: {real_str}\n")
        f.write(f"Y predicho: {pred_str}\n")
        f.write("\n")
    
    # Agregar el último conjunto de X y su Y predicho
    f.write(f"Último X sin Y:\n")
    last_X_original = data_original[-time_steps:]  # Obtener los valores originales del último conjunto de X
    last_X_str = "\n".join(map(str, last_X_original.flatten()))
    f.write(f"{last_X_str}\n")
    
    last_y_pred_str = str(np.round(predicted_last_y[0][0]).astype(int))
    f.write(f"Y predicho: {last_y_pred_str}\n")

print(f"Archivo guardado en: {output_file}")

# Ploteo de resultados
plt.figure(figsize=(14, 7))

# Ploteo del valor real contra el valor predicho
plt.plot(y_real, color='blue', label='Valor Real')
plt.plot(y_pred, color='red', label='Valor Predicho')

plt.title('Comparación entre el valor real y el valor predicho')
plt.xlabel('Tiempo')
plt.ylabel('Valor')
plt.legend()
plt.show()
