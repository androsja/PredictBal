import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from sklearn.preprocessing import MinMaxScaler
import os
import matplotlib.pyplot as plt
import glob
import sys

# Verificar si se pasó el nombre del archivo como argumento
if len(sys.argv) < 2:
    print("Por favor, proporciona el nombre del archivo como argumento.")
    sys.exit(1)

# Obtener el nombre del archivo desde los argumentos de la línea de comandos
file_name = sys.argv[1]

# Asegurarse de que el archivo esté en la carpeta 'historic_data'
file_path = os.path.join('historic_data', file_name)

# Verificar si el archivo existe en la carpeta 'historic_data'
if not os.path.exists(file_path):
    print(f"El archivo {file_path} no existe. Verifica el nombre y vuelve a intentarlo.")
    sys.exit(1)

# Leer los datos del fichero con un separador personalizado
data = pd.read_csv(file_path, sep=" - ", header=None, engine='python')
data = data.values

# Escalar los datos entre 0 y 1
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
checkpoint_dir = 'models_ai/checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

# Buscar el último checkpoint
checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, 'model_epoch_*.keras')))
latest_checkpoint = checkpoint_files[-1] if checkpoint_files else None

# Si existe un checkpoint previo, cargar el modelo desde ahí y obtener el número de epoch
if latest_checkpoint:
    print(f"Cargando el modelo desde: {latest_checkpoint}")
    model = load_model(latest_checkpoint)
    initial_epoch = int(latest_checkpoint.split('_')[-1].split('.')[0])  # Obtener el número de epoch del nombre del archivo
else:
    # Definir el modelo LSTM
    model = Sequential()

    # Añadir más capas LSTM con más neuronas y Dropout
    model.add(LSTM(units=2048, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.3))
    model.add(LSTM(units=1024, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(units=512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(units=512))
    model.add(Dropout(0.3))

    # Capa de salida con 5 neuronas para predecir el siguiente conjunto de 5 números
    model.add(Dense(units=5))

    # Compilar el modelo
    model.compile(optimizer='adam', loss='mean_squared_error')

    initial_epoch = 0  # Iniciar desde el primer epoch si no hay checkpoints previos

# Definir el callback para guardar el modelo en cada epoch
checkpoint_callback = ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, 'model_epoch_{epoch:02d}.keras'),
    save_weights_only=False,
    save_best_only=False,
    save_freq='epoch'
)

# Callback personalizado para eliminar checkpoints antiguos después de cada epoch
class CleanupCheckpointCallback(Callback):
    def __init__(self, checkpoint_dir, num_checkpoints_to_keep=5):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.num_checkpoints_to_keep = num_checkpoints_to_keep

    def on_epoch_end(self, epoch, logs=None):
        # Buscar los archivos de checkpoint en el directorio
        checkpoint_files = sorted(glob.glob(os.path.join(self.checkpoint_dir, 'model_epoch_*.keras')))
        # Verificar si hay más checkpoints de los permitidos
        if len(checkpoint_files) > self.num_checkpoints_to_keep:
            for file_to_remove in checkpoint_files[:-self.num_checkpoints_to_keep]:
                os.remove(file_to_remove)
                print(f"Checkpoint eliminado: {file_to_remove}")

# Crear la instancia del callback
cleanup_callback = CleanupCheckpointCallback(checkpoint_dir)

# Entrenar el modelo desde el último epoch
model.fit(X, y, epochs=2000, batch_size=64, callbacks=[checkpoint_callback, cleanup_callback], initial_epoch=initial_epoch)

# Predecir los valores
y_pred_scaled = model.predict(X)
y_pred = scaler.inverse_transform(y_pred_scaled)
y_real = scaler.inverse_transform(y)
X_real = [scaler.inverse_transform(x) for x in X]

# Predicción del último conjunto de X que no tiene Y
last_X = data_scaled[-time_steps:]  # Toma los últimos 'time_steps' elementos
last_X = np.expand_dims(last_X, axis=0)  # Ajustar la dimensión para la predicción
predicted_last_y_scaled = model.predict(last_X)
predicted_last_y = scaler.inverse_transform(predicted_last_y_scaled)

# Guardar los datos en un archivo con el formato solicitado
output_file = os.path.join('models_ai', f'{os.path.splitext(os.path.basename(file_name))[0]}_predict_detailed.txt')

with open(output_file, 'w') as f:
    for i in range(len(X_real)):
        f.write(f"X[{i}]:\n")
        for seq in X_real[i]:
            seq_str = " - ".join(map(str, seq.astype(int)))
            f.write(f"{seq_str}\n")
        
        real_str = " - ".join(map(str, y_real[i].astype(int)))
        pred_str = " - ".join(map(str, np.round(y_pred[i]).astype(int)))
        f.write(f"Y: {real_str}\n")
        f.write(f"Y predicho: {pred_str}\n")
        f.write("\n")
    
    # Agregar el último conjunto de X y su Y predicho
    f.write(f"Último X sin Y:\n")
    for seq in scaler.inverse_transform(last_X[0]):
        seq_str = " - ".join(map(str, seq.astype(int)))
        f.write(f"{seq_str}\n")
    
    last_y_pred_str = " - ".join(map(str, np.round(predicted_last_y[0]).astype(int)))
    f.write(f"Y predicho: {last_y_pred_str}\n")

print(f"Archivo guardado en: {output_file}")

# Ploteo de resultados
plt.figure(figsize=(14, 7))

# Se toma solo la primera serie de los datos predichos para hacer la comparación
plt.plot(y_real[:, 0], color='blue', label='Valor Real')
plt.plot(y_pred[:, 0], color='red', label='Valor Predicho')

plt.title('Comparación entre el valor real y el valor predicho')
plt.xlabel('Tiempo')
plt.ylabel('Valor')
plt.legend()
plt.show()
