import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Leer el archivo de datos
data = pd.read_csv('historic_data/bal_results_i.txt', sep='-', header=None)

# Renombrar las columnas para facilidad de uso
data.columns = ['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6']

# Eliminar espacios en blanco
data = data.applymap(lambda x: int(str(x).strip()))

# Preparar los datos para la red neuronal
X = data[['Num1', 'Num2', 'Num3', 'Num4', 'Num5']].values
y = data['Num6'].values

# One-hot encoding de las características
onehot_encoder = OneHotEncoder(categories=[range(1, 44)]*5, sparse_output=False)
X_encoded = onehot_encoder.fit_transform(X)

# One-hot encoding del objetivo
y_encoded = OneHotEncoder(categories=[range(1, 17)], sparse_output=False)
y_encoded = y_encoded.fit_transform(y.reshape(-1, 1))

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

# Definir el modelo para predecir el sexto número
model = Sequential()
model.add(Dense(128, input_dim=X_encoded.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(y_encoded.shape[1], activation='softmax'))

# Compilar el modelo
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, epochs=50, batch_size=10, validation_split=0.2)

# Hacer predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Convertir las predicciones de one-hot encoding a etiquetas
y_pred_labels = np.argmax(y_pred, axis=1) + 1
y_test_labels = np.argmax(y_test, axis=1) + 1

# Mostrar algunas predicciones
print("Predicciones:", y_pred_labels[:10])
print("Valores reales:", y_test_labels[:10])

# Función para predecir el próximo conjunto de números
def predict_next_numbers(model, onehot_encoder):
    # Generar números únicos aleatorios para las primeras cinco columnas
    first_five_numbers = np.random.choice(range(1, 44), 5, replace=False)
    first_five_numbers_sorted = np.sort(first_five_numbers).reshape(1, -1)
    
    # Preprocesar los números de entrada
    first_five_encoded = onehot_encoder.transform(first_five_numbers_sorted)
    
    # Hacer la predicción para el sexto número
    y_pred = model.predict(first_five_encoded)
    y_pred_label = np.argmax(y_pred, axis=1) + 1
    
    return list(first_five_numbers_sorted[0]) + [y_pred_label[0]]

# Ejemplo de predicción
next_numbers = predict_next_numbers(model, onehot_encoder)
print("Próximos números predichos:", next_numbers)
