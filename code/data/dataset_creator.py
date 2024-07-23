# File: code/data/dataset_creator.py
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from data.data_repository import DataRepository

class DatasetCreator:
    def __init__(self, data, time_step, use_sixth_column=False):
        self.time_step = time_step
        self.data = data
        self.use_sixth_column = use_sixth_column
        self.data_repository = DataRepository()
        self.scalers = {}

        print(f"Inicializando DatasetCreator con data: {self.data.head()}")

    def normalize_data(self, data):
        df_scaled = pd.DataFrame(index=data.index)
        for column in data.columns:
            scaler = MinMaxScaler()
            df_scaled[column] = scaler.fit_transform(data[column].values.reshape(-1, 1)).flatten()
            self.scalers[column] = scaler
            print(f"Scaler fitted for {column}: min={scaler.data_min_}, max={scaler.data_max_}")
        return df_scaled

    def create_dataset(self):
        print("Iniciando la creación del dataset...")

        if self.use_sixth_column:
            data_subset = self.data.iloc[:, [5]]  # Usar solo la sexta columna
        else:
            data_subset = self.data.iloc[:, :5]  # Usar solo las primeras 5 columnas
        print(data_subset.head())

        self.data_normalized = self.normalize_data(data_subset)

        X, Y = [], []
        for i in range(len(self.data_normalized) - self.time_step):
            a = self.data_normalized.iloc[i:(i + self.time_step), :].values
            X.append(a)
            Y.append(self.data_normalized.iloc[i + self.time_step, :].values)

        self.X = np.array(X)
        self.Y = np.array(Y)

        print("Forma final de X:", self.X.shape)
        print("Forma final de Y:", self.Y.shape)
        print("Columnas en los datos normalizados:", self.data_normalized.columns)

        self.data_repository.save_data(self.X, self.Y, self.data_normalized, self.time_step)

        return self.X, self.Y, self.scalers
