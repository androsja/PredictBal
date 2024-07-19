# File: code/data/dataset_creator.py
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler  # Cambiado a MinMaxScaler
from data.data_repository import DataRepository
class DatasetCreator:
    def __init__(self, data, time_step):
        self.time_step = time_step
        self.data = data
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

        data_subset = self.data.iloc[:, :5]  # Usar solo las primeras 5 columnas
        print(data_subset.head())

        self.data_normalized = self.normalize_data(data_subset)

        X_first_five, Y_first_five = [], []
        for i in range(len(self.data_normalized) - self.time_step):
            a_first_five = self.data_normalized.iloc[i:(i + self.time_step), :5].values
            X_first_five.append(a_first_five)
            Y_first_five.append(self.data_normalized.iloc[i + self.time_step, :5].values)

        self.X_first_five = np.array(X_first_five)
        self.Y_first_five = np.array(Y_first_five)

        print("Forma final de X_first_five:", self.X_first_five.shape)
        print("Forma final de Y_first_five:", self.Y_first_five.shape)
        print("Columnas en los datos normalizados:", self.data_normalized.columns)

        self.data_repository.save_data(self.X_first_five, self.Y_first_five, self.data_normalized, self.time_step)

        return self.X_first_five, self.Y_first_five, self.scalers
