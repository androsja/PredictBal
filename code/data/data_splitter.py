# File: code/data/data_splitter.py
import numpy as np
from data.dataset_creator_factory import DatasetCreatorFactory

class DataSplitter:
    def __init__(self, time_step):
        self.time_step = time_step

    def split_data(self, train_ratio=0.8, val_ratio=0.1):
        print("Get data for dataset Creator")
        dataset_creator = DatasetCreatorFactory().create_dataset_creator(self.time_step)

        print("Init for split_data")
        X_first_five, y_first_five, X_sixth, y_sixth, scalers = dataset_creator.create_dataset()

        train_size = int(train_ratio * len(X_first_five))
        val_size = int(val_ratio * len(X_first_five))
        test_size = len(X_first_five) - train_size - val_size

        X_train_five = X_first_five[:train_size]
        y_train_five = y_first_five[:train_size]
        X_val_five = X_first_five[train_size:train_size + val_size]
        y_val_five = y_first_five[train_size:train_size + val_size]
        X_test_five = X_first_five[train_size + val_size:]
        y_test_five = y_first_five[train_size + val_size:]

        X_train_sixth = X_sixth[:train_size]
        y_train_sixth = y_sixth[:train_size]
        X_val_sixth = X_sixth[train_size:train_size + val_size]
        y_val_sixth = y_sixth[train_size:train_size + val_size]
        X_test_sixth = X_sixth[train_size + val_size:]
        y_test_sixth = y_sixth[train_size + val_size:]

        return {
            'all_data': (X_first_five, y_first_five, X_sixth, y_sixth),
            'train_five': (X_train_five, y_train_five),
            'val_five': (X_val_five, y_val_five),
            'test_five': (X_test_five, y_test_five),
            'train_sixth': (X_train_sixth, y_train_sixth),
            'val_sixth': (X_val_sixth, y_val_sixth),
            'test_sixth': (X_test_sixth, y_test_sixth),
            'scalers': scalers
        }
