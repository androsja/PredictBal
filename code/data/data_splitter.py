# File: code/data/data_splitter.py
import numpy as np
from data.dataset_creator_factory import DatasetCreatorFactory
class DataSplitter:
    def __init__(self, time_step):
        self.time_step = time_step

    def split_data(self, train_ratio=0.97, val_ratio=0.03):
        dataset_creator = DatasetCreatorFactory().create_dataset_creator(self.time_step)
        print("DatasetCreator created.")

        X_first_five, y_first_five, scalers = dataset_creator.create_dataset()
        print("Dataset created.")
        print(f"X_first_five shape: {X_first_five.shape}")
        print(f"y_first_five shape: {y_first_five.shape}")

        train_size = int(train_ratio * len(X_first_five))
        val_size = int(val_ratio * len(X_first_five))
        test_size = len(X_first_five) - train_size - val_size

        X_train_five = X_first_five[:train_size]
        y_train_five = y_first_five[:train_size]
        X_val_five = X_first_five[train_size:train_size + val_size]
        y_val_five = y_first_five[train_size:train_size + val_size]
        X_test_five = X_first_five[train_size + val_size:]
        y_test_five = y_first_five[train_size + val_size:]

        print(f"Train shapes: X={X_train_five.shape}, y={y_train_five.shape}")
        print(f"Validation shapes: X={X_val_five.shape}, y={y_val_five.shape}")
        print(f"Test shapes: X={X_test_five.shape}, y={y_test_five.shape}")

        return {
            'all_data': (X_first_five, y_first_five),
            'train_five': (X_train_five, y_train_five),
            'val_five': (X_val_five, y_val_five),
            'test_five': (X_test_five, y_test_five),
            'scalers': scalers
        }
