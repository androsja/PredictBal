# File: code/data/data_splitter.py
import numpy as np
from data.dataset_creator_factory import DatasetCreatorFactory

class DataSplitter:
    def __init__(self, time_step):
        self.time_step = time_step

    def split_data(self, train_ratio=1.0, val_ratio=0.0):
        dataset_creator = DatasetCreatorFactory().create_dataset_creator(self.time_step)
        print("DatasetCreator created.")

        X_first_five, y_first_five, scalers = dataset_creator.create_dataset()
        print("Dataset created.")
        print(f"X_first_five shape: {X_first_five.shape}")
        print(f"y_first_five shape: {y_first_five.shape}")

        total_size = len(X_first_five)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size

        X_train_five = X_first_five[:train_size]
        y_train_five = y_first_five[:train_size]

        if val_ratio > 0.0:
            X_val_five = X_first_five[train_size:train_size + val_size]
            y_val_five = y_first_five[train_size:train_size + val_size]
        else:
            X_val_five = np.array([]).reshape(0, self.time_step, X_first_five.shape[2])
            y_val_five = np.array([]).reshape(0, y_first_five.shape[1])

        if test_size > 0:
            X_test_five = X_first_five[train_size + val_size:]
            y_test_five = y_first_five[train_size + val_size:]
        else:
            X_test_five = np.array([]).reshape(0, self.time_step, X_first_five.shape[2])
            y_test_five = np.array([]).reshape(0, y_first_five.shape[1])

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