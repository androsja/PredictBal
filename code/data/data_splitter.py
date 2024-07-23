# File: code/data/data_splitter.py
import numpy as np
from data.dataset_creator_factory import DatasetCreatorFactory

class DataSplitter:
    def __init__(self, time_step, use_sixth_column=False):
        self.time_step = time_step
        self.use_sixth_column = use_sixth_column

    def split_data(self, train_ratio=1.0, val_ratio=0.0):
        dataset_creator = DatasetCreatorFactory().create_dataset_creator(self.time_step, self.use_sixth_column)
        print("DatasetCreator created.")

        X, y, scalers = dataset_creator.create_dataset()
        print("Dataset created.")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")

        total_size = len(X)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size

        X_train = X[:train_size]
        y_train = y[:train_size]

        X_val = np.array([]).reshape(0, self.time_step, X.shape[2])
        y_val = np.array([]).reshape(0, y.shape[1])

        X_test = np.array([]).reshape(0, self.time_step, X.shape[2])
        y_test = np.array([]).reshape(0, y.shape[1])

        print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
        print(f"Validation shapes: X={X_val.shape}, y={y_val.shape}")
        print(f"Test shapes: X={X_test.shape}, y={y_test.shape}")

        return {
            'all_data': (X, y),
            'train_five': (X_train, y_train),
            'val_five': (X_val, y_val),
            'test_five': (X_test, y_test),
            'scalers': scalers
        }
