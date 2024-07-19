# File: code/data/data_inspector.py
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

class DataInspector:
    def __init__(self, data_splits):
        print("Initializing DataInspector...")
        self.X_all_data_five, self.y_all_data_five = data_splits['all_data']
        self.X_train_five, self.y_train_five = data_splits['train_five']
        self.X_val_five, self.y_val_five = data_splits['val_five']
        self.X_test_five, self.y_test_five = data_splits['test_five']
        self.scalers = data_splits['scalers']

    def inspect_data(self):
        print("Inspecting data...")
        data_info = {
            'Dataset': ['All Data (First Five)', 'Train (First Five)', 'Validation (First Five)', 'Test (First Five)'],
            'X Shape': [self.X_all_data_five.shape, self.X_train_five.shape, self.X_val_five.shape, self.X_test_five.shape],
            'Y Shape': [self.y_all_data_five.shape, self.y_train_five.shape, self.y_val_five.shape, self.y_test_five.shape],
            'Number of Samples': [len(self.X_all_data_five), len(self.X_train_five), len(self.X_val_five), len(self.X_test_five)]
        }

        df_info = pd.DataFrame(data_info)
        print("Data Summary:")
        print(df_info.to_string(index=False))

        self.print_sample_data()

    def print_sample_data(self):
        print("\nSample Data (First and Last Rows):")

        def format_sample(X, y, name):
            df_X = pd.DataFrame(X.reshape(X.shape[0], -1))  # Flattening for better readability
            df_y = pd.DataFrame(y)
            sample_data = pd.concat([df_X, df_y], axis=1)
            print(f"\n{name} Data (First 1 and Last 1):")
            print(pd.concat([sample_data.head(1), sample_data.tail(1)]))

        format_sample(self.X_all_data_five, self.y_all_data_five, "All (First Five)")
        format_sample(self.X_train_five, self.y_train_five, "Train (First Five)")
        format_sample(self.X_val_five, self.y_val_five, "Validation (First Five)")
        format_sample(self.X_test_five, self.y_test_five, "Test (First Five)")
