# File: code/data/data_inspector.py
import pandas as pd
import numpy as np

class DataInspector:
    def __init__(self, data_splits):
        self.X_all_data_five, self.y_all_data_five, self.X_all_data_sixth, self.y_all_data_sixth = data_splits['all_data']
        self.X_train_five, self.y_train_five = data_splits['train_five']
        self.X_val_five, self.y_val_five = data_splits['val_five']
        self.X_test_five, self.y_test_five = data_splits['test_five']
        self.X_train_sixth, self.y_train_sixth = data_splits['train_sixth']
        self.X_val_sixth, self.y_val_sixth = data_splits['val_sixth']
        self.X_test_sixth, self.y_test_sixth = data_splits['test_sixth']
        self.scalers = data_splits['scalers']

    def inspect_data(self):
        data_info = {
            'Dataset': ['All Data (First Five)', 'Train (First Five)', 'Validation (First Five)', 'Test (First Five)',
                        'All Data (Sixth)', 'Train (Sixth)', 'Validation (Sixth)', 'Test (Sixth)'],
            'X Shape': [self.X_all_data_five.shape, self.X_train_five.shape, self.X_val_five.shape, self.X_test_five.shape,
                        self.X_all_data_sixth.shape, self.X_train_sixth.shape, self.X_val_sixth.shape, self.X_test_sixth.shape],
            'Y Shape': [self.y_all_data_five.shape, self.y_train_five.shape, self.y_val_five.shape, self.y_test_five.shape,
                        self.y_all_data_sixth.shape, self.y_train_sixth.shape, self.y_val_sixth.shape, self.y_test_sixth.shape],
            'Number of Samples': [len(self.X_all_data_five), len(self.X_train_five), len(self.X_val_five), len(self.X_test_five),
                                  len(self.X_all_data_sixth), len(self.X_train_sixth), len(self.X_val_sixth), len(self.X_test_sixth)]
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

        format_sample(self.X_all_data_sixth, self.y_all_data_sixth, "All (Sixth)")
        format_sample(self.X_train_sixth, self.y_train_sixth, "Train (Sixth)")
        format_sample(self.X_val_sixth, self.y_val_sixth, "Validation (Sixth)")
        format_sample(self.X_test_sixth, self.y_test_sixth, "Test (Sixth)")
