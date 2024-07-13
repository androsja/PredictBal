import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
class DataRepository:
    def __init__(self, base_dir='/Users/jflorezgaleano/Documents/JulianFlorez/PredictBalGit/models_ia'):
        self.base_dir = base_dir

    def save_data(self, X, Y, data_normalized, time_step):
        file_prefix = f"timestep_{time_step}"

        X_file = os.path.join(self.base_dir, f"{file_prefix}_X.npy")
        Y_file = os.path.join(self.base_dir, f"{file_prefix}_Y.npy")
        normalized_file = os.path.join(self.base_dir, f"{file_prefix}_normalized.csv")

        np.save(X_file, X)
        np.save(Y_file, Y)
        data_normalized.to_csv(normalized_file)

        print(f"Data saved:\nX file: {X_file}\nY file: {Y_file}\nNormalized file: {normalized_file}")
