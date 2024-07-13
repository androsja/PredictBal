import matplotlib.pyplot as plt
import numpy as np

class DataPlotter:
    @staticmethod
    def plot_data(X, Y, title):
        plt.figure(figsize=(10, 6))
        for i in range(X.shape[2]):
            plt.plot(X[:, 0, i], label=f'Feature {i}')
        plt.plot(Y, label='Target', linestyle='--')
        plt.title(title)
        plt.legend()
        plt.show()
