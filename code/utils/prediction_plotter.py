import os
import matplotlib.pyplot as plt
import pandas as pd

class PredictionPlotter:
    def __init__(self, symbol, market_data, prediction_data, hyperparams, save_dir):
        self.symbol = symbol
        self.market_data = market_data
        self.prediction_data = prediction_data
        self.hyperparams = hyperparams
        self.save_dir = save_dir

    def plot(self, filter_months=None):
        self.market_data.index = pd.to_datetime(self.market_data.index)
        self.prediction_data['date'] = pd.to_datetime(self.prediction_data['date'])

        if filter_months:
            end_date = self.market_data.index.max()
            start_date = end_date - pd.DateOffset(months=filter_months)
            self.market_data = self.market_data[start_date:end_date]
            self.prediction_data = self.prediction_data[self.prediction_data['date'] >= start_date]

        plt.figure(figsize=(14, 7))
        plt.plot(self.market_data.index, self.market_data['4. close'], label='Actual')
        plt.scatter(self.prediction_data['date'], self.prediction_data['prediction'], color='red', s=50, marker='o', label='Predicted')
        hyperparams_str = ', '.join(f"{key}={value}" for key, value in self.hyperparams.items())
        plt.title(f'Predicciones con {hyperparams_str}')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)

        save_path = os.path.join(self.save_dir, f'prediction_{filter_months}.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Saved prediction plot at {save_path}")
