import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

class ModelEvaluator:
    def __init__(self, ticker):
        self.ticker = ticker

    def plot_predictions(self, train_data, seq_length, final_predictions, all_close_values):
        # Define the y-axis limits
        y_min = min(all_close_values.min(), final_predictions.min()) * 0.8
        y_max = max(all_close_values.max(), final_predictions.max()) * 1.2

        plt.figure(figsize=(14, 5))
        plt.plot(all_close_values, color='blue', label='True Close Price')
        plt.plot(range(len(train_data) + seq_length, len(train_data) + seq_length + len(final_predictions)), final_predictions, color='red', label='Predicted Close Price')
        plt.axvline(x=len(train_data) + seq_length, color='green', linestyle='--', label='Prediction Start')
        plt.title(f'{self.ticker} Close Price Prediction')
        plt.ylim(y_min, y_max)
        plt.xlabel('Days')
        plt.ylabel('Close Price')
        plt.legend()

        # Save the plot to the output folder
        plot_path = os.path.join('output', self.ticker, f'stock_price_prediction_{self.ticker}.png')
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path)
        plt.close()

    def evaluate_model(self, true_close, final_predictions):
        # Calculate additional metrics
        mae = mean_absolute_error(true_close, final_predictions)
        mse = mean_squared_error(true_close, final_predictions)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((true_close - final_predictions) / true_close)) * 100

        # Save metrics to a file
        metrics_file_path = os.path.join('output', self.ticker, f'{self.ticker}_metrics.txt')
        os.makedirs(os.path.dirname(metrics_file_path), exist_ok=True)
        with open(metrics_file_path, 'w') as f:
            f.write(f'MAE: {mae:.4f}\n')
            f.write(f'MSE: {mse:.4f}\n')
            f.write(f'RMSE: {rmse:.4f}\n')
            f.write(f'MAPE: {mape:.4f}%\n')

        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape
        }
