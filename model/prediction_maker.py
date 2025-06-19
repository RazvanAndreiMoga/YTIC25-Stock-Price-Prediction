import os
import numpy as np

class PredictionMaker:
    def __init__(self, ticker, model, scaler, seq_length):
        self.ticker = ticker
        self.model = model
        self.scaler = scaler
        self.seq_length = seq_length

    def make_predictions(self, X_test, scaled_data, test_data, train_data):
        # Make predictions
        predictions = self.model.predict(X_test)

        # Inverse transform the predictions
        full_scaler_predictions = np.zeros((predictions.shape[0], scaled_data.shape[1]))
        full_scaler_predictions[:, 0] = predictions[:, 0]
        inverse_predictions = self.scaler.inverse_transform(full_scaler_predictions)
        final_predictions = inverse_predictions[:, 0]

        # Extract true 'Close' values from test data
        true_close = self.scaler.inverse_transform(test_data)[:, 0][self.seq_length:]

        # Combine true close values from the entire dataset
        all_close_values = self.scaler.inverse_transform(scaled_data)[:, 0]

        # Latest known price from the training data
        latest_known_price = all_close_values[58]

        # Compare predictions with the latest known price and save to a text file
        trend_file_path = os.path.join('output', self.ticker, f'{self.ticker}_trend.txt')
        os.makedirs(os.path.dirname(trend_file_path), exist_ok=True)
        with open(trend_file_path, 'w') as f:
            for i, predicted_price in enumerate(final_predictions):
                trend = "Bullish" if predicted_price > latest_known_price else "Bearish"
                f.write(f"Day {i + 1}: Predicted Close = {predicted_price:.2f}, Actual Close = {latest_known_price:.2f}, Trend = {trend}\n")

        return final_predictions, true_close, all_close_values
