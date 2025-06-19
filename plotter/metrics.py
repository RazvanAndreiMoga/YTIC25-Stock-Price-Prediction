import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import re
import os

# Lists of models and tickers
models = ["deepseek_r1_0528", "phi_4", "llama_4_maverick", "gemini_2.5_flash_preview_05_20"]
tickers = ["AMZN", "META", "GOOG", "MSFT"]

def extract_predictions(ticker, model):
    filename = f"prediction_{ticker}_{model}.txt"
    with open(filename, 'r') as file:
        content = file.read()

    number_pattern = r'(?<![\d-])\$?\s*(\d+(?:\.\d+)?)\b(?!\s*-)'
    matches = re.findall(number_pattern, content)

    numeric_values = []
    for num in matches:
        try:
            numeric_values.append(float(num.replace(",", "")))
        except ValueError:
            continue

    if len(numeric_values) < 3:
        raise ValueError(f"Not enough numeric values found in {filename} to extract 3 predictions.")

    return np.array(numeric_values[-3:])

# Define the target dates
target_dates = pd.to_datetime(["2024-03-18", "2024-03-19", "2024-03-20"])

# Iterate over each model and ticker
for model in models:
    for ticker in tickers:
        try:
            predicted = extract_predictions(ticker, model)
            print(f"\nModel: {model} | Ticker: {ticker}")
            print("Predicted values:", predicted)

            file_path = f"merged_data_{ticker}.csv"
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            real = df[df['Date'].isin(target_dates)]["Price"].values

            print("Prices on specified dates:", real)

            mae = mean_absolute_error(real, predicted)
            rmse = np.sqrt(mean_squared_error(real, predicted))
            smape = np.mean(2 * np.abs(predicted - real) / (np.abs(predicted) + np.abs(real))) * 100
            r2 = r2_score(real, predicted)

            real_diff = np.diff(real, prepend=real[0])
            pred_diff = np.diff(predicted, prepend=real[0])
            da = np.mean(np.sign(real_diff[1:]) == np.sign(pred_diff[1:]))

            output_filename = f"metrics_{model}_{ticker}.txt"
            with open(output_filename, "w") as f:
                f.write(f"Model: {model}\n")
                f.write(f"Ticker: {ticker}\n")
                f.write(f"MAE: {mae:.4f}\n")
                f.write(f"RMSE: {rmse:.4f}\n")
                f.write(f"sMAPE: {smape:.2f}%\n")
                f.write(f"R2 Score: {r2:.4f}\n")
                f.write(f"Directional Accuracy (DA): {da:.4f}\n")

            print(f"Metrics saved to {output_filename}")
        except Exception as e:
            print(f"Error processing model {model}, ticker {ticker}: {e}")
