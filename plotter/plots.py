import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Configuration
TICKERS = ["AMZN", "MSFT", "META", "GOOG"]  # Add more tickers here
# MODELS = ["deepseek_r1_0528", "phi_4", "llama_4_maverick", "gemini_2.5_flash_preview_05_20"]  # Add more models here
MODELS = ["LSTM"]
# DATA_DIR = "."
OUTPUT_DIR = "plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Date ranges
start_date = "2024-03-04"
cutoff_date = "2024-03-18"
end_date = "2024-03-20"

# Iterate over each ticker-model combination
for ticker in TICKERS:
    # Load actual data once per ticker
    # merged_data_path = os.path.join(f"input/{ticker}/hist_{ticker}.csv")
    # df_actual = pd.read_csv(merged_data_path, parse_dates=["Date"])
    # df_actual_filtered = df_actual[(df_actual["Date"] >= start_date) & (df_actual["Date"] <= end_date)]

    # Prepare metric dictionary per ticker
    metrics_dict = {}

    for model in MODELS:
        # # --- Line Plot: Actual vs Prediction ---
        # pred_file = os.path.join(f"openrouter_basic_prompt/prediction_{ticker}_{model}.txt")
        # if not os.path.exists(pred_file):
        #     print(f"[WARNING] Missing prediction file: {pred_file}")
        #     continue

        # try:
        #     with open(pred_file, "r") as f:
        #         predicted_prices = [float(line.strip()) for line in f.readlines()]
        # except:
        #     print(f"[ERROR] Could not read prediction file: {pred_file}")
        #     continue

        # pred_dates = pd.date_range(start=cutoff_date, periods=len(predicted_prices))

        # plt.figure(figsize=(10, 6))
        # plt.plot(df_actual_filtered["Date"], df_actual_filtered["Price"], label="Actual Price", color="black", linewidth=2)
        # plt.plot(pred_dates, predicted_prices, label=f"Prediction - {model}", linestyle="--")
        # plt.axvline(pd.to_datetime(cutoff_date), color="red", linestyle="--", label="Prediction Start")

        # plt.xlabel("Date")
        # plt.ylabel("Price")
        # plt.title(f"{ticker} Price: Actual vs {model} Prediction")
        # plt.legend()
        # plt.grid(True)
        # plt.tight_layout()
        # plt.savefig(os.path.join(OUTPUT_DIR, f"price_comparison_{ticker}_{model}.png"))
        # plt.close()

        # --- Metrics Loading ---
        metrics_file = os.path.join(f"lstm_output/metrics_{model}_{ticker}.txt")
        if not os.path.exists(metrics_file):
            print(f"[WARNING] Missing metrics file: {metrics_file}")
            continue

        try:
            with open(metrics_file, "r") as f:
                lines = f.readlines()
            metrics = {
                "MAE": float(lines[2].split(":")[1].strip()),
                "RMSE": float(lines[3].split(":")[1].strip()),
                "sMAPE": float(lines[4].split(":")[1].strip().replace('%', '')),
                "R2": float(lines[5].split(":")[1].strip()),
                "DA": float(lines[6].split(":")[1].strip())
            }
            metrics_dict[model] = metrics
        except Exception as e:
            print(f"[ERROR] Failed parsing {metrics_file}: {e}")
            continue

    # --- Bar Plot: Model Metrics ---
    if metrics_dict:
        metrics_df = pd.DataFrame(metrics_dict).T  # Transpose so models are rows

        plt.figure(figsize=(12, 6))
        metrics_df.plot(kind="bar")
        plt.title(f"Model Metrics for {ticker}")
        plt.ylabel("Metric Value")
        plt.xticks(rotation=45)
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"model_metrics_{ticker}.png"))
        plt.close()
