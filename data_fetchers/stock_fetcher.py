import os
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from curl_cffi import requests


class StockFetcher:


    def __init__(self, start_date='2024-01-01', end_date='2024-03-31'):
        self.start_date = start_date
        self.end_date = end_date

    def fetch_historical_prices(self, ticker):
        # Define the paths for storing historical data and plots
        hist_dir = os.path.join('input', ticker)
        hist_path = os.path.join(hist_dir, f'hist_{ticker}.csv')
        
        # Check if the historical data file already exists
        if os.path.exists(hist_path):
            # Load historical data from CSV
            hist = pd.read_csv(hist_path, index_col=0, parse_dates=True)
        else:
            # Fetch historical data from Yahoo Finance
            session = requests.Session(impersonate="chrome")
            hist = yf.Ticker(ticker, session=session).history(start=self.start_date, end=self.end_date)[['Close', 'Volume']]
            hist.index = hist.index.tz_localize(None)  # Remove timezone information
            
            # Ensure the directory exists and save to CSV
            os.makedirs(hist_dir, exist_ok=True)
            hist.to_csv(hist_path)
        
        # Create output directory for plots
        plot_dir = os.path.join('output', ticker)
        os.makedirs(plot_dir, exist_ok=True)
        
        # Plotting the data
        fig, ax1 = plt.subplots(figsize=(14, 5))
        
        # Plot Close Price on the primary y-axis
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Close Price', color='tab:blue')
        ax1.plot(hist.index, hist['Close'], color='tab:blue', label='Close Price')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        
        # Create a secondary y-axis for Volume
        ax2 = ax1.twinx()
        ax2.set_ylabel('Volume', color='tab:orange')
        ax2.plot(hist.index, hist['Volume'], color='tab:orange', label='Volume')
        ax2.tick_params(axis='y', labelcolor='tab:orange')
        
        # Title and layout adjustments
        plt.title(f'{ticker} Stock Price and Volume Over Time')
        fig.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(plot_dir, f'stock_price_volume_{ticker}.png')
        fig.savefig(plot_path)
        plt.close(fig)
        
        return hist