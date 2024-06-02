import os
import yfinance as yf
import matplotlib.pyplot as plt

class StockFetcher:
    def __init__(self, start_date='2024-01-01', end_date='2024-03-31'):
        self.start_date = start_date
        self.end_date = end_date

    def fetch_historical_prices(self, ticker):
        # Fetch historical stock data
        hist = yf.Ticker(ticker).history(start=self.start_date, end=self.end_date)[['Close', 'Volume']]
        hist.index = hist.index.tz_localize(None)
        
        # Plotting Close price and Volume on the same plot with different y-axes
        fig, ax1 = plt.subplots(figsize=(14, 5))

        # Plot Close price on the left y-axis
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Close Price', color='tab:blue')
        ax1.plot(hist.index, hist['Close'], color='tab:blue', label='Close Price')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        # Create a second y-axis to plot Volume
        ax2 = ax1.twinx()
        ax2.set_ylabel('Volume', color='tab:orange')
        ax2.plot(hist.index, hist['Volume'], color='tab:orange', label='Volume')
        ax2.tick_params(axis='y', labelcolor='tab:orange')

        # Add title and show the plot
        plt.title(f'{ticker} Stock Price and Volume Over Time')
        fig.tight_layout()

        # Save the plot to the output folder
        plot_path = os.path.join('output', ticker, f'stock_price_volume_{ticker}.png')
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        fig.savefig(plot_path)
        plt.close(fig)

        return hist
