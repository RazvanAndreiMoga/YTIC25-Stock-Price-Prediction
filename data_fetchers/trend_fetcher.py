import os
import pandas as pd
import time
import matplotlib.pyplot as plt
from pytrends import dailydata

class TrendFetcher:
    def __init__(self, ticker, max_retries=5, wait_time=60):
        self.ticker = ticker
        self.max_retries = max_retries
        self.wait_time = wait_time

    def fetch_data(self, start_year, start_mon, stop_year, stop_mon, geo=''):
        file_path = f'input/{self.ticker}/{self.ticker}_daily.csv'
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if not os.path.exists(file_path):
            retries = 0
            while retries < self.max_retries:
                try:
                    res = dailydata.get_daily_data(self.ticker, start_year, start_mon, stop_year, stop_mon, geo)
                    print("Data fetched successfully.")
                    
                    # Select the relevant columns
                    data = res[[self.ticker + '_unscaled']]
                    
                    # Save the data to a CSV file
                    data.to_csv(file_path, header=True)
                    break
                except Exception as e:
                    retries += 1
                    print(f"Error encountered: {e}. Retrying {retries}/{self.max_retries}...")
                    time.sleep(self.wait_time)
            else:
                raise Exception("Max retries exceeded. Could not fetch data.")
        else:
            # Load the data from the existing file
            data = pd.read_csv(file_path, index_col=0)
            print("Data loaded from existing file.")
        
        # Plot the trend data using matplotlib
        plt.figure(figsize=(14, 5))
        plt.plot(data.index, data[self.ticker + '_unscaled'], label='Web Search Interest')
        plt.title('Keyword Web Search Interest Over Time')
        plt.xlabel('Date')
        plt.ylabel('Interest')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.legend()

        # Save the plot to the output folder
        plot_path = os.path.join('output', self.ticker, f'keyword_trend_{self.ticker}.png')
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path)
        plt.close()

        return data
