import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from data_fetchers.stock_fetcher import StockFetcher
from data_fetchers.sentiment_fetcher import SentimentFetcher
from data_preprocessors.sentiment_analyzer import SentimentAnalyzer
from data_fetchers.trend_fetcher import TrendFetcher

class DataMerger:
    def __init__(self, ticker, start_date='2024-01-01', end_date='2024-03-31'):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date

    def merge_data(self):
        # Fetch historical prices
        price_fetcher = StockFetcher(self.start_date, self.end_date)
        hist = price_fetcher.fetch_historical_prices(self.ticker)
        
        # Fetch sentiment data
        sentiment_fetcher = SentimentFetcher('input/financial_data.db')
        sentiment_data = sentiment_fetcher.fetch_sentiment_data(self.ticker)
        
        # Analyze sentiment
        sentiment_analyzer = SentimentAnalyzer()
        sentiment_transposed = sentiment_analyzer.analyze_sentiment(sentiment_data, self.ticker)
        
        # Fetch trend data
        trend_fetcher = TrendFetcher(self.ticker, max_retries=10, wait_time=10)
        ticker_daily = trend_fetcher.fetch_data(start_year=2024, start_mon=1, stop_year=2024, stop_mon=3)

        # Align date formats and ensure indexes are consistent
        hist.index = pd.to_datetime(hist.index)
        sentiment_transposed.index = pd.to_datetime(sentiment_transposed.index)
        ticker_daily.index = pd.to_datetime(ticker_daily.index)

        # Merge DataFrames
        merged_df = pd.merge(hist, sentiment_transposed[[self.ticker]], left_index=True, right_index=True, how='left')
        merged_df = pd.merge(merged_df, ticker_daily[[self.ticker + '_unscaled']], left_index=True, right_index=True, how='left')
        merged_df = merged_df.rename(columns={'Close': 'Price'})
        merged_df = merged_df.rename(columns={self.ticker: 'Sentiment'})
        merged_df = merged_df.rename(columns={self.ticker + '_unscaled': 'Trend'})
        merged_df['Sentiment'] = merged_df['Sentiment'].fillna(0)
        merged_df['Trend'] = merged_df['Trend'].fillna(0)

                # Calculate the correlation matrix
        correlation_matrix = merged_df.corr()

        # Plot the correlation matrix as a heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Matrix')

        # Save the correlation matrix plot to the output folder
        corr_matrix_path = os.path.join('output', self.ticker, f'correlation_matrix_{self.ticker}.png')
        os.makedirs(os.path.dirname(corr_matrix_path), exist_ok=True)
        plt.savefig(corr_matrix_path)
        plt.close()

        return merged_df, correlation_matrix