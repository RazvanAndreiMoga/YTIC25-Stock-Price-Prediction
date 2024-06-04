import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from data_fetchers.stock_fetcher import StockFetcher
from data_fetchers.sentiment_fetcher import SentimentFetcher
from data_preprocessors.sentiment_analyzer import SentimentAnalyzer
from data_fetchers.trend_fetcher import TrendFetcher

class DataMerger:
    def __init__(self, ticker, price_df, sentiment_df, trend_df):
        self.ticker = ticker
        self.price_df = price_df
        self.sentiment_df = sentiment_df
        self.trend_df = trend_df

    def merge_data(self):
        # Align date formats and ensure indexes are consistent
        self.price_df.index = pd.to_datetime(self.price_df.index)
        self.sentiment_df.index = pd.to_datetime(self.sentiment_df.index)
        self.trend_df.index = pd.to_datetime(self.trend_df.index)

        # Merge DataFrames
        merged_df = pd.merge(self.price_df, self.sentiment_df[[self.ticker]], left_index=True, right_index=True, how='left')
        merged_df = pd.merge(merged_df, self.trend_df[[self.ticker + '_unscaled']], left_index=True, right_index=True, how='left')
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