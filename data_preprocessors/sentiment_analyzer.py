import os
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

class SentimentAnalyzer:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()

    def analyze_sentiment(self, df, ticker):
        # Compute sentiment scores
        f = lambda title: self.vader.polarity_scores(title)['compound']
        df['compound'] = df.iloc[:, 2].apply(f)
        
        # Convert datetime to date
        df['date'] = pd.to_datetime(df['datetime'], unit='s').dt.strftime('%Y-%m-%d')
        df = df.drop(columns=['datetime'])
        
        # Calculate mean sentiment scores
        mean_df = df.groupby(['related', 'date']).mean(numeric_only=True).unstack()
        mean_df = mean_df.xs('compound', axis="columns").reset_index()

        # Melt the DataFrame to have 'date' as a column and 'related' as a variable
        mean_df = mean_df.melt(id_vars=['related'], var_name='date', value_name='compound')

        # Set the 'date' column as the index
        sentiment = mean_df.rename(columns={'compound': ticker})
        sentiment = sentiment.set_index('date')  # Set the 'date' row as index

        # Create the line plot using matplotlib
        plt.figure(figsize=(14, 5))
        for key, grp in mean_df.groupby(['related']):
            plt.plot(grp['date'], grp['compound'], label=key)
        plt.title('Sentiment Analysis Over Time')
        plt.xlabel('Date')
        plt.ylabel('Compound Sentiment Score')
        plt.legend(loc='best')
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save the plot to the output folder
        plot_path = os.path.join('output', ticker, f'sentiment_analysis_{ticker}.png')
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path)
        plt.close()

        return sentiment
