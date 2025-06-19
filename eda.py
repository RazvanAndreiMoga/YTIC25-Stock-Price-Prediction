import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_theme(style='whitegrid')
sns.set_palette("husl")

# Load the data
data = pd.read_csv('output/MSFT/merged_data_MSFT.csv', parse_dates=['Date'])
data.set_index('Date', inplace=True)

# 1. Summary Statistics
print("Summary Statistics:")
print(data.describe())

# 2. Time-Series Plots
plt.figure(figsize=(15, 12))

# Price
plt.subplot(2, 2, 1)
plt.plot(data.index, data['Price'], label='Price', color='blue')
plt.title('Microsoft Stock Price (Jan-Mar 2024)')
plt.ylabel('Price ($)')
plt.grid(True)

# Volume
plt.subplot(2, 2, 2)
plt.bar(data.index, data['Volume'], color='green', alpha=0.7)
plt.title('Trading Volume')
plt.ylabel('Volume')
plt.grid(True)

# Sentiment
plt.subplot(2, 2, 3)
plt.plot(data.index, data['Sentiment'], label='Sentiment', color='orange')
plt.title('Sentiment Score Over Time')
plt.ylabel('Sentiment Score')
plt.grid(True)

# Trend
plt.subplot(2, 2, 4)
plt.plot(data.index, data['Trend'], label='Trend', color='purple')
plt.title('Trend Indicator Over Time')
plt.ylabel('Trend Value')
plt.grid(True)

plt.tight_layout()
plt.savefig('time_series_plots.png')
plt.close()

# 3. Distributions & Boxplots
plt.figure(figsize=(15, 10))

# Price Distribution
plt.subplot(2, 2, 1)
sns.histplot(data['Price'], kde=True, color='blue')
plt.title('Price Distribution')
plt.xlabel('Price ($)')

# Volume Distribution
plt.subplot(2, 2, 2)
sns.histplot(data['Volume'], kde=True, color='green')
plt.title('Volume Distribution')
plt.xlabel('Volume')

# Sentiment Distribution
plt.subplot(2, 2, 3)
sns.histplot(data['Sentiment'], kde=True, color='orange')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment Score')

# Trend Distribution
plt.subplot(2, 2, 4)
sns.histplot(data['Trend'], kde=True, color='purple')
plt.title('Trend Distribution')
plt.xlabel('Trend Value')

plt.tight_layout()
plt.savefig('distributions.png')
plt.close()

# Boxplots
plt.figure(figsize=(15, 5))
sns.boxplot(data=data[['Price', 'Volume', 'Sentiment', 'Trend']])
plt.title('Boxplots of All Variables')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('boxplots.png')
plt.close()

# 4. Correlation Matrix + Heatmap
corr = data.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.close()

print("\nCorrelation Matrix:")
print(corr)

# 5. Rolling Averages (7-day)
rolling_window = 7

plt.figure(figsize=(15, 6))

# Price Rolling Average
plt.subplot(1, 2, 1)
plt.plot(data.index, data['Price'], label='Daily Price', alpha=0.3)
plt.plot(data.index, data['Price'].rolling(rolling_window).mean(), 
         label=f'{rolling_window}-Day Avg', color='red')
plt.title('Price with Rolling Average')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True)

# Volume Rolling Average
plt.subplot(1, 2, 2)
plt.bar(data.index, data['Volume'], alpha=0.3, label='Daily Volume')
plt.plot(data.index, data['Volume'].rolling(rolling_window).mean(), 
         label=f'{rolling_window}-Day Avg', color='green')
plt.title('Volume with Rolling Average')
plt.ylabel('Volume')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('rolling_averages.png')
plt.close()

# 6. Relationship Plots
plt.figure(figsize=(15, 12))

# Price vs Volume
plt.subplot(2, 2, 1)
sns.scatterplot(x='Volume', y='Price', data=data, hue=data.index.month)
plt.title('Price vs Volume')
plt.grid(True)

# Price vs Sentiment
plt.subplot(2, 2, 2)
sns.scatterplot(x='Sentiment', y='Price', data=data, hue=data.index.month)
plt.title('Price vs Sentiment')
plt.grid(True)

# Trend vs Price
plt.subplot(2, 2, 3)
sns.scatterplot(x='Trend', y='Price', data=data, hue=data.index.month)
plt.title('Price vs Trend')
plt.grid(True)

# Sentiment vs Trend
plt.subplot(2, 2, 4)
sns.scatterplot(x='Trend', y='Sentiment', data=data, hue=data.index.month)
plt.title('Sentiment vs Trend')
plt.grid(True)

plt.tight_layout()
plt.savefig('relationship_plots.png')
plt.close()

print("\nEDA complete! All plots saved as PNG files.")