import os
import matplotlib.pyplot as plt

class Plotter:
    def __init__(self, ticker):
        self.ticker = ticker

    def display_plots(self):
        # Display the first set of plots in a single figure with subplots
        plot_paths_1 = [
            os.path.join('output', self.ticker, f'stock_price_volume_{self.ticker}.png'),
            os.path.join('output', self.ticker, f'sentiment_analysis_{self.ticker}.png'),
            os.path.join('output', self.ticker, f'keyword_trend_{self.ticker}.png'),
            os.path.join('output', self.ticker, f'correlation_matrix_{self.ticker}.png')
        ]

        fig1, axes1 = plt.subplots(2, 2, figsize=(16, 16))
        axes1 = axes1.flatten()

        for ax, plot_path in zip(axes1, plot_paths_1):
            img = plt.imread(plot_path)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(os.path.basename(plot_path).replace('_', ' ').replace('.png', '').title(), fontsize=16)

        plt.tight_layout(pad=3.0)
        plt.show()

        # Display the second set of plots using plt.subplot
        plot_path_prediction = os.path.join('output', self.ticker, f'stock_price_prediction_{self.ticker}.png')
        metrics_path = os.path.join('output', self.ticker, f'{self.ticker}_metrics.txt')
        trend_path = os.path.join('output', self.ticker, f'{self.ticker}_trend.txt')

        # Create a new figure
        plt.figure(figsize=(16, 16))

        # Predicted Stock Prices
        plt.subplot(2, 2, 1)  # divide as 2x2, plot top left
        img = plt.imread(plot_path_prediction)
        plt.imshow(img)
        plt.axis('off')
        plt.title(os.path.basename(plot_path_prediction).replace('_', ' ').replace('.png', '').title(), fontsize=16)

        # Evaluation Metrics Bar Plot
        plt.subplot(2, 2, 2)  # divide as 2x2, plot top right
        metrics = {}
        with open(metrics_path, 'r') as f:
            for line in f:
                key, value = line.split(':')
                value = value.strip()
                if '%' in value:
                    value = value.replace('%', '')
                metrics[key.strip()] = float(value)

        plt.bar(metrics.keys(), metrics.values(), color=['blue', 'orange', 'green', 'red', 'purple'])
        plt.title('Evaluation Metrics', fontsize=16)
        plt.ylim(0, max(metrics.values()) * 1.2)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        # Predicted Trend
        plt.subplot(2, 1, 2)  # divide as 2x1, plot bottom
        with open(trend_path, 'r') as f:
            trend_lines = f.readlines()

        bullish_count = sum('Bullish' in line for line in trend_lines)
        bearish_count = sum('Bearish' in line for line in trend_lines)
        overall_trend = 'Bullish' if bullish_count > bearish_count else 'Bearish'

        plt.text(0.5, 0.5, f'Predicted Trend: {overall_trend}', ha='center', va='center', fontsize=20)
        plt.axis('off')
        plt.title('Predictions', fontsize=16)

        plt.tight_layout(pad=3.0)
        plt.show()
