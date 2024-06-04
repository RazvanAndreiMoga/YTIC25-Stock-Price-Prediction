import sys
from data_fetchers.data_fetcher import DataMerger
from data_preprocessors.data_preprocessor import DataPreprocessor
from model.lstm_model import LSTMModel
from model.prediction_maker import PredictionMaker
from model.model_evaluator import ModelEvaluator
from data_fetchers.stock_fetcher import StockFetcher
from data_fetchers.sentiment_fetcher import SentimentFetcher
from data_preprocessors.sentiment_analyzer import SentimentAnalyzer
from data_fetchers.trend_fetcher import TrendFetcher
from plotter.plotter import Plotter

def main(ticker='MSFT'):
    # Initialize the StockFetcher with the desired date range
    stock_fetcher = StockFetcher(start_date='2024-01-01', end_date='2024-03-31')
    
    # Fetch historical prices for the given ticker
    price_df = stock_fetcher.fetch_historical_prices(ticker)

    # Initialize the SentimentFetcher
    sentiment_fetcher = SentimentFetcher('input/financial_data.db')
    
    # Fetch sentiment data for the given ticker symbol
    sentiment_data = sentiment_fetcher.fetch_sentiment_data(ticker)

    # Initialize the SentimentAnalyzer
    sentiment_analyzer = SentimentAnalyzer()

    # Analyze the sentiment
    sentiment_df = sentiment_analyzer.analyze_sentiment(sentiment_data, ticker)

    # Initialize the TrendFetcher
    trend_fetcher = TrendFetcher(ticker=ticker, max_retries=10, wait_time=10)
    
    # Fetch trend data
    trend_df = trend_fetcher.fetch_data(start_year=2024, start_mon=1, stop_year=2024, stop_mon=3)

    # Initialize the DataMerger with the desired ticker and date range
    data_merger = DataMerger(ticker=ticker, price_df=price_df, sentiment_df=sentiment_df, trend_df=trend_df)

    # Merge data for the given ticker
    merged_df, correlation_matrix = data_merger.merge_data()

    # Initialize the DataPreprocessor with the desired sequence length
    preprocessor = DataPreprocessor(seq_length=10)

    # Preprocess the merged data to get training and testing sets
    X_train, y_train, X_test, y_test = preprocessor.preprocess(merged_df)

    # Initialize the LSTMModel with the input shape
    input_shape = (X_train.shape[1], X_train.shape[2])
    trainer = LSTMModel(ticker=ticker, input_shape=input_shape)

    # Train the model and get the average validation loss
    model, average_val_loss = trainer.train_model(X_train, y_train, X_test, y_test)

    # Initialize the PredictionMaker with the trained model, scaler, and ticker
    prediction_maker = PredictionMaker(ticker, model, preprocessor.scaler, preprocessor.seq_length)

    # Make predictions and compare with true values
    final_predictions, true_close, all_close_values = prediction_maker.make_predictions(X_test, preprocessor.normalize_data(merged_df), preprocessor.split_data(preprocessor.normalize_data(merged_df))[1], preprocessor.split_data(preprocessor.normalize_data(merged_df))[0])

    # Initialize the ModelEvaluator
    evaluator = ModelEvaluator(ticker=ticker)

    # Plot the predictions
    evaluator.plot_predictions(preprocessor.split_data(preprocessor.normalize_data(merged_df))[0], preprocessor.seq_length, final_predictions, all_close_values)

    # Evaluate the model and save metrics to a file
    evaluation_metrics = evaluator.evaluate_model(true_close, final_predictions)

    # Initialize the Plotter and display the plots
    plotter = Plotter(ticker=ticker)
    plotter.display_plots()

if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else 'MSFT'
    main(ticker)
