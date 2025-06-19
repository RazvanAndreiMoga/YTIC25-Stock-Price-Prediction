import sys
from data_fetchers.data_fetcher import DataMerger
from data_preprocessors.data_preprocessor import DataPreprocessor
from model.model_evaluator import ModelEvaluator
from data_fetchers.stock_fetcher import StockFetcher
from data_fetchers.sentiment_fetcher import SentimentFetcher
from data_preprocessors.sentiment_analyzer import SentimentAnalyzer
from data_fetchers.trend_fetcher import TrendFetcher
from plotter.plotter import Plotter
from model.hf_llm_model import HFLLMModel
from model.hf_llm_prediction_maker import HFLLMPredictionMaker
import json


def main(ticker='MSFT'):

    api_key=""

    # Initialize the StockFetcher
    stock_fetcher = StockFetcher(start_date='2024-01-01', end_date='2024-03-31')
    price_df = stock_fetcher.fetch_historical_prices(ticker)

    # Fetch sentiment data
    sentiment_fetcher = SentimentFetcher('input/financial_data.db')
    sentiment_data = sentiment_fetcher.fetch_sentiment_data(ticker)

    sentiment_analyzer = SentimentAnalyzer()
    sentiment_df = sentiment_analyzer.analyze_sentiment(sentiment_data, ticker)

    # Fetch trend data
    trend_fetcher = TrendFetcher(ticker=ticker, max_retries=10, wait_time=10)
    trend_df = trend_fetcher.fetch_data(start_year=2024, start_mon=1, stop_year=2024, stop_mon=3)

    # Merge all data
    data_merger = DataMerger(ticker=ticker, price_df=price_df, sentiment_df=sentiment_df, trend_df=trend_df)
    merged_df, correlation_matrix = data_merger.merge_data()

    # Preprocess merged data (mainly to normalize if needed)
    # preprocessor = DataPreprocessor(seq_length=10)
    # merged_df = preprocessor.normalize_data(merged_df)

    # hf_model = HFLLMModel()
    # predictor = HFLLMPredictionMaker(model=hf_model, seq_length=10)

    models = [
        "deepseek/deepseek-r1-0528", 
        "microsoft/phi-4-reasoning-plus",
        "meta-llama/llama-4-maverick"           
    ]
    for model in models:
        # Initialize the predictor
        api_key = api_key
        predictor = HFLLMPredictionMaker(api_key, model=model)

        # Make price prediction
        predicted_price, actual_price, result = predictor.make_prediction(merged_df, ticker)
        print(f"Predicted: ${predicted_price:.2f}, Actual: ${actual_price:.2f}")
        print(f"Error: ${result['absolute_error']:.2f} ({result['percentage_error']:.2f}%)")

        # Comprehensive analysis
        analysis = predictor.analyze_ticker_performance(ticker)
        print(f"Direction Accuracy: {analysis['direction_accuracy']:.2f}%")
        print(f"MAPE: {analysis['error_metrics']['mape']:.2f}%")
        print(f"RMSE: ${analysis['error_metrics']['rmse']:.2f}")

        # Quality report
        quality_report = predictor.get_prediction_quality_report(ticker, model)
        print(f"Quality Rating: {quality_report['quality_rating']}")
        print(f"Recommendations: {quality_report['recommendations']}")

        # Enhanced trading simulation with thresholds
        # trading_results = predictor.get_trading_simulation(ticker, price_threshold=0.03)
        # print(f"Strategy Return: {trading_results['total_return_pct']:.2f}%")
        # print(f"Buy & Hold Return: {trading_results['buy_hold_return_pct']:.2f}%")

        # predictor = HFLLMPredictionMaker(api_key="", model = model)
        # prediction, actual = predictor.make_prediction(merged_df)
        
        # print(f"🔮 {model} Prediction: {prediction}")
        # print(f"📈 Actual Movement: {actual}")
        # print("✅ Correct!" if prediction == actual else "❌ Incorrect.")

    # Get prediction
    # final_predictions, true_labels = predictor.make_predictions(merged_df)

    # Evaluate or display
    # print(f"\nPrediction: {final_predictions[0]}")


    # # 📊 Evaluate predictions
    # evaluator = ModelEvaluator(ticker=ticker)
    # evaluator.plot_predictions(preprocessor.split_data(merged_df)[0], 10, final_predictions, merged_df["Price"].tolist())
    # evaluation_metrics = evaluator.evaluate_model(true_labels, final_predictions)

    # # 📈 Plotting
    # plotter = Plotter(ticker=ticker)
    # plotter.display_plots()

if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else 'MSFT'
    main(ticker)
