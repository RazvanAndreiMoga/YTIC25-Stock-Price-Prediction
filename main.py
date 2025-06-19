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
from model.hf_llm_prediction_maker import HFLLMPredictionMaker


def main(ticker='MSFT'):
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
    preprocessor = DataPreprocessor(seq_length=10)
    merged_df = preprocessor.normalize_data(merged_df)

    # hf_model = HFLLMModel()
    # predictor = HFLLMPredictionMaker(model=hf_model, seq_length=10)

    predictor = HFLLMPredictionMaker(api_key="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")
    prediction, actual = predictor.make_prediction(merged_df)

    print(f"🔮 LLM Prediction: {prediction}")
    print(f"📈 Actual Movement: {actual}")
    print("✅ Correct!" if prediction == actual else "❌ Incorrect.")

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
