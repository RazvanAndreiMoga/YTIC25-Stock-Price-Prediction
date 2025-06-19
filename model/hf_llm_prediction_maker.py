import json
import requests
import os
import pandas as pd
from datetime import datetime
import numpy as np
from collections import defaultdict
import math

class HFLLMPredictionMaker:
    def __init__(self, api_key, model="openai/gpt-4o", results_dir="predictions"):
        self.api_key = api_key
        self.model = model
        self.results_dir = results_dir
        
        # Create results directory if it doesn't exist
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

    def make_prediction(self, df, ticker_symbol):
        if len(df) < 61:  # Need 61 rows to get 60 for analysis + 1 for actual
            raise ValueError("DataFrame must contain at least 61 rows.")

        # Use the last 61 rows (60 for input + 1 for actual result)
        recent_data = df.tail(61).reset_index(drop=True)
        # print(recent_data)
        # Format the first 60 rows as input
        input_data = recent_data.iloc[:56].to_dict(orient="records")
        current_price = recent_data.iloc[59]["Price"]
        # print(input_data)
        {json.dumps(input_data, indent=2)} 
        input_text = f"Given the following 56 days of stock data for {ticker_symbol}, predict the exact stock price for the next 5 days. Historical data: {json.dumps(input_data, indent=2)} Based on the patterns in price, volume, sentiment, and trend data, what will be the exact stock price for the next 5 days? Respond with only the numbers (the predicted prices). For example: 425.67, 423.35, 427.98, 424.56, 424.90"

        # Prepare the API request
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "your-site.com",
                "X-Title": "StockSentimentApp",
                "Content-Type": "application/json"
            },
            data=json.dumps({
                "model": self.model,
                "messages": [{"role": "user", "content": input_text}]
            })
        )

        if response.status_code != 200:
            raise RuntimeError(f"OpenRouter API error: {response.status_code} - {response.text}")

        result = response.json()
        prediction_text = result["choices"][0]["message"]["content"].strip()
        
        # Extract numerical prediction
        predicted_price = self._extract_price_from_response(prediction_text)
        
        # Get actual values
        price_today = float(recent_data.iloc[59]["Price"])  # Last day of input
        price_tomorrow = float(recent_data.iloc[60]["Price"])  # Next day (actual result)
        
        # Calculate directions for compatibility
        predicted_direction = "UP" if predicted_price > price_today else "DOWN"
        actual_direction = "UP" if price_tomorrow > price_today else "DOWN"
        direction_correct = predicted_direction == actual_direction
        
        # Calculate errors
        absolute_error = abs(predicted_price - price_tomorrow)
        percentage_error = abs((predicted_price - price_tomorrow) / price_tomorrow) * 100
        squared_error = (predicted_price - price_tomorrow) ** 2
        
        # Calculate percentage changes
        predicted_change_pct = ((predicted_price - price_today) / price_today) * 100
        actual_change_pct = ((price_tomorrow - price_today) / price_today) * 100

        # Save individual prediction result
        prediction_result = {
            "timestamp": datetime.now().isoformat(),
            "ticker": ticker_symbol,
            "model": self.model,
            "price_today": price_today,
            "predicted_price": predicted_price,
            "actual_price": price_tomorrow,
            "predicted_direction": predicted_direction,
            "actual_direction": actual_direction,
            "direction_correct": direction_correct,
            "predicted_change_pct": round(predicted_change_pct, 4),
            "actual_change_pct": round(actual_change_pct, 4),
            "absolute_error": round(absolute_error, 4),
            "percentage_error": round(percentage_error, 4),
            "squared_error": round(squared_error, 4),
            "date_predicted_for": recent_data.iloc[60]["Date"] if "Date" in recent_data.columns else None,
            "raw_model_response": prediction_text
        }
        
        self._save_prediction_result(ticker_symbol, prediction_result)
        
        return predicted_price, price_tomorrow, prediction_result

    def _extract_price_from_response(self, response_text):
        """Extract numerical price from model response"""
        import re
        
        # Remove common prefixes/suffixes and extract number
        cleaned = re.sub(r'[^\d.-]', ' ', response_text)
        numbers = re.findall(r'\d+\.?\d*', cleaned)
        
        if numbers:
            try:
                # Take the first reasonable number (usually the price)
                price = float(numbers[0])
                # Basic sanity check - stock prices shouldn't be negative or extremely high
                if 0 < price < 100000:
                    return price
            except ValueError:
                pass
        
        # Fallback: try to find dollar amounts
        dollar_matches = re.findall(r'\$?(\d+\.?\d*)', response_text)
        if dollar_matches:
            try:
                return float(dollar_matches[0])
            except ValueError:
                pass
        
        # If all else fails, return 0 (will be flagged as error)
        print(f"Warning: Could not extract valid price from: {response_text}")
        return 0.0

    def _save_prediction_result(self, ticker_symbol, result):
        """Save individual prediction result to ticker-specific file"""
        # Sanitize model name for file path (replace problematic characters)
        safe_model_name = self.model.replace("/", "_").replace("\\", "_").replace(":", "_")
        ticker_file = os.path.join(self.results_dir, f"{ticker_symbol}_{safe_model_name}_predictions.jsonl")
        
        with open(ticker_file, "a") as f:
            f.write(json.dumps(result) + "\n")

    def calculate_error_metrics(self, predictions_df):
        """Calculate comprehensive error metrics"""
        if len(predictions_df) == 0:
            return {}
        
        predicted = predictions_df['predicted_price'].values
        actual = predictions_df['actual_price'].values
        
        # Remove any zero predictions (parsing errors)
        valid_mask = predicted > 0
        predicted = predicted[valid_mask]
        actual = actual[valid_mask]
        
        if len(predicted) == 0:
            return {"error": "No valid predictions found"}
        
        # Mean Absolute Error
        mae = np.mean(np.abs(predicted - actual))
        
        # Mean Squared Error
        mse = np.mean((predicted - actual) ** 2)
        
        # Root Mean Squared Error
        rmse = np.sqrt(mse)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        # Mean Percentage Error (bias)
        mpe = np.mean((actual - predicted) / actual) * 100
        
        # R-squared (coefficient of determination)
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Additional metrics
        median_ae = np.median(np.abs(predicted - actual))
        max_error = np.max(np.abs(predicted - actual))
        std_error = np.std(predicted - actual)
        
        return {
            "mae": round(mae, 4),
            "mse": round(mse, 4),
            "rmse": round(rmse, 4),
            "mape": round(mape, 4),
            "mpe": round(mpe, 4),
            "r_squared": round(r_squared, 4),
            "median_absolute_error": round(median_ae, 4),
            "max_error": round(max_error, 4),
            "std_error": round(std_error, 4),
            "total_predictions": len(predicted),
            "valid_predictions": len(predicted)
        }

    def analyze_ticker_performance(self, ticker_symbol, model_name=None):
        """Analyze performance for a specific ticker with price prediction metrics"""
        if model_name:
            safe_model_name = model_name.replace("/", "_").replace("\\", "_").replace(":", "_")
            ticker_file = os.path.join(self.results_dir, f"{ticker_symbol}_{safe_model_name}_predictions.jsonl")
        else:
            # Try to find the file with current model name
            safe_model_name = self.model.replace("/", "_").replace("\\", "_").replace(":", "_")
            ticker_file = os.path.join(self.results_dir, f"{ticker_symbol}_{safe_model_name}_predictions.jsonl")
            
            # If not found, try without model name (backward compatibility)
            if not os.path.exists(ticker_file):
                ticker_file = os.path.join(self.results_dir, f"{ticker_symbol}_predictions.jsonl")
        
        if not os.path.exists(ticker_file):
            return f"No predictions found for {ticker_symbol}"
        
        predictions = []
        with open(ticker_file, "r") as f:
            for line in f:
                predictions.append(json.loads(line.strip()))
        
        if not predictions:
            return f"No predictions found for {ticker_symbol}"
        
        df = pd.DataFrame(predictions)
        
        # Direction accuracy (legacy compatibility)
        total_predictions = len(df)
        direction_correct = df['direction_correct'].sum()
        direction_accuracy = (direction_correct / total_predictions) * 100
        
        # Error metrics
        error_metrics = self.calculate_error_metrics(df)
        
        # Analyze by prediction direction
        up_predictions = df[df['predicted_direction'] == 'UP']
        down_predictions = df[df['predicted_direction'] == 'DOWN']
        
        up_direction_accuracy = (up_predictions['direction_correct'].sum() / len(up_predictions)) * 100 if len(up_predictions) > 0 else 0
        down_direction_accuracy = (down_predictions['direction_correct'].sum() / len(down_predictions)) * 100 if len(down_predictions) > 0 else 0
        
        # Price range analysis
        avg_predicted_price = df['predicted_price'].mean()
        avg_actual_price = df['actual_price'].mean()
        price_prediction_bias = avg_predicted_price - avg_actual_price
        
        analysis = {
            "ticker": ticker_symbol,
            "total_predictions": total_predictions,
            "direction_accuracy": round(direction_accuracy, 2),
            "up_predictions": len(up_predictions),
            "down_predictions": len(down_predictions),
            "up_direction_accuracy": round(up_direction_accuracy, 2),
            "down_direction_accuracy": round(down_direction_accuracy, 2),
            "avg_predicted_price": round(avg_predicted_price, 2),
            "avg_actual_price": round(avg_actual_price, 2),
            "price_prediction_bias": round(price_prediction_bias, 2),
            "error_metrics": error_metrics,
            "best_prediction_error": round(df['absolute_error'].min(), 2),
            "worst_prediction_error": round(df['absolute_error'].max(), 2),
            "avg_percentage_change_predicted": round(df['predicted_change_pct'].mean(), 2),
            "avg_percentage_change_actual": round(df['actual_change_pct'].mean(), 2)
        }
        
        return analysis

    def analyze_all_tickers(self):
        """Analyze performance across all tickers with comprehensive metrics"""
        if not os.path.exists(self.results_dir):
            return "No predictions directory found"
        
        # Look for files with model names and without
        all_files = [f for f in os.listdir(self.results_dir) if f.endswith('_predictions.jsonl')]
        
        if not all_files:
            return "No prediction files found"
        
        overall_stats = defaultdict(list)
        ticker_analyses = {}
        all_predictions = []
        
        for file in all_files:
            # Extract ticker from filename (handle both formats)
            if file.count('_') >= 2:  # Format: TICKER_MODEL_predictions.jsonl
                ticker = file.split('_')[0]
            else:  # Format: TICKER_predictions.jsonl
                ticker = file.replace('_predictions.jsonl', '')
                
            analysis = self.analyze_ticker_performance(ticker)
            
            if isinstance(analysis, dict):
                ticker_analyses[ticker] = analysis
                overall_stats['direction_accuracy'].append(analysis['direction_accuracy'])
                overall_stats['total_predictions'].append(analysis['total_predictions'])
                
                # Collect all predictions for overall error metrics
                ticker_file = os.path.join(self.results_dir, file)
                with open(ticker_file, "r") as f:
                    for line in f:
                        all_predictions.append(json.loads(line.strip()))
        
        # Calculate overall error metrics
        if all_predictions:
            all_df = pd.DataFrame(all_predictions)
            overall_error_metrics = self.calculate_error_metrics(all_df)
        else:
            overall_error_metrics = {}
        
        # Overall statistics
        avg_direction_accuracy = np.mean(overall_stats['direction_accuracy']) if overall_stats['direction_accuracy'] else 0
        total_predictions_all = sum(overall_stats['total_predictions'])
        
        # Find best and worst performing tickers by different metrics
        best_direction_ticker = max(ticker_analyses.items(), key=lambda x: x[1]['direction_accuracy']) if ticker_analyses else None
        best_mae_ticker = min(ticker_analyses.items(), key=lambda x: x[1]['error_metrics'].get('mae', float('inf'))) if ticker_analyses else None
        best_mape_ticker = min(ticker_analyses.items(), key=lambda x: x[1]['error_metrics'].get('mape', float('inf'))) if ticker_analyses else None
        
        summary = {
            "overall_direction_accuracy": round(avg_direction_accuracy, 2),
            "overall_error_metrics": overall_error_metrics,
            "total_predictions_across_all_tickers": total_predictions_all,
            "number_of_tickers": len(ticker_analyses),
            "best_direction_accuracy_ticker": {
                "ticker": best_direction_ticker[0] if best_direction_ticker else None,
                "accuracy": best_direction_ticker[1]['direction_accuracy'] if best_direction_ticker else None
            },
            "best_mae_ticker": {
                "ticker": best_mae_ticker[0] if best_mae_ticker else None,
                "mae": best_mae_ticker[1]['error_metrics'].get('mae') if best_mae_ticker else None
            },
            "best_mape_ticker": {
                "ticker": best_mape_ticker[0] if best_mape_ticker else None,
                "mape": best_mape_ticker[1]['error_metrics'].get('mape') if best_mape_ticker else None
            },
            "ticker_analyses": ticker_analyses
        }
        
        return summary

    def get_trading_simulation(self, ticker_symbol, initial_balance=10000, price_threshold=0.02, model_name=None):
        """
        Simulate trading based on model price predictions with configurable threshold
        price_threshold: minimum predicted percentage change to trigger a trade
        """
        if model_name:
            safe_model_name = model_name.replace("/", "_").replace("\\", "_").replace(":", "_")
            ticker_file = os.path.join(self.results_dir, f"{ticker_symbol}_{safe_model_name}_predictions.jsonl")
        else:
            # Try to find the file with current model name
            safe_model_name = self.model.replace("/", "_").replace("\\", "_").replace(":", "_")
            ticker_file = os.path.join(self.results_dir, f"{ticker_symbol}_{safe_model_name}_predictions.jsonl")
            
            # If not found, try without model name (backward compatibility)
            if not os.path.exists(ticker_file):
                ticker_file = os.path.join(self.results_dir, f"{ticker_symbol}_predictions.jsonl")
        
        if not os.path.exists(ticker_file):
            return f"No predictions found for {ticker_symbol}"
        
        predictions = []
        with open(ticker_file, "r") as f:
            for line in f:
                predictions.append(json.loads(line.strip()))
        
        if not predictions:
            return f"No predictions found for {ticker_symbol}"
        
        df = pd.DataFrame(predictions).sort_values('timestamp')
        
        balance = initial_balance
        shares = 0
        trades = []
        
        for _, row in df.iterrows():
            predicted_change = abs(row['predicted_change_pct'])
            
            # Only trade if predicted change exceeds threshold
            if predicted_change >= price_threshold * 100:
                if row['predicted_direction'] == 'UP' and balance > 0:
                    # Buy with all available balance
                    shares_bought = balance / row['price_today']
                    shares += shares_bought
                    balance = 0
                    trades.append({
                        'action': 'BUY',
                        'shares': shares_bought,
                        'price': row['price_today'],
                        'predicted_change': row['predicted_change_pct'],
                        'actual_change': row['actual_change_pct'],
                        'date': row.get('date_predicted_for', row['timestamp'])
                    })
                elif row['predicted_direction'] == 'DOWN' and shares > 0:
                    # Sell all shares
                    balance = shares * row['price_today']
                    trades.append({
                        'action': 'SELL',
                        'shares': shares,
                        'price': row['price_today'],
                        'predicted_change': row['predicted_change_pct'],
                        'actual_change': row['actual_change_pct'],
                        'date': row.get('date_predicted_for', row['timestamp'])
                    })
                    shares = 0
        
        # Final balance calculation
        final_value = balance + (shares * df.iloc[-1]['actual_price'])
        total_return = ((final_value - initial_balance) / initial_balance) * 100
        
        # Calculate buy-and-hold comparison
        initial_price = df.iloc[0]['price_today']
        final_price = df.iloc[-1]['actual_price']
        buy_hold_return = ((final_price - initial_price) / initial_price) * 100
        
        # Calculate additional metrics
        profitable_trades = [t for t in trades if self._is_trade_profitable(t, df)]
        win_rate = (len(profitable_trades) / len(trades)) * 100 if len(trades) > 0 else 0
        
        simulation_result = {
            "ticker": ticker_symbol,
            "model": self.model,
            "timestamp": datetime.now().isoformat(),
            "initial_balance": initial_balance,
            "final_value": round(final_value, 2),
            "total_return_pct": round(total_return, 2),
            "buy_hold_return_pct": round(buy_hold_return, 2),
            "outperformed_buy_hold": total_return > buy_hold_return,
            "outperformance": round(total_return - buy_hold_return, 2),
            "number_of_trades": len(trades),
            "win_rate": round(win_rate, 2),
            "price_threshold_used": price_threshold,
            "simulation_period_days": len(df),
            "avg_trade_return": round(np.mean([t.get('actual_change', 0) for t in trades]), 2) if trades else 0,
            "trades": trades[-10:]  # Show last 10 trades
        }
        
        # Save simulation result
        self._save_simulation_result(ticker_symbol, simulation_result)
        
        return simulation_result

    def _save_simulation_result(self, ticker_symbol, result):
        """Save trading simulation result to ticker-specific file"""
        safe_model_name = self.model.replace("/", "_").replace("\\", "_").replace(":", "_")
        simulation_file = os.path.join(self.results_dir, f"{ticker_symbol}_{safe_model_name}_simulations.jsonl")
        
        with open(simulation_file, "a") as f:
            f.write(json.dumps(result) + "\n")

    def _is_trade_profitable(self, trade, df):
        """Check if a trade was profitable based on actual price movements"""
        trade_date = trade.get('date')
        if not trade_date:
            return False
            
        # Find the corresponding row in predictions data
        matching_rows = df[df.get('date_predicted_for', df['timestamp']) == trade_date]
        if len(matching_rows) == 0:
            return False
            
        actual_change = matching_rows.iloc[0]['actual_change_pct']
        predicted_direction = trade['action']
        
        if predicted_direction == 'BUY':
            return actual_change > 0
        else:  # SELL
            return actual_change < 0

    def analyze_trading_performance(self, ticker_symbol, model_name=None):
        """Analyze historical trading simulation performance for a ticker"""
        if model_name:
            safe_model_name = model_name.replace("/", "_").replace("\\", "_").replace(":", "_")
            simulation_file = os.path.join(self.results_dir, f"{ticker_symbol}_{safe_model_name}_simulations.jsonl")
        else:
            safe_model_name = self.model.replace("/", "_").replace("\\", "_").replace(":", "_")
            simulation_file = os.path.join(self.results_dir, f"{ticker_symbol}_{safe_model_name}_simulations.jsonl")
        
        if not os.path.exists(simulation_file):
            return f"No trading simulations found for {ticker_symbol}"
        
        simulations = []
        with open(simulation_file, "r") as f:
            for line in f:
                simulations.append(json.loads(line.strip()))
        
        if not simulations:
            return f"No trading simulations found for {ticker_symbol}"
        
        df = pd.DataFrame(simulations)
        
        # Calculate performance metrics across all simulations
        avg_return = df['total_return_pct'].mean()
        avg_buy_hold = df['buy_hold_return_pct'].mean()
        avg_outperformance = df['outperformance'].mean()
        win_rate_vs_buy_hold = (df['outperformed_buy_hold'].sum() / len(df)) * 100
        best_simulation = df.loc[df['total_return_pct'].idxmax()]
        worst_simulation = df.loc[df['total_return_pct'].idxmin()]
        
        return {
            "ticker": ticker_symbol,
            "total_simulations": len(df),
            "avg_strategy_return": round(avg_return, 2),
            "avg_buy_hold_return": round(avg_buy_hold, 2),
            "avg_outperformance": round(avg_outperformance, 2),
            "win_rate_vs_buy_hold": round(win_rate_vs_buy_hold, 2),
            "best_simulation": {
                "return": round(best_simulation['total_return_pct'], 2),
                "threshold": best_simulation['price_threshold_used'],
                "trades": best_simulation['number_of_trades']
            },
            "worst_simulation": {
                "return": round(worst_simulation['total_return_pct'], 2),
                "threshold": worst_simulation['price_threshold_used'],
                "trades": worst_simulation['number_of_trades']
            },
            "avg_trades_per_simulation": round(df['number_of_trades'].mean(), 1),
            "avg_win_rate": round(df['win_rate'].mean(), 2)
        }

    def compare_trading_strategies(self, ticker_symbol, thresholds=[0.01, 0.02, 0.05], initial_balance=10000):
        """Compare trading performance across different price thresholds"""
        results = []
        
        for threshold in thresholds:
            result = self.get_trading_simulation(ticker_symbol, initial_balance, threshold)
            if isinstance(result, dict):
                results.append({
                    "threshold": threshold,
                    "return": result['total_return_pct'],
                    "outperformance": result['outperformance'],
                    "trades": result['number_of_trades'],
                    "win_rate": result['win_rate']
                })
        
        if not results:
            return f"No valid simulations for {ticker_symbol}"
        
        # Find best performing threshold
        best_strategy = max(results, key=lambda x: x['return'])
        
        comparison = {
            "ticker": ticker_symbol,
            "strategies_compared": len(results),
            "best_threshold": best_strategy['threshold'],
            "best_return": best_strategy['return'],
            "best_outperformance": best_strategy['outperformance'],
            "all_results": results,
            "recommendation": f"Use {best_strategy['threshold']} threshold for best performance ({best_strategy['return']:.2f}% return)"
        }
        
        # Save comparison results
        safe_model_name = self.model.replace("/", "_").replace("\\", "_").replace(":", "_")
        comparison_file = os.path.join(self.results_dir, f"{ticker_symbol}_{safe_model_name}_strategy_comparison.json")
        
        with open(comparison_file, "w") as f:
            json.dump(comparison, f, indent=2)
        
        return comparison
    
    def get_prediction_quality_report(self, ticker_symbol, model_name):
        """Generate a comprehensive quality report for predictions"""
        analysis = self.analyze_ticker_performance(ticker_symbol, model_name)
        
        if isinstance(analysis, str):
            return analysis
        
        # Determine quality rating based on multiple factors
        direction_acc = analysis['direction_accuracy']
        mape = analysis['error_metrics'].get('mape', 100)
        mae = analysis['error_metrics'].get('mae', float('inf'))
        
        # Quality scoring (0-100)
        direction_score = direction_acc
        mape_score = max(0, 100 - mape)  # Lower MAPE is better
        mae_score = max(0, 100 - (mae / analysis['avg_actual_price'] * 100))  # MAE as % of avg price
        
        overall_score = (direction_score + mape_score + mae_score) / 3
        
        if overall_score >= 80:
            quality_rating = "Excellent"
        elif overall_score >= 65:
            quality_rating = "Good"
        elif overall_score >= 50:
            quality_rating = "Fair"
        else:
            quality_rating = "Poor"
        
        return {
            "ticker": ticker_symbol,
            "quality_rating": quality_rating,
            "overall_score": round(overall_score, 2),
            "direction_accuracy": analysis['direction_accuracy'],
            "price_accuracy_metrics": analysis['error_metrics'],
            "recommendations": self._generate_recommendations(analysis),
            "summary": f"Model shows {quality_rating.lower()} performance for {ticker_symbol} with {direction_acc:.1f}% direction accuracy and {mape:.2f}% MAPE"
        }
    
    def _generate_recommendations(self, analysis):
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        if analysis['direction_accuracy'] < 55:
            recommendations.append("Consider collecting more diverse features or trying different model architectures")
        
        if analysis['error_metrics'].get('mape', 0) > 10:
            recommendations.append("High price prediction error - consider ensemble methods or feature engineering")
        
        if analysis['price_prediction_bias'] > 5:
            recommendations.append("Model consistently over-predicts prices - consider bias correction")
        elif analysis['price_prediction_bias'] < -5:
            recommendations.append("Model consistently under-predicts prices - consider bias correction")
        
        if analysis['up_direction_accuracy'] - analysis['down_direction_accuracy'] > 20:
            recommendations.append("Model is biased toward one direction - balance training data")
        
        return recommendations

# Example usage:
"""
# Initialize the predictor
api_key = "your_openrouter_api_key"
predictor = HFLLMPredictionMaker(api_key)

# Make price prediction
predicted_price, actual_price, result = predictor.make_prediction(your_dataframe, "AAPL")
print(f"Predicted: ${predicted_price:.2f}, Actual: ${actual_price:.2f}")
print(f"Error: ${result['absolute_error']:.2f} ({result['percentage_error']:.2f}%)")

# Comprehensive analysis
analysis = predictor.analyze_ticker_performance("AAPL")
print(f"Direction Accuracy: {analysis['direction_accuracy']:.2f}%")
print(f"MAPE: {analysis['error_metrics']['mape']:.2f}%")
print(f"RMSE: ${analysis['error_metrics']['rmse']:.2f}")

# Quality report
quality_report = predictor.get_prediction_quality_report("AAPL")
print(f"Quality Rating: {quality_report['quality_rating']}")
print(f"Recommendations: {quality_report['recommendations']}")

# Enhanced trading simulation with thresholds
trading_results = predictor.get_trading_simulation("AAPL", price_threshold=0.03)
print(f"Strategy Return: {trading_results['total_return_pct']:.2f}%")
print(f"Buy & Hold Return: {trading_results['buy_hold_return_pct']:.2f}%")
"""