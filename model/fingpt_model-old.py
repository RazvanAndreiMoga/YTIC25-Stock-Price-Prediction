import pandas as pd
from fingpt_forecaster.forecaster import FinGPTForecaster

class FinGPTModel:
    """
    Wrapper for the FinGPT-Forecaster model.
    Initializes the fine-tuned FinGPT model for making stock direction predictions.
    """
    def __init__(self):
        # Initialize the FinGPT-Forecaster model (LoRA-tuned LLaMA2-7B) for forecasting:contentReference[oaicite:6]{index=6}.
        self.forecaster = FinGPTForecaster()  

    def predict_text(self, prompt: str) -> str:
        """
        Generate a prediction string from the FinGPTForecaster given a text prompt.
        The prompt includes recent stock data and asks for next-day direction.
        """
        # Feed the prompt to the FinGPT model and get the response.
        # (Implementation detail: FinGPTForecaster should have a method to generate text. 
        #  Here we assume it returns a string answer to the prompt.)
        response = self.forecaster.predict(prompt)  # hypothetical usage; may vary by API
        return response

class FinGPTPredictionMaker:
    """
    Takes time-series stock data and uses FinGPTModel to predict future directions.
    Formats data into prompts, queries the model, and compares to ground truth.
    """
    def __init__(self, model: FinGPTModel, seq_length: int = 10):
        self.model = model
        self.seq_length = seq_length  # number of days in each prompt window

    def make_predictions(self, data: pd.DataFrame) -> (list, list):
        """
        Generate predictions and ground truth lists from input data.

        Parameters:
            data (pd.DataFrame): Must contain 'Date', 'Price', 'Volume', 'Sentiment', 'Trend'.

        Returns:
            predictions (list): Predicted 'bullish'/'bearish' for each prompt.
            ground_truth (list): Actual 'bullish'/'bearish' labels based on next-day price moves.
        """
        required_cols = {'Date', 'Price', 'Volume', 'Sentiment', 'Trend'}
        if not required_cols.issubset(data.columns):
            raise ValueError(f"Input data must contain columns: {required_cols}")

        predictions = []
        ground_truth = []
        n = len(data)
        
        # Iterate over each sliding window of length seq_length
        for start_idx in range(n - self.seq_length):
            end_idx = start_idx + self.seq_length
            window = data.iloc[start_idx:end_idx]      # Current 10-day window
            next_day = data.iloc[end_idx]             # Day to predict
            last_price = window.iloc[-1]['Price']     # Price on the last day of window
            next_price = next_day['Price']            # Actual next-day price

            # Build the natural-language prompt for these 10 days
            prompt_lines = []
            for _, row in window.iterrows():
                date = row['Date']
                price = row['Price']
                volume = row['Volume']
                sentiment = row['Sentiment']
                trend = row['Trend']
                # Format each day's data into a sentence.
                line = (f"On {date}, the price was ${price:.2f}, volume was {volume:.2f} million, "
                        f"sentiment score was {sentiment:.2f}, and trend score was {trend}.")
                prompt_lines.append(line)
            # Join the lines and append the question about next day’s direction
            prompt_text = " ".join(prompt_lines)
            prompt_text += (" What is the likely direction of the stock on the next day? "
                            "Reply with only 'bullish' or 'bearish'. "
                            "The model is not fine-tuned to give categorical outputs, so please try to infer and respond only with 'bullish' or 'bearish'.")
            
            # Query the FinGPT model
            raw_response = self.model.predict_text(prompt_text)
            response_lower = raw_response.lower()

            # Sanitize to 'bullish' or 'bearish'
            if 'bullish' in response_lower:
                pred = 'bullish'
            elif 'bearish' in response_lower:
                pred = 'bearish'
            else:
                # Default if unclear (could also use heuristic): default to 'bullish'
                pred = 'bullish'

            # Determine ground truth based on actual next-day price
            true_label = 'bullish' if next_price > last_price else 'bearish'

            predictions.append(pred)
            ground_truth.append(true_label)

        return predictions, ground_truth
