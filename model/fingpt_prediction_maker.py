class FinGPTPredictionMaker:
    def __init__(self, model, seq_length=10):
        self.model = model
        self.seq_length = seq_length

    def make_predictions(self, merged_df):
        # Take last N rows
        recent_data = merged_df.tail(self.seq_length)

        # Build prompt
        prompt = "You are a financial assistant. Given recent stock data with Date, Price, Volume, Sentiment, and Trend, predict whether the market will be bullish or bearish next.\n"
        prompt += "The model is not fine-tuned for exact binary classification, so answer based on general financial intuition from the data.\n\n"
        prompt += "Date,Price,Volume,Sentiment,Trend\n"

        for _, row in recent_data.iterrows():
            prompt += f"{row['Date']},{row['Price']},{row['Volume']},{row['Sentiment']},{row['Trend']}\n"

        prompt += "\nAnswer with one word: 'Bullish' or 'Bearish'.\n"

        # Generate response
        response = self.model.generate(prompt, max_new_tokens=10)

        # Extract label from response
        label = "Bullish" if "bullish" in response.lower() else "Bearish"
        return [label], ["Unknown (no true label)"]
