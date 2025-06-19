# YTIC25 Stock Price Prediction: Classical vs Modern Approaches

This research compares classical machine learning models with large language models (LLMs) for stock trend forecasting by incorporating sentiment analysis from financial news and Google Trends data. The primary objective is to evaluate how traditional ML approaches and LLMs perform when combining financial metrics with sentiment analysis and Google Trends data.

# Chosen Models & Stock Tickers: 
1 LSTM, 4 LLMs, 2 Prompts, 4 Stock Tickers for a total of 36 results to analyze. Chose cost effective models of big companies for ease of use and in order to test for bias.

# Summary of Findings:
- The LLMs given with a complex prompt provided the best results, even though they were given less data than the LSTM due to input context restraints;
- The integration of sentiment analysis and Google Trends data proved valuable in enhancing prediction accuracy, especially when given to LLMs that are trained to process human speech.
- The LLMs showed no bias towards their parent company.

# Future Work: 
- Integrate more diverse data sources, such as social media sentiment, economic indicators, and geopolitical events, to further improve the model’s accuracy.
- Implement real-time prediction systems that continuously update predictions based on incoming data.
 
The first iteration should include:
 
Domain: Define the entities for the conceptual model (2-3 entities).
Three functionalities: Search, filter, and sort.
Graphical User Interface (GUI): Implement a GUI for user interaction.
