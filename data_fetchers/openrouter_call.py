import requests
import json
import pandas as pd

# === Configuration ===
api_key=""
OPENROUTER_API_KEY = api_key  # Replace this
MODEL = "deepseek/deepseek-r1-0528"  # Change model name as needed (e.g. "anthropic/claude-3-opus")
YOUR_SITE_URL = "https://your-site.com"  # Optional
YOUR_SITE_NAME = "Your App Name"         # Optional
TICKER = "AMZN"

# === Load dataset ===
data_file = f"final_merged_{TICKER}.csv"
df = pd.read_csv(data_file)

print(f"{df}")
# === Construct prompt ===
prompt = (
    f"""
<role>
You are an expert financial analyst with 15+ years of experience in quantitative analysis, technical analysis, and market sentiment evaluation.
</role>

<task>
Your task is to predict stock prices for the next 3 trading days based on comprehensive historical data analysis.
</task>

<context>
You will receive 10 days of historical stock data with the following structure:
- Date: Trading date
- Headline: Key news headline for that date
- Price: Stock price in USD
- Volume: Trading volume
- Trend: Google Trends score (0-100)
</context>

<analysis_framework>
1. **Technical Analysis**: Examine price patterns, volume trends, and momentum indicators
2. **Sentiment Analysis**: Evaluate news headlines for market sentiment impact
3. **Trend Analysis**: Assess Google Trends data for public interest correlation
4. **Pattern Recognition**: Identify recurring patterns in the 10-day historical window
</analysis_framework>

<instructions>
- Calculate price momentum and volatility from the historical data
- Identify any correlation between news sentiment and price movements
- Assess volume patterns for institutional vs retail activity indicators
- Evaluate Google Trends correlation with price action
- Consider any emerging patterns or breakout/breakdown signals
- Account for weekend gaps and trading day adjustments
</instructions>

<output_requirements>
You MUST respond with ONLY the 3 predicted prices in USD. NO explanations, NO reasoning, NO additional text whatsoever.

Format your response as exactly 3 numbers with dollar signs, separated by commas:
$XX.XX, $XX.XX, $XX.XX

IMPORTANT: Do not include any other text, analysis, or commentary. Only the 3 prices.
</output_requirements>

<dataset>
{df}
</dataset>
"""
)

# === Make OpenRouter API call ===
response = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": YOUR_SITE_URL,
        "X-Title": YOUR_SITE_NAME,
    },
    data=json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    })
)

# === Handle response ===
if response.status_code == 200:
    result = response.json()
    prediction = result['choices'][0]['message']['content']
    print(f"Predicted Prices (next 3 days) for {TICKER} using {MODEL}:\n", prediction)

    # Save response to file
    model_short = MODEL.split("/")[-1].replace("-", "_")
    output_filename = f"prediction_{TICKER}_{model_short}.txt"
    with open(output_filename, "w") as f:
        f.write(prediction)

    print(f"\nSaved prediction to: {output_filename}")
else:
    print("Error:", response.status_code)
    print(response.text)
