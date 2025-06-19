import pandas as pd

# === Configuration ===
INPUT_FILE = "input/AMZN/AMZN_data.csv"
OUTPUT_FILE = "AMZN_processed_headlines.csv"
KEYWORD = "AMZN"
START_DATE = "2024-03-04"
END_DATE = "2024-03-15"

# === Load and preprocess ===
df = pd.read_csv(INPUT_FILE)
df['datetime'] = pd.to_datetime(df['datetime'], unit='s')
df['date'] = df['datetime'].dt.date

# Filter date range
df = df[(df['date'] >= pd.to_datetime(START_DATE).date()) & (df['date'] <= pd.to_datetime(END_DATE).date())]

# Filter by keyword in headline
df = df[df['headline'].str.contains(KEYWORD, case=False, na=False)]

# Group by date and sample up to 3 headlines per day
sampled = df.groupby('date', group_keys=False).apply(lambda x: x.sample(n=min(3, len(x)), random_state=42))

# Combine headlines into list per date
grouped = sampled.groupby('date')['headline'].apply(list).reset_index()

# Save final result
grouped.to_csv(OUTPUT_FILE, index=False)
print(f"Saved {len(grouped)} rows to '{OUTPUT_FILE}'")
