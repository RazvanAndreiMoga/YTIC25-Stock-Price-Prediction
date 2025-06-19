import pandas as pd
import time

# Load the dataset
df = pd.read_csv("input\GOOG\GOOG_data.csv")

# Convert Unix timestamp to datetime
df['datetime'] = pd.to_datetime(df['datetime'], unit='s')

start_date = pd.to_datetime("2024-03-04")
end_date = pd.to_datetime("2024-03-16")

# Filter rows within the date range
filtered_df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)]

# Select only date and headline columns
result_df = filtered_df[['datetime', 'headline']]

# Optionally, format datetime to date only
result_df['datetime'] = result_df['datetime'].dt.date

# Save to new CSV
result_df.to_csv("GOOG_filtered.csv", index=False)

print(f"Saved {len(result_df)} rows to 'filtered.csv'")

import pandas as pd

# Load the filtered dataset
df = pd.read_csv("GOOG_filtered.csv")

# Filter rows where 'GOOG' appears in the headline (case-insensitive)
GOOG_df = df[df['headline'].str.contains('GOOG', case=False, na=False)]

# Save the result to a new CSV
GOOG_df.to_csv("GOOG_headlines_2024_03_04_to_03_15.csv", index=False)

print(f"Saved {len(GOOG_df)} rows to 'GOOG_headlines_2024_03_04_to_03_15.csv'")

import pandas as pd

# Load the GOOG-related headlines
df = pd.read_csv("GOOG_headlines_2024_03_04_to_03_15.csv")

# Group by date and sample up to 3 rows per day (fewer if less available)
sampled_df = df.groupby('datetime', group_keys=False).apply(lambda x: x.sample(n=min(3, len(x)), random_state=42))

# Save the sampled data to a new CSV
sampled_df.to_csv("GOOG_sampled_headlines_per_day.csv", index=False)

print(f"Saved {len(sampled_df)} rows to 'GOOG_sampled_headlines_per_day.csv'")

