import pandas as pd

def merge_company_data(company):
    # File paths
    headlines_file = f"{company}_processed_headlines.csv"
    market_data_file = f"merged_data_{company}.csv"
    output_file = f"final_merged_{company}.csv"
    
    # Load files
    headlines_df = pd.read_csv(headlines_file)
    market_df = pd.read_csv(market_data_file)

    # Ensure date columns are datetime type
    headlines_df['date'] = pd.to_datetime(headlines_df['date'])
    market_df['Date'] = pd.to_datetime(market_df['Date'])

    # headlines_df already has headlines grouped as list per date in 'headline' column,
    # but since it's read from CSV, it's likely a string representation of list.
    # Convert string representation of list back to actual list:
    import ast
    headlines_df['headline'] = headlines_df['headline'].apply(ast.literal_eval)

    # Filter market data for the selected dates and select relevant columns
    market_filtered = market_df[market_df['Date'].isin(headlines_df['date'])]
    market_filtered = market_filtered[['Date', 'Price', 'Volume', 'Trend']]

    # Merge on date
    merged = pd.merge(headlines_df, market_filtered, left_on='date', right_on='Date')
    merged = merged.drop(columns='Date')

    # Rename date to Date for clarity
    merged = merged.rename(columns={'date': 'Date'})

    # Save result
    merged.to_csv(output_file, index=False)
    print(f"Saved merged file: {output_file}")

# Run for each company
for company in ['AMZN']:
    merge_company_data(company)
