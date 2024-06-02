import os
import sqlite3
import pandas as pd

class SentimentFetcher:
    def __init__(self, database_path='input/financial_data.db'):
        self.database_path = database_path

    def export_table_to_csv(self, table_name, csv_file_path):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

        # Connect to the SQLite database
        conn = sqlite3.connect(self.database_path)
        
        # Query to select all data from the table
        query = f"SELECT * FROM {table_name}"
        
        # Use pandas to read the SQL query into a DataFrame
        df = pd.read_sql_query(query, conn)
        
        # Export the DataFrame to a CSV file
        df.to_csv(csv_file_path, index=False)
        
        # Close the database connection
        conn.close()

    def fetch_sentiment_data(self, ticker):
        table_name = ticker + '_'
        csv_file_path = f'input/{ticker}/{ticker}_data.csv'
        
        # Export table to CSV
        self.export_table_to_csv(table_name, csv_file_path)
        
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file_path)
        return df
