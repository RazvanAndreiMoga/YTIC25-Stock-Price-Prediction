import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DataPreprocessor:
    def __init__(self, seq_length=10):
        self.seq_length = seq_length
        self.scaler = MinMaxScaler()

    def normalize_data(self, data):
        scaled_data = self.scaler.fit_transform(data)
        return scaled_data

    def create_sequences(self, data):
        X = []
        y = []
        for i in range(self.seq_length, len(data)):
            X.append(data[i-self.seq_length:i])
            y.append(data[i, 0])  # Predicting the 'Close' price
        return np.array(X), np.array(y)

    def split_data(self, data, train_size_ratio=0.8):
        train_size = int(len(data) * train_size_ratio)
        train_data = data[:train_size]
        test_data = data[train_size:]
        return train_data, test_data

    def preprocess(self, merged_df):
        # Normalize the data
        scaled_data = self.normalize_data(merged_df)
        
        # Split the data into train and test sets
        train_data, test_data = self.split_data(scaled_data)
        
        # Create sequences
        X_train, y_train = self.create_sequences(train_data)
        X_test, y_test = self.create_sequences(test_data)
        
        return X_train, y_train, X_test, y_test
