import numpy as np
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dropout, Dense # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # type: ignore
from sklearn.model_selection import KFold

class LSTMModel:
    def __init__(self, ticker, input_shape, n_splits=5, patience=10, epochs=50, batch_size=32):
        self.ticker = ticker
        self.input_shape = input_shape
        self.n_splits = n_splits
        self.patience = patience
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = self.create_model()

    def create_model(self):
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=self.input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def train_model(self, X_train, y_train, X_test, y_test):
        # Set up K-Fold Cross-Validation
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        val_scores = []

        for train_index, val_index in kf.split(X_train):
            X_train_cv, X_val_cv = X_train[train_index], X_train[val_index]
            y_train_cv, y_val_cv = y_train[train_index], y_train[val_index]

            model = self.create_model()

            # Define EarlyStopping and ModelCheckpoint callbacks
            early_stopping = EarlyStopping(monitor='val_loss', patience=self.patience, verbose=1)
            checkpoint = ModelCheckpoint(f'checkpoint/stock_price_model_{self.ticker}.keras', monitor='val_loss', save_best_only=True, verbose=1)

            # Train the model
            model.fit(X_train_cv, y_train_cv, epochs=self.epochs, batch_size=self.batch_size, validation_data=(X_val_cv, y_val_cv), callbacks=[early_stopping, checkpoint])

            # Load the best model and evaluate on validation set
            model.load_weights(f'checkpoint/stock_price_model_{self.ticker}.keras')
            val_loss = model.evaluate(X_val_cv, y_val_cv, verbose=0)
            val_scores.append(val_loss)

        # Average validation loss
        average_val_loss = np.mean(val_scores)
        print(f'Average validation loss: {average_val_loss:.4f}')

        # # Train the final model on the entire training data
        # model = self.create_model()
        # model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, validation_data=(X_test, y_test), callbacks=[early_stopping, checkpoint])

        # Load the best model
        model.load_weights(f'checkpoint/stock_price_model_{self.ticker}.keras')
        
        return model, average_val_loss
