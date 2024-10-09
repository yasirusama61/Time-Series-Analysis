#!/usr/bin/env python
# coding: utf-8

# This script trains an LSTM model on the preprocessed data.

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pandas as pd

# Load preprocessed data
def load_data(file_path):
    data = pd.read_csv(file_path)
    values = data.values
    n_train_samples = int(len(values) * 0.75)
    train = values[:n_train_samples, :]
    test = values[n_train_samples:, :]
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    return train_X, train_y, test_X, test_y

# Build and train the LSTM model
def build_and_train_lstm(train_X, train_y, test_X, test_y, epochs=100, batch_size=32):
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse')
    
    # Train the model
    history = model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, validation_data=(test_X, test_y), verbose=2, shuffle=False)
    
    # Plot the loss history
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('LSTM Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.show()
    return model

if __name__ == "__main__":
    # Example usage
    train_X, train_y, test_X, test_y = load_data("data/processed/processed_data.csv")
    model = build_and_train_lstm(train_X, train_y, test_X, test_y)
    model.save("models/lstm_model.h5")
    print("Model training completed.")
