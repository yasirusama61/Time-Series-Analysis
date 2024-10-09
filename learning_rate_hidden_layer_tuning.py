#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from math import sqrt
from numpy import concatenate
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from pandas import DataFrame, concat

# Function to convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    agg = concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# Function to scale the dataset
def scale(train, test):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(train)
    train_scaled = scaler.transform(train)
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled

# Function to invert scaling for forecast
def invert_scale(scaler, X, yhat):
    new_row = [x for x in X] + [yhat]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]

# Function to fit LSTM network
def fit_lstm(train, batch_size, nb_epoch, neurons, learning_rate):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(1))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='mae', optimizer=optimizer)
    model.fit(X, y, epochs=nb_epoch, batch_size=batch_size, verbose=2, shuffle=False)
    return model

# Load dataset
def load_dataset(file_path):
    dataset = pd.read_excel(file_path, header=0, index_col=0)
    dataset = dataset[['Chemical A (mg/L)', 'Chemical B (mg/L)', 'OOO', 'pH', 'ph槽ORP', 'NaS_ORP', 'NaS', 'FeSO4']]
    return dataset

# Prepare data
def prepare_data(dataset, train_split=0.7):
    values = dataset.values
    encoder = LabelEncoder()
    values[:, 4] = encoder.fit_transform(values[:, 4])
    values = values.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    reframed = series_to_supervised(scaled, 1, 1)
    values = reframed.values
    n_train = int(len(values) * train_split)
    train = values[:n_train, :]
    test = values[n_train:, :]
    return scaler, train, test

# Main function to perform tuning
def tune_hyperparameters(file_path, learning_rates, neuron_counts, batch_size=64, nb_epoch=50):
    dataset = load_dataset(file_path)
    scaler, train, test = prepare_data(dataset)
    
    results = []
    for lr in learning_rates:
        for neurons in neuron_counts:
            print(f"Training with learning rate={lr}, neurons={neurons}")
            model = fit_lstm(train, batch_size, nb_epoch, neurons, lr)
            
            train_rmse = evaluate_model(model, scaler, train, batch_size)
            test_rmse = evaluate_model(model, scaler, test, batch_size)
            results.append({'Learning Rate': lr, 'Neurons': neurons, 'Train RMSE': train_rmse, 'Test RMSE': test_rmse})
            print(f'Learning Rate={lr}, Neurons={neurons}, Train RMSE={train_rmse:.3f}, Test RMSE={test_rmse:.3f}')
    
    return pd.DataFrame(results)

# Evaluate the model
def evaluate_model(model, scaler, dataset, batch_size):
    X, y = dataset[:, 0:-1], dataset[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    yhat = model.predict(X, batch_size=batch_size)
    predictions = [invert_scale(scaler, X[i], yhat[i, 0]) for i in range(len(yhat))]
    rmse = sqrt(mean_squared_error(y, predictions))
    return rmse

# Plot results
def plot_results(results_df):
    pivot_table = results_df.pivot("Learning Rate", "Neurons", "Test RMSE")
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", fmt=".3f")
    plt.title("RMSE for Different Learning Rates and Neuron Counts")
    plt.show()

if __name__ == "__main__":
    file_path = "202106_example.xlsx"
    learning_rates = [0.001, 0.01, 0.1]
    neuron_counts = [10, 20, 50]
    
    results_df = tune_hyperparameters(file_path, learning_rates, neuron_counts, batch_size=64, nb_epoch=50)
    results_df.to_csv('hyperparameter_tuning_results.csv', index=False)
    plot_results(results_df)
