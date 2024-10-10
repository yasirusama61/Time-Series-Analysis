#!/usr/bin/env python
# coding: utf-8

# This script preprocesses the data for the time series model.

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame, concat

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """Convert series to a supervised learning format."""
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    
    # Input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    
    # Forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    
    # Combine everything
    agg = concat(cols, axis=1)
    agg.columns = names
    
    # Drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# Load the dataset
def load_and_preprocess_data(file_path):
    df = pd.read_excel(file_path, header=0, index_col=0)
    df = df[['Heavy Metal concentration (mg/L)', 'Heavy Metal Input concentration (mg/L)', 'Electrical Conductivity', 
             'pH', 'pH_ORP', 'Chemical A_ORP', 'Chemical A', 'Chemical B']]
    return df

# Normalize and reframe data for supervised learning
def normalize_and_reframe(df):
    values = df.values.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    reframed = series_to_supervised(scaled, 1, 1)
    reframed.drop(reframed.columns[[9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)
    return reframed, scaler

if __name__ == "__main__":
    # Example usage
    file_path = "data/raw/202106_data.xlsx"
    df = load_and_preprocess_data(file_path)
    reframed, scaler = normalize_and_reframe(df)
    reframed.to_csv("data/processed/processed_data.csv", index=False)
    print("Data preprocessing completed.")
