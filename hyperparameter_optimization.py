#!/usr/bin/env python
# coding: utf-8
"""
PSO-LSTM: LSTM model optimization using Particle Swarm Optimization (PSO)
Author: Usama Yasir Khan
Description: This script demonstrates the use of PSO to optimize LSTM model parameters for time-series prediction tasks.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pyswarms as ps
from pyswarms.utils.plotters import plot_cost_history

# Plot settings
style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']  # For Chinese character support
plt.rcParams['axes.unicode_minus'] = False

# TensorFlow GPU settings (if applicable)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

# Load dataset
df = pd.read_excel("202106有機自動加藥數據統計(新).xlsx", header=0, index_col=0)
df = df[['Heavy_metal_concentration', 'OOO', 'pH', 'ph槽ORP', 'NaS_ORP', 'Chemical_A', 'Chemical_B']]
print(df.head())

# Splitting features and target
Y = df.iloc[:, 0]  # Target variable
X = df[['OOO', 'pH', 'ph槽ORP', 'Chemical_A_ORP', 'Chemical_A', 'Chemical_B']]  # Feature variables

# Train-test split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.25, random_state=42)

# Standardize the data
ss_x = StandardScaler()
Xtrain = ss_x.fit_transform(Xtrain)
Xtest = ss_x.transform(Xtest)

ss_y = StandardScaler()
Ytrain = ss_y.fit_transform(Ytrain.values.reshape(-1, 1))
Ytest = ss_y.transform(Ytest.values.reshape(-1, 1))

# Reshape input data for LSTM [samples, timesteps, features]
timesteps = 6  # Example: we can adjust this based on our data
features = 1    # Since each feature is one-dimensional
Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], timesteps, features))
Xtest = np.reshape(Xtest, (Xtest.shape[0], timesteps, features))

# Define LSTM model
def create_lstm_model():
    model = Sequential()
    model.add(LSTM(128, activation='relu', input_shape=(timesteps, features)))
    model.add(Dense(1))
    return model

# Compile the model using SGD optimizer
model = create_lstm_model()
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)

# Train the model
t0 = time.time()
result_sgd = model.fit(Xtrain, Ytrain, batch_size=4, epochs=50, verbose=0)
t1 = time.time()

# Print training results
print(f"LSTM Model Training Time: {t1 - t0:.2f} seconds")
print(f"Final Training Loss: {result_sgd.history['loss'][-1]:.6f}")

# Store model weights for PSO optimization
model_weights = [model.layers[0].get_weights(), model.layers[1].get_weights()]
shape = [
    model_weights[0][0].shape,  # LSTM weights shape
    model_weights[0][1].shape,  # LSTM bias shape
    model_weights[0][2].shape,  # LSTM recurrent weights shape
    model_weights[1][0].shape,  # Dense weights shape
    model_weights[1][1].shape   # Dense bias shape
]

# Function to calculate the error for PSO optimization
def calculate_error(vector_x):
    # Rebuild the model weights from PSO particles
    idx = 0
    model_weights[0][0] = vector_x[idx:idx + np.prod(shape[0])].reshape(shape[0])
    idx += np.prod(shape[0])
    model_weights[0][1] = vector_x[idx:idx + np.prod(shape[1])].reshape(shape[1])
    idx += np.prod(shape[1])
    model_weights[0][2] = vector_x[idx:idx + np.prod(shape[2])].reshape(shape[2])
    idx += np.prod(shape[2])
    model_weights[1][0] = vector_x[idx:idx + np.prod(shape[3])].reshape(shape[3])
    idx += np.prod(shape[3])
    model_weights[1][1] = vector_x[idx:idx + np.prod(shape[4])].reshape(shape[4])
    
    # Set new weights
    model.layers[0].set_weights([model_weights[0][0], model_weights[0][1], model_weights[0][2]])
    model.layers[1].set_weights([model_weights[1][0], model_weights[1][1]])
    
    # Predict and calculate error
    predictions = model.predict(Xtest)
    error = mean_squared_error(Ytest, predictions)
    return error

# Define PSO function
def swarm_func(x):
    n_particles = x.shape[0]
    errors = np.array([calculate_error(x[i]) for i in range(n_particles)])
    return errors

# PSO setup
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
optimizer = ps.single.GlobalBestPSO(n_particles=80, dimensions=np.sum([np.prod(s) for s in shape]), options=options)

# Perform optimization
t2 = time.time()
best_cost, best_pos = optimizer.optimize(swarm_func, iters=120)
t3 = time.time()

# Print PSO optimization results
print(f"PSO Optimization Time: {t3 - t2:.2f} seconds")
print(f"Best PSO Error: {best_cost:.6f}")

# Plot the cost history
plot_cost_history(optimizer.cost_history)
plt.show()
