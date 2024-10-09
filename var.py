# -*- coding: utf-8 -*-
"""
VAR Model for Time Series Forecasting

This script performs Vector Autoregression (VAR) for time series forecasting
of multiple variables related to industrial wastewater treatment. 
It includes data loading, model fitting, and predictions.
"""

# Import required packages
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.var_model import VAR

# Load the dataset
dataset = pd.read_excel("202106有機自動加藥數據統計(新).xlsx", header=0, index_col=0)
# Select relevant columns
dataset = dataset[['銅濃(mg/L)', '銅在線濃度(mg/L)', 'OOO', 'pH', 'ph槽ORP', 'Chemical_A_ORP', 'Chemical_A', 'Chemical_B']]

# Display the first few rows of the dataset
print(dataset.head())

# Check data types for each column
print("\nData Types:\n", dataset.dtypes)

# Split the data into training and validation sets (80% train, 20% validation)
train_size = int(0.8 * len(dataset))
train = dataset[:train_size]
valid = dataset[train_size:]

# Fit the VAR model on the training set
model = VAR(endog=train)
model_fit = model.fit()

# Summary of the model
print("\nModel Summary:\n")
print(model_fit.summary())

# Make predictions on the validation set
predictions = model_fit.forecast(y=model_fit.y, steps=len(valid))

# Convert predictions to DataFrame for better readability
predicted_df = pd.DataFrame(predictions, index=valid.index, columns=valid.columns)

# Plotting the actual vs predicted values for the validation set
plt.figure(figsize=(12, 6))
for column in valid.columns:
    plt.plot(valid.index, valid[column], label=f'Actual {column}')
    plt.plot(predicted_df.index, predicted_df[column], linestyle='--', label=f'Predicted {column}')
plt.title('Actual vs Predicted Values for Validation Set')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid()
plt.tight_layout()
plt.show()

# Make final predictions on the entire dataset
final_model = VAR(endog=dataset)
final_model_fit = final_model.fit()
future_steps = 10  # Number of future steps to predict
final_predictions = final_model_fit.forecast(y=final_model_fit.y, steps=future_steps)

# Convert the final predictions to a DataFrame
future_index = pd.date_range(start=dataset.index[-1], periods=future_steps + 1, freq='D')[1:]
final_predicted_df = pd.DataFrame(final_predictions, index=future_index, columns=dataset.columns)

# Display the final forecasted values
print("\nFinal Forecasted Values:\n")
print(final_predicted_df)

# Plotting the future predictions
plt.figure(figsize=(12, 6))
for column in dataset.columns:
    plt.plot(dataset.index, dataset[column], label=f'Historical {column}')
    plt.plot(final_predicted_df.index, final_predicted_df[column], linestyle='--', label=f'Forecasted {column}')
plt.title('Future Forecasted Values')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid()
plt.tight_layout()
plt.show()
