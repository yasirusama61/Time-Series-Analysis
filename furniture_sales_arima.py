import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from pandas import DataFrame
from statsmodels.tsa.arima.model import ARIMA
from matplotlib import pyplot

warnings.filterwarnings("ignore")

# Set matplotlib parameters
plt.style.use('fivethirtyeight')
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['text.color'] = 'k'

# Read the data
df = pd.read_excel("Superstore.xls")
furniture = df.loc[df['Category'] == 'Furniture']

# Remove unnecessary columns
cols = ['Row ID', 'Order ID', 'Ship Date', 'Ship Mode', 'Customer ID', 'Customer Name', 
        'Segment', 'Country', 'City', 'State', 'Postal Code', 'Region', 'Product ID', 
        'Category', 'Sub-Category', 'Product Name', 'Quantity', 'Discount', 'Profit']
furniture.drop(cols, axis=1, inplace=True)

# Sort data by 'Order Date' and remove missing values
furniture = furniture.sort_values('Order Date')
furniture.isnull().sum()

# Set 'Order Date' as index
furniture = furniture.set_index('Order Date')

# Resample monthly sales data using the average value for each month
y = furniture['Sales'].resample('MS').mean()

# Plot Furniture Sales Time Series
y.plot(figsize=(15, 6))
plt.title('Furniture Sales Over Time')
plt.show()

# Decompose the time series data
from pylab import rcParams
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(y, model='additive')
decomposition.plot()
plt.show()

# Fit an ARIMA model (Autoregressive Integrated Moving Average)
model = ARIMA(y, order=(1, 1, 0))  
model_fit = model.fit()

# Display model summary
print(model_fit.summary())

# Plot residuals
residuals = DataFrame(model_fit.resid)
residuals.plot()
plt.title('Residuals')
pyplot.show()

# Plot density of residuals
residuals.plot(kind='kde')
plt.title('Density of Residuals')
pyplot.show()

# Display summary stats of residuals
print(residuals.describe())

# Model diagnostics plot
model_fit.plot_diagnostics(figsize=(16, 8))
plt.show()

# Calculate Mean Squared Error (MSE)
predictions = model_fit.fittedvalues
y_bar = predictions
summation = np.sum((y - y_bar) ** 2)
MSE = summation / len(y)
print("The Mean Square Error is:", MSE)

# Function to evaluate ARIMA parameters
def evaluate_arima_model(X, arima_order):
    # Prepare training dataset
    train_size = int(len(X) * 0.66)
    train, test = X[0:train_size], X[train_size:]
    model = ARIMA(train, order=arima_order)
    model_fit = model.fit()
    predictions = model_fit.forecast(steps=len(test))
    error = mean_squared_error(test, predictions)
    return error

# Function to evaluate combinations of p, d, and q for ARIMA
def evaluate_models(X, p_values, d_values, q_values):
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    mse = evaluate_arima_model(X, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print(f'ARIMA{order} MSE={mse:.3f}')
                except:
                    continue
    print(f'Best ARIMA{best_cfg} MSE={best_score:.3f}')

# Set values for p, d, q
p_values = [0, 1, 2, 4, 6, 8, 10]
d_values = range(0, 2)
q_values = range(0, 2)

# Evaluate ARIMA models with different parameter combinations
evaluate_models(furniture['Sales'].values, p_values, d_values, q_values)
