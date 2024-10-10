import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_excel("202106有機自動加藥數據統計(新)_1.xlsx", header=0, index_col=0)

# Selecting relevant features
features = ['Heavy Metal Input concentration (mg/L)', 'Electrical Conductivity', 'pH', 
            'pH_ORP', 'Chemical A_ORP', 'Chemical A', 'Chemical B']
X = df[features]
y = df['Heavy Metal concentration (mg/L)']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training a Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predicting and evaluating the baseline model performance
baseline_predictions = model.predict(X_test)
baseline_mse = mean_squared_error(y_test, baseline_predictions)
print(f'Baseline Mean Squared Error: {baseline_mse:.3f}')

# Function to perform sensitivity analysis
def sensitivity_analysis(model, X_test, feature_name, percentage_change=0.1):
    """
    Evaluate the sensitivity of the model output to changes in a specific input feature.

    Args:
        model: Trained machine learning model.
        X_test: Test dataset for evaluation.
        feature_name: The feature to vary for sensitivity analysis.
        percentage_change: The amount to vary the feature (default is 10%).

    Returns:
        mse_change: Change in MSE when the feature is perturbed.
    """
    X_test_modified = X_test.copy()
    
    # Calculate the amount of change for the specified feature
    change_amount = X_test[feature_name].std() * percentage_change
    
    # Increase the feature values by the change amount
    X_test_modified[feature_name] += change_amount
    predictions_high = model.predict(X_test_modified)
    mse_high = mean_squared_error(y_test, predictions_high)
    
    # Decrease the feature values by the change amount
    X_test_modified[feature_name] -= 2 * change_amount
    predictions_low = model.predict(X_test_modified)
    mse_low = mean_squared_error(y_test, predictions_low)
    
    # Calculate the average change in MSE
    mse_change = (abs(mse_high - baseline_mse) + abs(mse_low - baseline_mse)) / 2
    return mse_change

# Perform sensitivity analysis for each feature
sensitivity_results = {}
for feature in features:
    mse_change = sensitivity_analysis(model, X_test, feature, percentage_change=0.1)
    sensitivity_results[feature] = mse_change
    print(f'Sensitivity for {feature}: {mse_change:.3f}')

# Plotting the sensitivity results
plt.figure(figsize=(12, 8))
plt.barh(list(sensitivity_results.keys()), list(sensitivity_results.values()))
plt.xlabel('Change in Mean Squared Error (MSE)')
plt.title('Sensitivity Analysis: Impact of Features on Heavy Metal Concentration Prediction')
plt.show()
