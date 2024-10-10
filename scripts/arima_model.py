import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys

warnings.filterwarnings("ignore")


# Function to evaluate an ARIMA model for a given order
def evaluate_arima_model(X, arima_order):
    train_size = int(len(X) * 0.75)
    train, test = X[0:train_size], X[train_size:]
    history = list(train)
    predictions = []

    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])

    # Calculate out of sample error
    rmse = sqrt(mean_squared_error(test, predictions))
    return rmse


# Function to evaluate different combinations of ARIMA parameters
def evaluate_arima_hyperparameters(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None

    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    rmse = evaluate_arima_model(dataset, order)
                    if rmse < best_score:
                        best_score, best_cfg = rmse, order
                    print(f'ARIMA{order} RMSE={rmse:.3f}')
                except Exception as e:
                    print(f"Error evaluating ARIMA{order}: {e}")
                    continue

    print(f'Best ARIMA{best_cfg} RMSE={best_score:.3f}')


# PSO for ARIMA hyperparameter tuning
def pso_arima_tuning(X, n_particles=10, iterations=100, inertia=1.0):
    # Hyperparameters search space
    max_p, min_p = 10, 0
    max_d, min_d = 2, 0
    max_q, min_q = 2, 0

    # Initialize particles' positions
    positions = np.random.rand(n_particles, 3) * [(max_p - min_p), (max_d - min_d), (max_q - min_q)] + [min_p, min_d, min_q]
    velocities = np.zeros((n_particles, 3))
    personal_best_positions = positions.copy()
    personal_best_values = np.full(n_particles, sys.maxsize)
    global_best_position = np.zeros(3)
    global_best_value = sys.maxsize

    best_iter_values = []

    for i in range(iterations):
        for j in range(n_particles):
            arima_order = tuple(map(int, positions[j]))

            try:
                mse = evaluate_arima_model(X, arima_order)
            except Exception as e:
                print(f"Error during evaluation for particle {j}: {e}")
                continue

            # Update personal best
            if mse < personal_best_values[j]:
                personal_best_values[j] = mse
                personal_best_positions[j] = positions[j].copy()

            # Update global best
            if mse < global_best_value:
                global_best_value = mse
                global_best_position = positions[j].copy()

            # Update velocity and position
            rand1, rand2 = np.random.random(), np.random.random()
            velocities[j] = inertia * velocities[j] + 2 * rand1 * (personal_best_positions[j] - positions[j]) + 2 * rand2 * (global_best_position - positions[j])
            positions[j] += velocities[j]

            # Boundary conditions
            positions[j] = np.clip(positions[j], [min_p, min_d, min_q], [max_p, max_d, max_q])

        best_iter_values.append(global_best_value)
        print(f'Iteration {i + 1}/{iterations}, Best RMSE: {global_best_value:.3f}')

    print(f'Global Best ARIMA Order: {tuple(map(int, global_best_position))}, RMSE: {global_best_value:.3f}')
    plot_pso_iterations(best_iter_values)
    return global_best_position


# Plotting function for PSO iterations
def plot_pso_iterations(iter_values):
    plt.plot(iter_values, label='Best RMSE')
    plt.xlabel('Iteration')
    plt.ylabel('RMSE')
    plt.title('PSO Optimization Progress')
    plt.legend()
    plt.grid(True)
    plt.show()


# Main script execution
if __name__ == "__main__":
    # Load dataset
    df = pd.read_excel("202106有機自動加藥數據統計(新)_1.xlsx", header=0, index_col=0)
    df = df[['Copper concentration (mg/L)']]

    # Evaluate ARIMA models using grid search
    p_values = [0, 1, 2, 4, 6, 8, 10]
    d_values = range(0, 3)
    q_values = range(0, 3)
    evaluate_arima_hyperparameters(df.values, p_values, d_values, q_values)

    # Optimize ARIMA parameters using PSO
    pso_arima_tuning(df.values)
