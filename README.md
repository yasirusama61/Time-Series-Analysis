# Time Series Analysis Using Machine Learning

## Project Overview
This repository contains the code and data for a time series forecasting project aimed at predicting heavy metal concentrations in industrial wastewater. The study combines machine learning and statistical modeling techniques, including ARIMA (AutoRegressive Integrated Moving Average) and PSO-LSTM (Particle Swarm Optimization - Long Short-Term Memory), to accurately forecast heavy metal levels based on input features such as pH, chemical dosage, redox potential, and conductivity.

The goal of this project is to develop robust predictive models to optimize wastewater treatment processes, enabling better control of heavy metal removal and improving overall operational efficiency.

Below is a flowchart illustrating the stages of wastewater treatment and measurement:

![Wastewater Treatment Workflow](results/processflow.png)

## Models Implemented

### 1. **ARIMA (AutoRegressive Integrated Moving Average)**
   - Used to model and forecast heavy metal concentration in wastewater treatment.
   - Model parameters were tuned using grid search to find the optimal p, d, q values.
   - The following pseudocode illustrates the algorithmic approach:

### Algorithm: AutoRegressive Integrated Moving Average (ARIMA)

![ARIMA Algorithm](results/ARIMA_algorithm.png)

   - Input: Time series data
   - Output: Predicted future values

   1. Identify the model parameters (p, d, q) using ACF and PACF plots
   2. Perform differencing on the time series to make it stationary (if needed)
   3. Fit the ARIMA model with the identified parameters
   4. Generate forecasts using the fitted model
   5. Evaluate model performance using error metrics like MSE, MAE

### 2. **PSO-LSTM (Particle Swarm Optimization - Long Short-Term Memory)**
   - A hybrid model combining LSTM networks for capturing long-term dependencies in time series data and PSO for optimizing model hyperparameters.
   - Features such as pH, chemical dosage, redox potential, and conductivity were used as input variables to predict future concentrations of heavy metals.
   - The PSO algorithm helps to find optimal hyperparameters, including learning rate, number of neurons, and dropout rate, to improve the LSTM's performance.

## Repository Structure
      time-series-forecasting/
      ├── data/                          # Data folder for raw and preprocessed datasets
      │   ├── heavy_metal_data.xlsx      # Raw dataset of heavy metal concentrations
      │   └── processed_data.csv         # Preprocessed data for modeling
      ├── scripts/                       # Scripts for data preprocessing, modeling, and evaluation
      │   ├── arima_model.py             # Script for ARIMA time series forecasting
      │   ├── pso_lstm_model.py          # Script for PSO-LSTM model training
      │   ├── data_preprocessing.py      # Script for cleaning and preprocessing data
      │   └── hyperparameter_optimization.py # Script for optimizing hyperparameters
      ├── models/                        # Folder for saving trained models
      │   ├── arima_model.pkl            # Saved ARIMA model
      │   └── pso_lstm_model.h5          # Trained PSO-LSTM model
      ├── results/                       # Results like plots, metrics, and logs
      │   ├── arima_forecast_plot.png    # Visualization of ARIMA forecast
      │   ├── pso_lstm_loss_curve.png    # Loss curve for PSO-LSTM training
      │   ├── heavy_metal_prediction.png # Plot comparing predicted vs actual concentrations
      │   └── metrics.txt                # Performance metrics (e.g., MSE, MAE)
      ├── requirements.txt               # Python dependencies
      └── README.md                      # Project overview and instructions


## How to Run the Code
1. Clone the repository:
   - `git clone https://github.com/yasirusama61/Time-Series-Analysis.git`
   - `cd Time-Series-Analysis`
  
2. Install the required dependencies:
   - `pip install -r requirements.txt`

3. Data Preprocessing:
   - `python scripts/data_preprocessing.py`

3. Train the ARIMA model:
   - `python arima_model.py`
   
4.  Train the PSO-LSTM model:
   - `python PSO-LSTM.py`

5. Hyperparameter Optimization
   - `python scripts/hyperparameter_optimization.py`

## Data
The data used in this project was collected from an industrial wastewater treatment facility in collaboration with a company specializing in environmental protection and energy-saving technologies. The dataset includes features such as:

   - Heavy Metal Concentration (mg/L)
   - Heavy Metal Input Concentration (mg/L)
   - Electrical Conductivity
   - pH
   - pH_ORP (Oxidation-Reduction Potential)
   - Chemical Dosage Levels (Chemical A and B)

The original dataset is stored in the `data/heavy_metal_data.xlsx` file, and preprocessed data can be found in `data/processed_data.csv`.

### Optimization Techniques

#### Particle Swarm Optimization (PSO)

PSO is a population-based optimization algorithm inspired by the social behavior of bird flocking or fish schooling. In PSO, each particle represents a potential solution and adjusts its position based on its own experience and that of neighboring particles.

### Algorithm: Particle Swarm Optimization

![PSO Algorithm](results/pso_algorithm.png)



## Evaluation Metrics

The models are evaluated using the following metrics:
   - Mean Squared Error (MSE)
   - Mean Absolute Error (MAE)
   - Mean Squared Logarithmic Error (MSLE)
   - R-Squared (R²)



## Results
Results, including performance metrics and plots, are stored in the `results/` folder:

   - **ARIMA Forecast**: Shows the model's ability to capture trends and seasonality.
   - **PSO-LSTM Performance**: Visualizations such as training loss curves and predicted vs. actual plots.
   - **Evaluation Metrics**: A file summarizing the MSE, MAE, and R² scores for both models.
   - **Sensitivity Analysis**: A plot showing the influential parameters identified through sensitivity analysis.
   - **Batch Size Tuning**: Shows the effect of different batch sizes on RMSE.
   - **Block Number Tuning**: Illustrates the impact of varying the number of blocks in the hidden layer on RMSE.

## Hyperparameter Optimization

![Batch Size Tuning](results/batch_size_rmse.png)
![Block Number Tuning](results/block_number_rmse.png)

During the development of the LSTM model, hyperparameter tuning was performed to achieve the optimal settings for better prediction accuracy. The table below summarizes the optimal hyperparameter values used in the final model:

| Hyperparameters                     | Optimal Settings |
|-------------------------------------|------------------|
| Number of Epochs                    | 500              |
| Batch Size                          | 2                |
| Number of Blocks per Hidden Layer   | 1                |
| Dense Layer                         | 1                |
| Learning Rate                       | 0.1              |
| Dropout Ratio                       | 0.7              |
| Optimizer                           | Adam             |
| Activation Function                 | Hyperbolic Tangent|
| Training Loss                       | 0.0153           |
| Validation Loss                     | 0.0198           |

### Model Loss Curve

The following plot shows the Training and Validation loss over 500 epochs:

![Model Loss Curve](results/pso_lstm_loss_curve.png)

### Insights

- **Convergence:** Both the training and validation loss decrease significantly during the initial epochs, indicating that the model is learning effectively and reducing errors.
- **Stabilization:** After around 30-40 epochs, the losses begin to stabilize, suggesting that the model has reached a plateau and is no longer making substantial improvements.
- **Close Gap Between Training and Validation Loss:** The training and validation losses remain close throughout the training process, indicating good generalization and minimal overfitting.
- **Validation Loss Trends:** The slight fluctuations in the validation loss suggest some variations in performance on the validation set, but the overall trend remains consistent with the training loss.

These observations suggest that the model is well-tuned and exhibits a good balance between fitting the training data and generalizing to unseen validation data.

### LSTM Model Comparison

The figure below compares the predictions made by Univariate and Multivariate LSTM models against the actual heavy metal concentration.

![LSTM Comparison](results/lstm_comparison.png)

### Sensitivity Analysis Plot
![Sensitivity Analysis Plot](results/sensitivity_analysis_plot.png)

### ARIMA vs. LSTM Performance
In this project, both ARIMA (AutoRegressive Integrated Moving Average) and LSTM (Long Short-Term Memory) models were employed to forecast the heavy metal concentration in wastewater. The results were compared to evaluate the effectiveness of each model in time series forecasting:

- **ARIMA Model**: The ARIMA model demonstrated the capability to capture linear patterns in the time series data. It performed well for predicting the general trend and seasonality of the heavy metal concentration. However, ARIMA struggled with rapid fluctuations and non-linear relationships within the dataset, leading to some inaccuracies during sudden changes.

- **LSTM Model**: The LSTM model, with its ability to learn long-term dependencies and handle non-linear relationships, was better suited for capturing abrupt changes in heavy metal concentration. It provided more accurate predictions during periods of rapid change. However, the overall performance was similar to ARIMA in terms of capturing the main trend and seasonal variations.

- **Comparison Plot**: The plot below compares the predictions of ARIMA and LSTM against the actual heavy metal concentration values. Both models closely follow the actual trends, but LSTM shows a slight edge in accuracy, especially during rapid changes in concentration.

![ARIMA vs. LSTM Comparison](results/arima_vs_lstm_comparison.png)

# Performance Metrics

## Table 19. Predictive Index of Heavy Metal Concentration

         | Methodology      | Prediction Average | Prediction MSE | Prediction MAE | Prediction MSLE | R Square |
         |------------------|--------------------|----------------|----------------|-----------------|----------|
         | PSO-LSTM         | 0.048              | 0.021          | 0.064          | 0.0069          | 85%      |
         | LSTM             | 0.049              | 0.01           | 0.096          | 0.0025          | 85%      |
         | PSO-ARIMA        | 0.125              | 0.053          | 0.025          | 0.0373          | 90%      |
         | Univariate LSTM  | 0.120              | 0.02           | 0.07           | 0.0015          | 98%      |

### Key Observations:
   - PSO-LSTM and LSTM models show similar R² values of 85%, indicating comparable accuracy in explaining the variance in the data. However, the LSTM has a lower MSE, suggesting it has smaller prediction errors on average.
   - PSO-ARIMA performs better in terms of R² (90%) but shows a higher MSE and MSLE, suggesting that while it captures the trend well, its prediction errors may be larger than the LSTM models.
   - Univariate LSTM has the highest R² at 98%, indicating that it explains almost all the variability in the data. It also has the lowest MSLE, making it the best performer in terms of logarithmic error.

## Table 18. Performance Metric of Comparison Between All Methods

         | Analytical Methods | Training MSE | Testing MSE | MAE   | MSLE   |
         |--------------------|--------------|-------------|-------|--------|
         | PSO-LSTM           | 0.048        | 0.020       | 0.064 | 0.0069 |
         | PSO-ARIMA          | 0.238        | 0.238       | 0.025 | 0.0373 |
         | Grid Search        | 0.203        | 0.139       | 0.096 | 0.0025 |
         | LSTM               | 0.203        | 0.139       | 0.096 | 0.0025 |

### Key Observations:
   - PSO-LSTM shows the lowest testing MSE (0.020), indicating strong generalization to new data compared to other methods.
   - PSO-ARIMA has a significantly higher MSE, both in training and testing, which may indicate overfitting or difficulty in capturing the complexity of the data.
   - Grid Search LSTM and the plain LSTM models exhibit similar performance, with moderate testing MSE and MAE values. Both models have the lowest MSLE among all methods, highlighting their ability to handle smaller relative errors effectively.

![ Performance Metric Comparison](results/metrics.png)

### Conclusion
While both models demonstrated similar overall accuracy, the LSTM model's ability to handle non-linear relationships and rapid changes in data makes it a slightly better choice for this time series forecasting task. The combination of ARIMA and LSTM can also be considered for future work to leverage the strengths of both models.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
This project was conducted as part of a master's thesis in collaboration with a leading company in the environmental protection and energy-saving sector, and with guidance from Professor Huang Hao, Yuan Ze University. 
The original dataset was modified to protect proprietary information, following advice from the project supervisor.
