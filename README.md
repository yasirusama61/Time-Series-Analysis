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
      ├── .github/                         # GitHub Actions workflows for CI/CD (if applicable)
      │   └── workflows/
      │       └── python-package.yml       # Workflow for testing and building Python packages
      ├── data/                            # Data folder for raw and preprocessed datasets
      │   ├── heavy_metal_data.xlsx        # Placeholder for raw dataset (not publicly available)
      │   └── processed_data.csv           # Preprocessed data used for modeling (not included)
      ├── scripts/                         # Scripts for data processing, modeling, and evaluation
      │   ├── arima_model.py               # Script for ARIMA time series forecasting
      │   ├── train_lstm.py                # Script for training LSTM model
      │   ├── pso_lstm_model.py            # Script for PSO-LSTM model training
      │   ├── hyperparameter_optimization.py # Script for optimizing hyperparameters using PSO
      │   ├── learning_rate_hidden_layer_tuning.py # Script for tuning LSTM hyperparameters
      │   ├── preprocess_data.py           # Script for data cleaning and preprocessing
      │   ├── sensitivity_analysis.py      # Script for sensitivity analysis
      │   ├── var.py                       # Script for Vector Autoregression (VAR) analysis
      │   ├── test.py                      # Unit tests for the project
      │   └── furniture_sales_arima.py     # Additional ARIMA example using sales data
      ├── models/                          # Folder for saving trained models
      │   ├── arima_model.pkl              # Saved ARIMA model
      │   └── pso_lstm_model.h5            # Trained PSO-LSTM model
      ├── results/                         # Results such as plots, metrics, pseudocode, and logs
      │   ├── ARIMA_algorithm.png          # Pseudocode for ARIMA
      │   ├── arima_vs_lstm_comparison.png # Plot comparing ARIMA and LSTM predictions
      │   ├── batch_size_rmse.png          # Plot showing effect of batch size on RMSE
      │   ├── block_number_rmse.png        # Plot illustrating block number tuning for RMSE
      │   ├── lstm_comparison.png          # LSTM model comparison plot
      │   ├── metrics.png                  # Heatmap of performance metrics (e.g., MSE, MAE, R²)
      │   ├── processflow.png              # Visualization of the process flow
      │   ├── pso_algorithm.png            # Pseudocode for PSO algorithm
      │   ├── pso_lstm_loss_curve.png      # Loss curve for PSO-LSTM training
      │   └── sensitivity_analysis_plot.png # Plot showing feature sensitivity analysis
      ├── notebooks/                       # Jupyter notebooks for analysis and documentation
      │   ├── Tune_the_parameters_of_SVM_using_PSO.ipynb # Notebook for PSO-SVM tuning
      │   └── other_notebooks.ipynb        # Other analysis or tutorial notebooks
      ├── README.md                        # Project overview and instructions
      ├── requirements.txt                 # Python dependencies
      └── LICENSE                          # License for the project (if applicable)

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

The data used in this project was collected from an industrial wastewater treatment facility in collaboration with a company specializing in environmental protection and energy-saving technologies. Due to confidentiality agreements, the original dataset cannot be publicly shared. However, the analysis conducted in this project utilized features such as:

   - Heavy Metal Concentration (mg/L)
   - Heavy Metal Input Concentration (mg/L)
   - Electrical Conductivity
   - pH
   - pH_ORP (Oxidation-Reduction Potential)
   - Chemical Dosage Levels (Chemical A and B)

For privacy reasons, certain sensitive details have been anonymized or modified in the dataset used for analysis. The original raw data is not included in the repository. Only code and scripts for data processing, model training, and evaluation are provided.


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


### ARIMA vs. LSTM Performance
In this project, both ARIMA (AutoRegressive Integrated Moving Average) and LSTM (Long Short-Term Memory) models were employed to forecast the heavy metal concentration in wastewater. The results were compared to evaluate the effectiveness of each model in time series forecasting:

- **ARIMA Model**: The ARIMA model demonstrated the capability to capture linear patterns in the time series data. It performed well for predicting the general trend and seasonality of the heavy metal concentration. However, ARIMA struggled with rapid fluctuations and non-linear relationships within the dataset, leading to some inaccuracies during sudden changes.

- **LSTM Model**: The LSTM model, with its ability to learn long-term dependencies and handle non-linear relationships, was better suited for capturing abrupt changes in heavy metal concentration. It provided more accurate predictions during periods of rapid change. However, the overall performance was similar to ARIMA in terms of capturing the main trend and seasonal variations.

- **Comparison Plot**: The plot below compares the predictions of ARIMA and LSTM against the actual heavy metal concentration values. Both models closely follow the actual trends, but LSTM shows a slight edge in accuracy, especially during rapid changes in concentration.

![ARIMA vs. LSTM Comparison](results/arima_vs_lstm_comparison.png)

# Performance Metrics

## Predictive Index of Heavy Metal Concentration

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

## Performance Metric of Comparison Between All Methods

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

## Sensitivity Analysis

The following plot shows the impact of different features on predicting heavy metal concentration, measured by the change in Mean Squared Error (MSE) when each feature is excluded:

![Sensitivity Analysis Plot](results/sensitivity_analysis_plot.png)

### Insights

- **Key Influential Features:**
  - **Electrical Conductivity** has the highest impact, with a change in MSE of approximately 0.35. This indicates that it is a crucial predictor and highly correlated with the heavy metal concentration.
  - **Chemical A** is also significant, showing a change in MSE around 0.30. This suggests that its concentration or dosage greatly influences the prediction.

- **Moderate Impact Features:**
  - **pH_ORP** and **Chemical A_ORP** have a moderate effect on the prediction, with changes in MSE of approximately 0.22 and 0.15, respectively.
  - **Chemical B** shows a change in MSE of about 0.18, indicating it is an important predictor but less impactful than the top factors.

- **Less Influential Features:**
  - **Heavy Metal Input concentration** and **pH** show smaller changes in MSE, around 0.12 and 0.08, respectively. While they still contribute to the prediction, their impact is comparatively lower.

### Recommendations

- **Focus on Key Features:** Given the significant influence of electrical conductivity and Chemical A, these variables should be prioritized for monitoring and control to improve predictive accuracy.
- **Feature Engineering:** Consider adding interaction terms or derived features involving electrical conductivity, Chemical A, and pH_ORP to better capture complex relationships.
- **Further Investigation:** Explore why features like Heavy Metal Input concentration and pH have a lower impact, as this could reveal additional insights into the process.

These insights can guide the selection and engineering of features to enhance the model's predictive performance.

### Conclusion
While both models demonstrated similar overall accuracy, the LSTM model's ability to handle non-linear relationships and rapid changes in data makes it a slightly better choice for this time series forecasting task. The sensitivity analysis revealed that certain features, such as electrical conductivity and Chemical A concentration, have a significant impact on the prediction of heavy metal concentration. These findings suggest that focusing on the most influential features could further improve model accuracy. Incorporating interaction terms or engineered features based on these key variables may also enhance predictive performance.

For future work, combining the strengths of ARIMA and LSTM could be a promising approach. While ARIMA is effective for modeling linear trends and seasonality, LSTM excels in capturing complex patterns and sudden changes. A hybrid model leveraging these complementary strengths, along with the insights gained from sensitivity analysis, may yield even better forecasting results.

## Recommendations for Real-World Application

The PSO-LSTM model can be effectively used in industrial wastewater treatment to optimize processes and enhance decision-making. Here are some key recommendations:

### 1. **Real-Time Monitoring and Control**
   - **Integration with Monitoring Systems:** Use the model for real-time predictions of heavy metal concentrations to dynamically adjust treatment parameters, such as chemical dosage and flow rate.
   - **Automated Control Systems:** Enable automated process adjustments to maintain compliance with environmental standards and optimize treatment efficiency.

### 2. **Early Warning System**
   - **Anomaly Detection:** Identify unexpected spikes in heavy metal levels to prevent potential regulatory violations or equipment failures.
   - **Compliance Monitoring:** Predict when heavy metal concentrations approach regulatory limits and take preventive action.

### 3. **Optimizing Chemical Usage**
   - **Predictive Dosing Optimization:** Forecast the necessary chemical dosage to maintain target heavy metal levels, reducing chemical costs.
   - **Focus on Key Influencers:** Utilize sensitivity analysis results to prioritize influential parameters (e.g., electrical conductivity) for better control.

### 4. **Scenario Analysis**
   - **Simulate Operating Conditions:** Evaluate the impact of different treatment strategies and operating conditions on heavy metal removal efficiency.
   - **Assess Upgrades:** Predict the effects of potential system upgrades or changes in the treatment process.

### 5. **Deployment Considerations**
   - **Model Retraining:** Periodically update the model with new data to maintain accuracy.
   - **Cloud vs. Edge Deployment:** Choose deployment based on latency requirements—cloud for centralized processing or edge for faster local predictions.

These recommendations help guide the practical use of the PSO-LSTM model for optimizing wastewater treatment and ensuring compliance with regulatory standards.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
This project was conducted as part of a master's thesis in collaboration with a leading company in the environmental protection and energy-saving sector, and with guidance from Professor Huang Hao, Yuan Ze University. 
The original dataset was modified to protect proprietary information, following advice from the project supervisor.
