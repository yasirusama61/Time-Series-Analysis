## Project Overview
This repository contains code and data for time series forecasting using ARIMA and PSO-LSTM models. The main objective is to predict sales data and optimize model performance using Particle Swarm Optimization (PSO) with Long Short-Term Memory (LSTM) neural networks.

## Models Implemented
### 1. **ARIMA (AutoRegressive Integrated Moving Average)**
   - Used to model and forecast the sales of furniture from a dataset.
   - Model parameters were tuned using a grid search to find the best p, d, q values.
   - Includes a decomposition of the time series into trend, seasonality, and residuals.

### 2. **PSO-LSTM (Particle Swarm Optimization - Long Short-Term Memory)**
   - Implemented for time series prediction using PSO to optimize LSTM model parameters.
   - The LSTM model is used to predict time series data, while PSO improves model performance by adjusting weights.

## Files in the Repository
- **\`furniture_sales_arima.py\`**: Python script for ARIMA time series forecasting.
- **\`PSO-LSTM.py\`**: Python script implementing PSO optimization for LSTM models.
- **\`data/\`**: Folder containing the datasets used in the project.

## How to Run the Code
1. Clone the repository:
   \`\`\`bash
   git clone https://github.com/yasirusama61/Time-Series-Analysis.git
   \`\`\`
2. Install the required dependencies:
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`
3. Run the ARIMA model:
   \`\`\`bash
   python furniture_sales_arima.py
   \`\`\`
4. Run the PSO-LSTM model:
   \`\`\`bash
   python PSO-LSTM.py
   \`\`\`

## Data
The data used for these models comes from a dataset of furniture sales over several years. The data is processed and cleaned in the respective Python scripts.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

EOL

# Step 3: Stage the changes
echo "Staging changes..."
git add README.md

# Step 4: Commit the changes
echo "Committing changes..."
git commit -m "Updated README.md with project details"

# Step 5: Push the changes to GitHub
echo "Pushing changes to GitHub..."
git push origin main

# Step 6: Completion message
echo "README.md updated and changes pushed to GitHub successfully."
