import unittest
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Function to load and prepare data
def load_data(file_path):
    try:
        df = pd.read_excel(file_path)
        return df
    except Exception as e:
        raise ValueError(f"Error loading file: {e}")

# Function to process time-series data
def process_time_series(df, category):
    filtered_df = df.loc[df['Category'] == category]
    if filtered_df.empty:
        raise ValueError(f"No data found for category: {category}")
    
    filtered_df = filtered_df.sort_values('Order Date')
    filtered_df = filtered_df.set_index('Order Date')
    return filtered_df['Sales'].resample('MS').mean()

# Function to fit ARIMA model
def fit_arima_model(time_series):
    model = ARIMA(time_series, order=(1,1,0))
    model_fit = model.fit()
    return model_fit

# Unit Tests
class TestTimeSeriesAnalysis(unittest.TestCase):

    def setUp(self):
        # Prepare a small sample dataset
        data = {
            'Order Date': pd.date_range(start='1/1/2020', periods=4, freq='MS'),
            'Category': ['Furniture', 'Furniture', 'Furniture', 'Office Supplies'],
            'Sales': [200, 300, 150, 400]
        }
        self.df = pd.DataFrame(data)
        self.file_path = "Superstore.xls"  # Replace with your actual file path
    
    def test_load_data(self):
        # Test data loading
        df = load_data(self.file_path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue('Order Date' in df.columns)
    
    def test_process_time_series(self):
        # Test time series processing
        ts = process_time_series(self.df, 'Furniture')
        self.assertEqual(len(ts), 3)  # We expect 3 months of data for Furniture category
        self.assertEqual(ts.index.freqstr, 'MS')  # Make sure data is resampled by Month Start
    
    def test_fit_arima_model(self):
        # Test ARIMA model fitting
        ts = process_time_series(self.df, 'Furniture')
        model_fit = fit_arima_model(ts)
        self.assertTrue(hasattr(model_fit, 'aic'))  # ARIMA model should have AIC attribute after fitting
    
    def test_no_data_for_category(self):
        # Test error handling when no data is available for the category
        with self.assertRaises(ValueError):
            process_time_series(self.df, 'Technology')

    def test_invalid_file_loading(self):
        # Test error handling for invalid file loading
        with self.assertRaises(ValueError):
            load_data("invalid_file.xls")

if __name__ == "__main__":
    unittest.main()
