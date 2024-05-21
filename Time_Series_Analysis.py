# time_series_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load processed data
def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['date'])
    df.set_index('date', inplace=True)
    return df

# Time series decomposition
def decompose_time_series(df, column):
    decomposition = seasonal_decompose(df[column], model='additive', period=12)
    decomposition.plot()
    plt.show()
    return decomposition

# Plot ACF and PACF
def plot_acf_pacf(df, column, lags=40):
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(df[column], lags=lags, ax=ax[0])
    plot_pacf(df[column], lags=lags, ax=ax[1])
    plt.show()

# Augmented Dickey-Fuller test for stationarity
def adf_test(series):
    result = sm.tsa.adfuller(series)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    for key, value in result[4].items():
        print('Critical Values:')
        print(f'   {key}, {value}')

# Main function
def main():
    # File path
    processed_file = 'data/processed/processed_data.csv'
    
    # Load data
    df = load_data(processed_file)
    
    # Column to analyze
    column = 'sales'  # Replace with the actual sales column name in your dataset
    
    # Time series decomposition
    print("Decomposing time series...")
    decompose_time_series(df, column)
    
    # Plot ACF and PACF
    print("Plotting ACF and PACF...")
    plot_acf_pacf(df, column)
    
    # Augmented Dickey-Fuller test
    print("Performing Augmented Dickey-Fuller test...")
    adf_test(df[column])
    
    print("Time series analysis completed.")

if __name__ == "__main__":
    main()
