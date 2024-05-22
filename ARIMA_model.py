# arima_model.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load processed data
def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['date'])
    df.set_index('date', inplace=True)
    return df

# Train ARIMA model
def train_arima_model(df, column, order):
    model = sm.tsa.ARIMA(df[column], order=order)
    fitted_model = model.fit()
    return fitted_model

# Evaluate model
def evaluate_model(df, column, fitted_model):
    predictions = fitted_model.predict(start=df.index[0], end=df.index[-1], dynamic=False)
    mse = mean_squared_error(df[column], predictions)
    r2 = r2_score(df[column], predictions)
    return mse, r2, predictions

# Plot results
def plot_results(df, column, predictions):
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df[column], label='Actual Sales')
    plt.plot(df.index, predictions, label='Predicted Sales', alpha=0.7)
    plt.legend()
    plt.title('Actual vs Predicted Sales')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.show()

# Save model
def save_model(fitted_model, file_path):
    joblib.dump(fitted_model, file_path)

# Main function
def main():
    # File path
    processed_file = 'data/processed/processed_data.csv'
    model_file = 'models/sales_arima_model.pkl'
    
    # Load data
    df = load_data(processed_file)
    
    # Column to predict
    column = 'sales'  # Replace with the actual sales column name in your dataset
    
    # Define ARIMA order (p, d, q)
    order = (5, 1, 0)  # Adjust these parameters based on ACF, PACF, and model performance
    
    # Train ARIMA model
    print("Training ARIMA model...")
    fitted_model = train_arima_model(df, column, order)
    
    # Evaluate model
    print("Evaluating model...")
    mse, r2, predictions = evaluate_model(df, column, fitted_model)
    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')
    
    # Plot results
    plot_results(df, column, predictions)
    
    # Save model
    save_model(fitted_model, model_file)
    print(f"Model saved to {model_file}")

if __name__ == "__main__":
    main()
