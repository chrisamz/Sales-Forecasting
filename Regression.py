# model_building.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load processed data
def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['date'])
    df.set_index('date', inplace=True)
    return df

# Prepare data for regression model
def prepare_data(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train regression model
def train_regression_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2, y_pred

# Plot results
def plot_results(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.index, y_test, label='Actual Sales')
    plt.plot(y_test.index, y_pred, label='Predicted Sales', alpha=0.7)
    plt.legend()
    plt.title('Actual vs Predicted Sales')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.show()

# Save model
def save_model(model, file_path):
    joblib.dump(model, file_path)

# Main function
def main():
    # File path
    processed_file = 'data/processed/processed_data.csv'
    model_file = 'models/sales_regression_model.pkl'
    
    # Load data
    df = load_data(processed_file)
    
    # Column to predict
    target_column = 'sales'  # Replace with the actual sales column name in your dataset
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(df, target_column)
    
    # Train regression model
    print("Training regression model...")
    model = train_regression_model(X_train, y_train)
    
    # Evaluate model
    print("Evaluating model...")
    mse, r2, y_pred = evaluate_model(model, X_test, y_test)
    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')
    
    # Plot results
    plot_results(y_test, y_pred)
    
    # Save model
    save_model(model, model_file)
    print(f"Model saved to {model_file}")

if __name__ == "__main__":
    main()
