# data_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Load data
def load_data(sales_file, economic_file, seasonal_file):
    sales_data = pd.read_csv(sales_file, parse_dates=['date'])
    economic_data = pd.read_csv(economic_file, parse_dates=['date'])
    seasonal_data = pd.read_csv(seasonal_file, parse_dates=['date'])
    return sales_data, economic_data, seasonal_data

# Clean data
def clean_data(df):
    df = df.dropna()  # Drop rows with missing values
    df = df[df.select_dtypes(include=[np.number]).ge(0).all(1)]  # Remove negative values
    return df

# Feature engineering
def create_features(sales_data, economic_data, seasonal_data):
    # Merge dataframes on 'date' column
    data = sales_data.merge(economic_data, on='date', how='left')
    data = data.merge(seasonal_data, on='date', how='left')
    
    # Create additional time-related features
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['day_of_week'] = data['date'].dt.dayofweek
    data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
    
    return data

# Normalize data
def normalize_data(df):
    scaler = StandardScaler()
    df[df.columns.difference(['date'])] = scaler.fit_transform(df[df.columns.difference(['date'])])
    return df

# Save processed data
def save_data(df, file_path):
    df.to_csv(file_path, index=False)

# Main function
def main():
    # File paths
    sales_file = 'data/raw/sales_data.csv'
    economic_file = 'data/raw/economic_data.csv'
    seasonal_file = 'data/raw/seasonal_data.csv'
    processed_file = 'data/processed/processed_data.csv'
    
    # Load data
    sales_data, economic_data, seasonal_data = load_data(sales_file, economic_file, seasonal_file)
    
    # Clean data
    sales_data = clean_data(sales_data)
    economic_data = clean_data(economic_data)
    seasonal_data = clean_data(seasonal_data)
    
    # Feature engineering
    data = create_features(sales_data, economic_data, seasonal_data)
    
    # Normalize data
    data = normalize_data(data)
    
    # Save processed data
    save_data(data, processed_file)
    print(f"Processed data saved to {processed_file}")

if __name__ == "__main__":
    main()
