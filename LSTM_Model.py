# lstm_model.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import joblib

# Load processed data
def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['date'])
    df.set_index('date', inplace=True)
    return df

# Prepare data for LSTM
def prepare_data(df, column, time_steps):
    data = df[column].values
    data = data.reshape(-1, 1)

    # Scale data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(time_steps, len(scaled_data)):
        X.append(scaled_data[i-time_steps:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)

    # Reshape X for LSTM input
    X = X.reshape((X.shape[0], X.shape[1], 1))

    return X, y, scaler

# Build LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Plot results
def plot_results(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, color='blue', label='Actual Sales')
    plt.plot(y_pred, color='red', label='Predicted Sales')
    plt.title('Actual vs Predicted Sales')
    plt.xlabel('Time')
    plt.ylabel('Sales')
    plt.legend()
    plt.show()

# Save model and scaler
def save_model_and_scaler(model, scaler, model_path, scaler_path):
    model.save(model_path)
    joblib.dump(scaler, scaler_path)

# Main function
def main():
    # File path
    processed_file = 'data/processed/processed_data.csv'
    model_file = 'models/sales_lstm_model.h5'
    scaler_file = 'models/scaler.pkl'
    
    # Load data
    df = load_data(processed_file)
    
    # Column to predict
    column = 'sales'  # Replace with the actual sales column name in your dataset
    
    # Prepare data
    time_steps = 60  # Number of time steps to look back
    X, y, scaler = prepare_data(df, column, time_steps)
    
    # Split data into train and test sets
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Build LSTM model
    print("Building LSTM model...")
    model = build_lstm_model((X_train.shape[1], 1))
    
    # Train LSTM model
    print("Training LSTM model...")
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stop])
    
    # Evaluate model
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    y_pred_inverse = scaler.inverse_transform(y_pred)
    y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1))
    mse = mean_squared_error(y_test_inverse, y_pred_inverse)
    r2 = r2_score(y_test_inverse, y_pred_inverse)
    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')
    
    # Plot results
    plot_results(y_test_inverse, y_pred_inverse)
    
    # Save model and scaler
    save_model_and_scaler(model, scaler, model_file, scaler_file)
    print(f"Model saved to {model_file}")
    print(f"Scaler saved to {scaler_file}")

if __name__ == "__main__":
    main()
