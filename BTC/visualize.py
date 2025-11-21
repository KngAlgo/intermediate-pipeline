## AI GENERATED CHART SCRIPT

from tensorflow import keras
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from api import get_train_data

print("Loading model and scalers...")
model = keras.models.load_model('model.h5')
scaler_price = pickle.load(open('scaler_price.pkl', 'rb'))
scaler_vol = pickle.load(open('scaler_vol.pkl', 'rb'))

print("Fetching data...")
data = get_train_data()
data = data.drop(columns=['Dividends', 'Stock Splits'])

# Decide how many days to predict (last N days)
num_predictions = 100  # Change this to test more/fewer days
print(f"Making predictions for last {num_predictions} days...")

predictions = []
actuals = []
dates = []

# Start from day 60 (need 60 days history), predict until end
start_idx = len(data) - num_predictions
for i in range(start_idx, len(data)):
    # Get 60-day window ending at day i-1
    window = data.iloc[i-60:i].copy()

    # Scale the window
    window[['Open', 'High', 'Low', 'Close']] = scaler_price.transform(window[['Open', 'High', 'Low', 'Close']])
    window[['Volume']] = scaler_vol.transform(window[['Volume']])

    # Reshape for model
    window_array = window.values.reshape(1, 60, 5)

    # Predict
    pred_scaled = model.predict(window_array, verbose=0)

    # Inverse transform to get real price
    pred_real = scaler_price.inverse_transform([[pred_scaled[0,0], pred_scaled[0,0], pred_scaled[0,0], pred_scaled[0,0]]])
    predicted_price = pred_real[0, 3]

    # Get actual price
    actual_price = data.iloc[i]['Close']

    # Store results
    predictions.append(predicted_price)
    actuals.append(actual_price)
    dates.append(data.index[i])

# Calculate error metrics
predictions_arr = np.array(predictions)
actuals_arr = np.array(actuals)
mae = np.mean(np.abs(predictions_arr - actuals_arr))
rmse = np.sqrt(np.mean((predictions_arr - actuals_arr)**2))
mape = np.mean(np.abs((actuals_arr - predictions_arr) / actuals_arr)) * 100

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(dates, actuals, label='Actual Price', color='#2E86DE', linewidth=2)
plt.plot(dates, predictions, label='Predicted Price', color='#EE5A6F', linestyle='--', linewidth=2, alpha=0.8)

plt.xlabel('Date', fontsize=12)
plt.ylabel('Price (USD)', fontsize=12)
plt.title('BTC Price Prediction: Actual vs Predicted', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
plt.savefig('btc_prediction_comparison.png', dpi=300)
print("Chart saved as 'btc_prediction_comparison.png'")

# Show the plot
plt.show()

