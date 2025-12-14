from flask import Flask, render_template
import keras
import pickle
import numpy as np
from datetime import datetime
import sys
import os

# Add BTC directory to path to import api module
sys.path.append(os.path.join(os.path.dirname(__file__), 'BTC'))
from data.api import get_1hr_train_data, get_gecko_data
from data.indicators import add_technical_indicators
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# ============================================
# LOAD MODEL AND SCALERS AT STARTUP (ONCE!)
# ============================================
# This happens when the server starts, not on every request
# Loading model is expensive - we only want to do it once
print("Loading model and scalers...")
model = keras.models.load_model('saves/1hr_V2model.h5')
scaler_price = pickle.load(open('saves/scaler_price.pkl', 'rb'))
scaler_vol = pickle.load(open('saves/scaler_vol.pkl', 'rb'))
scaler_osc = pickle.load(open('saves/scaler_osc.pkl', 'rb'))
scaler_ind = pickle.load(open('saves/scaler_ind.pkl', 'rb'))
print("Model loaded successfully!")


def get_prediction():
    """
    Gets the BTC price prediction for tomorrow

    How it works:
    1. Fetch historical data (need enough for indicators to calculate)
    2. Add technical indicators (RSI, MACD, SMA, etc.)
    3. Preprocess: drop unnecessary columns to get 13 features
    4. Scale the data using the same scalers from training
    5. Get last 60 days and reshape: (1 sample, 60 timesteps, 13 features)
    6. Get prediction (scaled value)
    7. Inverse transform to get real dollar value
    """
    # Step 1: Get historical data
    data = get_1hr_train_data()

    # Step 2: Add technical indicators (same as training)
    data = add_technical_indicators(data)

    # Step 3: Drop the same columns as during training to get 13 features
    data = data.drop(columns=['Dividends', 'Stock Splits', 'Body', 'Stochastic',
                              'Stochastic_Signal', 'MACD_Histogram', 'Upper_Shadow',
                              'Lower_Shadow', 'EMA_12'])

    # Step 4: Fit scalers for indicators (these weren't saved during training)
    # We fit them on the full dataset to maintain consistenc

    data[['RSI']] = scaler_osc.transform(data[['RSI']])
    data[['MACD', 'MACD_Signal', 'ATR', 'Returns', 'Range', 'SMA_20', 'SMA_50']] = scaler_ind.transform(
        data[['MACD', 'MACD_Signal', 'ATR', 'Returns', 'Range', 'SMA_20', 'SMA_50']]
    )

    # Get last 60 days
    period = data[-168:].copy()

    # Step 5: Scale price and volume (MUST use transform, not fit_transform!)
    # We use transform() because we already fit the scaler during training
    # fit_transform() would learn NEW min/max values, which would be wrong
    period[['Open', 'High', 'Low', 'Close']] = scaler_price.transform(
        period[['Open', 'High', 'Low', 'Close']]
    )
    period[['Volume']] = scaler_vol.transform(period[['Volume']])

    # Note: Indicators are already scaled above

    # Step 6: Reshape for LSTM input
    # LSTM expects shape: (batch_size, timesteps, features)
    # We have: (60 days, 13 features) -> reshape to (1, 60, 13)
    period_reshaped = period.values.reshape(1, 168, 13)

    # Step 5: Get prediction (this is a scaled value between 0 and 1)
    pred = model.predict(period_reshaped, verbose=0)

    # Step 6: Inverse transform to get real price
    # We trained the model to predict Close price
    # Scaler expects 4 values [Open, High, Low, Close], so we duplicate pred
    pred_real = scaler_price.inverse_transform([
        [pred[0,0], pred[0,0], pred[0,0], pred[0,0]]
    ])

    # Return the Close price (index 3)
    return float(pred_real[0, 3])


@app.route('/')
def index():
    """
    Main route - displays BTC prediction

    When someone visits the website, this function:
    1. Gets current BTC price from CoinGecko
    2. Gets model's prediction for tomorrow
    3. Calculates the predicted change
    4. Renders HTML template with this data
    """
    try:
        # Get current price from CoinGecko API
        current_price = get_gecko_data()

        # Get model prediction for tomorrow's close
        prediction = get_prediction()

        # Calculate predicted change (dollar and percentage)
        change = prediction - current_price
        change_pct = (change / current_price) * 100

        # Determine if bullish or bearish
        sentiment = "BULLISH" if change > 0 else "BEARISH"

        # Pass all data to HTML template
        return render_template('index.html',
                             current_price=f"${current_price:,.2f}",
                             prediction=f"${prediction:,.2f}",
                             change=f"${change:+,.2f}",  # +/- sign
                             change_pct=f"{change_pct:+.2f}%",
                             sentiment=sentiment,
                             timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    except Exception as e:
        # If something goes wrong, show error page
        return render_template('index.html',
                             error=str(e),
                             timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == '__main__':
    # Run the Flask development server
    # debug=True enables auto-reload when you change code
    # host='0.0.0.0' makes it accessible from other devices on network
    app.run(debug=True, host='0.0.0.0', port=5000)
