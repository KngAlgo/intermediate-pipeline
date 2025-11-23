from flask import Flask, render_template
import keras
import pickle
import numpy as np
from datetime import datetime
import sys
import os

# Add BTC directory to path to import api module
sys.path.append(os.path.join(os.path.dirname(__file__), 'BTC'))
from data.api import get_train_data, get_gecko_data

app = Flask(__name__)

# ============================================
# LOAD MODEL AND SCALERS AT STARTUP (ONCE!)
# ============================================
# This happens when the server starts, not on every request
# Loading model is expensive - we only want to do it once
print("Loading model and scalers...")
model = keras.models.load_model('BTC/model.h5')
scaler_price = pickle.load(open('BTC/scaler_price.pkl', 'rb'))
scaler_vol = pickle.load(open('BTC/scaler_vol.pkl', 'rb'))
print("Model loaded successfully!")


def get_prediction():
    """
    Gets the BTC price prediction for tomorrow

    How it works:
    1. Fetch last 60 days of historical data (our model needs 60-day window)
    2. Preprocess: drop unnecessary columns
    3. Scale the data using the same scalers from training
    4. Reshape into format model expects: (1 sample, 60 timesteps, 5 features)
    5. Get prediction (scaled value)
    6. Inverse transform to get real dollar value
    """
    # Step 1: Get last 60 days of data
    data = get_train_data()
    period = data[-60:].copy()  # Last 60 days

    # Step 2: Drop columns we didn't train on
    period = period.drop(columns=['Dividends', 'Stock Splits'])

    # Step 3: Scale the data (MUST use transform, not fit_transform!)
    # We use transform() because we already fit the scaler during training
    # fit_transform() would learn NEW min/max values, which would be wrong
    period[['Open', 'High', 'Low', 'Close']] = scaler_price.transform(
        period[['Open', 'High', 'Low', 'Close']]
    )
    period[['Volume']] = scaler_vol.transform(period[['Volume']])

    # Step 4: Reshape for LSTM input
    # LSTM expects shape: (batch_size, timesteps, features)
    # We have: (60 days, 5 features) -> reshape to (1, 60, 5)
    period_reshaped = period.values.reshape(1, 60, 5)

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
