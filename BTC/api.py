import yfinance as yf
import requests
import pandas as pd

def get_train_data():
    btc = yf.Ticker('BTC-USD')
    historical = btc.history(period='5y')
    return historical

def get_gecko_data():
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {'ids': 'bitcoin', 'vs_currencies': 'usd'}
    response = requests.get(url, params=params)
    data = response.json()
    return data['bitcoin']['usd']

def get_current_price():
    try:
        url = "https://api.coingecko.com/api/v3/simple/price"
        response = requests.get(url, params={'ids': 'bitcoin', 'vs_currencies': 'usd'}, timeout=10)
        response.raise_for_status()  # Raises error for bad status codes
        data = response.json()
        return data['bitcoin']['usd']
    except requests.exceptions.RequestException as e:
        print(f"API Error: {e}")
        return None