from data.api import get_1hr_train_data
from data.indicators import add_technical_indicators, get_feature_list
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

def grab_data():
    train_data = get_1hr_train_data()
    train_data = add_technical_indicators(train_data)
    train_data = train_data.drop(columns=['Dividends', 'Stock Splits', 'Body', 'Stochastic', 'Stochastic_Signal', 'MACD_Histogram', 'Upper_Shadow', 'Lower_Shadow', 'EMA_12'])

    scaler_price = MinMaxScaler()
    scaler_vol = MinMaxScaler()
    scaler_osc = StandardScaler()
    scaler_ind = StandardScaler()

    # stochastic, stochastic signal, macd histogram, ema12, range ***, body, upper_shadow, lower_shadeow

    train_data[['Open', 'High', 'Low', 'Close']] = scaler_price.fit_transform(train_data[['Open', 'High', 'Low', 'Close']])
    train_data[['Volume']] = scaler_vol.fit_transform(train_data[['Volume']])
    train_data[['RSI']] = scaler_osc.fit_transform(train_data[['RSI']])
    train_data[['MACD', 'MACD_Signal', 'ATR', 'Returns', 'Range', 'SMA_20', 'SMA_50']] = scaler_ind.fit_transform(train_data[['MACD', 'MACD_Signal', 'ATR', 'Returns', 'Range', 'SMA_20', 'SMA_50']])

    return train_data, scaler_price, scaler_vol, scaler_osc, scaler_ind

train_data, scaler_price, scaler_vol, scaler_osc, scaler_ind = grab_data()
pickle.dump(scaler_price, open('saves/scaler_price.pkl', 'wb'))
pickle.dump(scaler_vol, open('saves/scaler_vol.pkl', 'wb'))
pickle.dump(scaler_osc, open('saves/scaler_osc.pkl', 'wb'))
pickle.dump(scaler_ind, open('saves/scaler_ind.pkl', 'wb'))

def create_sequences(data, seq_length=168):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i]) # last 60 days of data
        y.append(data[i, 3]) # next day close price index 3
    return np.array(X), np.array(y)

data_array = train_data.values
X, y = create_sequences(data_array, 168)

x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

def export_merged_data_to_csv():
    data = get_train_data()
    data = data.drop(columns=['Dividends', 'Stock Splits'])
    data_with_indicators = add_technical_indicators(data)
    data_with_indicators.to_csv('saves/btc_data_with_indicators.csv')
    print(f"Saved {len(data_with_indicators)} rows to btc_data_with_indicators.csv")

