from api import get_train_data, get_binance_data, get_current_price
from sklearn.preprocessing import MinMaxScaler
import pickle
from sklearn.model_selection import train_test_split
import numpy as np

def grab_data():
    train_data = get_train_data()
    train_data = train_data.drop(columns=['Dividends', 'Stock Splits'])

    scaler_price = MinMaxScaler()
    scaler_vol = MinMaxScaler()

    train_data[['Open', 'High', 'Low', 'Close']] = scaler_price.fit_transform(train_data[['Open', 'High', 'Low', 'Close']])
    train_data[['Volume']] = scaler_vol.fit_transform(train_data[['Volume']])

    return train_data, scaler_price, scaler_vol

train_data, scaler_price, scaler_vol = grab_data()
pickle.dump(scaler_price, open('scaler_price.pkl', 'wb'))
pickle.dump(scaler_vol, open('scaler_vol.pkl', 'wb'))

def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i]) # last 60 days of data
        y.append(data[i, 3]) # next day close price index 3
    return np.array(X), np.array(y)

data_array = train_data.values
X, y = create_sequences(data_array, 60)

x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)