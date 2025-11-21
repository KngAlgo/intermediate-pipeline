from api import get_train_data, get_binance_data, get_current_price
from sklearn.preprocessing import MinMaxScaler
import pickle

train_data = get_train_data()
train_data = train_data.drop(columns=['Dividends', 'Stock Splits'])

scaler_price = MinMaxScaler()
scaler_vol = MinMaxScaler()

train_data[['Open']] = scaler_price.fit_transform(train_data[['Open']])
train_data[['High']] = scaler_price.fit_transform(train_data[['High']])
train_data[['Low']] = scaler_price.fit_transform(train_data[['Low']])
train_data[['Close']] = scaler_price.fit_transform(train_data[['Close']])
train_data[['Volume']] = scaler_vol.fit_transform(train_data[['Volume']])

print(train_data.head())

pickle.dump(scaler_price, open('scaler_price.pkl', 'wb'))
pickle.dump(scaler_vol, open('scaler_vol.pkl', 'wb'))

def create_sequences(data, seq_length=60):
    pass