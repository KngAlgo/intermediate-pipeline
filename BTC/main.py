from tensorflow import keras
from api import get_train_data
import pickle
import pandas as pd

model = keras.models.load_model('model.h5')

scaler_price = pickle.load(open('scaler_price.pkl', 'rb'))
scaler_vol = pickle.load(open('scaler_vol.pkl', 'rb'))
# load each scaler in from processing to maintain scaled values

data = get_train_data() # grab data

period = data[-60:] # get last 60 days
period = period.drop(columns=['Dividends', 'Stock Splits'])

period[['Open', 'High', 'Low', 'Close']] = scaler_price.transform(period[['Open', 'High', 'Low', 'Close']])
period[['Volume']] = scaler_vol.transform(period[['Volume']])

period_arr = period.values
period_reshaped = period_arr.reshape(1, 60, 5)

pred = model.predict(period_reshaped)
pred_real = scaler_price.inverse_transform([[pred[0,0], pred[0,0], pred[0,0], pred[0,0]]])
pred_price = pred_real[0,3]
