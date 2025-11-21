from process import x_train, x_val, y_train, y_val
from tensorflow import keras
from keras import layers, Model

input = keras.Input(shape=(x_train.shape[1], x_train.shape[2]))
# model needs shape of timesteps (60), features (5)

# return_sequences=True to pass 3D output to next LSTM
# Using default activation='tanh' for cuDNN acceleration
x = layers.LSTM(16, return_sequences=True)(input)
x = layers.Dropout(0.2)(x)

# return_sequences=True to pass 3D output to next LSTM
x = layers.LSTM(32, return_sequences=True)(x)
x = layers.Dropout(0.2)(x)

# Last LSTM: return_sequences=False (default) to get 2D output for Dense layer
x = layers.LSTM(64)(x)
x = layers.Dropout(0.2)(x)

output = layers.Dense(1, activation='linear')(x)
model = Model(input, output)

model.compile(loss='mae', metrics=['mae'], optimizer='adam')

model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, callbacks=[keras.callbacks.ModelCheckpoint("model.h5", monitor='val_loss')])