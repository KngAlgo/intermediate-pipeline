from data.process import x_train, x_val, y_train, y_val
from tensorflow import keras
from keras import layers, Model
import mlflow
from mlflow.models import infer_signature

mlflow.set_experiment("First MLFlow setup")

with mlflow.start_run(run_name='Setup Run'):
    # if doing a new run on same experiment, change run_name
    # don't forget to change param dictionary too if adding new params
    # don't change the model name unless using different architecture

    mlflow.log_params({
        "lstm_units_1": 16,
        "lstm_units_2": 32,
        "lstm_units_3": 64,
        "dropout_rate": 0.2,
        "optimizer": "adam",
        "loss": "mae",
        "epochs": 100,
        "sequence_length": 60,
        "num_features": x_train.shape[2],
    })

    input = keras.Input(shape=(x_train.shape[1], x_train.shape[2]))
    # model needs shape of timesteps (60), features (13)

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

    mlflow.tensorflow.autolog()
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=75, callbacks=[keras.callbacks.ModelCheckpoint("saves/model.h5", monitor='val_loss')])

    mlflow.tensorflow.log_model(model=model, artifact_path="tf_model", registered_model_name="BTC-TF MODEL", input_example=x_train[:5])