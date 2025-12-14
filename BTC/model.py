from data.process import x_train, x_val, y_train, y_val
from tensorflow import keras
from keras import layers, Model
import mlflow
from mlflow.models import infer_signature

mlflow.set_experiment("Hourly Data Structure")

with mlflow.start_run(run_name='First 1HR Structure Test'):
    # if doing a new run on same experiment, change run_name
    # don't forget to change param dictionary too if adding new params
    # don't change the model name unless using different architecture

    mlflow.set_tag("mlflow.note.content", "Not using bidirectional to save resources")

    mlflow.log_params({
        "lstm_units_1": 32,
        "lstm_units_2": 64,
        "lstm_units_3": 128,
        "dropout_rate_1": 0.2,
        "dropout_rate_2": 0.2,
        "dropout_rate_3": 0.25,
        "optimizer": "adam",
        "loss": "mae",
        "epochs": 100,
        "sequence_length": 168,
        "num_features": x_train.shape[2],
        "batch_size": 128
    })

    input = keras.Input(shape=(x_train.shape[1], x_train.shape[2]))
    # model needs shape of timesteps (168), features (13)

    # return_sequences=True to pass 3D output to next LSTM
    # Using default activation='tanh' for cuDNN acceleration
    x = layers.LSTM(32, return_sequences=True)(input)
    x = layers.Dropout(0.2)(x)

    # return_sequences=True to pass 3D output to next LSTM
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.Dropout(0.5)(x)

    # Last LSTM: return_sequences=False (default) to get 2D output for Dense layer
    x = layers.LSTM(128)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Dense(64, activation='relu')(x)

    output = layers.Dense(1, activation='linear')(x)
    model = Model(input, output)

    model.compile(loss='mae', metrics=['mae'], optimizer='adam')

    mlflow.tensorflow.autolog()
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=128, epochs=45, callbacks=[keras.callbacks.ModelCheckpoint("saves/1hr_V2model.h5", monitor='val_loss')])

    mlflow.tensorflow.log_model(model=model, artifact_path="tf_model", registered_model_name="BTC-TF-V2 MODEL", input_example=x_train[:5])