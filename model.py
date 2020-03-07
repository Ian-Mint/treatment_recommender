import keras
from keras.layers import LSTM, Dense, Masking, TimeDistributed


def build_model(batch_size, n_timesteps, n_features) -> keras.Model:
    model = keras.Sequential([
        Masking(mask_value=-1, input_shape=(n_timesteps, n_features)),
        LSTM(
            8,
            return_sequences=True,
            input_shape=(n_timesteps, n_features),
            dropout=0,
            recurrent_dropout=0,
            stateful=False,
        ),
        TimeDistributed(Dense(1)),
    ])

    model.compile(optimizer='rmsprop', loss='mse')

    print(model.summary())
    return model

# TODO: make it stateful. Requires a loop with a call to model.reset_states after every iteration (epoch)
