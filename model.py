import keras
from keras.layers import Input, LSTM, Dense, Masking, TimeDistributed


def build_model(batch_size, n_timesteps, n_features) -> keras.Model:
    model = keras.Sequential()

    model.add(Masking(mask_value=-1, input_shape=(n_timesteps, n_features)))
    model.add(LSTM(16, return_sequences=True, input_shape=(n_timesteps, n_features)))
    model.add(TimeDistributed(Dense(1)))
    model.compile(optimizer='rmsprop', loss='mse')

    print(model.summary())
    return model

# TODO: make it stateful. Requires a loop with a call to model.reset_states after every iteration (epoch)
