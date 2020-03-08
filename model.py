import keras
from keras.layers import LSTM, Dense, Masking, TimeDistributed


def build_model(width, batch_size, n_timesteps, n_features, n_layers=1,
                dropout=0, recurrent_dropout=0,) -> keras.Model:
    """

    :param n_layers:
    :param width:
    :param batch_size:
    :param n_timesteps:
    :param n_features:
    :param dropout:
    :param recurrent_dropout:
    :return:
    """
    model = keras.Sequential()
    model.add(Masking(mask_value=-1, batch_input_shape=[batch_size, n_timesteps, n_features]))
    for i in range(n_layers):
        model.add(
            LSTM(width,
                 return_sequences=True,
                 dropout=dropout,
                 recurrent_dropout=recurrent_dropout,
                 stateful=True,)
        )
    model.add(TimeDistributed(Dense(1)))
    model.compile(optimizer='rmsprop', loss='mse')

    print(model.summary())
    return model

# TODO: make it stateful. Requires a loop with a call to model.reset_states after every iteration (epoch)
