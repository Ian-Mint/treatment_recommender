from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense, Masking, TimeDistributed, ThresholdedReLU


def build_model(width, batch_size, n_timesteps, n_features, n_layers=1,
                dropout=0, recurrent_dropout=0, output_threshold=0) -> keras.Model:
    """

    :param output_threshold: output values below this will be set to zero
    :param n_layers:
    :param width:
    :param batch_size:
    :param n_timesteps:
    :param n_features:
    :param dropout:
    :param recurrent_dropout:
    :return:
    """
    lstm_kwargs = {
        'return_sequences': True,
        'dropout': dropout,
        'recurrent_dropout': recurrent_dropout,
        'stateful': True,
    }

    model = keras.Sequential()
    model.add(Masking(mask_value=-1, batch_input_shape=[batch_size, n_timesteps, n_features]))

    for i in range(n_layers):
        model.add(LSTM(width, **lstm_kwargs))

    model.add(TimeDistributed(Dense(1)))
    model.add(TimeDistributed(ThresholdedReLU(1)))
    model.compile(optimizer='rmsprop', loss='mse')

    print(model.summary())
    return model
