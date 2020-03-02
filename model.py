import keras
import keras.backend as k
from keras.layers import Input, LSTM, Dense, Masking
import tensorflow.python as tf
import numpy as np


# sess = k.get_session()
# sess = tf.debug.LocalCLIDebugWrapperSession(sess)
# k.set_session(sess)

def build_model(batch_size, n_timesteps, n_features) -> keras.Model:
    model = keras.Sequential()

    model.add(Masking(mask_value=np.nan, input_shape=[n_timesteps, n_features]))
    model.add(LSTM(1, return_sequences=False))
    # model.add(Dense(1, input_shape=[n_timesteps, n_features]))

    model.compile(optimizer='rmsprop', loss='mse')

    print(model.summary())
    return model

# TODO: make it stateful. Requires a loop with a call to model.reset_states after every iteration (epoch)
