import keras
from keras.layers import Input, LSTM, Dense
import numpy as np


def build_model(data_length) -> keras.Model:
    # Only process ids for which no data is nan. This will exclude most ids until we get to the start of their data.

    bp_input = Input(shape=(data_length, 1), name='bp')
    lstm = LSTM(32)(bp_input)

    # Possibly useful for combining the non-sequential with the sequential data
    # static_input = Input(shape=(5,), name='static_input')
    # x = keras.layers.concatenate([lstm, static_input])

    x = lstm
    vasopressin_output = Dense(1, activation='sigmoid', name='vasopressin_output')(x)
    # fluid_output = Dense(1, activation='sigmoid', name='fluid_output')(x)

    # Create model from inputs and outputs
    # model = keras.Model(inputs=[bp_input, static_input], outputs=[vasopressin_output, fluid_output])
    model = keras.Model(inputs=[bp_input, ], outputs=[vasopressin_output, ])

    # Compile - For a mean squared error regression problem. `loss_weights` control the impact of each output.
    # Different loss functions can be used by passing a dict or list to `loss`.
    model.compile(optimizer='rmsprop', loss='mse', loss_weights=[1, 0.2])

    print(model.summary())
    return model
