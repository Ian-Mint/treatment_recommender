import numpy as np
import keras
import pickle

from data_loader import Data
from model import build_model
import config

name = ''
lookback = 4
epochs = 20
batch_size = 30
width = 32  # LSTM width
layers = 2  # LSTM layers
run_name = f'batch{batch_size}-epochs{epochs}-lstm{layers}x{width}.{name}'

data = Data(lookback=lookback, batch_size=batch_size)
assert not np.isnan(data.features).any()

n_samples = data.train.features.shape[0]
n_features = data.train.features.shape[2]
model = build_model(width, batch_size, lookback, n_features, layers)

# tensorboard_callback = keras.callbacks.TensorBoard(log_dir=config.tensorboard_log_path, histogram_freq=1)
checkpoint = keras.callbacks.callbacks.ModelCheckpoint('models/model', monitor='val_loss', verbose=0,
                                                       save_best_only=False, save_weights_only=False, mode='auto',
                                                       period=5)
history = []
for i in range(epochs):
    history.append(
        model.fit(x=data.train.features,
                  y=data.train.vasopressin,
                  batch_size=batch_size,
                  epochs=1,
                  validation_data=(data.validate.features, data.validate.vasopressin),
                  verbose=2,
                  shuffle=False,
                  callbacks=[checkpoint, ])
    )
    model.reset_states()

model.save(f'models/{run_name}')
with open(f'logs/{run_name}.history', 'wb') as f:
    pickle.dump(history, f)
