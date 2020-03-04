import numpy as np
import keras
import pickle

from data_loader import Data
from model import build_model
import config


run_name = 'batch50-epochs100-lstm16.clean'


def create_dataset(d: np.ndarray, look_back: int = 1):
    assert d.ndim == 3

    d = d.reshape(d.shape[0] * d.shape[1])
    x, y = [], []
    for i in range(len(d) - look_back - 1):
        a = d[i:(i + look_back), 0]
        x.append(a)
        y.append(d[i + look_back, 0])
    return np.array(x), np.array(y)


data = Data()
assert not np.isnan(data.features).any()
batch_size = 11  # other prime factors of len(data.hadm_id)==2068 are 4, 47

n_samples = data.train.features.shape[0]
n_features = data.train.features.shape[2]
model = build_model(batch_size, data.maxlen, n_features)

# tensorboard_callback = keras.callbacks.TensorBoard(log_dir=config.tensorboard_log_path, histogram_freq=1)
checkpoint = keras.callbacks.callbacks.ModelCheckpoint('models/model', monitor='val_loss', verbose=0,
                                                       save_best_only=False, save_weights_only=False, mode='auto',
                                                       period=5)
history = model.fit(x=data.train.features,
                    y=data.train.vasopressin.reshape(n_samples, data.maxlen, 1),
                    batch_size=30,
                    epochs=100,
                    validation_split=0.1,
                    verbose=2,
                    shuffle=False,
                    callbacks=[checkpoint,])

model.save(f'models/{run_name}')
with open(f'logs/{run_name}.history') as f:
    pickle.dump(history, f)
