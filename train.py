import numpy as np
import keras

from data_loader import Data
from model import build_model
import config


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
batch_size = 11  # other prime factors of len(data.hadm_id)==2068 are 4, 47

n_samples = data.features.shape[0]
n_features = data.features.shape[2]
model = build_model(batch_size, data.maxlen, n_features)

# TODO: figure out where these nans are coming from
data.features[np.isnan(data.features)] = -1

history = model.fit(x=data.features,
                    y=data.vasopressin.reshape(n_samples, data.maxlen, 1),
                    batch_size=11,
                    epochs=200,
                    validation_split=0.1,
                    verbose=2,
                    shuffle=False,)

model.save('model/batch11-epochs200-lstm16')
