import keras
import numpy as np

from data_loader import Data
from model import build_model


np.random.seed(0)

bp_mean_index = 60

data = Data()
model = build_model(data.maxlen)

# model.fit([data.features, data.demographics, data.elixhauser], [data.vasopressin, data.fluids],
model.fit([data.features, ], [data.vasopressin, ],
          epochs=50, batch_size=32)
