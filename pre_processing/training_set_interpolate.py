import pickle
import pandas as pd
import numpy as np
from numba import njit
import time

with open('../data/training_set.pkl', 'rb') as f:
    lab_data = pickle.load(f)

for k, array in lab_data.items():
    if len(array.shape) == 1:
        array = array.reshape([1, len(array)])
    for i, row in enumerate(array):
        for j, element in enumerate(row):
            if element in (-1, np.nan, None):
                if j == 0:
                    array[i, j] = np.nan
                else:
                    array[i, j] = array[i, j - 1]
    lab_data[k] = array

with open('../data/training_set.pkl', 'wb') as f:
    pickle.dump(lab_data, f)
