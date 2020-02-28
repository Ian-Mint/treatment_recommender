import pickle
import pandas as pd
import numpy as np
from typing import Union
# from keras.preprocessing import sequence

import config


class Data:
    def load(self):
        # load static features
        self.demographics = load_demographics(config.demographics_path)
        self.elixhauser = load_pickle(config.elixhauser_path)

        # load time-series features and labels
        self.fluids = load_pickle(config.fluids_path)
        self.vasopressin = load_pickle(config.vasopressin_path)
        self.time_series = load_pickle(config.main_data_path)

        # Reduce elixhauser and demographics to include keys in time-series data
        self.demographics = {k: v for k, v in self.demographics.items() if k in self.fluids.keys()}
        self.elixhauser = {k: v for k, v in self.elixhauser.items() if k in self.fluids.keys()}

    def combine(self):
        np.array()

    def to_sequence(self):
        pass

    def encode(self, data: Union[pd.Series, pd.DataFrame, np.ndarray], encoding: dict) \
            -> Union[pd.Series, pd.DataFrame, np.ndarray]:
        for k, v in encoding.items():
            data[data == k] = v
        return data


def load_demographics(path: str):
    data = pd.read_csv(path, names=['hadm_id', 'sex', 'age'])
    data = data.set_index(data['hadm_id']).drop('hadm_id', axis=1)
    return data.to_dict('index')


def load_pickle(path: str) -> dict:
    with open(path, 'rb') as f:
        return pickle.load(f)


d = Data()
d.load()
