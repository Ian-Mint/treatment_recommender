import pickle
import pandas as pd
import numpy as np
from typing import Union
from keras.preprocessing.sequence import pad_sequences

import config


class Data:
    def __init__(self):
        # load static features
        self.demographics = load_demographics(config.demographics_path)
        self.elixhauser = load_pickle(config.elixhauser_path)

        # load time-series features and labels
        self.fluids = load_pickle(config.fluids_path)
        self.vasopressin = load_pickle(config.vasopressin_path)
        self.features = load_pickle(config.main_data_path)

        self.hadm_ids = set(self.features.keys())
        self.bad_hadm_ids = set()  # used during data validation
        self.maxlen = max((x.shape[0] for x in self.features.values()))
        self.process_time_series_data()

        # Reduce elixhauser and demographics to include keys in time-series data
        self.demographics = {k: v for k, v in self.demographics.items() if k in self.fluids.keys()}
        self.elixhauser = {k: v for k, v in self.elixhauser.items() if k in self.fluids.keys()}
        self.features = pad_sequences(self.features.values(), maxlen=self.maxlen, dtype='float', value=np.nan)
        # TODO: nan might need to be replaced -> self.time_series[self.time_series.isnan()] = -1

    def process_time_series_data(self):
        """
        Concatenate the vasopressin and fluid labels onto the time_series data.

        Then, drop the *first* label time step and the *last* feature time step. This is because labels will correspond
        to the feature data from the previous time step. So, in the first time step, the label is from the second time
        step, and in the last time step, we have no label, so we drop that step. This also addresses the issue of
        dosages going to zero if the patient dies.
        """
        for k in self.hadm_ids:
            time_chunks = self.features[k].shape[0]

            # validate data lengths
            bad_id_v, self.vasopressin[k] = self.validate_time_series(k, time_chunks, self.vasopressin[k])
            bad_id_fl, self.fluids[k] = self.validate_time_series(k, time_chunks, self.fluids[k])
            if bad_id_v or bad_id_fl:
                continue

            # append labels
            new_series = np.append(self.features[k], self.vasopressin[k].reshape(time_chunks, 1), axis=1)
            new_series = np.append(new_series, self.fluids[k].reshape(time_chunks, 1), axis=1)

            # shift data
            self.features[k] = drop_last_time_step(new_series)
            self.vasopressin[k] = drop_first_time_step(self.vasopressin[k])
            self.fluids[k] = drop_last_time_step(self.fluids[k])

        self.drop_bad_hadm_ids()

    def validate_time_series(self, hadm_id, time_chunks, labels):
        """
        Validates that the time series labels is of compatible size with the data. If the difference is small enough,
        it is corrected and returned. If the difference is too large, the hadm_id is dropped.

        :param hadm_id:
        :param time_chunks:
        :param labels:
        :return:
        """
        bad_id = False

        # Truncate from the end if the number of time steps is wrong (not sure why this happens)
        # Drop the hadm_id if the difference is greater than 1
        if time_chunks != labels.shape[0]:
            if abs(labels.shape[0] - time_chunks) > 1:
                bad_id = True
                self.bad_hadm_ids.add(hadm_id)
            else:
                labels = labels[:time_chunks]
        return bad_id, labels

    def drop_bad_hadm_ids(self):
        """
        There are two bad ids in the dataset
        TODO: figure out why!

        :return:
        """
        for hadm_id in self.bad_hadm_ids:
            self.demographics.pop(hadm_id)
            self.elixhauser.pop(hadm_id)
            self.fluids.pop(hadm_id)
            self.vasopressin.pop(hadm_id)
            self.features.pop(hadm_id)
            self.hadm_ids.discard(hadm_id)


def drop_first_time_step(array):
    if len(array.shape) == 1:
        return array[1:]
    elif len(array.shape) == 2:
        return array[1:, :]
    else:
        raise ValueError("array must be 1 or 2 dimensions")


def drop_last_time_step(array):
    if len(array.shape) == 1:
        return array[:-1]
    elif len(array.shape) == 2:
        return array[:-1, :]
    else:
        raise ValueError("array must be 1 or 2 dimensions")

def load_demographics(path: str):
    data = pd.read_csv(path, names=['hadm_id', 'sex', 'age'])
    data = data.set_index(data['hadm_id']).drop('hadm_id', axis=1)
    return data.to_dict('index')


def load_pickle(path: str) -> dict:
    with open(path, 'rb') as f:
        return pickle.load(f)

Data()