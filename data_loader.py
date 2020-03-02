import pickle
import random
from functools import reduce
from typing import Union, List, Tuple
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences

import config

np.random.seed(0)
bp_mean_index = 60


class Data:
    def __init__(self, splits: Tuple[float] = (0.7, 0.1, 0.2)):
        """
        Loads, cleans, and splits the data for use in sequence training
        :param splits: tuple(<train>, <validate>, <test>)
        """
        assert sum(splits) == 1
        self.splits = splits

        # load static features
        self.demographics = load_demographics(config.demographics_path)
        self.elixhauser = load_pickle(config.elixhauser_path)

        # load time-series features and labels
        self.fluids = load_pickle(config.fluids_path)
        self.vasopressin = load_pickle(config.vasopressin_path)
        self.features = load_pickle(config.main_data_path)

        self.hadm_ids = list(self.features.keys())
        self._bad_hadm_ids = set()  # used during data validation

        self.maxlen = max((x.shape[0] for x in self.features.values()))
        self._process_time_series_data()
        self._split_hadm_ids()

        # Reduce elixhauser and demographics to include keys in time-series data
        self.demographics = {k: v for k, v in self.demographics.items() if k in self.hadm_ids}
        self.elixhauser = {k: v for k, v in self.elixhauser.items() if k in self.hadm_ids}

        # convert to lists, then truncate all starting nans, then post-pad
        self._convert_dicts_to_lists()
        self._truncate_to_first_valid_data()
        self._pad_time_series(padding='post')

        self.elixhauser = np.array(self.elixhauser)
        self.demographics = np.array(self.demographics)
        # TODO: nan might need to be replaced -> self.features[self.features.isnan()] = -1
        # TODO: May need to add processing to pick the first time step where there are no nans
        # TODO: consider scaling using sklearn.MinMaxScaler

        self.train = Split(self, self.train_id_idx)
        self.validate = Split(self, self.validation_id_idx)
        self.test = Split(self, self.test_id_idx)

    def _pad_time_series(self, padding='post'):
        """
        Pads all of the timeseries data to the same length. Pad parameters can be set in `kwargs`
        """
        kwargs = {
            'maxlen': self.maxlen,
            'dtype': float,
            'value': -1,  # nan does not work with keras Masking layers
            'padding': padding,  # post is required for CuDNN
        }
        self.features = pad_sequences(self.features, **kwargs)
        self.vasopressin = pad_sequences(self.vasopressin, **kwargs)
        self.fluids = pad_sequences(self.fluids, **kwargs, )

    # noinspection PyTypeChecker
    def _truncate_to_first_valid_data(self):
        """
        Looks for the first not-nan value across all features and labels for each hadm-id. In practice, truncations
        happen because there are missing values in `features`. The labels are 100% or nearly 100% complete.
        """
        first_idx_vasopressin = first_not_nan_idx(self.vasopressin)
        first_idx_fluids = first_not_nan_idx(self.fluids)
        first_idx_features = np.array([first_not_nan_idx(x).max() for x in self.features])
        max_idx = reduce(np.maximum, [first_idx_vasopressin, first_idx_fluids, first_idx_features])

        for i, start_idx in enumerate(max_idx):
            self.vasopressin[i] = self.vasopressin[i][start_idx:]
            self.fluids[i] = self.fluids[i][start_idx:]
            self.features[i] = self.features[i][start_idx:]


    # noinspection PyTypeChecker
    def _convert_dicts_to_lists(self):
        """
        Converts all of the dictionaries to lists in order of `hadm_ids`. `hadm_ids` are shuffled randomly.
        """
        random.shuffle(self.hadm_ids)

        self.features = dict_to_list_by_key(self.features, self.hadm_ids)
        self.vasopressin = dict_to_list_by_key(self.vasopressin, self.hadm_ids)
        self.fluids = dict_to_list_by_key(self.fluids, self.hadm_ids)
        self.demographics = dict_to_list_by_key(self.demographics, self.hadm_ids)
        self.elixhauser = dict_to_list_by_key(self.elixhauser, self.hadm_ids)

    def _split_hadm_ids(self):
        """
        Based on the number of admission ids, randomly selects a subset for training, testing and validation based on
        `self.splits`, which should be fractional inputs adding to `1`. Outputs are `test_id_idx`, `validation_id_idx`
        and `train_id_idx` where each of these is a numpy array sorted in ascending order.
        """
        remaining_id_idx = set(range(len(self.hadm_ids)))
        n_train = int(len(remaining_id_idx) * self.splits[0])
        n_validate = int(len(remaining_id_idx) * self.splits[1])

        self.train_id_idx = set(np.random.choice(list(remaining_id_idx), size=n_train, replace=False))
        remaining_id_idx = remaining_id_idx.difference(self.train_id_idx)
        self.train_id_idx = np.array(sorted(list(self.train_id_idx)))

        self.validation_id_idx = set(np.random.choice(list(remaining_id_idx), size=n_validate, replace=False))
        remaining_id_idx = remaining_id_idx.difference(self.validation_id_idx)
        self.validation_id_idx = np.array(sorted(list(self.validation_id_idx)))

        self.test_id_idx = remaining_id_idx
        self.test_id_idx = np.array(sorted(list(self.test_id_idx)))

    def _process_time_series_data(self):
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
                self._bad_hadm_ids.add(hadm_id)
            else:
                labels = labels[:time_chunks]
        return bad_id, labels

    def drop_bad_hadm_ids(self):
        """
        There are two bad ids in the dataset
        TODO: figure out why!
        """
        for hadm_id in self._bad_hadm_ids:
            self.demographics.pop(hadm_id)
            self.elixhauser.pop(hadm_id)
            self.fluids.pop(hadm_id)
            self.vasopressin.pop(hadm_id)
            self.features.pop(hadm_id)
            self.hadm_ids.remove(hadm_id)


class Split:
    def __init__(self, data: Data, sample_indexes: Union[List[int], np.ndarray]):
        self.hadm_ids = np.array(data.hadm_ids)[sample_indexes]
        self.features = np.array(data.features)[sample_indexes]
        self.fluids = np.array(data.fluids)[sample_indexes]
        self.vasopressin = np.array(data.vasopressin)[sample_indexes]
        self.demographics = np.array(data.demographics)[sample_indexes]
        self.elixhauser = np.array(data.elixhauser)[sample_indexes]


def first_not_nan_idx(array: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
    """
    Returns the highest index among features and labels that is nan, plus 1. This is where the data will be sliced.

    :param array: 2-d array
    :return: an index
    """
    if isinstance(array, np.ndarray):
        assert array.ndim == 2
        array = array.astype(float).T
        some_nan = np.isnan(array).any()
    elif isinstance(array, list):
        some_nan = any([np.isnan(x).any() for x in array])
    else:
        raise TypeError(f"{type(array)} not valid for Union[np.ndarray, List[np.ndarray]]")

    dummy_min = 3e9
    ret = np.array([np.min(np.where(np.logical_not(np.isnan(array[i]))), initial=dummy_min) for i in range(len(array))])
    ret[ret == dummy_min] = 0
    return ret


def dict_to_list_by_key(d: dict, keys) -> list:
    return [d[k] for k in keys]


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
