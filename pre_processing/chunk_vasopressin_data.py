#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import pickle

from pre_processing.pre_process import VasopressinTimeStepsDf, timestamp_to_int_seconds_series
from pre_processing.db_tools import *


period_seconds = 4 * 3600

sepsis_admissions = pd.read_sql("select * from sepsis_admissions_data", connection)
sepsis_inputevents_mv = pd.read_sql("select * from sepsis_inputevents_mv", connection)
item_ids = pd.read_sql("select itemid, label from d_items", connection)

sepsis_admissions['total_time'] = sepsis_admissions['dischtime'] - sepsis_admissions['admittime']
sepsis_admissions['time_chunks'] = (timestamp_to_int_seconds_series(sepsis_admissions['total_time']) // period_seconds + 1)

vasopressin_ids = np.array([1136, 2445, 30051, 222315])
vasopressin_events = sepsis_inputevents_mv.loc[sepsis_inputevents_mv['itemid'].isin(vasopressin_ids)]
vasopressin_events = VasopressinTimeStepsDf(vasopressin_events)
vasopressin_events.process(sepsis_admissions)

vasopressin_chunked = vasopressin_events.chunkify(sepsis_admissions, period_seconds)

with open('../data/vasopressin_chunked.pkl', 'wb') as f:
    pickle.dump(vasopressin_chunked, f)
