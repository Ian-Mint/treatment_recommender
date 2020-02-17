#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pickle

from pre_processing.config import *
from pre_processing.pre_process import VasopressinTimeStepsDf, load_admissions, load_sepsis_inputevents_mv
from pre_processing.db_tools import connection

sepsis_admissions = load_admissions()
sepsis_inputevents_mv = load_sepsis_inputevents_mv()
connection.close()

vasopressin_ids = np.array([1136, 2445, 30051, 222315])
vasopressin_events = sepsis_inputevents_mv.loc[sepsis_inputevents_mv['itemid'].isin(vasopressin_ids)]
vasopressin_events = VasopressinTimeStepsDf(vasopressin_events)
vasopressin_events.process(sepsis_admissions)

vasopressin_chunked = vasopressin_events.chunkify(sepsis_admissions, period_seconds)

with open('../data/vasopressin_chunked.pkl', 'wb') as f:
    pickle.dump(vasopressin_chunked, f)
