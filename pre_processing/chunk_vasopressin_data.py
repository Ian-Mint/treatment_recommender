#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pickle

from pre_processing.config import *
from pre_processing.pre_process import VasopressinTimeStepsDf, load_admissions
from pre_processing.db_tools import connection

sepsis_admissions = load_admissions()
vasopressin_events = VasopressinTimeStepsDf(sepsis_admissions)
connection.close()

vasopressin_chunked = vasopressin_events.chunkify(sepsis_admissions, period_seconds)

with open('../data/vasopressin_chunked.pkl', 'wb') as f:
    pickle.dump(vasopressin_chunked, f)
