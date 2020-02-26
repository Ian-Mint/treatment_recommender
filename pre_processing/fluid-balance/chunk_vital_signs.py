#!/usr/bin/env python
# coding: utf-8
import pickle

from pre_processing.config import *
from pre_processing.pre_process import load_admissions, VitalsTimeSteps
from pre_processing.db_tools import connection

sepsis_admissions = load_admissions()
vitals_events = VitalsTimeSteps(sepsis_admissions)
connection.close()

vitals_chunked = vitals_events.chunkify(period_seconds)
#
# with open('../data/vitals_chunked.pkl', 'wb') as f:
#     pickle.dump(vitals_chunked, f)
