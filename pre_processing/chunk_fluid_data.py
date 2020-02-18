#!/usr/bin/env python
# coding: utf-8
import pickle

from pre_processing.config import *
from pre_processing.pre_process import load_admissions, FluidTimeSteps
from pre_processing.db_tools import connection

sepsis_admissions = load_admissions()
fluid_events = FluidTimeSteps(sepsis_admissions)
connection.close()

fluids_chunked = fluid_events.chunkify(period_seconds)

with open('../data/fluids_chunked.pkl', 'wb') as f:
    pickle.dump(fluids_chunked, f)
