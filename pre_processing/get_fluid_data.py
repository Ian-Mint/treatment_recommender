#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np

from pre_processing.pre_process import VasopressinTimeStepsDf
from pre_processing.db_tools import *


PERIOD_SECONDS = 4 * 3600
sepsis_admissions = pd.read_sql("select * from sepsis_admissions_data", connection)
sepsis_inputevents_mv = pd.read_sql("select * from sepsis_inputevents_mv", connection)
item_ids = pd.read_sql("select itemid, label from d_items", connection)

vasopressin_ids = np.array([1136, 2445, 30051, 222315])
vasopressin_events = sepsis_inputevents_mv.loc[sepsis_inputevents_mv['itemid'].isin(vasopressin_ids)]
vasopressin_events = VasopressinTimeStepsDf(vasopressin_events)
vasopressin_events.process(sepsis_admissions)
