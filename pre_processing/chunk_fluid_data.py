#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pickle

from pre_processing.config import *
from pre_processing.pre_process import VasopressinTimeStepsDf, load_admissions, load_sepsis_fluid_events_mv
from pre_processing.db_tools import connection

# sepsis_admissions, sepsis_inputevents_mv = load_admissions()
sepsis_fluid_events = load_sepsis_fluid_events_mv()
connection.close()
