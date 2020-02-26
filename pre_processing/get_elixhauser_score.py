#!/usr/bin/env python
# coding: utf-8
import pickle
import pandas as pd

from pre_processing.db_tools import connection


query = """
select
    eas.hadm_id,
    (eas.elixhauser_vanwalraven + eas.elixhauser_sid30 + eas.elixhauser_sid29) / 3.0 as score
from
    elixhauser_ahrq_score eas
inner join
    sepsis_admissions a on eas.hadm_id = a.hadm_id;
"""
df = pd.read_sql(query, connection)
connection.close()

df['score'] = df['score'].round().astype(int)
df.set_index(df['hadm_id'], inplace=True)
df.drop('hadm_id', axis=1, inplace=True)

scores = df.to_dict()
with open('../data/elixhauser_score.pkl', 'wb') as f:
    pickle.dump(scores['score'], f)
