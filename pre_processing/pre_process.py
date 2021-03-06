import pandas as pd
import numpy as np
import time
from multiprocessing import Pool, cpu_count
from functools import partial, reduce
from typing import List, Tuple, Dict
from abc import abstractmethod

from pre_processing.config import period_seconds
from pre_processing.db_tools import connection


class TimeSteps:
    def __init__(self, sepsis_admissions: pd.DataFrame):
        self.df = pd.DataFrame(self.load_df())
        self.sepsis_admissions = sepsis_admissions
        self.process()

    def drop_if_endtime_before_starttime(self):
        endtime_before_starttime_idx = self.df[self.df['endtime'] < self.df['starttime']].index
        self.df.drop(index=endtime_before_starttime_idx, inplace=True)

    def __instancecheck__(self, instance):
        result = True
        result &= isinstance(instance, type(self))
        result &= isinstance(instance, pd.DataFrame)
        return result

    def process(self):
        """
        Performs data cleaning and processing steps
        """
        self.drop_if_endtime_before_starttime()

        gr = self.df.groupby('hadm_id')[['hadm_id', 'starttime', 'endtime']]
        result = gr.apply(
            self.time_norm,
            groupby_header='hadm_id',
            modifier_header='admittime',
            operate_on_header=['starttime', 'endtime']
        )
        self.df[['starttime', 'endtime']] = timestamp_to_int_seconds_series(result)

    def time_norm(self,
                  df: pd.DataFrame,
                  groupby_header: str,
                  modifier_header: str,
                  operate_on_header: List[str]) -> pd.DataFrame:
        """
        To be passed to groupby.apply
        Subtracts the value of `modifier_df` at index specified by the current grouped value from the entire dataframe
        group passed as df.

        :param operate_on_header: The list of `df` headers on which to operate
        :param df: The parameter passed by the pandas `groupby` function
        :param modifier_header: The header of the modifier to be applied to `df`
        :param groupby_header: The header by which `df` was grouped
        :return: `df` minus the modifier value
        """
        modifier_df = self.sepsis_admissions
        hadm_id = self.get_groupby_id(df, groupby_header)

        modifier = modifier_df.loc[modifier_df[groupby_header] == hadm_id, modifier_header].array
        if not len(modifier) == 1:
            raise AssertionError(
                f"The length of the modifier for group {df[groupby_header]} of header {groupby_header} equals {len(modifier)}"
            )
        return df[operate_on_header] - modifier[0]

    @classmethod
    def chunk_group_worker(cls,
                           args,
                           ref_df: pd.DataFrame,
                           period: int, ) -> Tuple[int, np.ndarray]:
        """
        To be passed to a pool and operate on data grouped by `hadm_id`
        :param args: [0] The group name, [1] the dataframe on which to operate
        :param period: The chunk period in seconds
        :param ref_df: a series indexed by `groupby_header`, which contains the total number of chunks
        """
        hadm_id = args[0]
        assert isinstance(hadm_id, int)
        df = args[1]
        assert isinstance(df, pd.DataFrame)

        num_chunks = ref_df.loc[hadm_id].item()
        chunked_amount = []

        for i in range(num_chunks):
            amount = 0
            current_chunk = [i * period, (i + 1) * period]

            amount += cls.amount_if_fully_contained_in_chunk(current_chunk, df)
            amount += cls.amount_if_end_in_chunk_but_not_start(current_chunk, df)
            amount += cls.amount_if_start_in_chunk_but_not_end(current_chunk, df)
            amount += cls.amount_if_surrounds_chunk(current_chunk, df)

            chunked_amount.append(amount)

        return hadm_id, np.array(chunked_amount)

    @abstractmethod
    def load_df(self):
        raise NotImplementedError()

    @staticmethod
    def get_groupby_id(df, groupby_header):
        groupby_id = df[groupby_header]
        assert groupby_id.nunique() == 1  # Something has gone terribly wrong if this asserts
        return groupby_id.array[0]

    @staticmethod
    def amount_if_fully_contained_in_chunk(current_chunk, df):
        full_in_idx = both_in(df, ['starttime', 'endtime'], current_chunk)
        amount = df.loc[full_in_idx, 'amount'].sum()
        return amount

    @staticmethod
    def amount_if_surrounds_chunk(current_chunk, df):
        surrounds_idx = surrounds(df, ['starttime', 'endtime'], current_chunk)
        surrounds_df = df.loc[surrounds_idx, ['rate']]
        chunk_length = current_chunk[1] - current_chunk[0]
        amount = sum(surrounds_df['rate'] * chunk_length / 3600)
        return amount

    @staticmethod
    def amount_if_start_in_chunk_but_not_end(current_chunk, df):
        start_in_idx = first_in(df, ['starttime', 'endtime'], current_chunk)
        start_in_df = df.loc[start_in_idx, ['starttime', 'rate']]
        start_in_df['timediff'] = current_chunk[1] - start_in_df['starttime']
        amount = sum(start_in_df['rate'] * start_in_df['timediff'] / 3600)
        return amount

    @staticmethod
    def amount_if_end_in_chunk_but_not_start(current_chunk, df):
        end_in_idx = second_in(df, ['starttime', 'endtime'], current_chunk)
        end_in_df = df.loc[end_in_idx, ['endtime', 'rate']]
        end_in_df['timediff'] = end_in_df['endtime'] - current_chunk[0]
        amount = sum(end_in_df['rate'] * end_in_df['timediff'] / 3600)
        return amount

    def chunkify(self, period):
        time_chunks_ref = self.sepsis_admissions[['hadm_id', 'time_chunks']].set_index('hadm_id')

        gr = self.df.groupby('hadm_id')[['hadm_id', 'amount', 'rate', 'starttime', 'endtime']]
        df_chunked = apply_parallel(gr,
                                    self.chunk_group_worker,
                                    ref_df=time_chunks_ref,
                                    period=period)
        return df_chunked


class FluidTimeSteps(TimeSteps):
    """
    Implements methods to chunk fluid input
    """

    def load_df(self):
        query = """
            select
                im.hadm_id,
                im.starttime,
                im.endtime,
                im.itemid,
                im.amount,
                im.amountuom,
                im.rate,
                im.rateuom,
                im.orderid,
                im.linkorderid,
                im.patientweight
            from inputevents_mv im
            inner join sepsis_admissions sa on
                im.hadm_id = sa.hadm_id
            inner join itemid_fluid_intake_mv ifim on 
                im.itemid = ifim.itemid;
        """
        return pd.read_sql(query, connection)

    def process(self):
        self.remove_pre_admission_intake()
        self.convert_rate_to_ml_hr()
        self.convert_ml_kg_hr_to_ml_hr()
        self.add_rate_if_none()
        super(FluidTimeSteps, self).process()

    def remove_pre_admission_intake(self):
        drop_index = self.df.loc[self.df['amountuom'] == 'L'].index
        self.df.drop(index=drop_index, inplace=True)

    def convert_rate_to_ml_hr(self):
        df_ml_min_idx = (self.df['rateuom'] == 'mL/min')
        self.df.loc[df_ml_min_idx, 'rate'] = self.df.loc[df_ml_min_idx, 'rate'] * 60
        self.df.loc[df_ml_min_idx, 'rateuom'] = 'mL/hour'

    def convert_ml_kg_hr_to_ml_hr(self):
        df_weight_rate_idx = (self.df['rateuom'] == 'mL/kg/hour')
        self.df.loc[df_weight_rate_idx, 'rate'] = self.df.loc[df_weight_rate_idx, 'rate'] * self.df.loc[
            df_weight_rate_idx, 'patientweight']
        self.df.loc[df_weight_rate_idx, 'rateuom'] = 'mL/hour'

    def add_rate_if_none(self):
        df_no_rate = self.df['rate'].isna()
        timediff = timestamp_to_int_seconds_series(self.df['endtime'] - self.df['starttime'])
        self.df.loc[df_no_rate, 'rate'] = (self.df.loc[df_no_rate, 'amount'] / timediff.loc[df_no_rate]) / 3600
        self.df.loc[df_no_rate, 'rateuom'] = 'mL/hour'


class VasopressinTimeSteps(TimeSteps):
    """
    Implements methods to chunk vasopressin data
    """

    def load_df(self):
        query = """
            select * from
                sepsis_inputevents_mv
            where
                itemid in (1136, 2445, 30051, 222315)
        """
        return pd.read_sql(query, connection)


class VitalsTimeSteps(TimeSteps):
    """
    Implements methods to chunk vital signs data
    """

    @classmethod
    def parse_blood_pressure(cls):
        pass

    @classmethod
    def parse_temperature(cls):
        pass

    # These tuples define the order in which to pick certain duplicate measurements
    bp_systolic_priority = (220050, 225309, 220179)
    bp_diastolic_priority = (220051, 225310, 220180)
    bp_mean_priority = (220052, 225312, 220181)
    temperature_priority = (223762, 223761, 226329, 227632, 227634)
    hr_priority = (220045, )
    category_priorities = {
        'bp_systolic': bp_systolic_priority,
        'bp_diastolic': bp_diastolic_priority,
        'bp_mean': bp_mean_priority,
        'temperature': temperature_priority,
        'hr': hr_priority,
    }

    def load_df(self):
        itemids = ', '.join((str(v) for v in reduce(tuple.__add__, self.category_priorities.values())))
        query = f"""
            select 
                hadm_id,
                itemid,
                charttime,
                valuenum as value,
                valueuom as unit
            from
                sepsis_vital_signs_filtered
            where itemid in ({itemids}) 
        """
        return pd.read_sql(query, connection)

    def process(self):
        """
        Performs data cleaning and processing steps
        Needs to override the base because Vitals has charttime only, not start and end time
        """
        self.df.dropna(inplace=True)
        self.farenheit_to_celcius('value')
        self.group_measurements()

        gr = self.df.groupby('hadm_id')[['hadm_id', 'charttime']]
        result = gr.apply(
            self.time_norm,
            groupby_header='hadm_id',
            modifier_header='admittime',
            operate_on_header=['charttime']
        )
        self.df['charttime'] = timestamp_to_int_seconds_series(result)

    def group_measurements(self):
        """
        Adds a 'group' column for measurements that are duplicated.
        Later, pick the measurement of highest priority in each group for each chunk.
        """
        self.df['group'] = np.nan
        for category_name, category_list in self.category_priorities.items():
            self.df.loc[self.df['itemid'].isin(category_list), 'group'] = category_name

        assert all(self.df['group'].notna())

    def farenheit_to_celcius(self, column):
        f_filter = self.df['unit'] == '?F'

        self.df.loc[f_filter, column] = (self.df.loc[f_filter, column] - 32) / 1.8
        self.df.loc[f_filter, 'unit'] = '?C'

    def chunkify(self, period):
        time_chunks_ref = self.sepsis_admissions[['hadm_id', 'time_chunks']].set_index('hadm_id')

        gr = self.df[['hadm_id', 'itemid', 'value', 'charttime', 'group']].groupby('hadm_id')
        df_chunked = apply_parallel(gr,
                                    self.chunk_group_worker,
                                    ref_df=time_chunks_ref,
                                    period=period)
        return df_chunked

    @classmethod
    def chunk_group_worker(cls,
                           args,
                           ref_df: pd.DataFrame,
                           period: int, ) -> Tuple[int, dict]:
        """
        To be passed to a pool and operate on data grouped by `hadm_id`
        :param args: [0] The group name, [1] the dataframe on which to operate
        :param period: The chunk period in seconds
        :param ref_df: a series containing the number of chunks for each hadm_id
        """
        t = time.time()

        hadm_id = args[0]
        assert isinstance(hadm_id, int)
        df = args[1]
        assert isinstance(df, pd.DataFrame)

        num_chunks = ref_df.loc[hadm_id].item()

        def chunk_items(groups_df: pd.DataFrame) -> pd.Series:
            assert isinstance(groups_df, pd.DataFrame)
            chunked_amount = []

            for i in range(num_chunks):
                current_chunk = [i * period, (i + 1) * period]

                in_idx = in_chunk(groups_df, 'charttime', current_chunk)
                amount = groups_df.loc[in_idx, 'value'].mean()

                # If amount is nan, replace with previous value, unless this is the first chunk
                if np.isnan(amount):
                    if chunked_amount:
                        chunked_amount.append(chunked_amount[-1])
                    else:
                        chunked_amount.append(np.nan)
                else:
                    chunked_amount.append(amount)
            return pd.Series(chunked_amount, name=groups_df['group'].iloc[0])

        gr = df[['group', 'itemid', 'charttime', 'value']].groupby('group').apply(chunk_items)
        gr = dict(zip(gr.index, gr.values))

        print(f"{hadm_id} | time: {time.time() - t:.2f} seconds")
        return hadm_id, gr


def in_chunk(df, column: str, comps: List[int]) -> pd.Series:
    assert len(comps) == 2
    assert isinstance(column, str)
    before_chunk_end = df[column] < comps[1]
    after_chunk_start = df[column] > comps[0]
    return before_chunk_end & after_chunk_start


def both_in(df, columns: List[str], comps: List[int]) -> pd.Series:
    assert len(comps) == 2
    starts_after_chunk_start = df[columns[0]] > comps[0]
    ends_before_chunk_end = df[columns[1]] < comps[1]
    return starts_after_chunk_start & ends_before_chunk_end


def first_in(df, columns: List[str], comps: List[int]) -> pd.Series:
    assert len(comps) == 2
    starts_in_chunk = (df[columns[0]] > comps[0]) & (df[columns[0]] < comps[1])
    ends_after_chunk = (df[columns[1]] > comps[1])
    return starts_in_chunk & ends_after_chunk


def second_in(df, columns: List[str], comps: List[int]) -> pd.Series:
    assert len(comps) == 2
    starts_before_chunk = df[columns[0]] < comps[0]
    ends_in_chunk = (df[columns[1]] > comps[0]) & (df[columns[1]] < comps[1])
    return starts_before_chunk & ends_in_chunk


def surrounds(df, columns: List[str], comps: List[int]) -> pd.Series:
    assert len(comps) == 2
    starts_before_chunk = df[columns[0]] < comps[0]
    ends_after_chunk = df[columns[1]] > comps[1]
    return starts_before_chunk & ends_after_chunk


def timestamp_to_int_seconds_series(s: pd.Series) -> pd.Series:
    # TODO: error in some version of pandas - needs more investigation
    return s.astype(int) // 1_000_000_000


def apply_parallel(gr, func, **kwargs):
    with Pool(cpu_count()) as p:
        ret_list = p.map(partial(func, **kwargs), [(name, df) for name, df in gr])
    return dict(ret_list)


def _debug_apply_parallel(gr, func, **kwargs):
    """
    Not actually parallel, used so that breakpoints can be set in `func`
    """
    ret_list = map(partial(func, **kwargs), [(name, df) for name, df in gr])
    return dict(ret_list)


def load_admissions():
    sepsis_admissions = pd.read_sql("select * from sepsis_admissions_data", connection)
    sepsis_admissions['total_time'] = sepsis_admissions['dischtime'] - sepsis_admissions['admittime']
    sepsis_admissions['time_chunks'] = (
            timestamp_to_int_seconds_series(sepsis_admissions['total_time']) // period_seconds + 1)

    return sepsis_admissions
