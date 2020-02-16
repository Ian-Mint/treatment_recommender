import pandas as pd
from multiprocessing import Pool, cpu_count
from functools import partial
from typing import List


class TimeStepsDf(pd.DataFrame):
    @staticmethod
    def time_norm(df: pd.DataFrame,
                  modifier_df: pd.DataFrame,
                  groupby_header: str,
                  modifier_header: str,
                  operate_on_header: List[str]) -> pd.DataFrame:
        """
        To be passed to groupby.apply
        Subtracts the value of `modifier_df` at index specified by the current grouped value from the entire dataframe
        group passed as df.

        :param operate_on_header: The list of `df` headers on which to operate
        :param modifier_df: Must have one column titled the same as `modifier header` and one the same as `groupby_header`
                            Values under `groupby_header` must be unique or an `AssertionError` will be raised.
        :param df: The parameter passed by the pandas `groupby` function
        :param modifier_header: The header of the modifier to be applied to `df`
        :param groupby_header: The header by which `df` was grouped
        :return: `df` minus the modifier value
        """
        hadm_id = get_groupby_id(df, groupby_header)

        modifier = modifier_df.loc[modifier_df[groupby_header] == hadm_id, modifier_header].array
        if not len(modifier) == 1:
            raise AssertionError(
                f"The length of the modifier for group {df[groupby_header]} of header {groupby_header} equals {len(modifier)}"
            )
        return df[operate_on_header] - modifier[0]


def get_groupby_id(df, groupby_header):
    groupby_id = df[groupby_header]
    assert groupby_id.nunique() == 1  # Something has gone terribly wrong if this asserts
    return groupby_id.array[0]


class VasopressinTimeStepsDf(TimeStepsDf):
    def process(self, sepsis_admissions):
        gr = self.groupby('hadm_id')[['hadm_id', 'starttime', 'endtime']]
        result = gr.apply(
            self.time_norm,
            modifier_df=sepsis_admissions,
            groupby_header='hadm_id',
            modifier_header='admittime',
            operate_on_header=['starttime', 'endtime']
        )
        self[['starttime', 'endtime']] = timestamp_to_int_seconds_series(result)

    def chunkify(self, sepsis_admissions, period):
        self.period = period
        time_chunks_ref = sepsis_admissions[['hadm_id', 'time_chunks']].set_index('hadm_id')

        gr = self.groupby('hadm_id')[['hadm_id', 'amount', 'rate', 'starttime', 'endtime']]
        vasopressin_chunked = apply_parallel(gr,
                                             get_vasopressin_by_chunk,
                                             ref_df=time_chunks_ref,
                                             groupby_header='hadm_id',
                                             period=self.period)
        return vasopressin_chunked


def get_vasopressin_by_chunk(df: pd.DataFrame,
                             ref_df: pd.DataFrame,
                             groupby_header: str,
                             period: int, ) -> pd.Series:
    """

    :param df: the group passed by `pd.apply`
    :param ref_df: a series indexed by `groupby_header`, which contains the total number of chunks
    :param groupby_header: The header by which `df` was grouped
    """
    hadm_id = get_groupby_id(df, groupby_header)
    num_chunks = ref_df.loc[hadm_id].item()
    chunked_amount = []

    for i in range(num_chunks):
        amount = 0
        current_chunk = [i * period, (i + 1) * period]

        amount += amount_if_fully_contained_in_chunk(current_chunk, df)
        amount += amount_if_end_in_chunk_but_not_start(current_chunk, df)
        amount += amount_if_start_in_chunk_but_not_end(current_chunk, df)
        amount += amount_if_surrounds_chunk(current_chunk, df)

        chunked_amount.append(amount)

    return pd.Series(chunked_amount)


def amount_if_surrounds_chunk(current_chunk, df):
    surrounds_idx = surrounds(df, ['starttime', 'endtime'], current_chunk)
    surrounds_df = df.loc[surrounds_idx, ['rate']]
    chunk_length = current_chunk[1] - current_chunk[0]
    amount = sum(surrounds_df['rate'] * chunk_length / 3600)
    return amount


def amount_if_start_in_chunk_but_not_end(current_chunk, df):
    start_in_idx = first_in(df, ['starttime', 'endtime'], current_chunk)
    start_in_df = df.loc[start_in_idx, ['starttime', 'rate']]
    start_in_df['timediff'] = current_chunk[1] - start_in_df['starttime']
    amount = sum(start_in_df['rate'] * start_in_df['timediff'] / 3600)
    return amount


def amount_if_end_in_chunk_but_not_start(current_chunk, df):
    end_in_idx = second_in(df, ['starttime', 'endtime'], current_chunk)
    end_in_df = df.loc[end_in_idx, ['endtime', 'rate']]
    end_in_df['timediff'] = end_in_df['endtime'] - current_chunk[0]
    amount = sum(end_in_df['rate'] * end_in_df['timediff'] / 3600)
    return amount


def amount_if_fully_contained_in_chunk(current_chunk, df):
    full_in_idx = both_in(df, ['starttime', 'endtime'], current_chunk)
    amount = df.loc[full_in_idx, 'amount'].sum()
    return amount


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


def timestamp_to_int_seconds_series(s: pd.Series):
    return s.astype(int) // 1_000_000_000


def apply_parallel(gr, func, **kwargs):
    with Pool(cpu_count()) as p:
        ret_list = p.map(partial(func, **kwargs), [group for _, group in gr])
    return pd.concat(ret_list)
