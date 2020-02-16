import pandas as pd
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
        hadm_id = df[groupby_header]
        assert hadm_id.nunique() == 1  # Something has gone terribly wrong if this asserts
        hadm_id = hadm_id.array[0]

        modifier = modifier_df.loc[modifier_df[groupby_header] == hadm_id, modifier_header].array
        if not len(modifier) == 1:
            raise AssertionError(
                f"The length of the modifier for group {df[groupby_header]} of header {groupby_header} equals {len(modifier)}"
            )
        return df[operate_on_header] - modifier[0]


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

        self[['starttime', 'endtime']] = result