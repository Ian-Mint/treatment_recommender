-- Ian Pegg
-- 2020-02-17

create materialized view sepsis_vital_signs as
    select ce.subject_id, ce.hadm_id, ce.icustay_id, ce.itemid, ce.charttime, ce.value, ce.valuenum, ce.valueuom from chartevents ce
    inner join (select hadm_id from sepsis_admissions) sa on (
        sa.hadm_id = ce.hadm_id
        and ce.error=0
        and ce.itemid in (select i.itemid from d_items i where i.category='Routine Vital Signs')
        );