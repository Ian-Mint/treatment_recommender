-- Ian Pegg
-- 2020-02-17

drop materialized view if exists sepsis_vital_signs_filtered cascade;
create materialized view sepsis_vital_signs as
    select
        ce.subject_id,
        ce.hadm_id,
        ce.icustay_id,
        ce.itemid,
        ce.charttime,
        ce.value,
        ce.valuenum,
        ce.valueuom
    from
        chartevents ce
    inner join
        (select hadm_id from sepsis_admissions) sa on
            (
                sa.hadm_id = ce.hadm_id
                and ce.error=0
                and ce.itemid in (select i.itemid from d_items i where i.category='Routine Vital Signs')
            );


create materialized view sepsis_vital_signs_filtered as
    select
        svs.*
    from
        sepsis_vital_signs svs
    inner join
        (
            select
                itemid,
                count(itemid)
            from
                sepsis_vital_signs
            group by itemid
            having
                count(itemid) > 1000
        ) c
        on c.itemid = svs.itemid;
