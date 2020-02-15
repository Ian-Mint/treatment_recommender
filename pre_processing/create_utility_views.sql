drop materialized view sepsis_admissions_data;
create materialized view sepsis_admissions_data as
    select
        a.admittime,
        a.hadm_id
    from
        admissions a
inner join sepsis_admissions sa on a.hadm_id = sa.hadm_id;