-- Ian Pegg
-- 2020-02-17


DROP MATERIALIZED VIEW IF EXISTS sepsis_demographics CASCADE;
CREATE MATERIALIZED VIEW sepsis_demographics as
    select
        sa.hadm_id,
        p.gender,
        (select extract(epoch from (a.admittime - p.dob)) / (60*60*24*365)) :: int as age
    from
        patients p
    inner join
        sepsis_admissions sa on p.subject_id = sa.subject_id
    inner join
        admissions a on sa.hadm_id = a.hadm_id;
