CREATE MATERIALIZED VIEW sepsis_demographics as
    select p.subject_id, p.gender, p.dob from patients p
    inner join (select subject_id from sepsis_admissions) sa on p.subject_id = sa.subject_id;
