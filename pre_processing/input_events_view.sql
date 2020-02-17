-- Ian Pegg
-- 2020-02-17

drop materialized view sepsis_inputevents_mv;
create materialized view sepsis_inputevents_mv as
    select
        imv.hadm_id,
        imv.starttime,
        imv.endtime,
        imv.itemid,
        imv.amount,
        imv.amountuom,
        imv.rate,
        imv.rateuom,
        imv.orderid,
        imv.linkorderid,
        imv.totalamount,
        imv.totalamountuom,
        imv.statusdescription
    from inputevents_mv imv
inner join sepsis_admissions sa on
    imv.hadm_id = sa.hadm_id
    and (imv.ordercategoryname='02-Fluids (Crystalloids)'
         OR imv.secondaryordercategoryname='02-Fluids (Crystalloids)');

-- TODO: Figure out how to filter inputevents_cv for fluids
-- create materialized view sepsis_inputevents_cv as
--     select
--         icv.hadm_id,
--         icv.charttime,
--         icv.itemid,
--         icv.amount,
--         icv.amountuom,
--         icv.rate,
--         icv.rateuom,
--         icv.orderid,
--         icv.linkorderid
--     from
--         inputevents_cv icv
-- inner join sepsis_admissions sa on
--     icv.hadm_id = sa.hadm_id
--     and (icv.ordercategoryname='02-Fluids (Crystalloids)'
--          OR icv.secondaryordercategoryname='02-Fluids (Crystalloids)');