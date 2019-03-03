# -*- coding: utf-8 -*-
"""
Created on Thu May 17 15:12:53 2018

@author: Shugui
"""

# =============================================================================
# below feature lists works for Others_table. 
# 'OTHER_MILEAGE_VAR'#69;'SUM_OTHER_TIMES_BY_MON'#75
# =============================================================================

### Numerical columns ###
floatCol=['ACTUAL_AMOUNT_TIMES',
'ASC_CHANGES_COUNT_BY_MON',
'AVG_INTERVAL_DAYS_BY_MON',
'AVG_INTERVAL_MILEAGE_BY_MON',
'AVG_REPAIR_AMOUNT_BY_MON',
'AVG_REPAIR_INTERVAL_DAYS',
'AVG_REPAIR_INTERVAL_MILEAGE',
'CIVILIZED_DR_SCORE',
'COST_DR_SCORE',
'DIFF_MILE',
'FAKE_AMOUNT_TIMES_BY_MON',
'FIRST_MAINTEN_ASC',
'GREEN_DR_SCORE',
'HOBBY',
'HOLIDAY_POINTNO',
'HOLIDAY_STOPCOUNT',
'HOLIDAY_STOPDAYS',
'IF_COMMERCIAL_INSU',
'IF_COMPULSORY_INSU',
'INDEXN',
'IS_BLACKLIST',
'IS_COMPANY_PS',
'IS_HOME_PS',
'IS_INTERNET',
'IS_MATCHXML',
'IS_UPDATELOG',
'LARGE_AMOUNT_TIMES',
'LAST_ASC_MONTHRATE',
'LAST_INTERVAL_DAYS',
'LAST_INTERVAL_MILEAGE',
'LAST_MAINTAIN_MILEAGE',
'LAST_MAINTEN_ASC',
'LAST_REPAIR_AMOUNT',
'LAST_REPAIR_MILEAGE',
'MAINTAIN_MILEAGE',
'MAX_MILEAGE_BY_MONEND',
'MOST_ASC_MONTHRATE',
'MT_DAYS_AIRCONDITION_FILTER',
'MT_DAYS_AIR_FILTER',
'MT_DAYS_ENGINEOIL',
'MT_DAYS_ENGINEOIL_FILTER',
'MT_DAYS_FUELOIL_ADDITIVE',
'MT_DAYS_FUELOIL_FILTER',
'MT_MILE_AIRCONDITION_FILTER',
'MT_MILE_AIR_FILTER',
'MT_MILE_ENGINEOIL',
'MT_MILE_ENGINEOIL_FILTER',
'MT_MILE_FUELOIL_ADDITIVE',
'MT_MILE_FUELOIL_FILTER',
'MT_OVERDUE',
'MT_OVERDUE_5TIMES',
'NEAREST_HOME_4S',
'NEAREST_HOME_DIS',
'NEAREST_WORK_4S',
'NEAREST_WORK_DIS',
'NO_OVERTIMES',
'OTHER_MILEAGE_VAR',
'OVERTIMES',
'RESIDUAL_VALUE_SCORE',
'SAFE_DR_SCORE',
'SMALL_AMOUNT_TIMES',
'SUM_OTHER_TIMES_BY_MON',
'SUM_REPAIR_TIMES_BY_MON',
'TOTAL_REPAIR_AMOUNT',
'VEHICLE_PURCHASE_PRICE',
'VEH_AGE',
'WEEKEND_STOPCOUNT',
'WEEKEND_STOPDAYS',
'WEEKEND_STOPSECONDS',
'WORKDAY_STOPCOUNT',
'WORKDAY_STOPDAYS',
'WORKDAY_STOPSECONDS',
'ZEBRA_SCORE']

### attribute columns ###
charCol=['BRAND',
'CUST_TYPE',
'FAMILY_INCOME',
'GENDER',
'INDUSTRY',
'IS_INSU_PLAN',
'LICENSE',
'LINK_CITY',
'MATERIAL_CN',
'MODEL',
'MT_STATUS',
'MT_TAG',
'PARTS_TYPE_ID',
'PURCHASE_ASC',
'PURCHASE_CITY',
'REMAIN_MONTH',
'SERIES',
'VEHICLE_TAG',
'VEH_TYPE',
'VIN',
'level']

#'''target column'''
Y_col=['IS_LEAVE']