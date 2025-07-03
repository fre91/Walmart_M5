# import librariesx
from dataclasses import dataclass
import polars as pl
import numpy as np 
from polars import selectors as cs 

from sklearn.mixture import GaussianMixture

from datetime import date
from pathlib import Path
from package.datapreparation import DataPreparation
import os
from package.utils import get_path_to_latest_file

prodlocs                  = get_path_to_latest_file('3.interim', 'prodlocs')
sales                    = get_path_to_latest_file('3.interim', 'sales')

calender_features   = get_path_to_latest_file('3.interim', 'calender_features')
calender_event_bool_features    = get_path_to_latest_file('3.interim', 'calendar_event_bool_features')
snap_features            = get_path_to_latest_file('3.interim', 'snap_features')
# price_feature = get_path_to_latest_file('price')
calender_event_gmm_features   = get_path_to_latest_file('3.interim', 'calendar_event_gmm_features')
 
prodlocs_output = (
    DataPreparation(prodlocs)
    .load_data(lazy=True)
)

#prodlocs_output.collect().result.head(4)
snap_output = (
    DataPreparation(snap_features)
    .load_data(lazy=True)
)
calender_features_output = (
    DataPreparation(calender_features)
    .load_data(lazy=True)
)
calender_event_bool_features_output = (
    DataPreparation(calender_event_bool_features)
    .load_data(lazy=True)
)
calender_event_gmm_features_output = (
    DataPreparation(calender_event_gmm_features)
    .load_data(lazy=True)
)

sales_output = (
    DataPreparation(sales)
    .load_data(lazy=True)
)

#sales_output.collect().result.head(4)

regressors_output = (
    snap_output
    .join(
        calender_features_output
        , on = ['date']
        , how = 'left'
    )
    .join(
        calender_event_bool_features_output
        , on = ['date']
        , how = 'left'
    )
    .join(
        calender_event_gmm_features_output
        , on = ['date']
        , how = 'inner'
    )
)

regressor_selection = (
     regressors_output
     .modify_data(
        lambda data:
            data
            .select(
                 pl.col('date')
                 , pl.col('state').alias('state_id')
                 , pl.col('trend')
                 , pl.col('trend_guassian_splines')
                 , pl.col('wday_guassian_splines')
                 , pl.col('weekday_bool')
                 , pl.col('year_month_trigonometric')
                 , pl.col('holiday_bool') 
                 , pl.col('holiday_pre_post_guassian_splines')
                 , pl.col('snap_bool')
             )
    )
)

#regressor_selection.collect().result.head(4)

DesignMatrix = (
    sales_output
    .load_data(lazy=True)
    .join(
            (
                prodlocs_output
                .select_columns(['id', 'item_id','dept_id','cat_id','store_id','state_id'])
            )
        , on = ['id']
        , how = 'inner'
    )
    .join(
        regressor_selection
        , on = ['date', 'state_id']
        , how = 'inner'
    )
)



#DesignMatrix.collect()
#DesignMatrix.result.head(10)
# Update so the we write a sorted parquest file on id and date

DesignMatrix.write_parquet(
        sink=True,
        name='designmatrix',
        path='4.processed',
        subfolder= 'designmatrix'
)