# import librariesx
import polars as pl
import numpy as np 
from polars import selectors as cs 

from sklearn.mixture import GaussianMixture

from datetime import date
from pathlib import Path
from package.datapreparation import DataAnalytics,DataPreparation
import os



def create_trading_periods(df:pl.DataFrame, ordinal_col , n_components=3):
    
    
    if isinstance(df, pl.LazyFrame):
        df = df.collect()
    elif not isinstance(df, pl.DataFrame):
        raise TypeError("Input must be either a Polars DataFrame or LazyFrame")
    
    X = df[ordinal_col].to_numpy().reshape(-1,1)
    
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(X)

    latent_features = gmm.predict_proba(X)
    col_name = [f'{ordinal_col}_{i+1}'for i in range(n_components)]

    return df.with_columns(
        pl.from_numpy(latent_features, col_name).to_struct().alias(f'{ordinal_col}_regimes')
    ).lazy()

def get_path_to_latest_file(subfolder_name):
    project_root    = Path(__file__).parent.parent
    base_path       = project_root / "data/3.interim"
    subfolder_name  = subfolder_name
    folder_path     = os.path.join(base_path, subfolder_name)
    file_list       = os.listdir(folder_path)
    ## Later apply logic for name cohesion and date coheasion
    path            = os.path.join(folder_path, max(file_list))
    
    return path

def delist(data, x, split='_', index =[0], col_name='name'):
    return data.with_columns(
            pl.concat_str(
                (pl.col(x).str.split(split).list.get(idx) for idx in index)
                , separator = '_'
            ).alias(col_name)
        )


prodloc                  = get_path_to_latest_file('prodlocs')
sales                    = get_path_to_latest_file('sales')

calender_trig_features   = get_path_to_latest_file('calender_trig_features')
holiday_bool_features    = get_path_to_latest_file('holiday_bool_features')
snap_features            = get_path_to_latest_file('snap_features')
# price_feature = get_path_to_latest_file('price')
event_pre_post_effect_guassian_features   = get_path_to_latest_file('event_pre_post_effect_guassian_features')
# regression weight vector weighting the 


snap_output = (
    DataPreparation(snap_features)
    .load_data(lazy=True)
)
calender_trig_features_output = (
    DataPreparation(calender_trig_features)
    .load_data(lazy=True)
)
holiday_output = (
    DataPreparation(holiday_bool_features)
    .load_data(lazy=True)
)
guassian_event_output = (
    DataPreparation(event_pre_post_effect_guassian_features)
    .load_data(lazy=True)
)

sales_prep = DataAnalytics(
    get_path_to_latest_file('sales')
)

regressors_output = (
    snap_output
    .join(
        calender_trig_features_output
        , on = 'date'
        , how = 'left'
    )
    .join(
        holiday_output
        , on = 'date'
        , how = 'left'
    )
    .join(
        guassian_event_output
        , on = 'date'
        , how = 'inner'
    )
)


#print(regressors_output.collect().result)

regressor_selection = (
     regressors_output
     .modify_data(
        lambda data:
            data
            .select(
                 pl.col('date')
                 , pl.col('state')
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




DesignMatrix = (
    sales_prep
    .load_data(lazy=True)
    .calculate_out_of_stock_periods(binomial_threshold=0.0001)
    .modify_data(
        lambda data:
            data 
           .pipe(delist,x='id',split='_',index=[3],col_name='state')
        #    .pipe(delist,x='id',split='_',index=[0,1],col_name='category_dep')
        #    .pipe(delist,x='id',split='_',index=[3,4],col_name='state_store')
        #    .pipe(delist,x='id',split='_',index=[3],col_name='state')
        #    .pipe(delist,x='id',split='_',index=[0,1,2,3],col_name='item_state')
        #    .pipe(delist,x='id',split='_',index=[2],col_name='item')

    )
    .join(
        regressor_selection
        , on = ['date', 'state']
        , how = 'inner'
    )
)

# Update so the we write a sorted parquest file on id and date

DesignMatrix.write_parquet(
        sink=True,
        name='designmatrix',
        path='4.processed',
        subfolder= 'designmatrix'
)