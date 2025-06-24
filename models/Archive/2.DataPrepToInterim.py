"""
Data Preparation Pipeline for Walmart M5 Forecasting

This script transforms raw data into interim datasets for the Walmart M5 forecasting project.
It handles various data transformations including:
- Product location metadata processing
- Sales data normalization
- Event calendar feature engineering (pre and post events)
- SNAP (food stamp) features
- Calendar features with cyclical encoding

Author: Fredrik Hornell
"""

# import librariesx
import polars as pl
import numpy as np 
from polars import selectors as cs 

from datetime import date
from package.datapreparation import DataPreparation
import pyarrow 


from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from datetime import date

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





# Usage Example
DataPrepSalesRaw    = DataPreparation(r"/Users/fredrik.hornell/Python/Private/Walmart_M5/data/2. raw/DataPrepSalesRaw_20241215_111210.parquet")
DataPrepSalesRaw1   = DataPreparation(r"/Users/fredrik.hornell/Python/Private/Walmart_M5/data/2. raw/DataPrepSalesRaw_20241215_111210.parquet")
DataPrepProdLocRaw  = DataPreparation(r"/Users/fredrik.hornell/Python/Private/Walmart_M5/data/2. raw/DataPrepProdLocRaw_20241215_111210.parquet")
DataPrepSnapRaw     = DataPreparation(r"/Users/fredrik.hornell/Python/Private/Walmart_M5/data/2. raw/DataPrepCalendarRaw_20241215_111217.parquet")
DataPrepCalendarRaw = DataPreparation(r"/Users/fredrik.hornell/Python/Private/Walmart_M5/data/2. raw/DataPrepCalendarRaw_20241215_111217.parquet")
DataPrepCalendarRaw1 = DataPreparation(r"/Users/fredrik.hornell/Python/Private/Walmart_M5/data/2. raw/DataPrepCalendarRaw_20241215_111217.parquet")
DataPrepCalendarRaw2 = DataPreparation(r"/Users/fredrik.hornell/Python/Private/Walmart_M5/data/2. raw/DataPrepCalendarRaw_20241215_111217.parquet")
DataPrepPriceRaw    = DataPreparation(r"/Users/fredrik.hornell/Python/Private/Walmart_M5/data/2. raw/DataPrepPriceRaw_20241215_111217.parquet")

### product loacation metadata

"""
Creates a dataset with product location metadata including:
- First and last sales dates
- Total sales
- Generated date range for each product-location combination
"""
DataPrepProdLocInterim = (
    # Filter out sales where sales are 0 and group by id
    # Then calculate first and last sales date and total sales
    # Then generate a date range for each id
    DataPrepProdLocRaw
    .load_data(lazy=True)
    .join(
        DataPrepSalesRaw        
        .load_data(lazy=True)
        .modify_data(
            lambda data: data.select(
                pl.col('id'),
                pl.col('date').cast(pl.Date),
                pl.col('sales')
            )
            .filter(pl.col('sales') > 0)
            .group_by('id')
            .agg(
                pl.col('date').min().cast(pl.Date).alias('first_sales_date'),
                pl.col('date').max().cast(pl.Date).alias('last_sales_date'),
                pl.col('sales').sum().alias('total_sales')
        )
        )
        .modify_data(
            lambda data: data.with_columns(
                pl.col('first_sales_date')
                .map_elements(
                    lambda first_sales_date: pl.date_range(
                        start=first_sales_date,
                        end=date(2016, 5, 22), # This is the last date in the dataset
                        interval="1d",
                        eager=True
                    ).to_list()
                    , return_dtype=pl.List(pl.Date)
                )
                .alias('prodloc_daterange')
            )
        )
        , on='id'
        , how='inner'
    )
    .update_schema()
)

### prepare product location sales data constrained by the min/max sales date. 

DataPrepProdLocInterimCopy = DataPrepProdLocInterim

"""
Normalizes sales data by joining with product location date ranges
to ensure consistent date coverage across all products
"""
DataPrepSalesInterim = (
    # inner join sales with prodloc_daterange to normalize the daterange in 
    # DataPrepSalesRaw to the date range of each id defined
    # by the prodloc_daterange column in DataPrepProdLocInterim 
    DataPrepSalesRaw1
    .load_data(lazy=True)
    .join(
         DataPrepProdLocInterimCopy
        .select_columns(
            [
                'id',
                'prodloc_daterange'
            ]
        )
        .modify_data(
                lambda data: data.explode('prodloc_daterange').rename({'prodloc_daterange': 'date'})
            )
        , on=['id', 'date']
        , how='inner'
    )
    .update_schema()
)

DataPrepEventPrePostInterim = (
    DataPrepCalendarRaw
    .load_data(lazy=True)
    .modify_data(
        lambda data: (
        pl.concat(
            [
                data
                .select(
                    pl.col("date"),
                    pl.col("event_type_1").alias('event_type'),
                    pl.col("event_name_1").alias('event_name'),
                )
                ,
                data
                .select(
                    pl.col("date"),
                    pl.col("event_type_2").alias('event_type'),
                    pl.col("event_name_2").alias('event_name'), 
                )
            ]
        )
        )
        .drop_nulls()
        .select(
            pl.col('date')
            , (
                pl.lit('event').str.to_uppercase() +
                pl.lit('_') +
                pl.col('event_type').str.to_uppercase() +
                pl.lit('_') +
                pl.col('event_name').str.to_uppercase()
               
               ).alias('event')

        )
        .group_by('event')
        .agg(
            pl.col('date')
        )
    )
)

GaussianSplineEventLong = (
    DataPrepEventPrePostInterim
    .modify_data(
        lambda data:
            data
            .explode('date')
            .rename({'date':'event_date'})
            .with_columns(
                pl.date_ranges(
                    start=(pl.col('event_date') - pl.duration(days=4)),
                    end=(pl.col('event_date') + pl.duration(days=4))
                ).alias('date')
            )
            .explode('date')
            .with_columns(
                event_period =  pl.int_range(pl.len(), dtype=pl.UInt32).over('event','event_date')
            )
            .pipe(create_trading_periods,ordinal_col='event_period' ,n_components=3)
    )
)

GaussianSplineEventWide = (
    GaussianSplineEventLong
    .modify_data(
        lambda data:
            data.unnest("event_period_regimes")
    )
    .pivot_and_lazy(
        on='event'
        , index = 'date'
        , values = ['event_period_1','event_period_2','event_period_3']
    )
)

GaussianSplineEventWide2 = (
    DataPrepCalendarRaw1.load_data(lazy=True)
    .select_columns(['date'])
    .join(
        GaussianSplineEventWide
        , on = 'date'
        , how = 'left'
    )
    .modify_data(
       lambda data:
            data
            .fill_null(0)
            .select(
                pl.col('date')
                ,pl.struct(pl.all().exclude('date')).alias("event_struct")
        )
    )   
)


"""
Event Feature Engineering Pipeline

This section processes calendar events to create pre-event and post-event features:

Pre-Event Features (DataPrepCalendarInterimEventPreEffectsLong):
- Creates countdown features for 7 days before each event (6 days prior + event day)
- For each event, generates features like EVENT_TYPE_NAME_7 through EVENT_TYPE_NAME_1
  where the number indicates days until the event

Post-Event Features (DataPrepCalendarInterimEventPostEffectsLong):
- Creates count-up features for 7 days after each event
- For each event, generates features like EVENT_TYPE_NAME_1 through EVENT_TYPE_NAME_7
  where the number indicates days since the event

The pipeline:
1. Processes both event_type_1 and event_type_2 from calendar data
2. Standardizes event names in uppercase format (EVENT_TYPE_NAME)
3. Generates date ranges for pre/post event periods
4. Creates wide-format features through pivoting
5. Final output is two separate datasets for pre and post event effects

Example feature: EVENT_SPORTING_SUPERBOWL_3 (3 days until/after Super Bowl)
"""
holiday_day = ['NewYear',
 'LentStart',
 'Eid al-Fitr',
 'Thanksgiving',
 'OrthodoxEaster',
 'ColumbusDay',
 'Ramadan starts',
 'PresidentsDay',
 'IndependenceDay',
 'ValentinesDay',
 'SuperBowl',
 'Purim End',
 'Easter',
 'Chanukah End',
 'EidAlAdha',
 'LentWeek2',
 'OrthodoxChristmas',
 'Pesach End',
 'Christmas']


holiday = (
    DataPrepCalendarRaw1
    .load_data(lazy=True)
    .select_columns('date')
    .join(
        DataPrepCalendarRaw2
        .load_data(lazy=True)
        .modify_data(
            lambda data :
                data
                .select(pl.col('date'),pl.col('event_name_1'))
                .drop_nulls()
                .with_columns(pl.lit(1).alias('value'))
        )
        .pivot_and_lazy(
            index = 'date'
            , on= 'event_name_1'
            , values='value'
        )
        , on = 'date'
        , how= 'left'
    )
    .modify_data(
        lambda data :
            data
            .fill_null(0)
            .select(
                pl.col('date')
                , pl.struct(pl.all().exclude('date')).alias('holidays')
            )
    )
)


"""
Post-Event Feature Engineering Pipeline

This pipeline processes calendar events to create features tracking the aftermath of events:

1. Data Processing Steps:
   - Combines event_type_1 and event_type_2 from calendar data
   - Standardizes event names to uppercase format (EVENT_TYPE_NAME)
   - Creates a 7-day window after each event occurrence
   - Generates sequential counters (1-7) for days following each event

2. Feature Generation:
   - For each event, creates features like EVENT_TYPE_NAME_1 through EVENT_TYPE_NAME_7
   - The number suffix indicates days since the event occurred
   - Example: EVENT_SPORTING_SUPERBOWL_3 means "3 days after Super Bowl"

3. Output Format:
   - Final dataset is pivoted to wide format
   - Each row represents a date
   - Columns represent different event aftermath periods
   - Values are binary indicators (1 for active post-event day, 0 otherwise)

Parameters:
    DataPrepCalendarRaw: LazyFrame
        Input calendar data containing event information

Returns:
    LazyFrame
        Wide-format dataset with post-event effect features
"""

DataPrepEventPostInterim = (
    DataPrepCalendarRaw
    .load_data(lazy=True)
    .modify_data(
        lambda data: (
        pl.concat(
            [
                data
                .select(
                    pl.col("date"),
                    pl.col("event_type_1").alias('event_type'),
                    pl.col("event_name_1").alias('event_name'),
                )
                ,
                data
                .select(
                    pl.col("date"),
                    pl.col("event_type_2").alias('event_type'),
                    pl.col("event_name_2").alias('event_name'), 
                )
            ]
        )
        )
        .drop_nulls()
        .select(
            pl.col('date')
            , (
                pl.lit('event').str.to_uppercase() +
                pl.lit('_') +
                pl.col('event_type').str.to_uppercase() +
                pl.lit('_') +
                pl.col('event_name').str.to_uppercase()
               
               ).alias('event')

        )
        .group_by('event')
        .agg(
            pl.col('date')
        )
    )
    .update_schema()
)

DataPrepCalendarInterimEventPostEffectsLong = (
    DataPrepEventPostInterim
    .modify_data(
        lambda data:
            data
            .explode('date')
            .with_columns(
                pl.date_ranges(
                    start=(pl.col('date') + pl.duration(days=1)),
                    end=(pl.col('date') + pl.duration(days=7))
                ).alias('range')
            )
            .explode('range')
            .with_columns(
                (pl.lit(1) + pl.int_range(pl.len(), dtype=pl.UInt32).over('event','date')).cast(pl.String).alias("test")
            )
            .with_columns(
                pl.concat_str([pl.col('event') ,pl.lit('_'),pl.col('test')]).alias('feature')
            )
    )
)

DataPrepInterimEventPostEffectsFeatureWide = (
    DataPrepCalendarInterimEventPostEffectsLong
    .select_columns(pl.col('*').exclude('test'))
    .pivot_and_lazy(
        index = ['range','event']
        , on = 'feature'
        , aggregate_function='len'
    )
)

"""
Processes SNAP (Supplemental Nutrition Assistance Program) data
for California, Texas, and Wisconsin
"""
DataPrepSnapInterimLong = (
    DataPrepSnapRaw
    .load_data(lazy=True)
    .select_columns(['date', 'snap_CA', 'snap_TX', 'snap_WI'])
    .modify_data(
        lambda data:
            data
            .melt(id_vars='date')
            .rename({'variable': 'snap'})
            .with_columns(
                pl.col('snap').str.replace('snap_', '')
            )
            
    )
)

### datapreperations calender features

"""
Creates calendar features including:
- Basic date components (day of month, day of year)
- Cyclical encodings using sine and cosine transformations
- Both yearly and monthly cyclical features
"""
DataPrepCalenderInterim = (
    DataPrepCalendarRaw
    .load_data(lazy=True)
    .modify_data(
        lambda data:
            data
            .with_columns(
                pl.col("date").dt.day().alias("day_of_month"),
                pl.col("date").dt.ordinal_day().alias("day_of_year")
            )
            .select(pl.col('*').exclude(['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2', 'snap_CA', 'snap_WI', 'snap_TX']))
    )
    .modify_data(
        lambda data :
            data
            .with_columns(
                (pl.lit(2)* np.pi * pl.lit(1) * pl.col("day_of_year").mod(365.25)/pl.lit(365.25)).sin().alias('sin_1') ,
                (pl.lit(2)* np.pi * pl.lit(2) * pl.col("day_of_year").mod(365.25)/pl.lit(365.25)).sin().alias('sin_2') ,
                (pl.lit(2)* np.pi * pl.lit(3) * pl.col("day_of_year").mod(365.25)/pl.lit(365.25)).sin().alias('sin_3') ,
                (pl.lit(2)* np.pi * pl.lit(1) * pl.col("day_of_year").mod(365.25)/pl.lit(365.25)).cos().alias('cos_1') ,
                (pl.lit(2)* np.pi * pl.lit(2) * pl.col("day_of_year").mod(365.25)/pl.lit(365.25)).cos().alias('cos_2') ,
                (pl.lit(2)* np.pi * pl.lit(3) * pl.col("day_of_year").mod(365.25)/pl.lit(365.25)).cos().alias('cos_3') ,
                (pl.lit(2)* np.pi * pl.lit(1) * pl.col("day_of_month")/pl.lit(30.5)).sin().alias('month_sin_1') ,
                (pl.lit(2)* np.pi * pl.lit(2) * pl.col("day_of_month")/pl.lit(30.5)).sin().alias('month_sin_2') ,
                (pl.lit(2)* np.pi * pl.lit(3) * pl.col("day_of_month")/pl.lit(30.5)).sin().alias('month_sin_3') ,
                (pl.lit(2)* np.pi * pl.lit(1) * pl.col("day_of_month")/pl.lit(30.5)).cos().alias('month_cos_1') ,
                (pl.lit(2)* np.pi * pl.lit(2) * pl.col("day_of_month")/pl.lit(30.5)).cos().alias('month_cos_2') ,
                (pl.lit(2)* np.pi * pl.lit(3) * pl.col("day_of_month")/pl.lit(30.5)).cos().alias('month_cos_3') ,
            )
    )
)



"""
Exports all processed interim datasets to parquet files
and cleans up memory by setting variables to None
"""
# Export product location data
DataPrepProdLocInterim.collect().write_parquet(sink=False, name='DataPrepProdLocInterim',path='3. interim')
DataPrepProdLocInterim = None

# Export sales data
DataPrepSalesInterim.collect().write_parquet(sink=False, name='DataPrepSalesInterim',path='3. interim')
DataPrepSalesInterim = None

# Export event features
holiday.write_parquet(sink=True, name='DataPrepInterimEventholidayWide',path='3. interim')
DataPrepInterimEventPreEffectsFeatureWide = None

DataPrepInterimEventPostEffectsFeatureWide.write_parquet(sink=True, name='DataPrepInterimEventPostEffectsFeatureWide',path='3. interim')
DataPrepInterimEventPostEffectsFeatureWide = None

# Export SNAP features
DataPrepSnapInterimLong.write_parquet(sink=True, name='DataPrepSnapInterimLong',path='3. interim')
DataPrepSnapInterimLong = None

# Export calendar features
DataPrepCalenderInterim.write_parquet(sink=True, name='DataPrepCalenderInterim',path='3. interim')
DataPrepCalenderInterim = None

GaussianSplineEventLong.write_parquet(sink=True,name='GaussianSplineEvent',path='3. interim')
GaussianSplineEventLong= None


GaussianSplineEventWide.write_parquet(name="GaussianSplineEventWide", path='3. interim')
GaussianSplineEventWide = None

GaussianSplineEventWide2.write_parquet(name="GaussianSplineEventWide", path='3. interim')
GaussianSplineEventWide = None

