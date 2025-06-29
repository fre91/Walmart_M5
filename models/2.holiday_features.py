"""
Event/holiday Feature Processing

Creates event-specific features for the dataset.
"""

import polars as pl
from pathlib import Path
from package.datapreparation import DataPreparation
from package.utils import get_path_to_latest_file




def create_event_features():
    
    DataPrepCalendarRaw = DataPreparation(get_path_to_latest_file('2.raw', 'calendar_raw'))
    
    DataPrepCalendarRaw1 = DataPreparation(get_path_to_latest_file('2.raw', 'calendar_raw'))


    calender = (
        DataPrepCalendarRaw1
        .load_data(lazy=True)
        .select_columns('date')
    )
    
    calender.schema
    holidays = (
                DataPrepCalendarRaw
                .load_data(lazy=True)
                .modify_data(
                    lambda data:
                        pl.concat(
                            [
                                data
                                .select(pl.col('date'), pl.col('event_name_1').alias('event'))
                                .drop_nulls()
                                .with_columns(pl.lit(1).alias('value'))
                            ,
                                data
                                .select(pl.col('date'), pl.col('event_name_2').alias('event'))
                                .drop_nulls()
                                .with_columns(pl.lit(1).alias('value'))
                            
                            ]
                        )
                )
                .pivot_and_lazy(
                    index='date',
                    on='event',
                    values='value',
                    aggregate_function = pl.any,
                )
        )
    

    holiday_output = (
        calender
        .join(
            holidays
            , on = ['date']
            , how ='left'
        )
        .modify_data(
            lambda data:
                data
                .fill_null(0)
                .select(
                    pl.col('date'),
                    pl.struct(pl.all().exclude('date')).alias('holiday_bool')
                )
        )
    )


    holiday_output.write_parquet(
        sink=True,
        name='event_features',
        path='3.interim',
        subfolder= 'event_features'
    )

if __name__ == "__main__":
    create_event_features() 