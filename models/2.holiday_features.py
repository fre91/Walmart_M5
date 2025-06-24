"""
Holiday Feature Processing

Creates holiday-specific features for the dataset.
"""

import polars as pl
from pathlib import Path
from package.datapreparation import DataPreparation




def create_holiday_features():
    project_root = Path(__file__).parent.parent
    
    DataPrepCalendarRaw = DataPreparation(
        project_root / "data/2.raw/DataPrepCalendarRaw_20241215_111217.parquet"
    )
    
    DataPrepCalendarRaw1 = DataPreparation(
        project_root / "data/2.raw/DataPrepCalendarRaw_20241215_111217.parquet"
    )


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
                    values='value'
                )
        )
    

    holiday_output = (
        calender
        .join(
            holidays
            , on = 'date'
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
    holiday_output.collect().result


    holiday_output.write_parquet(
        sink=True,
        name='holiday_bool_features',
        path='3.interim',
        subfolder= 'holiday_bool_features'
    )

if __name__ == "__main__":
    create_holiday_features() 