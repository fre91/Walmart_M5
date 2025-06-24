"""
Post-Event Feature Engineering

Creates features tracking the aftermath of events for 7 days after each occurrence.
"""

import polars as pl
from pathlib import Path
from package.datapreparation import DataPreparation
from package.utils import get_path_to_latest_file

def create_event_post_effects():
    project_root = Path(__file__).parent.parent
    
    calendar_raw = DataPreparation(get_path_to_latest_file('2.raw', 'DataPrepCalendarRaw'))

    event_post_interim = (
        calendar_raw
        .load_data(lazy=True)
        .modify_data(
            lambda data: (
                pl.concat([
                    data.select(
                        pl.col("date"),
                        pl.col("event_type_1").alias('event_type'),
                        pl.col("event_name_1").alias('event_name'),
                    ),
                    data.select(
                        pl.col("date"),
                        pl.col("event_type_2").alias('event_type'),
                        pl.col("event_name_2").alias('event_name'),
                    )
                ])
            )
            .drop_nulls()
            .select(
                pl.col('date'),
                (
                    pl.lit('event').str.to_uppercase() +
                    pl.lit('_') +
                    pl.col('event_type').str.to_uppercase() +
                    pl.lit('_') +
                    pl.col('event_name').str.to_uppercase()
                ).alias('event')
            )
            .group_by('event')
            .agg(pl.col('date'))
        )
    )

    post_effects_long = (
        event_post_interim
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
                    pl.concat_str([pl.col('event'), pl.lit('_'), pl.col('test')]).alias('feature')
                )
        )
    )

    post_effects_wide = (
        post_effects_long
        .select_columns(pl.col('*').exclude('test'))
        .pivot_and_lazy(
            index=['range', 'event'],
            on='feature',
            aggregate_function='len'
        )
    )

    post_effects_wide.write_parquet(
        sink=True,
        name='event_post_features',
        path='3.interim',
        subfolder= 'event_post_effect_features'
    )

if __name__ == "__main__":
    create_event_post_effects() 