"""
Gaussian Spline Event Processing

Creates Gaussian mixture model-based features for events.
"""

import polars as pl
from pathlib import Path
from sklearn.mixture import GaussianMixture
from package.datapreparation import DataPreparation
from package.utils import get_path_to_latest_file

def create_gaussian_splines(df: pl.DataFrame, ordinal_col, n_components=3):
    if isinstance(df, pl.LazyFrame):
        df = df.collect()
    elif not isinstance(df, pl.DataFrame):
        raise TypeError("Input must be either a Polars DataFrame or LazyFrame")
    
    X = df[ordinal_col].to_numpy().reshape(-1,1)
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(X)
    
    latent_features = gmm.predict_proba(X)
    col_name = [f'{ordinal_col}_{i+1}' for i in range(n_components)]
    
    return df.with_columns(
        pl.from_numpy(latent_features, col_name).to_struct().alias(f'{ordinal_col}_guassian_splines')
    ).lazy()

def create_gaussian_spline_features():
    project_root = Path(__file__).parent.parent
    
    calendar_raw = DataPreparation(get_path_to_latest_file('2.raw', 'calendar_raw'))
    DataPrepCalendarRaw1 = DataPreparation(get_path_to_latest_file('2.raw', 'calendar_raw'))
    event_pre_post = (
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
                    pl.col('event_name').str.to_lowercase()
                ).alias('event')
            )
            .group_by('event')
            .agg(pl.col('date'))
        )
    )

    spline_long = (
        event_pre_post
        .modify_data(
            lambda data:
                data
                .explode('date')
                .rename({'date': 'event_date'})
                .with_columns(
                    pl.date_ranges(
                        start=(pl.col('event_date') - pl.duration(days=4)),
                        end=(pl.col('event_date') + pl.duration(days=4))
                    ).alias('date')
                )
                .explode('date')
                .with_columns(
                    spline=pl.int_range(pl.len(), dtype=pl.UInt32).over('event', 'event_date')
                )
                .pipe(create_gaussian_splines, ordinal_col='spline', n_components=3)
        )
    )
    #print((
    #    spline_long
    #    .collect()
    #    .result.unnest('spline_regimes')
    #    .unpivot(index=['event','event_date','date','spline'])
    #    .with_columns(event_spline_name=pl.col('event')+"_"+pl.col('variable'))
    #    
    #    .pivot(on='event_spline_name',index='date' , values ='value')
    #    
    #    ))
    
    spline_wide = (
        spline_long
        .modify_data(lambda data: 
            data
            .unnest("spline_guassian_splines")
            .unpivot(index=['event','event_date','date','spline'])
            .with_columns(event_spline_name=pl.col('event')+"_"+pl.col('variable'))
        )
        .pivot_and_lazy(
           on='event_spline_name',
            index='date',
            values='value'
        )
    )
    #print(spline_wide.collect().result.head(4))
    
    spline_wide2 = (
        DataPrepCalendarRaw1
        .load_data(lazy=True)
        .select_columns(['date'])
        .join(
            spline_wide,
            on=['date'],
            how='left'
        )
        .modify_data(
            lambda data:
                data
                .fill_null(0)
                .select(
                    pl.col('date'),
                    pl.struct(pl.all().exclude('date')).alias("holiday_pre_post_guassian_splines")
                )
        )
    )
    #print(spline_wide2.collect().result.head(4))
    
    spline_wide2.write_parquet(
        name="calendar_event_gmm_features"
        , path='3.interim'
        ,  subfolder= 'calendar_event_gmm_features'
        )
    
if __name__ == "__main__":
    create_gaussian_spline_features() 