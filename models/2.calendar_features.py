"""
Calendar Features Processing

Creates calendar features including:
- Basic date components (day of month, day of year)
- Cyclical encodings using sine and cosine transformations
- Both yearly and monthly cyclical features
"""

import polars as pl
import numpy as np
from pathlib import Path
from package.datapreparation import DataPreparation
from datetime import date
from sklearn.mixture import GaussianMixture
from package.utils import get_path_to_latest_file



def create_gaussian_splines(df:pl.DataFrame, ordinal_col , n_components=3):
    
    
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
        pl.from_numpy(latent_features, col_name).to_struct().alias(f'{ordinal_col}_guassian_splines')
    ).lazy()






def create_calendar_features():
    # Get the project root (two levels up from the current script)
    project_root = Path(__file__).parent.parent
    
    # Print the paths to debug
    print(f"Project root: {project_root}")
    print(f"Full path: {project_root / 'data/2.raw/DataPrepCalendarRaw_20241215_111217.parquet'}")
    
    calendar_raw = DataPreparation(get_path_to_latest_file('2.raw', 'DataPrepCalendarRaw'))

    calendar_interim = (
        calendar_raw
        .load_data(lazy=True)
        .modify_data(
            lambda data:
                data
                .with_columns(
                    pl.col("date").dt.day().alias("day_of_month"),
                    pl.col("date").dt.ordinal_day().alias("day_of_year")
                )
                .select(pl.col('*').exclude([
                    'event_name_1', 'event_type_1', 'event_name_2', 
                    'event_type_2', 'snap_CA', 'snap_WI', 'snap_TX'
                ]))
        )
        .modify_data(
            lambda data:
                data
                .with_columns([
                    (pl.lit(2) * np.pi * pl.lit(i) * pl.col("day_of_year").mod(365.25)/pl.lit(365.25)).sin().alias(f'sin_{i}')
                    for i in range(1, 4)
                ] + [
                    (pl.lit(2) * np.pi * pl.lit(i) * pl.col("day_of_year").mod(365.25)/pl.lit(365.25)).cos().alias(f'cos_{i}')
                    for i in range(1, 4)
                ] + [
                    (pl.lit(2) * np.pi * pl.lit(i) * pl.col("day_of_month")/pl.lit(30.5)).sin().alias(f'month_sin_{i}')
                    for i in range(1, 4)
                ] + [
                    (pl.lit(2) * np.pi * pl.lit(i) * pl.col("day_of_month")/pl.lit(30.5)).cos().alias(f'month_cos_{i}')
                    for i in range(1, 4)
                ])
        )
        .modify_data(
            lambda data:
                data
                .sort(by='date')
                .set_sorted('date')
                .with_row_index(name='trend')
                .with_columns(
                    pl.col('trend').cast(pl.Int64)
                )
                .pipe(
                    create_gaussian_splines
                    , ordinal_col = 'wday'
                    , n_components= 3
                    
                )
                .pipe(
                    create_gaussian_splines
                    , ordinal_col = 'trend'
                    , n_components= 12
                )
                .with_columns(
                     monday = pl.when(pl.col("weekday")=="Monday").then(pl.lit(1)).otherwise(0)
                     , tuesday = pl.when(pl.col("weekday")=="Tuesday").then(pl.lit(1)).otherwise(0)
                     , wednesday = pl.when(pl.col("weekday")=="Wednesday").then(pl.lit(1)).otherwise(0)
                     , thursday = pl.when(pl.col("weekday")=="Thursday").then(pl.lit(1)).otherwise(0)
                     , friday = pl.when(pl.col("weekday")=="Friday").then(pl.lit(1)).otherwise(0)
                     , saturday = pl.when(pl.col("weekday")=="Saturday").then(pl.lit(1)).otherwise(0)
                     , sunday = pl.when(pl.col("weekday")=="Sunday").then(pl.lit(1)).otherwise(0)
                )
                .with_columns( 
                    pl.struct(
                        "sin_1", "sin_2", "sin_3", 
                        "cos_1", "cos_2", "cos_3",
                        "month_sin_1", "month_sin_2", "month_sin_3",
                        "month_cos_1", "month_cos_2", "month_cos_3"
                    ).alias("year_month_trigonometric")
                ,
                pl.struct(
                    "monday","tuesday", "wednesday", "thursday", "friday", "saturday" , "sunday"
                ).alias("weekday_bool")
                )
        )
    )
    



    calendar_interim.write_parquet(
        sink=True,
        name='calender_trig_features',
        path='3.interim',
        subfolder= 'calender_trig_features'
    )


if __name__ == "__main__":
    create_calendar_features() 
    
    # Adjust this path to the actual file generated
    file_path = Path("3.interim/calender_trig_features") / "calender_trig_features_<timestamp>.parquet"

    # Load the file
    df = pl.read_parquet(r"/Users/fredrik.hornell/Python/Private/Walmart_M5/data/3.interim/calender_trig_features/calender_trig_features_20250505_212843.parquet")

    # Show the schema (column names and types)
    print(df.schema)

    # Show the first few rows
    df.head(10)