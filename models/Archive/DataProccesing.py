"""
Walmart M5 Sales Forecasting Data Processing and Model Training
This module handles the data processing, feature engineering, and model training for Walmart sales forecasting.
It implements a rolling window forecasting approach using LASSO regression.

Main components:
- Data loading and preprocessing
- Rolling window split creation
- LASSO model training and prediction
- Model evaluation and diagnostics

TODO:
1. Implement naive forecast using previous year's values
2. Create modular training loop function
3. Add feature importance analysis using residuals
4. Enhance model evaluation metrics and coefficient analysis
"""


# import librariesx
import polars as pl
import numpy as np 
from polars import selectors as cs 

from datetime import date
from package.datapreparation import DataPreparation
import pyarrow 

import altair as alt

from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from datetime import date


# import libraries
import polars as pl
import numpy as np 
from polars import selectors as cs 
from datetime import timedelta  # Add this import
import polars_ols as pls 

from datetime import date
from package.datapreparation import DataAnalytics

import matplotlib.pyplot as plt
import seaborn as sns

import hvplot.polars
import panel as pn
import panel.widgets as pnw
import holoviews as hv

pn.extension(comms="vscode")

#pl.read_parquet("/Users/fredrik.hornell/Python/Private/Walmart_M5/data/4. processed/DesignMatrix2.parquet")

# Usage Example
df = (
    pl.scan_parquet("/Users/fredrik.hornell/Python/Private/Walmart_M5/data/4. processed/DesignMatrix2.parquet")
    .select(pl.col('*').exclude('id','zero_sales_ind','category_dep','state','item_state','item')) # analysis on state store level
    .group_by(
        pl.all().exclude('sales','OOS')
    )
    .agg(
        pl.col('sales').sum().alias('sales')
        , pl.col('OOS').sum().alias('OOS_COUNT')
        , pl.col('OOS').len().alias('ProductPlaces')
    )
    .sort(by='date')
    .set_sorted('date')
    .collect(engine="streaming") # engine="streaming"
)

cols = ['date',
 'state_store',
 'trend_regimes',
 'wday_regimes',
 'trig_features',
 'event_struct',
 'state_snap',
 'sales',
 'OOS_COUNT',
 'Days'
]

struct_cols = [ 'trend_regimes',
 'wday_regimes',
 'trig_features',
 'event_struct',
 'holidays'
 ]

outlier_cols = []


def create_rolling_train_test_splits(df, start_date, end_date, training_years=3, forecast_months=1):
    """
    Creates rolling window splits for time series forecasting.
    
    Args:
        df (pl.DataFrame): Input DataFrame with date column and features
        start_date (date): Start date for the first training window
        end_date (date): End date for the entire analysis
        training_years (int): Number of years for training window
        forecast_months (int): Number of months to forecast
    
    Returns:
        list: List of tuples (train_df, test_df, test_start_date) for each split
    
    TODO:
        - Add previous year's values as naive forecast baseline
        - Include data quality checks for each split
    """
    splits = []
    current_date = start_date
    
    while current_date + timedelta(days=30) <= end_date:
        # Calculate window boundaries
        train_start = current_date - timedelta(days=training_years * 365)
        test_end = current_date + timedelta(days=30)  # Approximately one month
        
        # Create train and test sets
        train_df = df.filter(
            (pl.col('date') >= train_start) & 
            (pl.col('date') < current_date)
        )
        
        test_df = df.filter(
            (pl.col('date') >= current_date) & 
            (pl.col('date') < test_end)
        )
        
        splits.append((train_df, test_df, current_date))
        
        # Move forward one month for the next split
        current_date = test_end
    
    return splits

def train_evaluate_model(train_df, test_df, features, alpha=0.0001):
    """
    Trains LASSO model and evaluates predictions for a given split.
    
    Args:
        train_df (pl.DataFrame): Training data
        test_df (pl.DataFrame): Test data
        features (list): List of feature names
        alpha (float): LASSO regularization parameter
    
    Returns:
        tuple: (predictions, coefficients, evaluation_metrics)
    
    TODO:
        - Add multiple evaluation metrics (RMSE, MAE, MAPE)
        - Include feature importance analysis
        - Compare against naive forecast
        - Add residual analysis
    """
    # TODO: Implement this function to replace current training loop

def analyze_feature_importance(model, features, residuals):
    """
    Analyzes feature importance using coefficient values and residual patterns.
    
    Args:
        model: Trained model object
        features (list): List of feature names
        residuals (pl.Series): Model residuals
    
    Returns:
        pl.DataFrame: Feature importance metrics and diagnostics
    
    TODO:
        - Implement feature coverage analysis
        - Add residual correlation analysis
        - Create visualization functions
    """
    # TODO: Implement this function

def evaluate_model_performance(actual, predicted, naive_forecast=None):
    """
    Calculates comprehensive model performance metrics.
    
    Args:
        actual (pl.Series): Actual values
        predicted (pl.Series): Model predictions
        naive_forecast (pl.Series, optional): Naive forecast values
    
    Returns:
        dict: Dictionary of performance metrics
    
    TODO:
        - Add multiple error metrics
        - Include comparison with naive forecast
        - Add statistical tests
    """
    # TODO: Implement this function


def create_trading_periods(df:pl.DataFrame, ordinal_col , n_components=3):
    
    if isinstance(df, pl.LazyFrame):
        df = df.collect()
    elif not isinstance(df, pl.DataFrame):
        raise TypeError("Input must be either a Polars DataFrame or LazyFrame")
    
    X = df[ordinal_col].to_numpy().reshape(-1,1)
    
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(X)

    latent_features = gmm.predict_proba(X)
    cluster_labels = gmm.predict(X)
    col_name = [f'{ordinal_col}_{i+1}'for i in range(n_components)]

    return df.with_columns([
        pl.from_numpy(latent_features, col_name).to_struct().alias(f'{ordinal_col}_regimes'),
        pl.Series(f'{ordinal_col}_cluster', cluster_labels)
    ]).lazy()



from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN


# Example usage
start_date = date(2013, 1,29)  # Adjust based on your data
end_date = date(2016, 1, 29)  # Adjust based on your data

splits = create_rolling_train_test_splits(df, start_date, end_date)

# Iterate through the splits
for index, (train_df, test_df, test_start_date) in enumerate(splits):
    print(f"\nSplit {index} at {test_start_date}")
    print(f"Training set: {train_df['date'].min()} to {train_df['date'].max()}")
    print(f"Test set: {test_df['date'].min()} to {test_df['date'].max()}")
    
    # Here you would:
    # 1. Train your model on train_df
    # 2. Make predictions on test_df
    # 3. Move to the next split



X = []

for i in struct_cols:
    X.extend(df.select(i).to_series().struct.fields)

lasso_expr = (
    pl.col("sales").least_squares.lasso(*X, alpha=0.01, add_intercept=True).over("state_store")
)
lasso_coef = (
    pl.col("sales").least_squares.lasso(*X, alpha=0.01, add_intercept=True,  mode="coefficients").over("state_store").alias("coefficients_group")
)
lasso_resi = (
    pl.col("sales").least_squares.lasso(*X, alpha=0.01, add_intercept=True,  mode="residuals").over("state_store").alias("residuals_group")
)


test_train = (
    splits[35][0]
    .unnest(
        'trend_regimes'
        ,'wday_regimes'
        ,'trig_features'
        ,'event_struct'
        ,'holidays'
    )
    .with_columns(
        fit = lasso_expr
        , coefficients_group = lasso_coef
        , residual =lasso_resi
    )
)

test_train.sort(by='date').filter(pl.col('state_store')=='TX_2').to_pandas().plot(x='date', y=['sales', 'fit'])


test_train.select(pl.col('state_store'),pl.col('coefficients_group')).unnest('coefficients_group').unique().unpivot(index='state_store').sort(by='value').filter(pl.col('value')==0).select("variable").unique().to_series().to_list()

residual_frame = (
    test_train
    .filter(
        (pl.col('residual') <= pl.col('residual').quantile(0.01)) | 
        (pl.col('residual') >= pl.col('residual').quantile(0.99))
        )
    .with_columns(
    pl.col('date').dt.month().alias('month')
    , pl.col('date').dt.day().alias('day_month')
    )
    .group_by(['month','day_month'])
    .agg(
        pl.len().alias('residual_count')
        , pl.col('residual').abs().sum().log().alias('residual_sum')
        , pl.col('residual').abs().mean().log().alias('residual_mean')
        , pl.col('date').unique()
        , pl.col('state_store')
    )
).sort(by='residual_count')


numeric_cols = ['residual_sum', 'residual_mean','residual_count']

clustered_df = (
    residual_frame
    .pipe(
        create_trading_periods
        , 'residual_sum'
        , 3
    )
    .pipe(
        create_trading_periods
        , 'residual_mean'
        , 3
    )
     .pipe(
        create_trading_periods
        , 'residual_count'
        , 3
    )
    .collect()
)

# The clusters can then be used as features in your model
#X.extend(clustered_df.select('dbscan_clusters').to_series().struct.fields)


clustered_df.hvplot.scatter(
    x='residual_mean'
    , y ='residual_sum'
    , by ='residual_sum_cluster'

)

DataPrepCalendarRaw = DataPreparation(r"/Users/fredrik.hornell/Python/Private/Walmart_M5/data/2. raw/DataPrepCalendarRaw_20241215_111217.parquet")


test = DataPrepCalendarRaw.load_data().collect().result.with_columns(
    pl.col('date').dt.month().alias('month')
    , pl.col('date').dt.day().alias('day_month')
)

residual_frame.join(
    test
    , how = 'inner'
    , on = ['month' , 'day_month']
).sort(by='residual_count').select(pl.all().exclude(['event_name_2','event_type_2'])).drop_nulls().select('event_name_1').unique().to_series().to_list()


test_pred = (
    splits[35][1]
    .unnest(
        'trend_regimes'
        ,'wday_regimes'
        ,'trig_features'
        ,'event_struct'
        ,'holidays'
    )
    .join(
        test_train.select('state_store','coefficients_group').unique(maintain_order=True)
        , on = 'state_store'
        , how = 'inner'
    )
    .with_columns(
        "state_store",
        pl.col("coefficients_group").least_squares.predict(
          *X  , add_intercept=True,
          name="predictions_test"
        )
    )
)

test_pred.sort(by='date').filter(pl.col('state_store')=='CA_2').to_pandas().plot(x='date', y=['sales','predictions_test'])


def train_evaluate_model(train_df, test_df, features, alpha=0.0001):
    """
    Trains LASSO model and evaluates predictions for a given split.
    
    Args:
        train_df (pl.DataFrame): Training data
        test_df (pl.DataFrame): Test data
        features (list): List of feature names
        alpha (float): LASSO regularization parameter
    
    Returns:
        tuple: (predictions, coefficients, evaluation_metrics)
    
    TODO:
        - Add multiple evaluation metrics (RMSE, MAE, MAPE)
        - Include feature importance analysis
        - Compare against naive forecast
        - Add residual analysis
    """
    # TODO: Implement this function to replace current training loop

def analyze_feature_importance(model, features, residuals):
    """
    Analyzes feature importance using coefficient values and residual patterns.
    
    Args:
        model: Trained model object
        features (list): List of feature names
        residuals (pl.Series): Model residuals
    
    Returns:
        pl.DataFrame: Feature importance metrics and diagnostics
    
    TODO:
        - Implement feature coverage analysis
        - Add residual correlation analysis
        - Create visualization functions
    """
    # TODO: Implement this function

def evaluate_model_performance(actual, predicted, naive_forecast=None):
    """
    Calculates comprehensive model performance metrics.
    
    Args:
        actual (pl.Series): Actual values
        predicted (pl.Series): Model predictions
        naive_forecast (pl.Series, optional): Naive forecast values
    
    Returns:
        dict: Dictionary of performance metrics
    
    TODO:
        - Add multiple error metrics
        - Include comparison with naive forecast
        - Add statistical tests
    """
    # TODO: Implement this function




