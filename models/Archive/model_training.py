"""
Walmart M5 Sales Forecasting Model Training
This module implements a rolling window forecasting approach using LASSO regression.

Classes:
- WalmartSalesModel: Main class handling model training and evaluation
- DataProcessor: Handles data preprocessing and feature engineering
"""

import polars as pl
import numpy as np
from datetime import date, timedelta
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from typing import List, Tuple, Dict, Optional

class DataProcessor:
    """Handles data preprocessing and feature engineering"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        
    def load_data(self) -> pl.DataFrame:
        """Load and prepare initial dataset"""
        df = (
            pl.scan_parquet(self.data_path)
            .select(pl.col('*').exclude('id','zero_sales_ind','category_dep','state','item_state','item'))
            .group_by(pl.all().exclude('sales','OOS'))
            .agg(
                pl.col('sales').sum().alias('sales'),
                pl.col('OOS').sum().alias('OOS_COUNT'),
                pl.col('OOS').len().alias('ProductPlaces')
            )
            .sort(by='date')
            .set_sorted('date')
            .collect(engine="streaming")
        )
        return df

    @staticmethod
    def create_trading_periods(df: pl.DataFrame, ordinal_col: str, n_components: int = 3) -> pl.LazyFrame:
        """Create trading periods using Gaussian Mixture Model"""
        if isinstance(df, pl.LazyFrame):
            df = df.collect()
        
        X = df[ordinal_col].to_numpy().reshape(-1, 1)
        gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
        gmm.fit(X)
        
        latent_features = gmm.predict_proba(X)
        col_name = [f'{ordinal_col}_{i+1}' for i in range(n_components)]
        
        return df.with_columns([
            pl.from_numpy(latent_features, col_name).to_struct().alias(f'{ordinal_col}_regimes'),
            pl.Series(f'{ordinal_col}_cluster', gmm.predict(X))
        ]).lazy()

class WalmartSalesModel:
    """Handles model training, prediction, and evaluation"""
    
    def __init__(self, training_years: int = 3, forecast_months: int = 1):
        self.training_years = training_years
        self.forecast_months = forecast_months
        
    def create_rolling_splits(self, df: pl.DataFrame, start_date: date, end_date: date) -> List[Tuple]:
        """Creates rolling window splits for time series forecasting"""
        splits = []
        current_date = start_date
        
        while current_date + timedelta(days=30) <= end_date:
            train_start = current_date - timedelta(days=self.training_years * 365)
            test_end = current_date + timedelta(days=30)
            
            train_df = df.filter(
                (pl.col('date') >= train_start) & 
                (pl.col('date') < current_date)
            )
            
            test_df = df.filter(
                (pl.col('date') >= current_date) & 
                (pl.col('date') < test_end)
            )
            
            splits.append((train_df, test_df, current_date))
            current_date = test_end
        
        return splits

    def train_model(self, train_df: pl.DataFrame, features: List[str], alpha: float = 0.01) -> Dict:
        """Train LASSO model on given features"""
        lasso_expr = pl.col("sales").least_squares.lasso(
            *features, alpha=alpha, add_intercept=True
        ).over("state_store")
        
        coefficients = pl.col("sales").least_squares.lasso(
            *features, alpha=alpha, add_intercept=True, mode="coefficients"
        ).over("state_store").alias("coefficients_group")
        
        residuals = pl.col("sales").least_squares.lasso(
            *features, alpha=alpha, add_intercept=True, mode="residuals"
        ).over("state_store").alias("residuals_group")
        
        return {
            "predictions": lasso_expr,
            "coefficients": coefficients,
            "residuals": residuals
        }

    def evaluate_predictions(self, actual: pl.Series, predicted: pl.Series) -> Dict:
        """Calculate model performance metrics"""
        mse = ((actual - predicted) ** 2).mean()
        mae = (actual - predicted).abs().mean()
        mape = ((actual - predicted).abs() / actual).mean() * 100
        
        return {
            "mse": mse,
            "rmse": np.sqrt(mse),
            "mae": mae,
            "mape": mape
        }

# Usage example:
if __name__ == "__main__":
    data_path = "/path/to/DesignMatrix2.parquet"
    processor = DataProcessor(data_path)
    df = processor.load_data()
    
    model = WalmartSalesModel()
    splits = model.create_rolling_splits(
        df,
        start_date=date(2013, 1, 29),
        end_date=date(2016, 1, 29)
    )
    
    # Example feature columns
    struct_cols = ['trend_regimes', 'wday_regimes', 'trig_features', 'event_struct', 'holidays']
    features = []
    for col in struct_cols:
        features.extend(df.select(col).to_series().struct.fields)
    
    # Train and evaluate for one split
    train_df, test_df, test_date = splits[0]
    model_results = model.train_model(train_df, features) 