import polars as pl
from datetime import datetime
from polars import selectors as cs 
import numpy as np
import os

class DataPreparation:
    def __init__(self, file_path):
        """Initialize with the file path."""
        self.file_path = file_path
        self.data = None  # Initialize data attribute
        self.result = None
        self.schema = None  # Initialize schema attribute

    def load_data(self, lazy=True):
        """Load data from the specified file path."""
        if lazy:
            self.data = pl.scan_parquet(self.file_path)
        else:
            self.data = pl.read_parquet(self.file_path)
        # Update schema attribute after loading data
        if isinstance(self.data, pl.LazyFrame):
            self.schema = self.data.schema
        else:
            self.schema = self.data.schema
        return self  # Return self to enable method chaining
    def transform_to_long_format(self, drop_columns=None, keep_id_column='id', value_column_name='sales', date_column_name='day'):
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        if drop_columns is None:
            drop_columns = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
        self.data = (
            self.data
            .drop(drop_columns)
            .melt(id_vars=keep_id_column)
            .rename({'variable': date_column_name, 'value': value_column_name})
            .with_columns([
                pl.col(value_column_name).cast(pl.Int16),
            ])
        )
        return self
    def hash_column(self, column: str, new_column_name: str):
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        self.data = self.data.with_columns(
            pl.col(column).hash(seed=4).alias(new_column_name)
        )
        return self
    def select_columns(self, columns: list):
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        self.data = self.data.select(columns)
        return self
    def modify_data(self, expression):
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        self.data = expression(self.data)
        return self
    def collect(self):
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        if isinstance(self.data, pl.LazyFrame):
            self.result = self.data.collect()
        else:
            self.result = self.data
        return self
    def show_graph(self):
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        if isinstance(self.data, pl.LazyFrame):
            self.data.show_graph()
        else:
            raise TypeError("show_graph is only available for LazyFrame.")
        return self
    def fetch(self, limit: int = 10):
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        if isinstance(self.data, pl.LazyFrame):
            return self.data.fetch(limit)
        else:
            return self.data.head(limit)
    def update_schema(self):
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        if isinstance(self.data, pl.LazyFrame):
            self.schema = self.data.collect_schema()
        else:
            self.schema = self.data.schema
        return self
    def pivot_and_lazy(self,index=None,on=None,values=None, aggregate_function=None):
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        # Ensure required arguments are not None
        if index is None or on is None or values is None or aggregate_function is None:
            raise ValueError("index, on, values, and aggregate_function must all be provided.")
        if isinstance(self.data, pl.LazyFrame):
            df = self.data.collect()
        else:
            df = self.data
        self.data = (
            df
            .pivot(
                index=index
                , on= on
                , values=values
                , aggregate_function=aggregate_function
            )
            .fill_null(0)
            .lazy()
        )
        return self
    def join(self, other, on: list, how: str = 'inner'):
        if self.data is None or other.data is None:
            raise ValueError("Data not loaded in one or both instances. Call load_data() first.")
        valid_hows = {'inner', 'left', 'right', 'full', 'semi', 'anti', 'cross'}
        if how not in valid_hows:
            raise ValueError(f"Invalid join type: {how}. Must be one of {valid_hows}")
        self.data = self.data.join(other.data, on=on, how=how)  # type: ignore
        return self
    def write_parquet(self, sink=True, name=None, path='1.external', subfolder=None):
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        if name is None:
            raise ValueError("Name of file not provided.")
        date_now = datetime.now().strftime("%Y%m%d_%H%M%S")
        path_map = {
            '1.external': r"/Users/fredrik.hornell/Python/Private/Walmart_M5/data/1.external/",
            '2.raw': r"/Users/fredrik.hornell/Python/Private/Walmart_M5/data/2.raw/",
            '3.interim': r"/Users/fredrik.hornell/Python/Private/Walmart_M5/data/3.interim/",
            '4.processed': r"/Users/fredrik.hornell/Python/Private/Walmart_M5/data/4.processed/"
        }
        if path not in path_map:
            raise ValueError(f"Invalid path option: {path}.")
        base_path = path_map[path]
        if subfolder:
            full_path = os.path.join(base_path, subfolder)
            os.makedirs(full_path, exist_ok=True)
            file_path = os.path.join(full_path, f"{name}_{date_now}.parquet")
        else:
            file_path = os.path.join(base_path, f"{name}_{date_now}.parquet")
        if sink:
            if isinstance(self.data, pl.LazyFrame):
                self.data.sink_parquet(file_path, row_group_size=100000)
            else:
                self.data.write_parquet(file_path)
        else:
            if self.result is not None and hasattr(self.result, 'write_parquet'):
                self.result.write_parquet(file_path)
            else:
                raise TypeError("Result is not a DataFrame and cannot be written to parquet.")
        return self
    def calculate_out_of_stock_periods(self, list_of_ids=None, binomial_threshold=0.001):
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        id_col, time_col, target_col = self.data.collect_schema().names()
        binomial_threshold = binomial_threshold
        if list_of_ids is None:
            df_zero_ind = (self.data.lazy()
                .select(pl.col('*'), pl.when(pl.col(target_col) == 0).then(1).otherwise(0).alias('zero_sales_ind'))
            )
        else:
            df_zero_ind = (self.data.lazy()
                .filter(pl.col(id_col).is_in(list_of_ids))
                .select(pl.col('*'), pl.when(pl.col(target_col) == 0).then(1).otherwise(0).alias('zero_sales_ind'))
            )
        df_zero_sales_period = ( 
            df_zero_ind
            .sort([id_col, time_col])
            .select(
                pl.col('*'),
                (
                    pl.col('zero_sales_ind') -
                    pl.col('zero_sales_ind').shift(1).over(id_col)).alias('delta')
            )
            .select(
                    pl.col('*').exclude('delta'),
                    pl.when(pl.col('delta')==-1).then(0).otherwise(pl.col('delta')).alias('delta')
                )
            .select(
                    pl.col('*'),
                    (pl.col('delta').cum_sum().over(id_col) * pl.col('zero_sales_ind')).alias('zero_sales_period')
                )
            .filter(pl.col('zero_sales_ind')==1)
        )
        df_zero_sales_probability = (
            df_zero_ind.group_by(id_col)
            .agg(pl.col('zero_sales_ind')
            .mean().alias('prob_zero_sales'))
        )
        df_zero_sales_period_aggregated =(
            df_zero_sales_period
            .group_by('id','zero_sales_period').agg(pl.col('zero_sales_ind').sum().alias('nbr_zero_sales_in_a_row')) 
            .join(df_zero_sales_probability, on=[id_col])
            .with_columns(
                pl.when(
                    (
                        pl.col('prob_zero_sales') ** pl.col('nbr_zero_sales_in_a_row') * (1 - pl.col('prob_zero_sales')) ** (pl.col('nbr_zero_sales_in_a_row') - pl.col('nbr_zero_sales_in_a_row'))
                    ) <= binomial_threshold
                ).then(1).otherwise(0).alias('OOS')
            )
        )
        self.data = (
            df_zero_ind 
            .join(
                    (  
                        df_zero_sales_period
                        .join(  # Join the calculated prob for the zero sales periods
                            df_zero_sales_period_aggregated
                            , on=[id_col,'zero_sales_period']
                            )
                        .select([id_col, time_col, pl.col('OOS')])
                    )
                    , on=[id_col, time_col] 
                    , how='left'
                )
            .with_columns(pl.col('OOS').fill_null(strategy="zero").alias('OOS'))
        )
        return self 