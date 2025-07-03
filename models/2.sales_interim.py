"""
Sales Data Normalization

Normalizes sales data by joining with product location date ranges
to ensure consistent date coverage across all products.
"""

import polars as pl
from package.datapreparation import DataPreparation
from pathlib import Path
from package.utils import get_path_to_latest_file

def create_sales_interim():
    
    DataPrepSalesRaw = DataPreparation(get_path_to_latest_file('2.raw', 'sales_raw'))
    DataPrepProdLocInterim = DataPreparation(get_path_to_latest_file('3.interim', 'prodlocs'))

    sales_interim = (
        DataPrepSalesRaw
        .load_data(lazy=True)
        .join(
            DataPrepProdLocInterim
            .load_data(lazy=True)
            .select_columns(['id', 'prodloc_daterange'])
            .modify_data(
                lambda data: data.explode('prodloc_daterange').rename({'prodloc_daterange': 'date'})
            )
            , on=['id', 'date']
            , how='inner'
        )
        .calculate_out_of_stock_periods(binomial_threshold=0.0001)
        #.calculate_out_of_stock_periods(binomial_threshold=0.001) / fix so that I chain Call these 
        #.calculate_out_of_stock_periods(binomial_threshold=0.01)  / fix so that I chain Call these 
    )

    sales_interim.write_parquet(
        sink=True,
        name='sales',
        path='3.interim',
        subfolder='sales'
    )

if __name__ == "__main__":
    create_sales_interim() 