"""
Product Location Metadata Processing

Creates a dataset with product location metadata including:
- First and last sales dates
- Total sales
- Generated date range for each product-location combination
"""

import polars as pl
from pathlib import Path
from datetime import date
from package.utils import get_path_to_latest_file
from package.datapreparation import DataPreparation

def create_prod_loc_interim():
    project_root = Path(__file__).parent.parent
    
    DataPrepProdLocRaw = DataPreparation(get_path_to_latest_file('2.raw', 'DataPrepProdLocRaw'))
    DataPrepSalesRaw = DataPreparation(get_path_to_latest_file('2.raw', 'DataPrepSalesRaw'))

    prod_loc_interim = (
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
                            end=date(2016, 5, 22),
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
    )

    prod_loc_interim.write_parquet(
        sink=True, 
        name='ProdLocs',
        path='3.interim',
        subfolder = 'prodlocs'
    )

if __name__ == "__main__":
    create_prod_loc_interim() 