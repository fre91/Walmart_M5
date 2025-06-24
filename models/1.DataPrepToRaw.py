# import librariesx
import polars as pl
from datetime import date
from package.datapreparation import DataPreparation

# Usage Example
DataPrepSales = DataPreparation(r"/Users/fredrik.hornell/Python/Private/Walmart_M5/data/1. external/sales_train_evaluation.parquet")
DataPrepProdLoc = DataPreparation(r"/Users/fredrik.hornell/Python/Private/Walmart_M5/data/1. external/sales_train_evaluation.parquet")
DataPrepCalendar = DataPreparation(r"/Users/fredrik.hornell/Python/Private/Walmart_M5/data/1. external/calendar.parquet")
DataPrepPrice = DataPreparation(r"/Users/fredrik.hornell/Python/Private/Walmart_M5/data/1. external/sell_prices.parquet")


DataPrepProdLocRaw = (
    DataPrepProdLoc.load_data(lazy=True)
    #.hash_column(column='id', new_column_name='id_seq_num')
    .select_columns(
        ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    )
    .update_schema()
)

DataPrepSalesRaw = (
    DataPrepSales.load_data(lazy=True)
    #.hash_column(column='id', new_column_name='id_seq_num')
    .transform_to_long_format(
        drop_columns=['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
        keep_id_column='id',
        value_column_name='sales',
        date_column_name= 'd'
    )
   .join(
        DataPrepCalendar
        .load_data(lazy=True)
        .select_columns(['d','date'])
        , on='d'
        , how='left'
    )
   .select_columns(['id','date','sales'])
   .update_schema()
)

DataPrepPriceRaw = (
    DataPrepPrice
    .load_data(lazy=True)
    .join(
        DataPrepProdLocRaw,
        on=['item_id','store_id'],
        how='left'
    )
    .join(
        DataPrepCalendar
        .load_data(lazy=True)
        .select_columns(['wm_yr_wk','date'])
        ,
        on=['wm_yr_wk'],
        how='left'
    )
    .select_columns(['id','date','sell_price'])
    .update_schema()
)

DataPrepCalendarRaw = (
    DataPrepCalendar
    .load_data(lazy=True)
)

DataPrepProdLocRaw.write_parquet(sink=True, name='DataPrepProdLocRaw',path='2.raw')
DataPrepSalesRaw.write_parquet(sink=True, name='DataPrepSalesRaw',path='2.raw')
DataPrepCalendarRaw.write_parquet(sink=True, name='DataPrepCalendarRaw',path='2.raw')
DataPrepPriceRaw.write_parquet(sink=True, name='DataPrepPriceRaw',path='2.raw')

DataPrepCalendarRaw = None
DataPrepProdLocRaw = None
DataPrepSalesRaw = None
DataPrepPriceRaw = None


