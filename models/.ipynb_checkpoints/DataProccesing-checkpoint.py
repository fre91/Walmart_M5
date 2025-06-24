# import librariesx
import polars as pl
import numpy as np 
from polars import selectors as cs 

from datetime import date
from package.datapreparation import DataAnalytics

import matplotlib.pyplot as plt
import seaborn as sns

import hvplot.polars

import panel as pn
import panel.widgets as pnw
pn.extension(comms="vscode")

# Usage Example
DataPrepSalesProcessing    = DataAnalytics(r"/Users/fredrik.hornell/Python/Private/Walmart_M5/data/4. processed/DataPrepSalesOOS_20241222_220143.parquet")




def delist(data, x, split='_', index =[0], col_name='name'):
    return data.with_columns(
            pl.concat_str(
                (pl.col(x).str.split(split).list.get(idx) for idx in index)
                , separator = '_'
            ).alias(col_name)
        )


df = (
    DataPrepSalesProcessing.load_data()
    .modify_data(
        lambda data:
            (
                data
                .pipe(delist,x='id',split='_',index=[0,1],col_name='category_dep')
                .pipe(delist,x='id',split='_',index=[3,4],col_name='state_store')
            )
    )
    .modify_data(
        lambda data:
            data
            .select(pl.col('*').exclude('id'))
            .group_by(['category_dep','state_store','date'])
            .agg(
                pl.col('sales').sum().alias('sales')
                , pl.col('sales').mean().alias('mean_sales')
                , pl.col('sales').len().alias('prodloc_count')
                , pl.col('zero_sales_ind').sum().alias('zero_sales_ind')
                , pl.col('zero_sales_ind').mean().alias('mean_zero_sales_ind')
                , pl.col('OOS').sum().alias('OOS')
                , pl.col('OOS').mean().alias('mean_OOS')
            )
            
    )
    .collect()
)


widget = pn.widgets.FloatSlider(start=0.0, end=10.0, value=2.0, step=0.1)

pn.widgets.Select(
    options=
    [
        "HOUSEHOLD_1", "HOUSEHOLD_2", "HOUSEHOLD_3",
        "HOBBIES_1", "HOBBIES_2", "HOBBIES_3",
        "FOODS_1", "FOODS_1", "FOODS_1"
    ]
)
pn.Column(widget, widget.param.value)

df.result.hvplot.line(
    x='date'
    , y= 'sales'
    , by='state_store'
    )