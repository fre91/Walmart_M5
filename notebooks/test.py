

pl_plot = (
    DataPrepSalesInterimOOS
    .result
    .pipe(delist,x='id',split='_',index=[1],col_name='department')
    #.pipe(delist,x='id',split='_',idx=2,col_name='item')
    #.pipe(delist,x='id',split='_',idx=3,col_name='state')
    #.pipe(delist,x='id',split='_',idx=4,col_name='store')
    #.pipe(delist,x='id',split='_',idx=5,col_name='eval')'
    .pipe(delist,x='id',split='_',index=[0,1],col_name='category_dep')
    .pipe(delist,x='id',split='_',index=[3,4],col_name='state_store')
    .group_by(['department','category_dep','state_store'])
    .agg(
        (pl.col('OOS').sum().truediv(pl.col('zero_sales_ind').sum()) * 100).alias('pct_oos_of_zero_sale')
        , (pl.col('OOS').sum().truediv(pl.col('zero_sales_ind').len()) * 100).alias('pct_oos_of_total_sales')
        , (pl.col('zero_sales_ind').sum().truediv(pl.col('zero_sales_ind').len()) * 100).alias('pct_zero_of_total_sales')
        , pl.col('OOS').sum().alias('oos_zero_sale')
        , pl.col('zero_sales_ind').sum().sub(pl.col('OOS').sum()).alias('non_oss_zero_sale')
        , pl.col('zero_sales_ind').sum().alias('zero_sales_ind')
        , pl.col('zero_sales_ind').len().sub(pl.col('zero_sales_ind').sum()).alias('sales_ind')
        , pl.col('zero_sales_ind').len().alias('count_possible_sale')
        , pl.col('sales').mean().alias('mean')
        , pl.col('sales').filter(pl.col('OOS')==0).mean().alias('mean_excl_oos')
        , pl.col('sales').median().alias('median')
        , pl.col('sales').filter(pl.col('OOS')==0).median().alias('median_excl_oos')
        , pl.col('sales').sum().log10().alias('log10Sum')
        , pl.col('sales').var().alias('var')
        , pl.col('sales').filter(pl.col('OOS')==0).var().alias('var_excl_oos')
        , pl.col('sales').std().alias('std')
        , pl.col('sales').filter(pl.col('OOS')==0).std().alias('std_excl_oos')
        #, pl.col('sales').filter()
    )
    #.pipe(delist,x='id',split='_',index=[0,1],col_name='category_dep')
    #.pipe(delist,x='id',split='_',index=[3,4],col_name='state_store')
    
)

pl_plot


DataPrepSalesInterim.result