from models.package.utils import get_path_to_latest_file

lasso_expr = (
    pl.col("sales")
    .least_squares
    .lasso(*X, alpha=0.0001, add_intercept=True).over("state", "store_id")
)

reg_wighted = (
    pl.col("sales")
    .least_squares
    .ols(*X, alpha=0.0001, add_intercept=True,  mode="residuals",sample_weights=pl.col("sample_weights")).over("state", "store_id")
)

lasso_coef = (
    pl.col("sales")
    .least_squares
    .lasso(*X, alpha=0.0001, add_intercept=True,  mode="coefficients")
    .over("state", "store_id").alias("coefficients_group")
)
lasso_resi = (
    pl.col("sales")
    .least_squares
    .lasso(*X, alpha=0.0001, add_intercept=True,  mode="residuals")
    .over("state", "store_id").alias("residuals_group")
)



test_train.select('state','store_id','coefficients_group').unnest('coefficients_group').unpivot(index=['state','store_id'])



test_train.sort(by='date').to_pandas().plot.scatter(x='reg_wighted', y= 'sum1' )
test_train.sort(by='date').to_pandas().plot.scatter(x='residual', y= 'sum1' )


test_train.sort(by='date').to_pandas().plot(x='date', y= ['reg_wighted','residual'] )


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
        , pl.col('state').unique()
        , pl.col('store_id')
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




holiday_bool_features = get_path_to_latest_file('3.interim','holiday_bool_features')

DataPrepCalendarRaw = DataPreparation(holiday_bool_features)


test = (
    DataPrepCalendarRaw
    .load_data()
    .collect()
    .result
    .with_columns(
        pl.col('date').dt.month().alias('month')
        , pl.col('date').dt.day().alias('day_month')
    )
    .unnest('holidays')
    .unpivot(index=['date','day_month','month'])
    .filter(pl.col('value')==1)
)

residual_frame.join(
    test
    , how = 'inner'
    , on = ['month' , 'day_month']
).sort(by='residual_count').select('variable').unique().to_series().to_list()


test_train.select('state','store_id','coefficients_group').unique(maintain_order=True).unnest('coefficients_group')