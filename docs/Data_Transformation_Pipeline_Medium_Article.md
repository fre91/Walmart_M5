# Building a Robust Data Transformation Pipeline for Walmart M5 Forecasting: From Raw Data to Design Matrix

*How we transformed messy retail data into a machine learning-ready design matrix using Polars and advanced feature engineering techniques*

---

## Introduction

In the competitive world of retail forecasting, the quality of your data transformation pipeline can make or break your predictive models. This article walks through a comprehensive data transformation pipeline developed for the Walmart M5 forecasting competition, demonstrating how we transformed raw retail data into a sophisticated design matrix ready for machine learning.

## The Challenge: Walmart M5 Dataset

The M5 dataset contains hierarchical sales data from Walmart stores across three US states (California, Texas, and Wisconsin) over 1,941 days. The data includes:
- **Sales data**: Daily unit sales for 3,049 products across 10 stores
- **Calendar data**: Date features, events, holidays, and SNAP (Supplemental Nutrition Assistance Program) indicators
- **Price data**: Selling prices for products
- **Product metadata**: Hierarchical product classifications (item → department → category)

Our goal was to transform this complex, multi-dimensional data into a clean, feature-rich design matrix suitable for advanced forecasting models.

## Pipeline Architecture

Our transformation pipeline follows a structured approach with clear separation of concerns:

```
Raw Data (1.external) → Processed Raw (2.raw) → Feature Engineering (3.interim) → Design Matrix (4.processed)
```

Let's dive into each stage:

## Stage 1: Raw Data Processing (`1.DataPrepToRaw.py`)

The first stage standardizes and cleans the raw data, preparing it for feature engineering.

### Sales Data Transformation
```python
DataPrepSalesRaw = (
    DataPrepSales.load_data(lazy=True)
    .transform_sales_to_long_format(
        drop_columns=['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
        keep_id_column='id',
        value_column_name='sales',
        date_column_name='d'
    )
    .join(
        DataPrepCalendar.load_data(lazy=True).select_columns(['d','date']),
        on=['d'],
        how='left'
    )
    .select_columns(['id','date','sales'])
)
```

**Key transformations:**
- Convert wide-format sales data (columns as dates) to long format
- Join with calendar data to get proper date objects
- Standardize column names and data types

### Price Data Enrichment
```python
DataPrepPriceRaw = (
    DataPrepPrice.load_data(lazy=True)
    .join(DataPrepProdLocRaw, on=['item_id','store_id'], how='left')
    .join(
        DataPrepCalendar.load_data(lazy=True).select_columns(['wm_yr_wk','date']),
        on=['wm_yr_wk'],
        how='left'
    )
    .select_columns(['id','date','sell_price'])
)
```

**Key transformations:**
- Join price data with product location metadata
- Align weekly price data with daily calendar dates
- Create unified product-location-date identifiers

## Stage 2: Feature Engineering (`2.*.py`)

This stage creates sophisticated features that capture temporal patterns, seasonal effects, and event impacts.

### 2.1 Calendar Features (`2.calendar_features.py`)

Calendar features capture the fundamental temporal patterns in retail data:

```python
calendar_interim = (
    calendar_raw.load_data(lazy=True)
    .with_columns([
        pl.col("date").dt.day().alias("day_of_month"),
        pl.col("date").dt.ordinal_day().alias("day_of_year")
    ])
    .with_columns([
        # Yearly cyclical features
        (pl.lit(2) * np.pi * pl.lit(i) * pl.col("day_of_year").mod(365.25)/pl.lit(365.25)).sin().alias(f'sin_{i}')
        for i in range(1, 4)
    ] + [
        (pl.lit(2) * np.pi * pl.lit(i) * pl.col("day_of_year").mod(365.25)/pl.lit(365.25)).cos().alias(f'cos_{i}')
        for i in range(1, 4)
    ])
)
```

**Key features created:**
- **Basic temporal features**: Day of month, day of year
- **Cyclical encodings**: Sine and cosine transformations for yearly and monthly patterns
- **Gaussian splines**: Non-linear representations of weekday and trend patterns
- **Weekday indicators**: Boolean features for each day of the week

### 2.2 Holiday and Event Features (`2.holiday_features.py`)

Events and holidays significantly impact retail sales:

```python
holidays = (
    pl.concat([
        data.select(pl.col('date'), pl.col('event_name_1').alias('event')).drop_nulls(),
        data.select(pl.col('date'), pl.col('event_name_2').alias('event')).drop_nulls()
    ])
    .pivot_and_lazy(
        index='date',
        on='event',
        values='value',
        aggregate_function=pl.any
    )
)
```

**Key features created:**
- **Event indicators**: Boolean features for each unique event
- **Event post-effects**: Features tracking 7 days after each event
- **Gaussian spline events**: Non-linear event impact modeling

### 2.3 SNAP Features (`2.snap_features.py`)

SNAP (Supplemental Nutrition Assistance Program) benefits are distributed on specific days, creating predictable sales patterns:

```python
snap_interim = (
    calendar_raw.load_data(lazy=True)
    .select_columns(['date', 'snap_CA', 'snap_TX', 'snap_WI'])
    .unpivot(index='date', on=['snap_CA', 'snap_TX', 'snap_WI'])
    .rename({'variable': 'state', 'value': 'snap_bool'})
)
```

**Key features created:**
- **State-specific SNAP indicators**: Boolean features for each state's SNAP distribution days
- **Long-format SNAP data**: Enables state-specific modeling

### 2.4 Product Location Features (`2.prod_loc_interim.py`)

Product-location metadata helps understand sales patterns:

```python
prod_loc_interim = (
    DataPrepProdLocRaw.load_data(lazy=True)
    .join(
        DataPrepSalesRaw.load_data(lazy=True)
        .filter(pl.col('sales') > 0)
        .group_by('id')
        .agg([
            pl.col('date').min().alias('first_sales_date'),
            pl.col('date').max().alias('last_sales_date'),
            pl.col('sales').sum().alias('total_sales')
        ])
    )
)
```

**Key features created:**
- **Sales history**: First/last sales dates and total sales volume
- **Date ranges**: Complete date coverage for each product-location combination

### 2.5 Sales Normalization (`2.sales_interim.py`)

Normalize sales data to ensure consistent date coverage:

```python
sales_interim = (
    DataPrepSalesRaw.load_data(lazy=True)
    .join(
        DataPrepProdLocInterim.load_data(lazy=True)
        .explode('prodloc_daterange').rename({'prodloc_daterange': 'date'}),
        on=['id', 'date'],
        how='inner'
    )
    .calculate_out_of_stock_periods(binomial_threshold=0.0001)
)
```

**Key transformations:**
- **Date alignment**: Ensure all products have data for the same date range
- **Out-of-stock detection**: Identify periods with zero sales

## Stage 3: Design Matrix Creation (`3.DesignMatric.py`)

The final stage assembles all features into a unified design matrix:

```python
DesignMatrix = (
    sales_prep.load_data(lazy=True)
    .pipe(delist, x='id', split='_', index=[3], col_name='state')
    .join(
        regressor_selection,
        on=['date', 'state'],
        how='inner'
    )
)
```

**Key transformations:**
- **Feature selection**: Choose the most relevant features for modeling
- **State extraction**: Parse state information from product IDs
- **Final join**: Combine sales data with all engineered features

## Advanced Techniques Used

### 1. Gaussian Mixture Models for Non-linear Patterns

We used Gaussian Mixture Models to capture complex temporal patterns:

```python
def create_gaussian_splines(df, ordinal_col, n_components=3):
    X = df[ordinal_col].to_numpy().reshape(-1,1)
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(X)
    latent_features = gmm.predict_proba(X)
    return df.with_columns(
        pl.from_numpy(latent_features, col_name).to_struct().alias(f'{ordinal_col}_guassian_splines')
    )
```

### 2. Cyclical Encoding for Temporal Features

Cyclical encoding prevents discontinuities in temporal features:

```python
# Yearly cyclical features
(pl.lit(2) * np.pi * pl.lit(i) * pl.col("day_of_year").mod(365.25)/pl.lit(365.25)).sin().alias(f'sin_{i}')
```

### 3. Lazy Evaluation with Polars

Throughout the pipeline, we use Polars' lazy evaluation for memory efficiency:

```python
calendar_interim = (
    calendar_raw.load_data(lazy=True)  # Lazy loading
    .modify_data(lambda data: ...)     # Lazy transformations
    .write_parquet(sink=True, ...)     # Only executes when needed
)
```

## Performance Benefits

### 1. Memory Efficiency
- **Lazy evaluation**: Operations are only executed when needed
- **Streaming processing**: Large datasets processed without loading entirely into memory
- **Optimized joins**: Polars' efficient join algorithms

### 2. Processing Speed
- **Rust-based backend**: Polars provides near-native performance
- **Parallel processing**: Automatic parallelization of operations
- **Vectorized operations**: Efficient array-based computations

### 3. Data Quality
- **Type safety**: Strong typing prevents data type errors
- **Null handling**: Consistent null value treatment
- **Schema validation**: Automatic schema checking and updates

## Key Lessons Learned

### 1. Modular Design
Breaking the pipeline into focused modules makes it:
- **Maintainable**: Easy to modify individual components
- **Testable**: Each stage can be tested independently
- **Reusable**: Components can be reused across projects

### 2. Feature Engineering Strategy
- **Domain knowledge**: Understanding retail patterns is crucial
- **Temporal features**: Cyclical encoding captures seasonal patterns effectively
- **Event modeling**: Events and holidays require sophisticated feature engineering

### 3. Data Pipeline Best Practices
- **Reproducibility**: Timestamped outputs ensure reproducible results
- **Error handling**: Robust error handling prevents pipeline failures
- **Documentation**: Clear documentation enables team collaboration

## Conclusion

This data transformation pipeline demonstrates how to build a production-ready system for retail forecasting. By combining modern data engineering tools (Polars) with domain-specific feature engineering, we created a robust foundation for machine learning models.

The key to success lies in:
1. **Understanding the domain**: Retail forecasting has unique challenges
2. **Choosing the right tools**: Polars provides the performance and flexibility needed
3. **Modular design**: Breaking complex transformations into manageable pieces
4. **Feature engineering**: Creating features that capture the underlying patterns

This pipeline serves as a template for other time series forecasting projects, demonstrating how to transform raw data into a machine learning-ready format while maintaining performance and scalability.

---

*The complete codebase and additional documentation can be found in the project repository. This pipeline was developed for the Walmart M5 forecasting competition and can be adapted for other retail forecasting applications.* 