# Building a Robust Data Transformation Pipeline for Walmart M5 Forecasting: From Raw Data to Feature Engineering

*How we transformed messy retail data into a machine learning-ready dataset using Polars, lazy evaluation, and advanced feature engineering techniques*

---

## Introduction

In the competitive world of retail forecasting, the quality of your data transformation pipeline can make or break your predictive models. This article walks through a comprehensive data transformation pipeline developed for the Walmart M5 forecasting competition, demonstrating how we transformed raw retail data into a sophisticated feature-rich dataset ready for machine learning.

We'll explore the `DataPreparation` class that serves as the foundation for all transformations, the power of Polars for efficient data processing, and the sophisticated feature engineering techniques that capture the complex patterns in retail data.

## The Challenge: Walmart M5 Dataset

The M5 dataset contains hierarchical sales data from Walmart stores across three US states (California, Texas, and Wisconsin) over 1,941 days. The data includes:
- **Sales data**: Daily unit sales for 3,049 products across 10 stores
- **Calendar data**: Date features, events, holidays, and SNAP (Supplemental Nutrition Assistance Program) indicators
- **Price data**: Selling prices for products
- **Product metadata**: Hierarchical product classifications (item → department → category)

Our goal was to transform this complex, multi-dimensional data into a clean, feature-rich dataset suitable for advanced forecasting models.

## Why Polars? The Power of Lazy Evaluation

Before diving into our pipeline, let's understand why we chose **Polars** as our data processing engine and how lazy evaluation transforms our approach to data engineering.

### Polars: The Modern Data Processing Engine

Polars is a lightning-fast DataFrame library implemented in Rust, designed for high-performance data manipulation. Unlike pandas, Polars offers:

- **Rust-based backend**: Near-native performance for data operations
- **Memory efficiency**: Optimized memory usage through columnar storage
- **Parallel processing**: Automatic parallelization of operations
- **Type safety**: Strong typing prevents data type errors
- **Lazy evaluation**: The game-changer for large-scale data processing

### Lazy Evaluation: Building Computation Graphs

Lazy evaluation is the cornerstone of our data processing strategy. Instead of executing operations immediately, Polars builds a **computation graph** that only executes when needed:

```python
# Eager evaluation (immediate execution)
df = pl.read_parquet("large_file.parquet")  # Loads entire file into memory
result = df.filter(pl.col("sales") > 0)     # Executes immediately

# Lazy evaluation (builds computation graph)
df = pl.scan_parquet("large_file.parquet")  # Creates a scan operation
result = df.filter(pl.col("sales") > 0)     # Adds filter to computation graph
# Nothing executes until we call .collect() or .sink_parquet()
```

**Benefits of Lazy Evaluation:**
- **Memory efficiency**: Only loads data when necessary
- **Query optimization**: Polars can optimize the entire computation graph
- **Streaming processing**: Large datasets processed without loading entirely into memory
- **Reproducible pipelines**: Computation graphs can be saved and reused

## The DataPreparation Class: Our Foundation

The `DataPreparation` class is a sophisticated wrapper around Polars that provides a fluent, chainable interface for data transformations. It encapsulates common data processing operations and provides a consistent API across the entire pipeline.

### Class Architecture

```python
class DataPreparation:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None      # Holds the LazyFrame or DataFrame
        self.result = None    # Holds collected results
        self.schema = None    # Schema information
```

### Key Methods and Capabilities

#### 1. Data Loading with Lazy Evaluation

```python
def load_data(self, lazy=True):
    if lazy:
        self.data = pl.scan_parquet(self.file_path)  # Creates computation graph
    else:
        self.data = pl.read_parquet(self.file_path)  # Eager loading
    # Update schema information
    if isinstance(self.data, pl.LazyFrame):
        self.schema = self.data.schema
    else:
        self.schema = self.data.schema
    return self  # Enables method chaining
```

#### 2. Data Transformation Methods

**`transform_sales_to_long_format()`**
Converts wide-format sales data (dates as columns) to long format, essential for time series analysis:

```python
def transform_sales_to_long_format(self, drop_columns=['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], 
                                  keep_id_column='id', value_column_name='sales', date_column_name='day'):
    self.data = (
        self.data
        .drop(drop_columns)
        .melt(id_vars=keep_id_column)  # Wide to long transformation
        .rename({'variable': date_column_name, 'value': value_column_name})
        .with_columns([
            pl.col(value_column_name).cast(pl.Int16),
        ])
    )
    return self
```

**`modify_data(expression)`**
Generic method for applying custom transformations while maintaining lazy evaluation:

```python
def modify_data(self, expression):
    self.data = expression(self.data)  # Applies lambda function to data
    return self
```

#### 3. Data Operations

**`join(other, on, how='inner')`**
Performs joins between DataPreparation instances with validation:

```python
def join(self, other, on: list[str], how: str = 'inner'):
    valid_hows = {'inner', 'left', 'right', 'full', 'semi', 'anti', 'cross'}
    if how not in valid_hows:
        raise ValueError(f"Invalid join type: {how}")
    self.data = self.data.join(other.data, on=on, how=how)
    return self
```

**`pivot_and_lazy()`**
Performs pivot operations and converts to lazy evaluation:

```python
def pivot_and_lazy(self, index=None, on=None, values=None, aggregate_function=None):
    if isinstance(self.data, pl.LazyFrame):
        df = self.data.collect()
    else:
        df = self.data
    pivot_kwargs = dict(index=index, on=on, values=values)
    if aggregate_function is not None:
        pivot_kwargs['aggregate_function'] = aggregate_function
    self.data = (
        df
        .pivot(**pivot_kwargs)
        .fill_null(0)
        .lazy()  # Convert back to lazy evaluation
    )
    return self
```

#### 4. Output and Persistence

**`write_parquet()`**
Saves data with timestamped filenames and supports different pipeline stages:

```python
def write_parquet(self, sink=True, name=None, path='1.external', subfolder=None):
    date_now = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Path mapping for different pipeline stages
    path_map = {
        '1.external': os.path.join(project_root, 'data', '1.external'),
        '2.raw': os.path.join(project_root, 'data', '2.raw'),
        '3.interim': os.path.join(project_root, 'data', '3.interim'),
        '4.processed': os.path.join(project_root, 'data', '4.processed'),
    }
    # ... file path construction
    if sink:
        if isinstance(self.data, pl.LazyFrame):
            self.data.sink_parquet(file_path, row_group_size=100000)  # Efficient streaming write
        else:
            self.data.write_parquet(file_path)
    return self
```

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
    DataPrepSales.load_data(lazy=True)  # Lazy loading for memory efficiency
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

This stage creates features that capture temporal patterns, seasonal effects, and event impacts.

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

**Key features created from this:**
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

This is one of the most critical transformations, ensuring data quality and consistency:

```python
sales_interim = (
    DataPrepSalesRaw.load_data(lazy=True)
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
)
```

#### The Join Operation: Tackling Leading Days of Zero Sales

This join addresses a critical issue in retail data:

**What This Join Accomplishes:**
- **Leading Zero Elimination**: Eliminates periods before a product was introduced

**The Process:**
1. **Product Location Data**: Contains `prodloc_daterange` - a list of dates from first sale to the end of the dataset
2. **Explode Operation**: Converts the date range list into individual rows
3. **Inner Join**: Only keeps dates where both sales data and product availability exist
4. **Result**: Clean dataset with consistent date coverage across all products

#### The `calculate_out_of_stock_periods()` Method: Statistical Out-of-Stock Detection

This method implements an algorithm to detect unlikely periods of zero sales using binomial probability theory.

**Algorithm Overview:**
```python
def calculate_out_of_stock_periods(self, list_of_ids=None, binomial_threshold=0.001):
    # 1. Create zero sales indicator
    # 2. Identify zero sales periods
    # 3. Calculate probability of zero sales for each product
    # 4. Apply binomial test to detect unlikely zero sales periods
    # 5. Mark periods as out-of-stock (OOS)
```

**Step-by-Step Breakdown:**

**Step 1: Zero Sales Indicator**
```python
df_zero_ind = (self.data.lazy()
    .select(pl.col('*'), 
            pl.when(pl.col(target_col) == 0).then(1).otherwise(0).alias('zero_sales_ind'))
)
```

**Step 2: Zero Sales Period Identification**
```python
df_zero_sales_period = ( 
    df_zero_ind
    .sort([id_col, time_col])
    .select(
        pl.col('*'),
        (pl.col('zero_sales_ind') - pl.col('zero_sales_ind').shift(1).over(id_col)).alias('delta')
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
```

This operation:
- **Delta Calculation**: Identifies transitions between sales and zero sales
- **Period Grouping**: Groups consecutive zero sales days into periods
- **Period Labeling**: Assigns unique identifiers to each zero sales period

**Step 3: Probability Calculation**
```python
df_zero_sales_probability = (
    df_zero_ind.group_by(id_col)
    .agg(pl.col('zero_sales_ind').mean().alias('prob_zero_sales'))
)
```

**Step 4: Binomial Test Application**
```python
df_zero_sales_period_aggregated =(
    df_zero_sales_period
    .group_by('id','zero_sales_period').agg(pl.col('zero_sales_ind').sum().alias('nbr_zero_sales_in_a_row')) 
    .join(df_zero_sales_probability, on=[id_col])
    .with_columns(
        pl.when(
            (pl.col('prob_zero_sales') ** pl.col('nbr_zero_sales_in_a_row') * 
             (1 - pl.col('prob_zero_sales')) ** (pl.col('nbr_zero_sales_in_a_row') - pl.col('nbr_zero_sales_in_a_row'))
            ) <= binomial_threshold
        ).then(1).otherwise(0).alias('OOS')
    )
)
```

#### The Binomial Probability Formula:

The algorithm uses the binomial probability formula:
```
P(X = k) = C(n,k) * p^k * (1-p)^(n-k)
```

Where:
- `p` = historical probability of zero sales for the product
- `k` = number of consecutive zero sales days
- `n` = total number of days in the period

**Interpretation:**
- If the probability of observing `k` consecutive zero sales days is ≤ `binomial_threshold` (0.0001), the period is marked as out-of-stock
- This threshold represents a 0.01% chance - extremely unlikely under normal circumstances

## Practical Example

Consider a product with the following sales pattern:
```
Date    Sales   Zero_Ind
Day 1   5       0
Day 2   0       1
Day 3   0       1
Day 4   0       1
Day 5   3       0
Day 6   0       1
Day 7   0       1
Day 8   4       0
```

**Analysis:**
1. **Period 1**: Days 2-4 (3 consecutive zero sales)
2. **Period 2**: Days 6-7 (2 consecutive zero sales)
3. **Historical probability**: 5/8 = 0.625 (62.5% zero sales rate)

**Binomial Test:**
- Period 1: P(3 consecutive zeros) = 0.625³ = 0.244 (24.4% chance) → **Not OOS**
- Period 2: P(2 consecutive zeros) = 0.625² = 0.391 (39.1% chance) → **Not OOS**

However, if the historical probability was 0.1 (10% zero sales rate):
- Period 1: P(3 consecutive zeros) = 0.1³ = 0.001 (0.1% chance) → **OOS** (≤ 0.0001 threshold)
- Period 2: P(2 consecutive zeros) = 0.1² = 0.01 (1% chance) → **Not OOS**

## Conclusion

This data transformation pipeline demonstrates how to build a production-ready system for retail forecasting. By combining modern data engineering tools (Polars) with domain-specific feature engineering and sophisticated statistical methods, we created a robust foundation for machine learning models.

The key to success lies in:
1. **Understanding the domain**: Retail forecasting has unique challenges
2. **Choosing the right tools**: Polars provides the performance and flexibility needed
3. **Lazy evaluation**: Building computation graphs for memory efficiency
4. **Modular design**: Breaking complex transformations into manageable pieces
5. **Feature engineering**: Creating features that capture the underlying patterns
6. **Statistical rigor**: Using probability theory for data quality issues

The `DataPreparation` class serves as the backbone of this pipeline, providing a consistent, chainable interface that makes complex transformations readable and maintainable. The combination of lazy evaluation, statistical out-of-stock detection, and sophisticated feature engineering creates a powerful foundation for retail forecasting applications.

---

*The complete codebase and additional documentation can be found in the project repository: https://github.com/fre91/Walmart_M5/tree/main. This pipeline was developed for the Walmart M5 forecasting competition and can be adapted for other retail forecasting applications.* 