# import librariesx
import polars as pl
import numpy as np 
from polars import selectors as cs 
import polars_ols as pls 

from datetime import date
from pathlib import Path
from package.datapreparation import DataAnalytics,DataPreparation
import os
from datetime import timedelta  # Add this import



from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from datetime import date

import seaborn as sns
import hvplot.polars
import panel as pn
import panel.widgets as pnw
import holoviews as hv
import pandas as pd

import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go

from package.utils import get_path_to_latest_file

POLARS_VERBOSE=1

def create_trading_periods(df:pl.DataFrame, ordinal_col , n_components=3):
    
    
    if isinstance(df, pl.LazyFrame):
        df = df.collect()
    elif not isinstance(df, pl.DataFrame):
        raise TypeError("Input must be either a Polars DataFrame or LazyFrame")
    
    X = df[ordinal_col].to_numpy().reshape(-1,1)
    
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(X)

    latent_features = gmm.predict_proba(X)
    col_name = [f'{ordinal_col}_{i+1}'for i in range(n_components)]

    return df.with_columns(
        pl.from_numpy(latent_features, col_name).to_struct().alias(f'{ordinal_col}_regimes')
    )

def generate_sample_weights(df:pl.DataFrame):
    
    num_samples = df.len()

    
    return df.with_columns(
        sample_weights = pl.linear_space(start=0.25, end=1.0, num_samples=num_samples)
    )

def create_rolling_train_test_splits(df, start_date, end_date, training_years=3, forecast_days=30):
    """
    Creates rolling window splits for time series forecasting.
    
    Args:
        df (pl.DataFrame): Input DataFrame with date column and features
        start_date (date): Start date for the first training window
        end_date (date): End date for the entire analysis
        training_years (int): Number of years for training window
        forecast_months (int): Number of months to forecast
    
    Returns:
        list: List of tuples (train_df, test_df, test_start_date) for each split
    
    TODO:
        - Add previous year's values as naive forecast baseline
        - Include data quality checks for each split
    """
    splits = []
    current_date = start_date
    
    while current_date + timedelta(days=forecast_days) <= end_date:
        # Calculate window boundaries
        train_start = current_date - timedelta(days=training_years * 365)
        test_end = current_date + timedelta(days=forecast_days)  # Approximately one month
        
        df_filtered = (
            df
            .filter(pl.col('date').is_between(train_start,test_end))
            .pipe(create_trading_periods , ordinal_col = 'trend',n_components= 12)
            .unnest('trend_regimes')
            .sort(by='trend')
            
        )
        
        print("df_filtered:{}".format(df_filtered.shape))
        
        a = 0.0   # Minimum value (bottom cap)
        b = 1.0   # Maximum value (top cap)
        k = 0.01     # Steepness
        x0 = df_filtered.select('trend').median()
        
        
        sample_weights_frame = (
            df_filtered
            .select('date','trend', *holidays_features_cols)
            .unique()
            .filter(
                (pl.col('date') >= train_start) & 
                (pl.col('date') < current_date)
            )
            .sort(by='date')
            .with_columns(
                sample_weights = pl.linear_space(start=0.2, end=1, num_samples=pl.len())
                , sample_weights_x2= pl.linear_space(start=0.2, end=1, num_samples=pl.len()).mul(pl.linear_space(start=0.2, end=1, num_samples=pl.len()))
                , sample_weights_shifted_logistic = (a + (b - a) * (1 / (1 + (-(pl.col('trend') - x0) * k).exp())))
            )
            .with_columns(
                    sample_weights_x2 = pl.when(pl.sum_horizontal(*holidays_features_cols)>0).then(pl.lit(1)).otherwise(pl.col('sample_weights_x2'))
                    ,  sample_weights_shifted_logistic = pl.when(pl.sum_horizontal(*holidays_features_cols)>0).then(pl.lit(1)).otherwise(pl.col('sample_weights_shifted_logistic'))
                    ,  sample_weights = pl.when(pl.sum_horizontal(*holidays_features_cols)>0).then(pl.lit(1)).otherwise(pl.col('sample_weights'))
            )
            .select(pl.all().exclude(*holidays_features_cols))
        )
        # Create train and test sets
        train_df = (
            df_filtered
            .filter(
                (pl.col('date') >= train_start) & 
                (pl.col('date') < current_date)
            )
            .join(
                sample_weights_frame
                , how = 'left'
                , on = ['date','trend']
            )
            .fill_null(0)
            .with_columns(
                rolling_row_std=pl.col("sales").rolling_std_by("date", window_size="30d").over([ 'state','store_id']).fill_null(strategy="backward")
            )
            )
        
        print("train_df:{}".format(train_df.shape))
        
        test_df = (
            df_filtered
            .filter(
                (pl.col('date') >= current_date) & 
                (pl.col('date') < test_end)
            )
            .join(
                sample_weights_frame
                , how = 'left'
                , on = ['date','trend']
            )
            .fill_null(0)
             .with_columns(
                rolling_row_std=pl.col("sales").rolling_std_by("date", window_size="30d").over([ 'state','store_id']).fill_null(strategy="backward")
            )
        )
        
        print("test_df:{}".format(test_df.shape))
        
        splits.append((train_df, test_df, current_date))
        
        # Move forward one month for the next split
        current_date = test_end
        
        
    for index, (train_df, test_df, test_start_date) in enumerate(splits):
        print(f"\nSplit {index} at {test_start_date}")
        print(f"Training set: {train_df['date'].min()} to {train_df['date'].max()}")
        print(f"Test set: {test_df['date'].min()} to {test_df['date'].max()}")
    
    return splits

cols = [
#    'id',
    'date',
    'state',
    'store_id',
    'sales',
#    'zero_sales_ind',
    'OOS',
    'trend',
    'trend_regimes',
    'wday_regimes',
    'trig_features',
    'holidays',
    'event_struct',
    'snap'
]

df = (
    pl.scan_parquet(get_path_to_latest_file('4.processed','designmatrix'))
    .join(
        (
            pl.scan_parquet(get_path_to_latest_file('3.interim','prodlocs'))
            .select(pl.col('id'),pl.col('store_id'))
        )
        , on = 'id'
        , how ='inner'
    )
    .select(
        *cols
    )
    .group_by(pl.all().exclude('sales','OOS'))
    .agg(
        pl.col('sales').sum().alias('sales')
        , pl.col('OOS').sum().alias('OOS_COUNT')
        , pl.col('OOS').len().alias('ProductPlaces')
    )
    .collect(engine="streaming")
)

# After loading df and before using the column lists
column_groups = {
    "trend_regimes": df.select("trend_regimes").unnest("trend_regimes").columns,
    "wday_regimes": df.select("wday_regimes").unnest("wday_regimes").columns,
    "trig_features": df.select("trig_features").unnest("trig_features").columns,
    "holidays": df.select("holidays").unnest("holidays").columns,
    "event_struct": df.select("event_struct").unnest("event_struct").columns,
    "weekday_boolen": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
}

# Create the date range
date_range = pl.date_range(
    start= date(2010, 1,1),
    end= date(2022, 1,1),
    interval="1d",
    eager= True
)

# Create the DataFrame
weekday_boolen =(
    pl.DataFrame({"date": date_range})
    .with_columns(
        pl.col('date').dt.weekday().alias('weekday_nbr')
        ,pl.col('date').dt.weekday().replace_strict([1, 2, 3,4,5,6,7],column_groups["weekday_boolen"]).alias("weekday")   
    )
    .pivot(
        index = 'date'
        , on = "weekday"
        , aggregate_function= "len"
    )
    .fill_null(0)
)


df_unnested = (
    df.select(pl.all().exclude('trend_regimes')).unnest(*column_groups["trend_regimes"])
    .join(
        weekday_boolen
        , on='date'
        , how = 'inner'
    )
)

start_date = date(2013, 1,29)  # Adjust based on your data
end_date = date(2016, 1, 29)  # Adjust based on your data

splits = create_rolling_train_test_splits(df_unnested, start_date, end_date)


# forecasting + backtesting
# Define variables and forecast dimensions
Y = pl.col('sales')
X1 = [
    *column_groups["trend_regimes"],
    *column_groups["weekday_boolen"],
    *column_groups["trig_features"],
    *column_groups["holidays"],
    *column_groups["event_struct"]
]
X2 = [
    *column_groups["trend_regimes"],
    *column_groups["wday_regimes"],
    *column_groups["trig_features"],
    *column_groups["holidays"],
    *column_groups["event_struct"]
]
over = ['state','store_id'] # time-serie dimensions

lasso_expr = (Y.least_squares.lasso(*X2, alpha=0.0001, add_intercept=True).over(*over))
lasso_coef = (Y.least_squares.lasso(*X2, alpha=0.0001, add_intercept=True, mode="coefficients").over(*over))
lasso_resi = (Y.least_squares.lasso(*X2, alpha=0.0001, add_intercept=True, mode="residuals").over(*over))

weighted_lasso_expr = (Y.least_squares.lasso(*X1, alpha=0.0001, add_intercept=True, sample_weights=pl.col('sample_weights_shifted_logistic')).over(*over))
weighted_lasso_coef = (Y.least_squares.lasso(*X1, alpha=0.0001, add_intercept=True, sample_weights=pl.col('sample_weights_shifted_logistic') ,mode="coefficients").over(*over))
weighted_lasso_resi = (Y.least_squares.lasso(*X1, alpha=0.0001, add_intercept=True, sample_weights=pl.col('sample_weights_shifted_logistic'), mode="residuals").over(*over))


all_splits = []

for idx, inner in enumerate(splits):
    print(idx,inner[2])
    
    train = (
        inner[0]
        .with_columns(
            lasso_coefficients_group = lasso_coef,
            lasso_prediction = lasso_expr,
            lasso_residual = lasso_resi,
            weigthed_lasso_coefficients_group = weighted_lasso_coef,
            weigthed_lasso_prediction = weighted_lasso_expr,
            weigthed_lasso_residual = weighted_lasso_resi,
            set_type = pl.lit("train"),
            set_first_date = pl.lit( inner[2]), # sample split start dt
            set_index = pl.lit(idx)
            
        )
    )
    
    pred  = (
        inner[1]
        .join(
            train
            .select(*over, 'lasso_coefficients_group','weigthed_lasso_coefficients_group' ).unique(maintain_order=True), # ,'weigthed_lasso_coefficients_group'
            on = over,
            how = 'inner'
        )
        .with_columns(
            pl.col("lasso_coefficients_group")
            .least_squares
            .predict(
                *X2, 
                add_intercept=True,
                name="lasso_prediction"
            ),
            pl.col("weigthed_lasso_coefficients_group")
            .least_squares
            .predict(
                *X1, 
                add_intercept=True,
                name="weigthed_lasso_prediction"
            ),
        )
        .with_columns(
            lasso_residual = pl.col('sales') - pl.col('lasso_prediction')
            , weigthed_lasso_residual =  pl.col('sales') - pl.col('weigthed_lasso_prediction')
            , set_type = pl.lit("pred")
            , set_first_date = pl.lit( inner[2]) # sample split start dt
            , set_index = pl.lit(idx)
        )
    )

        # Concatenate train and pred
    combined = pl.concat([train, pred], how='align')
    # add the split
    all_splits.append(combined)

    

all_data = pl.concat(all_splits)
all_data = all_data.sort(by=['date','state', 'store_id'])
mask = (pl.col('store_id')=='CA_4') | (pl.col('set_type')=='pred')

import seaborn as sns
sns.set_theme(style="dark")

# Load the example tips dataset
cols = ['date','store_id','set_type','lasso_residual','lasso_prediction','weigthed_lasso_residual','weigthed_lasso_prediction','rolling_row_std','sales','set_index','snap' ,*column_groups["holidays"]]
df_plot = all_data.select(*cols)

sns.displot(
    data=df_plot 
    , x = 'prediction'
    , y = 'sales'
    , hue = 'set_type'
    , row = 'store_id'
    , kind = 'hist'

)
sns.displot(
    data=df_plot 
    , x = 'prediction'
    , y = 'residual'
    , hue = 'set_type'
    , row = 'store_id'
    , kind = 'hist'
)
sns.violinplot(
    data=df_plot 
    , x="store_id", y="residual"
    , hue="set_type"
    , split=True
    , inner="quart"
    , fill=False
    , palette={"train": "g", "pred": ".35"}
)
df_plot.group_by('set_type').agg(
    wape = pl.col("residual").abs().sum()/pl.col('sales').sum(),
    rmse = pl.col("residual").pow(2).mean().sqrt(),
    bias = pl.col('prediction').sum()/pl.col('sales').sum() ,
)

df_pred = df_plot.filter(pl.col('store_id')=='CA_2').unpivot(index=['store_id', 'date','set_type','set_index','snap' ,*column_groups["holidays"] ], on =['lasso_residual','lasso_prediction','weigthed_lasso_residual','weigthed_lasso_prediction','rolling_row_std','sales']).sort(by=['set_index','set_type','date','variable']).to_pandas()


df_pred.describe()


import plotly.express as px

# Assume df_pred is your pandas DataFrame as before
fig = px.line(
    df_pred,
    x="date",
    y="value",
    color="variable",
    line_dash="set_type",
    facet_row=None,
    title="Sales vs Prediction by Set Index",
    labels={"value": "Value", "date": "Date"},
    animation_frame="set_index",
    color_discrete_map={"sales": "blue", "prediction": "orange", "residual":"black"},
    line_dash_map={"train": "solid", "pred": "dash"},
    width=1200,
    height=800
)
fig.update_layout(transition={'duration': 500})
fig.show()

# List of time series variables and event columns
series_options = df_pred["variable"].unique()
event_cols = [
    'SuperBowl', 'ValentinesDay', 'PresidentsDay', 'LentStart', 'LentWeek2', 'StPatricksDay',
    'Purim End', 'OrthodoxEaster', 'Pesach End', 'Cinco De Mayo', "Mother's day", 'MemorialDay',
    'NBAFinalsStart', 'NBAFinalsEnd', "Father's day", 'IndependenceDay', 'Ramadan starts',
    'Eid al-Fitr', 'LaborDay', 'ColumbusDay', 'Halloween', 'EidAlAdha', 'VeteransDay',
    'Thanksgiving', 'Christmas', 'Chanukah End', 'NewYear', 'OrthodoxChristmas',
    'MartinLutherKingDay', 'Easter', 'snap'
]

# App layout
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("Sales vs Prediction by Set Index with Event Markers"),
    html.Div([
        html.Label("Select Series:"),
        dcc.Checklist(
            id='series-selector',
            options=[{'label': s, 'value': s} for s in series_options],
            value=list(series_options),
            inline=True
        ),
    ]),
    html.Div([
        html.Label("Select Events:"),
        dcc.Dropdown(
            id='event-selector',
            options=[{'label': e, 'value': e} for e in event_cols],
            value=[],
            multi=True,
            placeholder="Select events to display"
        ),
    ], style={"marginTop": "1em"}),
    html.Div([
        html.Label("Select Set Index:"),
        dcc.Slider(
            id='set-index-slider',
            min=int(df_pred["set_index"].min()),
            max=int(df_pred["set_index"].max()),
            step=1,
            value=int(df_pred["set_index"].min()),
            marks={int(i): str(i) for i in sorted(df_pred["set_index"].unique())},
            tooltip={"placement": "bottom", "always_visible": True}
        ),
    ], style={"marginTop": "1em"}),
    dcc.Graph(id='main-graph', style={"height": "700px", "width": "100%"}),
])

@app.callback(
    Output('main-graph', 'figure'),
    Input('series-selector', 'value'),
    Input('event-selector', 'value'),
    Input('set-index-slider', 'value')
)
def update_figure(selected_series, selected_events, selected_set_index):
    # Filter data for selected set_index and series
    filtered = df_pred[
        (df_pred["set_index"] == selected_set_index) &
        (df_pred["variable"].isin(selected_series))
    ]
    fig = go.Figure()

    # Add time series traces
    color_map = {"sales": "blue", "prediction": "orange", "residual": "black"}
    dash_map = {"train": "solid", "pred": "dash"}
    for var in selected_series:
        for stype in filtered["set_type"].unique():
            sub = filtered[(filtered["variable"] == var) & (filtered["set_type"] == stype)]
            fig.add_trace(go.Scatter(
                x=sub["date"], y=sub["value"],
                mode="lines",
                name=f"{var} ({stype})",
                line=dict(color=color_map.get(var, None), dash=dash_map.get(stype, "solid"))
            ))

    # Add event markers as vertical lines
    for event in selected_events:
        # Find dates where event occurs (for this set_index)
        event_dates = df_pred[
            (df_pred["set_index"] == selected_set_index) & (df_pred[event] == 1)
        ]["date"].unique()
        for d in event_dates:
            fig.add_vline(
                x=d,
                line_width=1,
                line_dash="dot",
                line_color="gray",
                annotation_text=event,
                annotation_position="top left"
            )

    fig.update_layout(
        title=f"Set Index: {selected_set_index}",
        xaxis_title="Date",
        yaxis_title="Value",
        legend_title="Variable (Set Type)",
        width=1200,
        height=700,
        template="plotly_white"
    )
    return fig

if __name__ == '__main__':
    app.run(debug=True)
    
    


