"""
SNAP Features Processing

Processes SNAP (Supplemental Nutrition Assistance Program) data
for California, Texas, and Wisconsin.
"""

import polars as pl
from pathlib import Path
from package.datapreparation import DataPreparation
from package.utils import get_path_to_latest_file

def create_snap_features():
    project_root = Path(__file__).parent.parent
    
    calendar_raw = DataPreparation(get_path_to_latest_file('2.raw', 'calendar_raw'))

    snap_interim = (
        calendar_raw
        .load_data(lazy=True)
        .select_columns(['date', 'snap_CA', 'snap_TX', 'snap_WI'])
        .modify_data(
            lambda data:
                data
                .unpivot(
                    index='date',
                    on=['snap_CA', 'snap_TX', 'snap_WI']
                )
                .rename({'variable': 'state', 'value': 'snap_bool' })
                .with_columns(
                    pl.col('state').str.replace('snap_', '')
                )
        )
    )

    
    
    
    snap_interim.write_parquet(
        sink=True,
        name='snap_features',
        path='3.interim',
        subfolder='snap_features'
    )



if __name__ == "__main__":
    create_snap_features() 