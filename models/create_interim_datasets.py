"""
Main orchestration script for creating all interim datasets.

This script coordinates the execution of all data processing steps
in the correct order to create the interim datasets.

you need to run this script from the project root - Walmart_M5
Walmart_M5: python -m models.create_interim_datasets

"""

from package.utils import get_path_to_latest_file
from package.datapreparation import DataPreparation

from models.prod_loc_interim import create_prod_loc_interim
from models.sales_interim import create_sales_interim
from models.holiday_features import create_holiday_features
from models.event_post_effects import create_event_post_effects
from models.snap_features import create_snap_features
from models.calendar_features import create_calendar_features
from models.gaussian_spline_events import create_gaussian_spline_features

def main():
    print("Creating interim datasets...")
    
    print("1. Processing product location data...")
    create_prod_loc_interim()
    
    print("2. Processing sales data...")
    create_sales_interim()
    
    print("3. Processing holiday features...")
    create_holiday_features()
    
    print("4. Processing event post-effects...")
    create_event_post_effects()
    
    print("5. Processing SNAP features...")
    create_snap_features()
    
    print("6. Processing calendar features...")
    create_calendar_features()
    
    print("7. Processing Gaussian spline features...")
    create_gaussian_spline_features()
    
    print("All interim datasets created successfully!")

if __name__ == "__main__":
    main()