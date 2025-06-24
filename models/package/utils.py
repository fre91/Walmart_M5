import polars as pl
from datetime import datetime
from polars import selectors as cs 
import numpy as np
import os
from pathlib import Path

def get_path_to_latest_file(*args):
    """
    Utility to get the path to the latest file in a directory. Supports two signatures:
    - get_path_to_latest_file(subfolder_name)
    - get_path_to_latest_file(parentfolder_name, subfolder_name)
    """
    project_root = Path(__file__).parent.parent.parent
    if len(args) == 1:
        # models/3.DesignMatric.py usage
        base_path = project_root / "data/3.interim"
        subfolder_name = args[0]
        folder_path = os.path.join(base_path, subfolder_name)
    elif len(args) == 2:
        # models/4.forecast.py usage
        parentfolder_name, subfolder_name = args
        base_path = project_root / "data" / parentfolder_name
        folder_path = os.path.join(base_path, subfolder_name)
    else:
        raise ValueError("Invalid arguments for get_path_to_latest_file")
    file_list = os.listdir(folder_path)
    path = os.path.join(folder_path, max(file_list))
    return path