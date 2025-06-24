import polars as pl
from datetime import datetime
from polars import selectors as cs 
import numpy as np
import os
from pathlib import Path

def get_path_to_latest_file(parentfolder=None, subfolder=None, filename=None):
    """
    Utility to get the path to a file in a directory.
    - If filename is given: returns the path to that file in the resolved folder.
    - If subfolder is given: uses data/3.interim/subfolder
    - If parentfolder and subfolder are given: uses data/{parentfolder}/{subfolder}
    - If only parentfolder is given: uses data/{parentfolder}
    - Raises ValueError if neither is provided.
    """
    project_root = Path(__file__).parent.parent.parent
    if parentfolder is None and subfolder is None:
        raise ValueError("You must provide at least a parentfolder or a subfolder.")
    if parentfolder is not None and subfolder is not None:
        folder_path = project_root / "data" / parentfolder / subfolder
    elif parentfolder is not None:
        folder_path = project_root / "data" / parentfolder
    elif subfolder is not None:
        folder_path = project_root / "data/3.interim" / subfolder
    else:
        raise ValueError("Invalid arguments for get_path_to_latest_file")
    if filename is not None:
        path = os.path.join(folder_path, filename)
    else:
        file_list = os.listdir(folder_path)
        path = os.path.join(folder_path, max(file_list))
    return path