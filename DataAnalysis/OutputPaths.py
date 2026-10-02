# -*- coding: utf-8 -*-
"""
Where analysis results go and what they are called.

Every analysis script writes into

    OUTPUT_ROOT / <name of the folder the input file is in> /

and names each file after its input file plus a tag for what it holds:

    <input file name>_<TAG>.csv

Example (LAS analysis of one run):

    input   .../LAS_0.7Torr/LAS_DataFrame_33a310dd_Laser_True_20260922_151050.csv
    output  Output/LAS_0.7Torr/LAS_DataFrame_33a310dd_Laser_True_20260922_151050_LAS.csv

so any output file can be traced straight back to the dataframe it came from.
The master list (MasterList/MasterList.py) lives in OUTPUT_ROOT / 'Master'.

Scripts in sub-folders of DataAnalysis import this with

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from OutputPaths import output_file
"""

import os
from pathlib import Path

# Default: an "Output" folder at the top of the repository, next to DataAnalysis.
# To use another location on one computer (e.g. a shared Google Drive folder),
# set the environment variable MINICHAMBER_OUTPUT to that folder, or edit this line.
OUTPUT_ROOT = Path(os.environ.get('MINICHAMBER_OUTPUT',
                                  Path(__file__).resolve().parents[1] / 'Output'))

MASTER_DIR = OUTPUT_ROOT / 'Master'


def output_dir(input_file):
    """OUTPUT_ROOT/<folder the input file is in>/, created if missing."""
    folder = Path(str(input_file).replace('\\', '/')).parent.name
    d = OUTPUT_ROOT / folder if folder else OUTPUT_ROOT
    d.mkdir(parents=True, exist_ok=True)
    return d


def output_file(input_file, tag, ext='.csv'):
    """OUTPUT_ROOT/<input folder>/<input file name>_<tag><ext>"""
    stem = Path(str(input_file).replace('\\', '/')).stem
    return output_dir(input_file) / f'{stem}_{tag}{ext}'
