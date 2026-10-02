# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 12:14:26 2026

@author: dptro

Simplest possible merge: stack every dataframe in a folder into one CSV.

No matching and no combining: every row of every file, one after the other,
with a 'source_file' column saying where each row came from. Columns a file
doesn't have are left empty. (MasterList.py is the version that matches
conditions and combines repeat measurements.)

Output: Output/Master/SimpleMerge_<folder name>.csv

Usage: set FOLDER (and PATTERN) below and run, or
    python SimpleMerge.py FOLDER [PATTERN]
"""

import sys
from pathlib import Path

import pandas as pd

try:
    _HERE = Path(__file__).resolve().parent
except NameError:                        # pasted into a console
    _HERE = Path.cwd()
sys.path.insert(0, str(_HERE.parent))    # DataAnalysis/, for OutputPaths
from OutputPaths import MASTER_DIR

FOLDER = r"D:\Data\NafisaData\CombinedRuns"   # every CSV you want, dropped in here
PATTERN = '*.csv'       # e.g. '*_LAS.csv' or 'LAS_DataFrame_*_Laser_True_*.csv'
RECURSIVE = False       # True: sub-folders too (careful: a run folder's
                        # sub-folders hold the large scope files)


def simple_merge(folder=FOLDER, pattern=PATTERN, recursive=RECURSIVE):
    folder = Path(folder)
    files = sorted(folder.rglob(pattern) if recursive else folder.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {folder}")

    frames = []
    for f in files:
        df = pd.read_csv(f, dtype={'run_id': str}, low_memory=False)
        df.insert(0, 'source_file', f.name)
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True, sort=False)

    MASTER_DIR.mkdir(parents=True, exist_ok=True)
    out = MASTER_DIR / f'SimpleMerge_{folder.name}.csv'
    merged.to_csv(out, index=False)
    print(f"{len(files)} files -> {len(merged)} rows, {merged.shape[1]} columns\n  {out}")
    return merged


if __name__ == '__main__':
    args = sys.argv[1:]
    merged = simple_merge(args[0] if args else FOLDER,
                          args[1] if len(args) > 1 else PATTERN)
