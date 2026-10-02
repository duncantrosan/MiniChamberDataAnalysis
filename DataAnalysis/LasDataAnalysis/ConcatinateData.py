# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 12:14:26 2026

@author: dptro
"""

import pandas as pd
from pathlib import Path

FOLDER = r"D:\Data\NafisaData\CombinedRuns"   # every summary CSV you want, dropped in here

files = list(Path(FOLDER).glob("*.csv"))
combined = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
combined.to_csv("combined_physics_summary.csv", index=False)

print(f"{len(files)} files -> {len(combined)} rows")