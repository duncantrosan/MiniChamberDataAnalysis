# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 14:02:18 2026

@author: dptro
"""

# -*- coding: utf-8 -*-
"""
Unified LAS analysis pipeline: argon plasma with N2 admixture.

DATAFRAME-DRIVEN VERSION  (+ full error tracking ported from las_single.py)
---------------------------------------------------------------------------
Scope files and run parameters are taken from the acquisition dataframes
(LAS_DataFrame_<run_id>_Laser_True_*.csv / _Laser_False_*.csv) instead of being
scraped out of filenames with a regex.

ERROR MODEL (new)
-----------------
Each per-period fit now carries an inflated statistical error:

    sigma_noise  = MAD-of-diff of the ABSORBANCE trace (estimate_noise), a
                   robust white-noise floor immune to the line, baseline drift
                   and spikes - replaces np.std of the plasma-off trace.
    inflate      = max(1, sqrt(chi2_red)) * excess_structure(resid)
                   covers residuals larger than the noise floor (chi2) and
                   residuals that are CORRELATED (fringing) which chi2 misses.

Per scope file the periods are Birge-combined, giving a statistical error that
already contains the replicate scatter. On top of that:

    frequency-axis jitter   from the Fabry-Perot peak-spacing scatter
                            (find_relative_frequency now carries peak_spacings,
                            glitch-filtered; frequency_axis_error decomposes it)
    systematic scale terms  L_PATH, A_KI, FSR  - multiplicative, common-mode,
                            move the ABSOLUTE scale without distorting trends.

add_physics() therefore produces, for BOTH T_gas and N_s, a *_stat and a *_sys
error column (and their quadrature *_err total). The line plots draw two nested
error bars: red = statistical only (trend-affecting), blue = total (stat + sys).

An aggregate error-budget bar chart (plot_error_budget_aggregate) averages the
error breakdown across every measurement.

No bootstrapping - the covariance+Birge+decomposition path only.

@author: dptro
"""

import contextlib
from datetime import datetime
from pathlib import Path, PurePosixPath
import glob
import importlib.util
import json
import os
import re
import subprocess
import sys
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.interpolate import griddata, PchipInterpolator
from scipy.stats import linregress
from scipy.spatial import cKDTree

try:
    _HERE = Path(__file__).resolve().parent
except NameError:                        # pasted into a console
    _HERE = Path.cwd()
sys.path.insert(0, str(_HERE.parent))    # DataAnalysis/: OutputPaths, MasterList
from OutputPaths import OUTPUT_ROOT, output_dir, output_file

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_ROOT = r"D:\Data"

# What to analyse: ONE run folder (the folder holding its LAS_DataFrame_*.csv
# files), or a FOLDER OF RUNS, e.g. one sub-folder per pressure, at any depth:
# then every run in it is analysed, one after the other (see "Folder of runs"
# below). Folders can also be given on the command line:
#     python LASAnalysisv6.py "D:\Data\NafisaData" ...
Location = r'D:\Data\NafisaData\LAS_1.0Torr_SmallAdmixture_1kHzLaserFreq_N1.3_very_fine'
TrueName = 'LAS_DataFrame_*_Laser_True_*.csv'
FalseName = 'LAS_DataFrame_*_Laser_False_*.csv'
DF_TRUE = os.path.join(glob.escape(Location), TrueName)
DF_FALSE = os.path.join(glob.escape(Location), FalseName)

# False: skip the slow fitting and reuse the period fits saved by the last run
# (<dataframe>_LAS_PeriodFits.csv), e.g. to try other filter settings.
RUN_RAW_PROCESSING = True

# --- Output --------------------------------------------------------------------
# Everything goes to Output/<run folder>/ (see DataAnalysis/OutputPaths.py),
# named after the Laser_True dataframe it came from:
#   <dataframe>_LAS.csv              the input dataframe, same rows, plus the
#                                    T_gas / N_s columns. The master list uses this.
#   <dataframe>_LAS_FinalTable.csv   clean table of inputs and results
#   <dataframe>_LAS_FileResults.csv  one row per scope file: fit summary, Birge
#                                    ratios, physics with stat/sys errors
#   <dataframe>_LAS_PeriodFits.csv   one row per sawtooth period (the raw fits)
#   <dataframe>_LAS_RunInfo.json     inputs, settings and warnings of the run
#   <dataframe>_LAS_Log.txt          everything the run printed
#   LAS_figures/                     plots
OUTPUT_TAG = 'LAS'

# Add <dataframe>_LAS.csv to the master list (Output/Master) after each run.
# Re-running a dataset replaces its old entries, it never double counts.
ADD_TO_MASTER = True

# --- Folder of runs ------------------------------------------------------------
# When Location holds several runs, each run's results still go to
# Output/<run folder>/, a run that fails is reported and the rest carry on, and
# Output/Batch_<folder name>/Batch_Summary.csv lists how every run went.
# Master list for the runs in the folder (instead of ADD_TO_MASTER):
#   'separate'  its own master list in Output/Batch_<folder name>/, holding
#               exactly the runs in that folder. To merge it into the main one:
#               python MasterList.py add Output/Batch_<folder name>/Master_AllMeasurements.csv
#   'main'      straight into the main master list, Output/Master/
#   None        no master list
BATCH_MASTER = 'separate'

# True: skip runs whose _LAS.csv is newer than their dataframe, so a stopped
# batch can be restarted, or new run folders added, without redoing the rest.
# Leave False after changing analysis settings, so every run is redone.
SKIP_DONE = False

# Show figures on screen (they are always saved). Folders of runs never show them.
SHOW_PLOTS = True

# Stems of files this script (or older versions of it) wrote; never read as input.
SKIP_SUFFIXES = ('_with_Tg_Ns', '_' + OUTPUT_TAG)

COLUMN_MAP = {
    'power':    'power',
    'pressure': 'pressure',
    'n2_flow':  'Nitrogen_Percent',
    'freq':     'frequency',
}

CARRY_COLS = ['run_id', 'Measured Pressure', 'delivered_power',
              'delivered_power_error', 'gamma', 'gamma_error',
              'forward_power', 'forward_power_error',
              'reverse_power', 'reverse_power_error',
              'Nitrogen_Gas_Flow', 'Argon_Gas_Flow']

SCOPE_COL     = 'ScopeFileLocation'
SCOPE_OFF_COL = 'ScopeFileLocation_Off_Loop'
RUN_ID_COL    = 'run_id'

SINGLE_REF_ON_PATH = r'LAS_1.0Torr_SmallAdmixture_1kHzLaserFreq_N1.3_very_fine\OscopeData_Laser_True\Plasma Off Measuremnt.csv'
SINGLE_REF_OFF_PATH = r'LAS_1.0Torr_SmallAdmixture_1kHzLaserFreq_N1.3_very_fine\OscopeData_Laser_False\Plasma Off Measuremnt.csv'

PAIR_STRATEGIES = ('path', 'basename', 'condition', 'params', 'fuzzy')
PAIR_PARAM_KEYS = ['power', 'pressure', 'n2_flow', 'freq']
PAIR_ROUND = {'power': 3, 'pressure': 3, 'n2_flow': 2, 'freq': 2}
FUZZY_TOL = {'power': 0.05, 'pressure': 0.05, 'n2_flow': 0.05, 'freq': 0.05}
TRIAL_RE = re.compile(r"TrialNumber_(?P<trial>\d+)")

# --- Oscilloscope channel mapping -------------------------------------------
FP_CHANNEL    = 'CH1'   # Fabry-Perot etalon fringes
RAMP_CHANNEL  = 'CH2'   # laser sweep sawtooth
DIODE_CHANNEL = 'CH3'   # transmitted intensity
BIAS_VOLTAGE  = 0.0

# --- Filtering ---------------------------------------------------------------
NSMALLEST_PER_GROUP = 3
CHI2_MAX = None

# Leading/trailing partial sawtooth periods (scope record cut mid-sweep) often
# miss most of the absorption line. A near-flat trace fits with a LOW chi2, so
# the nsmallest-chi2 cut picks them over the real full periods and drags area
# (N_s) down. Excluded before the chi2 ranking unless this is True.
INCLUDE_PARTIAL_PERIODS = False

GROUP_KEYS = ['row_uid']

# --- Fitting -----------------------------------------------------------------
FSR = 1.5e9              # Fabry-Perot free spectral range, Hz
F_GRID = np.arange(-5, 5, 0.002)   # GHz, plotting grid only (not used for chi2)
MIN_POINTS_PER_PERIOD = 50
MIN_PEAKS_PER_PERIOD = 3

# Width of the smoothing window used by excess_structure to detect correlated
# (non-white) residual structure. Well below points-per-period, well above the
# white-noise correlation length (1 sample).
STRUCTURE_WINDOW = 401

# --- systematic scale factors (ported from las_single) -----------------------
# Multiplicative, constant across a sweep: they move the ABSOLUTE scale of N_s
# (and, via the frequency axis, T_gas) without distorting trends in power,
# pressure or N2. Reported separately from the statistical error, and drawn as
# the blue (total) error bar on the line plots.
#
#   L, A_ki -> N_s only.  FSR -> N_s linearly AND T_gas quadratically.
L_PATH_REL_ERR = 0.10     # effective absorption length, 10 %
A_KI_REL_ERR   = 0.07     # Einstein coefficient; set from the NIST accuracy rating
FSR_REL_ERR    = 0.005    # etalon free spectral range uncertainty (0.5 %).
                          # Set to your real etalon-spacing spec (0.005-0.01).

# --- Plotting ----------------------------------------------------------------
SAVE_FIGURES = True
FIG_DIR = OUTPUT_ROOT / 'LAS_figures'    # main() points this at Output/<run>/LAS_figures

# Representative fits: one sheet of N random fits that went into the results
# (data, fitted line, residuals), saved as LAS_figures/representative_fits.png
# for every run, folder of runs included, and shown on screen for a single run.
# 0 or None turns it off. FIT_PLOT_SEED = None picks different fits every run.
N_REPRESENTATIVE_FITS = 10
FIT_PLOT_SEED = 0

# Old per-fit pop-up: show every Nth fit as it is made (None = off). These
# windows are closed again after each run of a folder of runs, so use the
# representative-fits sheet above to look at fits there.
PLOT_EVERY_N_FITS = None

# Settings copied into <dataframe>_LAS_RunInfo.json
RUN_INFO_SETTINGS = [
    'NSMALLEST_PER_GROUP', 'CHI2_MAX', 'INCLUDE_PARTIAL_PERIODS', 'GROUP_KEYS',
    'PAIR_STRATEGIES', 'FSR', 'FSR_REL_ERR', 'L_PATH', 'L_PATH_REL_ERR',
    'A_KI', 'A_KI_REL_ERR', 'LAMBDA_0', 'M_AR', 'G_LOWER', 'G_UPPER',
    'MIN_POINTS_PER_PERIOD', 'MIN_PEAKS_PER_PERIOD', 'STRUCTURE_WINDOW',
    'FP_CHANNEL', 'RAMP_CHANNEL', 'DIODE_CHANNEL', 'BIAS_VOLTAGE',
    'RUN_RAW_PROCESSING', 'DATA_ROOT', 'N_REPRESENTATIVE_FITS', 'FIT_PLOT_SEED',
]

# Problems met during a run; written to the run info file.
RUN_WARNINGS = []


# =============================================================================
# SPECTROSCOPIC CONSTANTS
# =============================================================================

LAMBDA_0 = 696.5431e-9      # m
L_PATH   = 0.04          # m
M_AR     = 40.0          # amu
G_LOWER  = 5
G_UPPER  = 3
A_KI     = 6.4e6         # s^-1

C = 2.9979e8
DOPPLER_CONST = 7.16e-7

K_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))


# =============================================================================
# MODEL FUNCTIONS
# =============================================================================

def gaussian_lin(x, A, x0, sig, b, m):
    """Gaussian on a linear background."""
    return A * np.exp(-(x - x0) ** 2 / (2 * sig ** 2)) + b + m * x


# =============================================================================
# NOISE AND RESIDUAL STRUCTURE  (ported from las_single)
# =============================================================================

def estimate_noise(y):
    """
    White-noise level of a trace, from the scatter of successive differences.

        sigma = 1.4826 * MAD(diff(y)) / sqrt(2)

    Differencing removes any smooth component (line profile, baseline drift),
    so only point-to-point noise survives; the MAD is robust to spikes; the
    sqrt(2) undoes the variance doubling from differencing.

    Replaces np.std() of the plasma-emission trace, which was (a) in volts not
    absorbance and (b) inflated by real emission drift across the sweep,
    deflating chi2 exactly when the plasma was brightest.
    """
    d = np.diff(np.asarray(y, dtype=float))
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return np.nan
    mad = np.median(np.abs(d - np.median(d)))
    return float(1.4826 * mad / np.sqrt(2.0))


def excess_structure(resid, w=STRUCTURE_WINDOW):
    """
    How much more residual survives smoothing than white noise would.

    Smoothing over w samples shrinks white noise by sqrt(w), so white residuals
    give std(smoothed)/std(raw) -> 1/sqrt(w) and this returns 1. Correlated
    residuals (etalon fringing, baseline curvature, wrong line shape) survive
    smoothing and score above 1 - the factor by which the 1/sqrt(N) averaging
    in the fit covariance is optimistic. Scale-free, so comparable between
    files. chi2 alone can't see this: residuals can sit at the noise amplitude
    (chi2_red = 1) and still be a coherent fringe.
    """
    r = np.asarray(resid, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 4 * w or r.std() <= 0:
        return 1.0
    sm = np.convolve(r, np.ones(w) / w, mode='valid')
    return float(max(1.0, sm.std() / r.std() * np.sqrt(w)))


# =============================================================================
# ERROR COMBINATION
# =============================================================================

def birge_combine(values, errors):
    """
    Combine repeat measurements with individual uncertainties.

        w_j     = 1 / u_j^2
        xbar    = sum(w_j x_j) / sum(w_j)
        u_int   = 1 / sqrt(sum(w_j))                          (internal)
        u_ext   = sqrt( sum(w_j (x_j - xbar)^2) /
                        ((n-1) sum(w_j)) )                    (external)
        R       = u_ext / u_int                               (Birge ratio)
        u_final = u_int * max(1, R)

    R ~ 1: replicates scatter consistently with their own fit errors.
    R >> 1: an unmodelled systematic dominates; internal error inflated to
    cover it. max(1, R) avoids deflating below the statistical floor when R < 1.

    Returns dict: mean, err, err_int, err_ext, birge, n, method
    """
    values = np.asarray(values, dtype=float)
    errors = np.asarray(errors, dtype=float)

    finite = np.isfinite(values)
    values, errors = values[finite], errors[finite]
    n = len(values)

    blank = dict(mean=np.nan, err=np.nan, err_int=np.nan,
                 err_ext=np.nan, birge=np.nan, n=0, method='none')
    if n == 0:
        return blank

    if n == 1:
        e = errors[0] if np.isfinite(errors[0]) else np.nan
        return dict(mean=values[0], err=e, err_int=e,
                    err_ext=np.nan, birge=np.nan, n=1, method='single')

    usable = np.isfinite(errors) & (errors > 0)
    if not usable.all():
        s = values.std(ddof=1)
        sem = s / np.sqrt(n)
        return dict(mean=values.mean(), err=sem, err_int=np.nan,
                    err_ext=sem, birge=np.nan, n=n, method='unweighted')

    w = 1.0 / errors ** 2
    W = w.sum()
    mean = float((w * values).sum() / W)
    err_int = float(1.0 / np.sqrt(W))
    err_ext = float(np.sqrt((w * (values - mean) ** 2).sum() / ((n - 1) * W)))
    birge = err_ext / err_int if err_int > 0 else np.nan
    err = err_int * max(1.0, birge) if np.isfinite(birge) else err_int

    return dict(mean=mean, err=err, err_int=err_int,
                err_ext=err_ext, birge=birge, n=n, method='birge')


# =============================================================================
# STAGE 0: DATAFRAME INDEX
# =============================================================================

def _as_path_list(spec):
    """Accept a path, a glob, or a list of either; return sorted real files."""
    if spec is None:
        return []
    items = [spec] if isinstance(spec, (str, os.PathLike)) else list(spec)
    out = []
    for item in items:
        item = str(item)
        hits = glob.glob(item, recursive=True)
        out.extend(hits if hits else ([item] if os.path.isfile(item) else []))
    out = [os.path.abspath(f) for f in out
           if not Path(f).stem.endswith(SKIP_SUFFIXES)]
    return sorted(dict.fromkeys(out))


def _normalise(rel):
    """Windows-style relative path from the dataframe -> platform-native Path."""
    return Path(str(rel).replace('\\', '/'))


def resolve_path(rel, data_root=DATA_ROOT, df_path=None, _cache={}):
    """
    Turn a path stored in a dataframe into a file that exists, trying:
      1. next to the dataframe: <its folder>/<last folder of rel>/<file name>,
         e.g. OscopeData_Laser_True/Scope_...csv. This keeps working after the
         run folder is moved, renamed or copied to another computer.
      2. rel itself, if absolute
      3. DATA_ROOT / rel
      4. rel below the dataframe's folder and each folder above it
    Returns None if nothing exists.
    """
    rel = _normalise(rel)
    key = (str(rel), str(data_root), str(df_path))
    if key in _cache:
        return _cache[key]

    candidates = []
    if df_path is not None:
        here = Path(df_path).resolve().parent
        if rel.parent.name:
            candidates.append(here / rel.parent.name / rel.name)
        candidates.append(here / rel.name)
    if rel.is_absolute():
        candidates.append(rel)
    if data_root:
        candidates.append(Path(data_root) / rel)
    if df_path is not None:
        candidates.extend(anc / rel for anc in [here, *here.parents])

    found = next((c for c in candidates if c.exists()), None)
    _cache[key] = found
    return found


def _laser_agnostic_path(p):
    """Relative scope path with the laser-loop token stripped."""
    s = str(p).replace('\\', '/').lower()
    return s.replace('_laser_true', '_laser').replace('_laser_false', '_laser')


def _prepare_keys(frame, column_map):
    """Attach every candidate pairing key to a dataframe, in place."""
    if RUN_ID_COL not in frame.columns:
        frame[RUN_ID_COL] = 'run0'
    frame['_basename'] = frame[SCOPE_COL].map(
        lambda p: PurePosixPath(str(p).replace('\\', '/')).name)
    frame['_pathkey'] = frame[SCOPE_COL].map(_laser_agnostic_path)
    for internal, source in column_map.items():
        vals = (pd.to_numeric(frame[source], errors='coerce')
                if source in frame.columns else pd.Series(np.nan, index=frame.index))
        frame[f'_p_{internal}'] = vals
        nd = PAIR_ROUND.get(internal)
        frame[f'_r_{internal}'] = vals.round(nd) if nd is not None else vals
    return frame


def _fuzzy_match(remaining, f_, tol=FUZZY_TOL, keys=PAIR_PARAM_KEYS):
    """Nearest-neighbour pairing on params, per-column tolerance, one-to-one."""
    use_keys = [k for k in keys if k in tol]
    cols = [f'_p_{k}' for k in use_keys]

    L = remaining[cols].to_numpy(dtype=float)
    R = f_[cols].to_numpy(dtype=float)
    scale = np.array([tol[k] for k in use_keys])

    valid_L = np.isfinite(L).all(axis=1)
    valid_R = np.isfinite(R).all(axis=1)
    if not valid_L.any() or not valid_R.any():
        return remaining.iloc[0:0]

    li_all = np.flatnonzero(valid_L)
    ri_all = np.flatnonzero(valid_R)
    tree = cKDTree(R[valid_R] / scale)
    dist, r_pos = tree.query(L[valid_L] / scale, k=1, p=np.inf)

    cand = sorted(zip(dist, li_all, ri_all[r_pos]), key=lambda x: x[0])
    used_l, used_r, l_final, r_final = set(), set(), [], []
    for d, li, ri in cand:
        if d > 1.0 or li in used_l or ri in used_r:
            continue
        used_l.add(li); used_r.add(ri)
        l_final.append(li); r_final.append(ri)

    if not l_final:
        return remaining.iloc[0:0]

    left = remaining.iloc[l_final].reset_index(drop=True)
    right = (f_.iloc[r_final][[SCOPE_COL, SCOPE_OFF_COL, '_df_source', RUN_ID_COL]]
             .reset_index(drop=True).add_suffix('_off'))
    return pd.concat([left, right], axis=1)


def _pair_frames(t, f_, column_map, strategies=PAIR_STRATEGIES):
    """Merge laser-on to laser-off, cascading through strategies."""
    right_cols = [SCOPE_COL, SCOPE_OFF_COL, '_df_source', RUN_ID_COL]

    remaining = t.reset_index(drop=False).rename(columns={'index': '_true_idx'})
    all_pairs = []
    counts = {}

    for strat in strategies:
        if len(remaining) == 0:
            break

        if strat == 'path':
            keys = ['_pathkey']
        elif strat == 'basename':
            keys = ['_basename']
        elif strat == 'fuzzy':
            matched = _fuzzy_match(remaining, f_)
            counts[strat] = len(matched)
            print(f"  strategy 'fuzzy' (tol={FUZZY_TOL}): paired {len(matched)} "
                  f"/ {len(remaining)} still unmatched")
            if len(matched) > 0:
                all_pairs.append(matched)
                remaining = remaining[~remaining['_true_idx'].isin(matched['_true_idx'])]
            continue
        elif strat == 'params':
            keys = [f'_r_{k}' for k in PAIR_PARAM_KEYS if k in column_map]
            if not keys or f_[keys].isna().all().any():
                print("  strategy 'params': unusable key columns - skipping")
                continue
        elif strat == 'condition':
            keys = [f'_r_{k}' for k in ('power', 'pressure', 'n2_flow')
                    if k in column_map]
            if not keys or f_[keys].isna().all().any():
                print("  strategy 'condition': unusable key columns - skipping")
                continue
        else:
            raise ValueError(f"unknown pairing strategy: {strat}")

        right = f_[keys + right_cols]
        dupes = int(right.duplicated(keys).sum())
        if dupes:
            print(f"  strategy '{strat}': {dupes} laser-off rows share a key; "
                  f"keeping the first of each")
            right = right.drop_duplicates(keys, keep='first')

        matched = remaining.merge(right, on=keys, how='inner', suffixes=('', '_off'))
        counts[strat] = len(matched)
        print(f"  strategy '{strat}': paired {len(matched)} / {len(remaining)} still unmatched")

        if len(matched) > 0:
            all_pairs.append(matched)
            remaining = remaining[~remaining['_true_idx'].isin(matched['_true_idx'])]

    if not all_pairs:
        raise RuntimeError(
            "No rows paired by any strategy in PAIR_STRATEGIES.\n"
            "  Print t['_pathkey'].head() and f_['_pathkey'].head() to see how the\n"
            "  two sides actually differ.")

    pairs = pd.concat(all_pairs, ignore_index=True).drop(columns=['_true_idx'])
    print(f"  TOTAL paired: {len(pairs)} / {len(t)}  (by strategy: {counts})")

    if len(remaining):
        print(f"  {len(remaining)} rows never matched by ANY strategy, e.g.:")
        show = [c for c in ('_basename', '_p_n2_flow', '_p_freq') if c in remaining.columns]
        print(remaining[show].head(10).to_string(index=False))

    return pairs, counts, None


def _read_run_dataframe(path):
    # run_id as text: an all-digit id like 34299012 would otherwise become a number
    df = pd.read_csv(path, dtype={RUN_ID_COL: str})
    df['_df_source'] = str(path)
    return df


def load_run_index(df_true=DF_TRUE, df_false=DF_FALSE, data_root=DATA_ROOT,
                   column_map=COLUMN_MAP, carry_cols=CARRY_COLS,
                   require_files=True):
    """Build the processing index from the acquisition dataframes."""
    print(f"\n{'=' * 60}\nBUILDING INDEX FROM DATAFRAMES\n{'=' * 60}\n")

    true_files = _as_path_list(df_true)
    false_files = _as_path_list(df_false)
    if not true_files:
        raise FileNotFoundError(f"No Laser_True dataframe matched: {df_true}")
    if not false_files:
        raise FileNotFoundError(f"No Laser_False dataframe matched: {df_false}")

    print(f"Laser_True  dataframes: {len(true_files)}")
    for f in true_files:
        print(f"   {f}")
    print(f"Laser_False dataframes: {len(false_files)}")
    for f in false_files:
        print(f"   {f}")

    t = pd.concat([_read_run_dataframe(f) for f in true_files], ignore_index=True)
    f_ = pd.concat([_read_run_dataframe(f) for f in false_files], ignore_index=True)
    print(f"\nRows: {len(t)} laser-on, {len(f_)} laser-off")

    for frame, tag, fallback in (
            (t, 'Laser_True', SINGLE_REF_ON_PATH),
            (f_, 'Laser_False', SINGLE_REF_OFF_PATH)):
        if SCOPE_COL not in frame.columns:
            raise KeyError(f"{tag} dataframe is missing ['{SCOPE_COL}']")
        if SCOPE_OFF_COL not in frame.columns:
            if fallback is None:
                raise KeyError(
                    f"{tag} dataframe is missing '{SCOPE_OFF_COL}' and no fallback "
                    f"path is set. Set SINGLE_REF_ON_PATH / SINGLE_REF_OFF_PATH.")
            print(f"  NOTE: {tag} has no '{SCOPE_OFF_COL}' column; broadcasting "
                  f"single reference file to every row: {fallback}")
            frame[SCOPE_OFF_COL] = fallback

    _prepare_keys(t, column_map)
    _prepare_keys(f_, column_map)

    print("\nPairing laser-on to laser-off:")
    pairs, strategy_counts, _ = _pair_frames(t, f_, column_map)
    print(f"Strategy breakdown: {strategy_counts}")
    print(f"  run_id on:  {sorted(t[RUN_ID_COL].unique())}")
    print(f"  run_id off: {sorted(f_[RUN_ID_COL].unique())}")

    idx = pd.DataFrame(index=pairs.index)
    for internal, source in column_map.items():
        idx[internal] = pairs[f'_p_{internal}'].to_numpy()
        if source not in t.columns:
            print(f"WARNING: column '{source}' -> '{internal}' not in dataframe; "
                  f"filling with NaN")

    idx['trial'] = (pairs['_basename']
                    .str.extract(TRIAL_RE)['trial']
                    .astype(float).fillna(0).astype(int))

    for c in carry_cols:
        if c in pairs.columns:
            idx[c] = pairs[c].to_numpy()

    # laser-off files are looked up next to the laser-off dataframe
    df_src = pairs['_df_source']
    df_src_off = pairs.get('_df_source_off', df_src)
    idx['path_on']  = [resolve_path(p, data_root, s)
                       for p, s in zip(pairs[SCOPE_COL], df_src)]
    idx['path_off'] = [resolve_path(p, data_root, s)
                       for p, s in zip(pairs[f'{SCOPE_COL}_off'], df_src_off)]
    idx['ref_on']   = [resolve_path(p, data_root, s)
                       for p, s in zip(pairs[SCOPE_OFF_COL], df_src)]
    idx['ref_off']  = [resolve_path(p, data_root, s)
                       for p, s in zip(pairs[f'{SCOPE_OFF_COL}_off'], df_src_off)]

    idx['basename'] = pairs['_basename'].to_numpy()
    idx['df_source'] = df_src.to_numpy()
    idx['run_id_off'] = pairs[f'{RUN_ID_COL}_off'].to_numpy()
    idx['row_uid'] = (pairs[RUN_ID_COL].astype(str) + '|' + pairs['_basename'])

    # One scope file on several rows: the collection code saved several
    # measurements under the same file name and each overwrote the last, so the
    # file only holds the LAST row's data. Fit it once and keep that row (its
    # power / gamma / pressure readings go with the data in the file).
    shared = idx['row_uid'].duplicated(keep=False)
    if shared.any():
        n_files = idx.loc[shared, 'row_uid'].nunique()
        when = (pd.to_datetime(pairs['timestamp'], errors='coerce')
                if 'timestamp' in pairs.columns
                else pd.Series(np.arange(len(pairs)), index=pairs.index))
        idx['_when'] = when.to_numpy()
        idx = (idx.sort_values('_when', kind='stable')
                  .drop_duplicates('row_uid', keep='last')
                  .drop(columns='_when')
                  .sort_index())
        msg = (f"{n_files} scope files are each listed on more than one row "
               f"({int(shared.sum())} rows). Each file only holds the last "
               f"measurement written to it, so it is fitted once and its "
               f"result goes to the latest of those rows only.")
        print(f"\nWARNING: {msg}")
        RUN_WARNINGS.append(msg)

    missing_cols = ['path_on', 'path_off', 'ref_on', 'ref_off']
    miss = idx[missing_cols].isna().any(axis=1)
    if miss.any():
        print(f"\nWARNING: {int(miss.sum())} rows have unresolvable file paths.")
        bad = pairs.loc[miss[miss].index, SCOPE_COL].head(3).tolist()
        for b in bad:
            print(f"   {b}")
        print(f"   DATA_ROOT is currently: {data_root}")
        RUN_WARNINGS.append(f"{int(miss.sum())} rows have scope files that could "
                            f"not be found, e.g. {bad[:1]}")
        if require_files:
            idx = idx[~miss].copy()
            print(f"   Dropped; {len(idx)} rows remain.")

    print(f"\nIndex ready: {len(idx)} scope files")
    for col in ('power', 'pressure', 'n2_flow', 'freq'):
        vals = np.sort(idx[col].dropna().unique())
        head = np.array2string(vals[:8], precision=3)
        print(f"  {col:9s}: {len(vals):3d} unique {head}"
              f"{' ...' if len(vals) > 8 else ''}")

    return idx.reset_index(drop=True)


# =============================================================================
# STAGE 1: RAW PROCESSING
# =============================================================================

def find_relative_frequency(mean_df, fsr=FSR, min_peaks=MIN_PEAKS_PER_PERIOD,
                            min_pts=1000, prom=0.1, plot=False, method="pchip",
                            poly_deg=2):
    """
    Build a relative frequency axis (Hz) from Fabry-Perot fringes.

    Each period dict now also carries 'peak_spacings': the single-FSR-normalised
    time gaps between cleaned FP peaks, GLITCH-FILTERED (gaps that don't round
    cleanly to an integer FSR are dropped). Their fractional scatter is the
    fractional uncertainty of that period's GHz-per-sample axis scale, used by
    frequency_axis_error().
    """
    time = mean_df.index.to_numpy()
    perot = mean_df[FP_CHANNEL].to_numpy()
    ramp = mean_df[RAMP_CHANNEL].to_numpy()

    dy = np.diff(ramp)
    rough, _ = find_peaks(-dy, height=0.6 * np.max(-dy))
    if len(rough) < 2:
        raise RuntimeError("Could not find sawtooth edges")
    est_period = np.median(np.diff(rough))
    edges, _ = find_peaks(-dy, height=0.5 * np.max(-dy),
                          distance=int(0.8 * est_period))

    p = perot - np.median(perot)
    span = np.percentile(p, 99) - np.percentile(p, 1)
    peaks, _ = find_peaks(p, prominence=prom * span, distance=40, width=2)

    period_ranges = []
    if edges[0] >= min_pts:
        period_ranges.append((0, edges[0], "leading partial"))
    for i in range(len(edges) - 1):
        period_ranges.append((edges[i], edges[i + 1], f"full {i}"))
    if (len(time) - edges[-1]) >= min_pts:
        period_ranges.append((edges[-1], len(time), "trailing partial"))

    frequency = np.full(len(time), np.nan)
    periods = []

    for lo, hi, label in period_ranges:
        pk = np.sort(peaks[(peaks >= lo) & (peaks < hi)])

        if len(pk) < min_peaks:
            print(f"    period '{label}': only {len(pk)} peaks - skipping")
            periods.append(None)
            continue

        peak_times = time[pk]

        # remove doublets
        d = np.diff(peak_times)
        med = np.median(d)
        keep = np.ones(len(peak_times), dtype=bool)
        keep[1:][d < 0.5 * med] = False
        peak_times = peak_times[keep]

        if len(peak_times) < min_peaks:
            print(f"    period '{label}': too few peaks after cleaning - skipping")
            periods.append(None)
            continue

        # integer FSR steps
        d = np.diff(peak_times)
        med = np.median(d)
        steps = np.rint(d / med).astype(int)
        steps[steps < 1] = 1
        freq_peaks = np.concatenate(([0.0], np.cumsum(steps))) * fsr

        # single-FSR-normalised spacings, glitch-filtered: keep only gaps whose
        # length rounds cleanly to an integer FSR. A gap far from an integer
        # multiple is a peak-finding glitch (doublet, near-miss mis-rounded),
        # not real spacing jitter - and with ~5-8 gaps per period one glitch
        # dominates std/mean and inflates the axis-jitter estimate ~10x.
        ratio = d / med
        clean = np.abs(ratio - np.rint(ratio)) < 0.25
        peak_spacings = (d / steps)[clean]

        sl = slice(lo, hi)
        t_sl = time[sl]

        if method == "linear":
            frequency[sl] = np.interp(t_sl, peak_times, freq_peaks,
                                      left=np.nan, right=np.nan)
        elif method == "pchip":
            interp_func = PchipInterpolator(peak_times, freq_peaks, extrapolate=False)
            frequency[sl] = interp_func(t_sl)
        elif method == "poly":
            coeffs = np.polyfit(peak_times, freq_peaks, poly_deg)
            f = np.polyval(coeffs, t_sl)
            f[(t_sl < peak_times[0]) | (t_sl > peak_times[-1])] = np.nan
            frequency[sl] = f
        else:
            raise ValueError(f"Unknown method '{method}'")

        periods.append({"slice": sl, "label": label,
                        "peak_times": peak_times, "peak_freqs": freq_peaks,
                        "peak_spacings": peak_spacings, "method": method})

    if np.all(np.isnan(frequency)):
        raise RuntimeError("Frequency calibration failed in every period")

    if plot:
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
        ax[0].plot(time, perot)
        ax[0].plot(time[peaks], perot[peaks], 'rx')
        for e in edges:
            ax[0].axvline(time[e], color='k', alpha=0.3)
        ax[0].set_ylabel("Etalon (V)")
        ax[1].plot(time, frequency)
        ax[1].set_ylabel("Rel. frequency (Hz)")
        ax[1].set_xlabel("Time (s)")
        plt.tight_layout()
        plt.show()

    return frequency, periods


def analyze_period(f_hz, y, sigma_noise, f_grid=F_GRID):
    """
    Fit a Gaussian with linear background to one sawtooth period.

    sigma_noise is on the ABSORBANCE scale (MAD-of-diff), so chi2 is
    dimensionless and comparable between files. curve_fit is called with
    absolute_sigma=True so the covariance is the pure statistical one tied to
    the measured noise floor. Errors are then inflated explicitly:

        u -> u * max(1, sqrt(chi2_red)) * excess_structure(resid)

    chi2 covers residuals larger than the noise floor; excess_structure covers
    residuals that are CORRELATED (fringing) which chi2 alone can't see. The
    max(1, .) clamp stops chi2_red < 1 from deflating the error below the
    measured noise.

    Area uncertainty keeps the A-sigma covariance cross term.
    """
    order = np.argsort(f_hz)
    f = f_hz[order] / 1e9          # GHz
    y = y[order]

    off0 = np.median(y)

    df = np.abs(np.median(np.diff(f)))
    if not np.isfinite(df) or df <= 0:
        df = (f.max() - f.min()) / max(len(f), 2)

    p0 = [y.max() - off0, f[np.argmax(y)], 1.0, off0, 0.0]
    lo = [0.0,    f.min(), df,                 -np.inf, -np.inf]
    hi = [np.inf, f.max(), f.max() - f.min(),   np.inf,  np.inf]

    popt, pcov = curve_fit(gaussian_lin, f, y, p0=p0, bounds=(lo, hi),
                           sigma=np.full(len(f), sigma_noise),
                           absolute_sigma=True, maxfev=10000)
    A, x0, sig, b, m = popt
    sig = abs(sig)

    # --- chi2 on native points ---
    resid = y - gaussian_lin(f, A, x0, sig, b, m)
    dof = len(f) - len(popt)
    chi2_red = float(np.sum((resid / sigma_noise) ** 2) / dof) if dof > 0 else np.nan

    # --- error inflation (ported from las_single) ---
    k_chi2 = np.sqrt(max(1.0, chi2_red)) if np.isfinite(chi2_red) else 1.0
    k_struct = excess_structure(resid)
    inflate = k_chi2 * k_struct

    # --- parameter uncertainties ---
    pcov_ok = np.all(np.isfinite(pcov))
    if pcov_ok:
        perr = np.sqrt(np.diag(pcov))
        A_err, sig_err = perr[0] * inflate, perr[2] * inflate
        dA, ds = sig * np.sqrt(2 * np.pi), A * np.sqrt(2 * np.pi)
        area_var = (dA ** 2 * pcov[0, 0] + ds ** 2 * pcov[2, 2]
                    + 2.0 * dA * ds * pcov[0, 2])
        area_err = float(np.sqrt(area_var)) * inflate if area_var > 0 else np.nan
    else:
        A_err = sig_err = area_err = np.nan

    spec = np.interp(f_grid, f - x0, y, left=np.nan, right=np.nan)
    y_fit = gaussian_lin(f_grid, A, 0.0, sig, b, m)

    return {
        "spec": spec, "spec_x": f_grid, "Fit": y_fit,
        "A": A, "A_err": A_err,
        "x0": x0, "sigma": sig, "sigma_err": sig_err,
        "offset": b, "slope": m,
        "fwhm": K_FWHM * sig,
        "fwhm_err": K_FWHM * sig_err,
        "area": A * sig * np.sqrt(2 * np.pi),
        "area_err": area_err,
        "Chi^2": chi2_red,
        "k_chi2": k_chi2, "k_struct": k_struct, "inflate": inflate,
        "n_points": len(f),
    }


def _load_scope(path, chans=(FP_CHANNEL, RAMP_CHANNEL, DIODE_CHANNEL),
                cache=False, _cache={}):
    """Read a scope CSV and average repeat sweeps onto the time axis."""
    key = str(path)
    if cache and key in _cache:
        return _cache[key]
    d = pd.read_csv(path)
    out = d.groupby('time')[list(chans)].mean()
    if cache:
        _cache[key] = out
    return out


def _reference_intensity(ref_on_path, ref_off_path, _cache={}):
    """Laser-only reference: (plasma off, laser on) - (plasma off, laser off)."""
    key = (str(ref_on_path), str(ref_off_path))
    if key not in _cache:
        on = _load_scope(ref_on_path, cache=True)
        off = _load_scope(ref_off_path, cache=True)
        if not on.index.equals(off.index):
            print("    NOTE: reference on/off time bases differ; "
                  "interpolating laser-off onto laser-on grid")
            off_i = np.interp(on.index.to_numpy(), off.index.to_numpy(),
                              off[DIODE_CHANNEL].to_numpy())
        else:
            off_i = off[DIODE_CHANNEL].to_numpy()
        _cache[key] = (on.index.to_numpy(),
                       on[DIODE_CHANNEL].to_numpy() - off_i)
    return _cache[key]


def process_index(index, plot_every=PLOT_EVERY_N_FITS):
    """
    Process every row of the dataframe index into per-period fit results.

    Each output row now also carries the frequency-axis jitter for its period
    ('rel_axis') and the inflation diagnostics, so the summary stage can build
    the statistical/systematic error split.
    """
    print(f"\n{'=' * 60}\nRAW PROCESSING\n{'=' * 60}\n")

    meta_cols = [c for c in index.columns
                 if c not in ('path_on', 'path_off', 'ref_on', 'ref_off')]

    rows = []
    n_plotted = 0
    failures = []

    for i, row in enumerate(index.to_dict('records'), start=1):
        name = row['basename']
        print(f"  [{i}/{len(index)}] {name}")

        try:
            mean_on = _load_scope(row['path_on'])
            mean_off = _load_scope(row['path_off'])
            t_ref, i_ref_full = _reference_intensity(row['ref_on'], row['ref_off'])
        except Exception as e:
            print(f"    ERROR reading scope files: {e}")
            failures.append((name, repr(e)))
            continue

        t_meas = mean_on.index.to_numpy()
        if not mean_on.index.equals(mean_off.index):
            i_off = np.interp(t_meas, mean_off.index.to_numpy(),
                              mean_off[DIODE_CHANNEL].to_numpy())
        else:
            i_off = mean_off[DIODE_CHANNEL].to_numpy()

        i_ref = (i_ref_full if np.array_equal(t_meas, t_ref)
                 else np.interp(t_meas, t_ref, i_ref_full))

        try:
            frequency, periods = find_relative_frequency(mean_on)
        except RuntimeError as e:
            print(f"    WARNING: frequency calibration failed: {e}")
            failures.append((name, str(e)))
            continue

        i_m = mean_on[DIODE_CHANNEL].to_numpy() - i_off + BIAS_VOLTAGE
        with np.errstate(divide='ignore', invalid='ignore'):
            absorbance = np.log((i_ref + BIAS_VOLTAGE) / i_m)

        meta = {c: row[c] for c in meta_cols}

        for k, p in enumerate(periods):
            if p is None:
                continue

            sl = p["slice"]
            f, y = frequency[sl], absorbance[sl]
            mask = np.isfinite(f) & np.isfinite(y)
            if mask.sum() < MIN_POINTS_PER_PERIOD:
                continue

            # MAD noise floor straight off the ABSORBANCE trace (dimensionless,
            # immune to emission drift) - replaces np.std of the plasma-off.
            sigma_noise = estimate_noise(y[mask])
            if not np.isfinite(sigma_noise) or sigma_noise <= 0:
                print(f"    period {k}: invalid sigma_noise ({sigma_noise}) - skipping")
                continue

            try:
                res = analyze_period(f[mask], y[mask], sigma_noise)
            except RuntimeError as e:
                print(f"    period {k}: fit failed - {e}")
                continue

            # per-period frequency-axis jitter from the FP peak spacings
            sp = np.asarray(p.get('peak_spacings', []), dtype=float)
            sp = sp[np.isfinite(sp) & (sp > 0)]
            rel_axis = _rel_axis_from_spacings(sp)

            rec = dict(meta)
            rec.update({
                "period": k,
                "period_label": p["label"],
                "fwhm": res["fwhm"], "fwhm_err": res["fwhm_err"],
                "area": res["area"], "area_err": res["area_err"],
                "A": res["A"], "A_err": res["A_err"],
                "sigma": res["sigma"], "sigma_err": res["sigma_err"],
                "offset": res["offset"], "slope": res["slope"],
                "Chi^2": res["Chi^2"],
                "k_chi2": res["k_chi2"], "k_struct": res["k_struct"],
                "inflate": res["inflate"],
                "sigma_noise": sigma_noise,
                "rel_axis": rel_axis,
                "n_points": res["n_points"],
            })
            rows.append(rec)

            if plot_every and len(rows) % plot_every == 0:
                _plot_fit_diagnostic(res, name)
                n_plotted += 1

    master = pd.DataFrame(rows)
    print(f"\n{'=' * 60}\nRAW RESULTS\n{'=' * 60}")
    print(f"Total fits:  {len(master)} from {len(index)} scope files "
          f"({n_plotted} diagnostic plots)")
    if failures:
        print(f"Failed files: {len(failures)}")
        for name, err in failures[:5]:
            print(f"   {name}: {err}")
        RUN_WARNINGS.append(f"{len(failures)} scope files failed: "
                            + "; ".join(f"{n}: {e}" for n, e in failures))
    if len(master):
        print(f"Files with >=1 fit: {master['row_uid'].nunique()}")
        if 'inflate' in master:
            print(f"  mean inflation: {master['inflate'].mean():.2f}x")
        if 'rel_axis' in master:
            print(f"  median rel_axis (freq jitter): "
                  f"{100*master['rel_axis'].median():.3f} %")
    return master


def _rel_axis_from_spacings(sp):
    """
    Robust fractional axis-scale jitter from single-FSR-normalised peak spacings.

    MAD (scaled to a sigma) not std, so a single surviving peak-finding glitch
    can't dominate a short (~5-8) per-period set; /sqrt(n) because the axis
    scale is the mean of n spacings. Returns NaN if too few clean spacings.
    """
    sp = np.asarray(sp, dtype=float)
    sp = sp[np.isfinite(sp) & (sp > 0)]
    if len(sp) >= 3:
        med = np.median(sp)
        mad = np.median(np.abs(sp - med)) * 1.4826
        return (mad / med) / np.sqrt(len(sp)) if med > 0 else np.nan
    if len(sp) == 2:
        return np.std(sp, ddof=1) / np.mean(sp) / np.sqrt(2)
    return np.nan


def _plot_fit_diagnostic(res, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(res['spec_x'], res['spec'], label='data', linewidth=2)
    plt.plot(res['spec_x'], res['Fit'], label='fit', linewidth=2)
    plt.xlabel('Relative frequency (GHz)')
    plt.ylabel('Absorbance')
    plt.title(filename, fontsize=9)
    plt.text(0.05, 0.95,
             f"$\\chi^2_\\nu$ = {res['Chi^2']:.2f}\n"
             f"FWHM = {res['fwhm']:.3f} $\\pm$ {res['fwhm_err']:.3f} GHz\n"
             f"inflate = {res['inflate']:.1f}x",
             transform=plt.gca().transAxes, va='top', ha='left',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.75))
    plt.legend()
    plt.tight_layout()
    plt.show()


# =============================================================================
# STAGE 2: FILTERING AND AGGREGATION
# =============================================================================

def filter_and_aggregate(results_df, nsmallest=NSMALLEST_PER_GROUP,
                         chi2_max=CHI2_MAX, group_keys=GROUP_KEYS,
                         index=None, include_partial=INCLUDE_PARTIAL_PERIODS):
    """
    Filter per-period fits and aggregate to one row per group (one per scope
    file), carrying Birge-combined uncertainties on FWHM and area, PLUS the
    frequency-axis jitter needed to split statistical error later.

    Partial periods are dropped first (see INCLUDE_PARTIAL_PERIODS). Done here
    rather than in process_index so it also applies to cached period fits.
    """
    print(f"\n{'=' * 60}\nFILTERING & AGGREGATION\n{'=' * 60}\n")

    diagnostics = {"n_raw": len(results_df), "stages": {}}

    stage = results_df.dropna(subset=['Chi^2']).copy()
    removed = len(results_df) - len(stage)
    diagnostics["stages"]["dropna_chi2"] = {"removed": removed, "remaining": len(stage)}
    print(f"After dropna(Chi^2):      removed {removed}, remaining {len(stage)}")

    if not include_partial:
        if 'period_label' in stage.columns:
            before = len(stage)
            partial = stage['period_label'].astype(str).str.contains('partial')
            stage = stage[~partial]
            diagnostics["stages"]["partial_periods"] = {
                "removed": before - len(stage), "remaining": len(stage)}
            print(f"After dropping partials:  removed {before - len(stage)}, "
                  f"remaining {len(stage)}")
        else:
            print("WARNING: no 'period_label' column; cannot drop partial periods")

    if chi2_max is not None:
        before = len(stage)
        stage = stage[stage['Chi^2'] <= chi2_max]
        diagnostics["stages"]["chi2_max"] = {
            "threshold": chi2_max, "removed": before - len(stage), "remaining": len(stage)}
        print(f"After chi2 <= {chi2_max}:        removed {before - len(stage)}, "
              f"remaining {len(stage)}")

    if nsmallest is not None:
        before = len(stage)
        stage = (stage.sort_values('Chi^2')
                      .groupby(group_keys, sort=False)
                      .head(nsmallest)
                      .reset_index(drop=True))
        diagnostics["stages"]["nsmallest"] = {
            "nsmallest": nsmallest, "removed": before - len(stage), "remaining": len(stage)}
        print(f"After nsmallest({nsmallest}):        removed {before - len(stage)}, "
              f"remaining {len(stage)}")
    else:
        print("Skipped nsmallest filter (keeping all)")

    filtered_df = stage.copy()
    if 'n2_flow' in filtered_df:
        print(f"\nN2 flows present: {sorted(filtered_df['n2_flow'].unique())}")
    print(f"chi2 distribution:\n{filtered_df['Chi^2'].describe()}\n")

    print(f"{'=' * 60}\nAGGREGATION\n{'=' * 60}\n")

    records = []
    for keys, g in filtered_df.groupby(group_keys, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = dict(zip(group_keys, keys))

        a = birge_combine(g['area'].to_numpy(), g['area_err'].to_numpy())
        w = birge_combine(g['fwhm'].to_numpy(), g['fwhm_err'].to_numpy())

        # frequency-axis jitter for this scope file: quadrature-mean of the
        # per-period rel_axis, as a fractional error on the width.
        if 'rel_axis' in g:
            ra = g['rel_axis'].to_numpy(dtype=float)
            ra = ra[np.isfinite(ra)]
            rel_axis_file = float(np.sqrt(np.mean(ra ** 2))) if len(ra) else np.nan
        else:
            rel_axis_file = np.nan

        rec.update({
            'area_mean': a['mean'], 'area_err': a['err'],
            'area_err_int': a['err_int'], 'area_err_ext': a['err_ext'],
            'area_birge': a['birge'], 'area_method': a['method'],
            'area_std': g['area'].std(ddof=1) if len(g) > 1 else np.nan,

            'fwhm_mean': w['mean'], 'fwhm_err': w['err'],
            'fwhm_err_int': w['err_int'], 'fwhm_err_ext': w['err_ext'],
            'fwhm_birge': w['birge'], 'fwhm_method': w['method'],
            'fwhm_std': g['fwhm'].std(ddof=1) if len(g) > 1 else np.nan,

            'rel_axis': rel_axis_file,
            'inflate_mean': g['inflate'].mean() if 'inflate' in g else np.nan,
            'n_fits': len(g),
            'chi2_median': g['Chi^2'].median(),
        })
        records.append(rec)

    summary = pd.DataFrame(records)
    diagnostics["summary_groups"] = len(summary)

    if index is not None and 'row_uid' in summary.columns:
        meta = index.drop(columns=['path_on', 'path_off', 'ref_on', 'ref_off'],
                          errors='ignore').drop_duplicates('row_uid')
        summary = summary.merge(meta, on='row_uid', how='left')
    elif index is None:
        for col in ('power', 'pressure', 'n2_flow', 'freq'):
            if col not in summary.columns and col in filtered_df.columns:
                lut = filtered_df.groupby(group_keys)[col].first().reset_index()
                summary = summary.merge(lut, on=group_keys, how='left')

    print(f"Aggregated into {len(summary)} groups (keys: {group_keys})")
    for col in ('fwhm_birge', 'area_birge'):
        vals = summary[col].dropna()
        if len(vals):
            print(f"  {col}: median {vals.median():.2f}, "
                  f"{(vals > 2).sum()}/{len(vals)} groups above 2")
    print("  (Birge >> 1 means replicate scatter exceeds the fit errors:\n"
          "   an unmodelled systematic is present in those conditions.)")

    return filtered_df, summary, diagnostics


def aggregate_conditions(summary, keys=('power', 'pressure', 'n2_flow', 'freq')):
    """Optional second pass: pool repeat scope files sharing a nominal condition."""
    keys = [k for k in keys if k in summary.columns]
    records = []
    for vals, g in summary.groupby(keys, sort=True):
        if not isinstance(vals, tuple):
            vals = (vals,)
        rec = dict(zip(keys, vals))
        a = birge_combine(g['area_mean'].to_numpy(), g['area_err'].to_numpy())
        w = birge_combine(g['fwhm_mean'].to_numpy(), g['fwhm_err'].to_numpy())
        rec.update({'area_mean': a['mean'], 'area_err': a['err'],
                    'area_birge': a['birge'],
                    'fwhm_mean': w['mean'], 'fwhm_err': w['err'],
                    'fwhm_birge': w['birge'],
                    'n_files': len(g), 'n_fits': int(g['n_fits'].sum()),
                    'chi2_median': g['chi2_median'].median()})
        records.append(rec)
    out = pd.DataFrame(records)
    print(f"\nPooled {len(summary)} files -> {len(out)} conditions on {keys}")
    return out


# =============================================================================
# STAGE 3: PHYSICS  (with statistical / systematic error split)
# =============================================================================

def add_physics(summary):
    """
    Convert fitted FWHM and area into gas temperature and metastable density,
    propagating uncertainties AND splitting each into a statistical and a
    systematic part.

    STATISTICAL (red bar - affects trends, shrinks as periods/files average):
        T_gas: 2 * T * fwhm_rel_stat, where fwhm_rel_stat combines the
               covariance+Birge width error with the frequency-axis jitter.
               (Both are per-measurement random contributions.)
        N_s:   area_rel_stat = area_err/area (covariance+Birge), plus the same
               frequency-axis jitter passed through linearly.

    SYSTEMATIC (blue bar minus red - common-mode, sets ABSOLUTE scale):
        T_gas: FSR only, and QUADRATICALLY -> 2 * T * FSR_REL_ERR.
        N_s:   sqrt(L^2 + A_ki^2 + FSR^2), all linear.

    TOTAL = quadrature(stat, sys). Columns:
        T_gas_K, T_gas_K_stat, T_gas_K_sys, T_gas_K_err
        N_s_cm3, N_s_cm3_stat, N_s_cm3_sys, N_s_cm3_err
    """
    s = summary.copy()

    # ---------------- gas temperature ----------------
    fwhm_hz = s['fwhm_mean'] * 1e9
    fwhm_hz_err = s['fwhm_err'] * 1e9

    s['fwhm_wavelength_m'] = fwhm_hz * LAMBDA_0 ** 2 / C
    s['T_gas_K'] = M_AR * (s['fwhm_wavelength_m'] / (DOPPLER_CONST * LAMBDA_0)) ** 2

    # relative width error: covariance+Birge, combined in quadrature with the
    # frequency-axis jitter (rel_axis is already a fractional width error)
    rel_fwhm_fit = (fwhm_hz_err / fwhm_hz).replace([np.inf, -np.inf], np.nan)
    rel_axis = s['rel_axis'] if 'rel_axis' in s else pd.Series(0.0, index=s.index)
    rel_axis = rel_axis.fillna(0.0)
    rel_fwhm_stat = np.sqrt(rel_fwhm_fit ** 2 + rel_axis ** 2)

    # T ~ width^2 -> relative error doubles
    s['T_gas_K_stat'] = 2.0 * s['T_gas_K'] * rel_fwhm_stat
    s['T_gas_K_sys']  = 2.0 * s['T_gas_K'] * FSR_REL_ERR      # FSR, quadratic
    s['T_gas_K_err']  = np.sqrt(s['T_gas_K_stat'] ** 2 + s['T_gas_K_sys'] ** 2)

    # ---------------- metastable density ----------------
    NS_CONST = 8.0 * np.pi * G_LOWER / (LAMBDA_0 ** 2 * G_UPPER * A_KI * L_PATH)

    s['N_s_m3'] = NS_CONST * s['area_mean'] * 1e9
    s['N_s_cm3'] = s['N_s_m3'] * 1e-6

    rel_area_fit = (s['area_err'] / s['area_mean']).replace([np.inf, -np.inf], np.nan)
    # frequency jitter passes through the area linearly too
    rel_area_stat = np.sqrt(rel_area_fit ** 2 + rel_axis ** 2)
    rel_ns_sys = np.sqrt(L_PATH_REL_ERR ** 2 + A_KI_REL_ERR ** 2 + FSR_REL_ERR ** 2)

    s['N_s_cm3_stat'] = s['N_s_cm3'] * rel_area_stat
    s['N_s_cm3_sys']  = s['N_s_cm3'] * rel_ns_sys
    s['N_s_cm3_err']  = np.sqrt(s['N_s_cm3_stat'] ** 2 + s['N_s_cm3_sys'] ** 2)

    # keep the plain _err aliases the rest of the pipeline expects
    s['N_s_m3_err'] = s['N_s_cm3_err'] * 1e6

    print(f"\n{'=' * 60}\nDERIVED QUANTITIES\n{'=' * 60}")
    print(f"T_gas: {s['T_gas_K'].min():.0f} - {s['T_gas_K'].max():.0f} K "
          f"(mean {s['T_gas_K'].mean():.0f} K)")
    rel_T = (s['T_gas_K_err'] / s['T_gas_K']).replace([np.inf, -np.inf], np.nan).dropna()
    rel_Ts = (s['T_gas_K_stat'] / s['T_gas_K']).replace([np.inf, -np.inf], np.nan).dropna()
    if len(rel_T):
        print(f"  median rel. uncertainty: total {rel_T.median()*100:.1f} %, "
              f"stat {rel_Ts.median()*100:.1f} %")

    print(f"N_s:   {s['N_s_cm3'].min():.2e} - {s['N_s_cm3'].max():.2e} cm^-3 "
          f"(mean {s['N_s_cm3'].mean():.2e})")
    rel_N = (s['N_s_cm3_err'] / s['N_s_cm3']).replace([np.inf, -np.inf], np.nan).dropna()
    rel_Ns = (s['N_s_cm3_stat'] / s['N_s_cm3']).replace([np.inf, -np.inf], np.nan).dropna()
    if len(rel_N):
        print(f"  median rel. uncertainty: total {rel_N.median()*100:.1f} %, "
              f"stat {rel_Ns.median()*100:.1f} %")

    if not (1e8 <= s['N_s_cm3'].median() <= 1e12):
        print("\n  !! N_s median is outside 1e8-1e12 cm^-3. Re-check g_lower,\n"
              "     g_upper, A_ki and the line identity.")
    return s


def build_final_table(summary):
    """Collapse the summary down to the columns worth looking at."""
    s = summary
    out = pd.DataFrame({
        'Power_Input_W':       s['power'],
        'Pressure_Input_Torr': s['pressure'],
        'N2_Percent_Input':    s['n2_flow'],
        'Frequency_Input_MHz': s['freq'],

        'Power_Measured_W':       s.get('delivered_power'),
        'Power_Measured_W_err':   s.get('delivered_power_error'),
        'Pressure_Measured_Torr': s.get('Measured Pressure'),

        'N2_Percent_Measured':    100 * s.get('Nitrogen_Gas_Flow', np.nan)
                                   / (s.get('Nitrogen_Gas_Flow', np.nan)
                                      + s.get('Argon_Gas_Flow', np.nan)),
        'N2_Flow_Measured_sccm':  s.get('Nitrogen_Gas_Flow'),
        'Ar_Flow_Measured_sccm':  s.get('Argon_Gas_Flow'),

        'Forward_Power_W':      s.get('forward_power'),
        'Forward_Power_W_err':  s.get('forward_power_error'),
        'Reflected_Power_W':    s.get('reverse_power'),
        'Reflected_Power_W_err':s.get('reverse_power_error'),
        'Gamma':                s.get('gamma'),
        'Gamma_err':            s.get('gamma_error'),

        'N_s_cm3':      s['N_s_cm3'],
        'N_s_cm3_stat': s['N_s_cm3_stat'],
        'N_s_cm3_sys':  s['N_s_cm3_sys'],
        'N_s_cm3_err':  s['N_s_cm3_err'],
        'T_gas_K':      s['T_gas_K'],
        'T_gas_K_stat': s['T_gas_K_stat'],
        'T_gas_K_sys':  s['T_gas_K_sys'],
        'T_gas_K_err':  s['T_gas_K_err'],
    })
    print(f"\nFinal table: {len(out)} rows, {len(out.columns)} columns")
    return out


# =============================================================================
# STAGE 3b: WRITE RESULTS BACK ONTO THE ACQUISITION DATAFRAME
# =============================================================================

PHYSICS_OUT_COLS = {
    'T_gas_K':      'T_gas_K',
    'T_gas_K_stat': 'T_gas_K_stat',
    'T_gas_K_sys':  'T_gas_K_sys',
    'T_gas_K_err':  'T_gas_K_err',
    'N_s_cm3':      'N_s_cm3',
    'N_s_cm3_stat': 'N_s_cm3_stat',
    'N_s_cm3_sys':  'N_s_cm3_sys',
    'N_s_cm3_err':  'N_s_cm3_err',
    'fwhm_mean':    'fwhm_GHz',
    'fwhm_err':     'fwhm_GHz_err',
    'area_mean':    'area_GHz',
    'area_err':     'area_GHz_err',
    'n_fits':       'n_periods_used',
    'chi2_median':  'chi2_median',
    'fwhm_birge':   'fwhm_birge',
}


def attach_to_dataframe(summary, df_true=DF_TRUE, cols=PHYSICS_OUT_COLS):
    """
    Append physics (with stat/sys errors) to each Laser_True dataframe and save
    it as Output/<run folder>/<dataframe>_LAS.csv: the input dataframe, same
    rows and columns, plus the result columns. The master list reads this file.

    Rows that share a scope file (see load_run_index) get the result only on
    the latest of them; the others get NaN and a note in 'LAS_note'.
    """
    print(f"\n{'=' * 60}\nAPPENDING PHYSICS TO ACQUISITION DATAFRAME\n{'=' * 60}\n")

    if 'row_uid' not in summary.columns:
        raise KeyError("summary has no row_uid; run filter_and_aggregate with "
                       "GROUP_KEYS=['row_uid']")

    keep = ['row_uid'] + [c for c in cols if c in summary.columns]
    phys = summary[keep].rename(columns=cols).drop_duplicates('row_uid')
    result_cols = [c for c in phys.columns if c != 'row_uid']

    outputs = []
    for path in _as_path_list(df_true):
        df = pd.read_csv(path, dtype={RUN_ID_COL: str})
        if RUN_ID_COL not in df.columns:
            df[RUN_ID_COL] = 'run0'
        basename = df[SCOPE_COL].map(
            lambda p: PurePosixPath(str(p).replace('\\', '/')).name)
        df['row_uid'] = df[RUN_ID_COL].astype(str) + '|' + basename

        merged = df.merge(phys, on='row_uid', how='left')

        shared = merged['row_uid'].duplicated(keep=False)
        if shared.any():
            when = (pd.to_datetime(merged['timestamp'], errors='coerce')
                    if 'timestamp' in merged.columns
                    else pd.Series(np.arange(len(merged)), index=merged.index))
            order = when.sort_values(kind='stable').index
            stale = merged.loc[order, 'row_uid'].duplicated(keep='last')
            stale = stale[stale].index
            merged.loc[stale, result_cols] = np.nan
            merged['LAS_note'] = ''
            merged.loc[stale, 'LAS_note'] = 'scope file overwritten by a later row'
            print(f"  {len(stale)} rows share their scope file with a later row "
                  f"and get no result (see LAS_note)")

        n_hit = merged['T_gas_K'].notna().sum() if 'T_gas_K' in merged else 0
        print(f"{Path(path).name}: {n_hit}/{len(merged)} rows got results")

        target = output_file(path, OUTPUT_TAG)
        merged.to_csv(target, index=False)
        print(f"  -> {target}")
        outputs.append(merged)

    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


# =============================================================================
# STAGE 3c: AGGREGATE ERROR BUDGET (averaged across all measurements)
# =============================================================================

def error_budget_components(summary):
    """
    Per-measurement, disjoint statistical/systematic error breakdown, as % of
    the value, so pieces add IN QUADRATURE to the quoted total. Averaged by
    plot_error_budget_aggregate over every row.

    Statistical (red family - trend-affecting):
        jitter   MAD fit-noise floor through the covariance = fwhm/area err_int.
        freq     frequency-axis (FP-spacing) jitter, rel_axis (doubled for T).
        fit      the rest of the full covariance+Birge error, in quadrature:
                 sqrt(full^2 - jitter^2), kept disjoint from freq/jitter.
                 (Zero when Birge R<=1: the fits already predicted the scatter.)

    Systematic (blue family - absolute scale):
        L, A_ki, FSR.  (T_gas: FSR only, quadratic. N_s: all three, linear.)

    Returns a DataFrame of per-row % contributions with columns
        t_jitter t_freq t_fit t_fsr t_stat t_sys t_tot
        n_jitter n_freq n_fit n_L n_A n_fsr n_stat n_sys n_tot
    """
    s = summary
    out = pd.DataFrame(index=s.index)

    # --- width (T_gas) relative pieces ---
    fw = s['fwhm_mean'].replace(0, np.nan)
    rel_full_w = (s['fwhm_err'] / fw)
    rel_jit_w  = (s['fwhm_err_int'] / fw) if 'fwhm_err_int' in s else pd.Series(np.nan, index=s.index)
    rel_freq_w = s['rel_axis'] if 'rel_axis' in s else pd.Series(0.0, index=s.index)
    rel_freq_w = rel_freq_w.fillna(0.0)
    # the full stat already includes freq (we combined them in add_physics), so
    # remove BOTH jitter and freq from full to get the 'fit/Birge' remainder
    full_w_tot = np.sqrt(rel_full_w ** 2 + rel_freq_w ** 2)   # matches add_physics stat
    rel_fit_w = np.sqrt((full_w_tot ** 2 - rel_jit_w ** 2 - rel_freq_w ** 2)
                        .clip(lower=0.0))

    out['t_jitter'] = 100 * 2.0 * rel_jit_w.fillna(0.0)
    out['t_freq']   = 100 * 2.0 * rel_freq_w
    out['t_fit']    = 100 * 2.0 * rel_fit_w.fillna(0.0)
    out['t_fsr']    = 100 * 2.0 * FSR_REL_ERR
    out['t_stat']   = np.sqrt(out['t_jitter']**2 + out['t_freq']**2 + out['t_fit']**2)
    out['t_sys']    = out['t_fsr']
    out['t_tot']    = np.sqrt(out['t_stat']**2 + out['t_sys']**2)

    # --- area (N_s) relative pieces ---
    ar = s['area_mean'].replace(0, np.nan)
    rel_full_a = (s['area_err'] / ar)
    rel_jit_a  = (s['area_err_int'] / ar) if 'area_err_int' in s else pd.Series(np.nan, index=s.index)
    rel_freq_a = rel_freq_w
    full_a_tot = np.sqrt(rel_full_a ** 2 + rel_freq_a ** 2)
    rel_fit_a = np.sqrt((full_a_tot ** 2 - rel_jit_a ** 2 - rel_freq_a ** 2)
                        .clip(lower=0.0))

    out['n_jitter'] = 100 * rel_jit_a.fillna(0.0)
    out['n_freq']   = 100 * rel_freq_a
    out['n_fit']    = 100 * rel_fit_a.fillna(0.0)
    out['n_L']      = 100 * L_PATH_REL_ERR
    out['n_A']      = 100 * A_KI_REL_ERR
    out['n_fsr']    = 100 * FSR_REL_ERR
    out['n_stat']   = np.sqrt(out['n_jitter']**2 + out['n_freq']**2 + out['n_fit']**2)
    out['n_sys']    = np.sqrt(out['n_L']**2 + out['n_A']**2 + out['n_fsr']**2)
    out['n_tot']    = np.sqrt(out['n_stat']**2 + out['n_sys']**2)

    return out


def plot_error_budget_aggregate(summary, fname='error_budget_aggregate.png'):
    """
    Averaged error budget across every measurement: two stacked bars (T_gas,
    N_s), each split into a red statistical family (jitter, freq, fit/Birge)
    and a blue systematic family (L, A_ki, FSR). Heights are the median % over
    all rows; blocks are drawn so the stack total equals the quadrature total.

    Red = statistical (trends), blue = systematic (absolute scale).
    """
    comp = error_budget_components(summary)
    med = comp.median(numeric_only=True)

    red  = plt.cm.Reds(np.linspace(0.45, 0.85, 3))    # jitter, freq, fit
    blue = plt.cm.Blues(np.linspace(0.45, 0.85, 3))   # L, A, FSR

    def _draw(ax, parts_stat, parts_sys, stat_tot, sys_tot, tot, title):
        parts = [p for p in parts_stat + parts_sys if p[1] > 1e-9]
        linsum = sum(p[1] for p in parts)
        scale = (tot / linsum) if linsum > 0 else 1.0
        bottom = 0.0
        for label, val, color in parts:
            h = val * scale
            ax.bar(0, h, bottom=bottom, width=0.6, color=color,
                   edgecolor='white', linewidth=1.2)
            if h > 0.03 * tot:
                ax.text(0, bottom + h / 2, f'{label}\n{val:.2f}%',
                        ha='center', va='center', fontsize=8,
                        color='white', fontweight='bold')
            bottom += h
        stat_h = sum(p[1] for p in parts_stat if p[1] > 1e-9) * scale
        if 0 < stat_h < bottom:
            ax.axhline(stat_h, color='k', linewidth=1.0, linestyle=':')
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])
        ax.set_ylabel('Relative error (%)', fontweight='bold')
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis='y', alpha=0.3)
        ax.text(0, bottom, f'  total {tot:.2f}%\n'
                           f'  (stat {stat_tot:.2f} / sys {sys_tot:.2f})',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_ylim(0, bottom * 1.18)

    fig, (axt, axn) = plt.subplots(1, 2, figsize=(11, 6.5))

    _draw(axt,
          [('jitter\n(MAD floor)', med['t_jitter'], red[0]),
           ('freq jitter\n(FP axis)', med['t_freq'], red[1]),
           ('fit / chi2\n/ Birge', med['t_fit'], red[2])],
          [('FSR', med['t_fsr'], blue[2])],
          med['t_stat'], med['t_sys'], med['t_tot'],
          f'T_gas   (median over {len(summary)} measurements)')

    _draw(axn,
          [('jitter\n(MAD floor)', med['n_jitter'], red[0]),
           ('freq jitter\n(FP axis)', med['n_freq'], red[1]),
           ('fit / chi2\n/ Birge', med['n_fit'], red[2])],
          [('L path', med['n_L'], blue[0]),
           ('A_ki', med['n_A'], blue[1]),
           ('FSR', med['n_fsr'], blue[2])],
          med['n_stat'], med['n_sys'], med['n_tot'],
          f'N_s   (median over {len(summary)} measurements)')

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=red[1],  label='statistical  (trend-affecting: '
                                              'jitter, freq, fit/Birge)'),
               Patch(facecolor=blue[1], label='systematic  (absolute scale: '
                                              'L, A_ki, FSR)')]
    fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('LAS aggregate error budget - red = statistical (trends), '
                 'blue = systematic (absolute)', fontweight='bold')
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _savefig(fname)
    _show()
    return comp


# =============================================================================
# STAGE 4: PLOTTING HELPERS
# =============================================================================

def _savefig(name, dpi=300):
    if not SAVE_FIGURES:
        return
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"  saved: {path}")


def _show():
    """Show the figure, or close it when nobody is watching (folders of runs)."""
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close('all')


# --- representative fits -------------------------------------------------------

FIT_DATA_COLOR = '#2a78d6'    # data points
FIT_LINE_COLOR = '#eb6834'    # fitted line


def _row_absorbance(row):
    """
    Frequency axis, sawtooth periods and absorbance of one index row, built the
    same way process_index builds them (so a re-fit gives the stored result).
    """
    mean_on = _load_scope(row['path_on'])
    mean_off = _load_scope(row['path_off'])
    t_ref, i_ref_full = _reference_intensity(row['ref_on'], row['ref_off'])

    t_meas = mean_on.index.to_numpy()
    i_off = (mean_off[DIODE_CHANNEL].to_numpy() if mean_on.index.equals(mean_off.index)
             else np.interp(t_meas, mean_off.index.to_numpy(),
                            mean_off[DIODE_CHANNEL].to_numpy()))
    i_ref = (i_ref_full if np.array_equal(t_meas, t_ref)
             else np.interp(t_meas, t_ref, i_ref_full))

    frequency, periods = find_relative_frequency(mean_on)
    i_m = mean_on[DIODE_CHANNEL].to_numpy() - i_off + BIAS_VOLTAGE
    with np.errstate(divide='ignore', invalid='ignore'):
        absorbance = np.log((i_ref + BIAS_VOLTAGE) / i_m)
    return frequency, periods, absorbance


def plot_representative_fits(index, filtered, n=None, seed=FIT_PLOT_SEED,
                             fname='representative_fits.png'):
    """
    One sheet of n random fits from the periods that went into the results:
    for each, the absorbance data with the fitted Gaussian, and the residual.

    Picks n different scope files at random, then one of each file's kept
    periods. Each is re-fitted from its scope file (about a second each), so
    this works after a cached run (RUN_RAW_PROCESSING = False) too.
    """
    n = N_REPRESENTATIVE_FITS if n is None else n
    if not n:
        return None
    uids = filtered['row_uid'].unique()
    if len(uids) == 0:
        print("  representative fits: no fits to show")
        return None

    rng = np.random.default_rng(seed)
    chosen = rng.choice(uids, size=min(n, len(uids)), replace=False)
    rows = index.drop_duplicates('row_uid').set_index('row_uid')

    print(f"\nRepresentative fits: re-fitting {len(chosen)} random periods "
          f"(of {len(uids)} scope files)")
    fits = []
    for uid in chosen:
        k = int(rng.choice(filtered.loc[filtered['row_uid'] == uid, 'period']))
        row = rows.loc[uid]
        try:
            frequency, periods, absorbance = _row_absorbance(row)
            sl = periods[k]['slice']
            f, y = frequency[sl], absorbance[sl]
            mask = np.isfinite(f) & np.isfinite(y)
            res = analyze_period(f[mask], y[mask], estimate_noise(y[mask]))
        except Exception as e:
            print(f"  {row['basename']} period {k}: could not re-fit ({e})")
            continue
        order = np.argsort(f[mask])
        res['f_data'] = f[mask][order] / 1e9 - res['x0']
        res['y_data'] = y[mask][order]
        res['label'] = periods[k]['label']
        res['row'] = row
        fits.append(res)
    if not fits:
        return None
    fits.sort(key=lambda r: (r['row']['power'], r['row']['n2_flow']))

    ncols = min(5, len(fits))
    nrows = int(np.ceil(len(fits) / ncols))
    fig = plt.figure(figsize=(3.9 * ncols, 3.7 * nrows))
    outer = fig.add_gridspec(nrows, ncols, hspace=0.5, wspace=0.3)
    for i, res in enumerate(fits):
        r, c = divmod(i, ncols)
        inner = outer[r, c].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
        ax = fig.add_subplot(inner[0])
        axr = fig.add_subplot(inner[1], sharex=ax)

        fd, yd = res['f_data'], res['y_data']
        dense = np.linspace(fd.min(), fd.max(), 600)
        pars = (res['A'], res['x0'], res['sigma'], res['offset'], res['slope'])
        curve = gaussian_lin(dense + res['x0'], *pars)
        resid = yd - gaussian_lin(fd + res['x0'], *pars)

        ax.plot(fd, yd, '.', ms=2.5, alpha=0.45, color=FIT_DATA_COLOR,
                label='data', rasterized=True)
        ax.plot(dense, curve, '-', lw=1.6, color=FIT_LINE_COLOR, label='fit')
        axr.plot(fd, resid, '.', ms=2.5, alpha=0.45, color=FIT_DATA_COLOR,
                 rasterized=True)
        axr.axhline(0, color=FIT_LINE_COLOR, lw=1.0)

        row = res['row']
        ax.set_title(f"{row['power']:.0f} W, {row['n2_flow']:.2f} % N$_2$, "
                     f"{row['freq']:.0f} MHz\n{res['label']}: "
                     f"$\\chi^2_\\nu$ {res['Chi^2']:.2f}, "
                     f"FWHM {res['fwhm']:.2f} GHz", fontsize=8.5)
        plt.setp(ax.get_xticklabels(), visible=False)
        for a in (ax, axr):
            a.grid(True, alpha=0.2, lw=0.6)
            a.tick_params(labelsize=8)
            for side in ('top', 'right'):
                a.spines[side].set_visible(False)
        if c == 0:
            ax.set_ylabel('Absorbance', fontsize=9)
            axr.set_ylabel('residual', fontsize=9)
        if r == nrows - 1 or i + ncols >= len(fits):
            axr.set_xlabel('Relative frequency (GHz)', fontsize=9)
        if i == 0:
            ax.legend(fontsize=8, frameon=False, loc='upper right')

    fig.suptitle(f"Representative fits: {len(fits)} random periods used in the "
                 f"results of {len(uids)} scope files", fontsize=12, y=0.995)
    _savefig(fname, dpi=200)
    _show()
    return fits


def check_density(data, x_col, y_col, title=""):
    nx, ny = data[x_col].nunique(), data[y_col].nunique()
    ok = nx >= 2 and ny >= 2
    print(f"{title}: unique {x_col}={nx}, {y_col}={ny}, points={len(data)} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def contour_plot(data, x_col, y_col, z_col, xlabel, ylabel, zlabel,
                 title, fname, cmap='viridis', fmt=None, method='cubic',
                 vmin=None, vmax=None):
    """Generic interpolated contour plot with the real samples overlaid."""
    d = data.dropna(subset=[x_col, y_col, z_col])
    if not check_density(d, x_col, y_col, f"  {title}"):
        print("  skipped: insufficient density\n")
        return

    xi = np.linspace(d[x_col].min(), d[x_col].max(), 80)
    yi = np.linspace(d[y_col].min(), d[y_col].max(), 80)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((d[x_col], d[y_col]), d[z_col], (Xi, Yi), method=method)

    if vmin is not None and vmax is not None:
        levels_f = np.linspace(vmin, vmax, 16)
        levels_l = np.linspace(vmin, vmax, 11)
    else:
        levels_f, levels_l = 15, 10

    fig, ax = plt.subplots(figsize=(10, 6))
    cf = ax.contourf(Xi, Yi, Zi, levels=levels_f, cmap=cmap, extend='both')
    cl = ax.contour(Xi, Yi, Zi, levels=levels_l,
                    colors='black', alpha=0.3, linewidths=0.5)
    ax.clabel(cl, inline=True, fontsize=8, **({'fmt': fmt} if fmt else {}))
    ax.scatter(d[x_col], d[y_col], c='red', s=18, marker='x',
               alpha=0.55, label='measurements')

    cbar = plt.colorbar(cf, ax=ax)
    cbar.set_label(zlabel, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    plt.tight_layout()
    _savefig(fname)
    _show()
    print()


def line_plot(data, x_col, y_col, series_col, xlabel, ylabel, title, fname,
              yerr_col=None, yerr_stat_col=None, series_stride=1,
              series_fmt="{:.2f}", series_unit=""):
    """
    Plot y vs x with one line per unique value of series_col.

    TWO nested error bars per point when both error columns are given:
        red  = statistical only (yerr_stat_col) - the trend-affecting error
        blue = total            (yerr_col)       - stat + systematic scale

    The blue (total) bar is drawn first and wider-capped so the red (stat) bar
    sits inside it; the marker/line colour still encodes the series value.
    Replicates are averaged over any remaining grouping columns, errors combined
    in quadrature.
    """
    d = data.dropna(subset=[x_col, y_col])
    if len(d) == 0:
        print(f"  {title}: no data\n")
        return

    series_vals = sorted(d[series_col].unique())[::series_stride]
    colors = plt.cm.viridis(np.linspace(0, 0.92, max(len(series_vals), 1)))

    def _quad(e):
        e = np.asarray(e, dtype=float)
        return np.sqrt(np.nansum(e ** 2)) / max(len(e), 1)

    fig, ax = plt.subplots(figsize=(11, 7))
    for val, color in zip(series_vals, colors):
        sub = d[d[series_col] == val]
        agg_map = {y_col: 'mean'}
        if yerr_col is not None:
            agg_map[yerr_col] = _quad
        if yerr_stat_col is not None and yerr_stat_col in sub.columns:
            agg_map[yerr_stat_col] = _quad
        agg = sub.groupby(x_col).agg(agg_map).reset_index().sort_values(x_col)

        x, y = agg[x_col], agg[y_col]

        # blue = total error (drawn first, underneath)
        if yerr_col is not None:
            ax.errorbar(x, y, yerr=agg[yerr_col], fmt='none',
                        ecolor='tab:blue', elinewidth=2.0, capsize=6,
                        capthick=2.0, alpha=0.55, zorder=2)
        # red = statistical-only error (drawn on top, inside the blue)
        if yerr_stat_col is not None and yerr_stat_col in agg.columns:
            ax.errorbar(x, y, yerr=agg[yerr_stat_col], fmt='none',
                        ecolor='tab:red', elinewidth=2.0, capsize=3,
                        capthick=2.0, alpha=0.8, zorder=3)

        # the series line/markers on top
        ax.plot(x, y, marker='o', color=color, linewidth=2.2, markersize=6,
                alpha=0.9, zorder=4,
                label=f"{series_fmt.format(val)}{series_unit}")

    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')

    # legend: series entries plus a one-off explanation of the two bar colours
    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    err_handles = [Line2D([0], [0], color='tab:red', lw=2, label='stat error'),
                   Line2D([0], [0], color='tab:blue', lw=2, label='total error')]
    ax.legend(handles + err_handles, labels + ['stat error', 'total error'],
              loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.tight_layout()
    _savefig(fname)
    _show()
    print()


def nearest(values, target):
    return min(values, key=lambda v: abs(v - target))


# =============================================================================
# STAGE 4b: ANALYSIS PLOTS
# =============================================================================

def make_all_plots(summary, target_pressure=1.0, target_power=40.0, target_freq=2420,
                    freq_col='freq'):
    print(f"\n{'=' * 60}\nPLOTS\n{'=' * 60}\n")

    pressures = summary['pressure'].dropna().unique()
    powers = summary['power'].dropna().unique()
    p_sel = nearest(pressures, target_pressure)
    w_sel = nearest(powers, target_power)
    print(f"Fixed pressure slice: {p_sel} Torr")
    print(f"Fixed power slice:    {w_sel} W\n")

    has_freq = freq_col in summary.columns and summary[freq_col].notna().any()
    f_sel = None
    if has_freq and target_freq is not None:
        freqs = summary[freq_col].dropna().unique()
        f_sel = nearest(freqs, target_freq)
        print(f"Fixed frequency slice: {f_sel} MHz\n")
        summary = summary[summary[freq_col] == f_sel]

    argon = summary[summary['n2_flow'] == 0.0]
    at_p = summary[summary['pressure'] == p_sel]
    at_w = summary[summary['power'] == w_sel]

    if len(argon):
        contour_plot(argon, 'power', 'pressure', 'T_gas_K',
                     'Power (W)', 'Pressure (Torr)', 'Gas temperature (K)',
                     'Gas Temperature vs Power & Pressure (pure Ar)',
                     'contour_Tg_power_pressure_argon.png', cmap='hot')
        contour_plot(argon, 'power', 'pressure', 'N_s_cm3',
                     'Power (W)', 'Pressure (Torr)', 'Metastable density (cm$^{-3}$)',
                     'Metastable Density vs Power & Pressure (pure Ar)',
                     'contour_Ns_power_pressure_argon.png',
                     cmap='plasma', fmt='%.1e')

    if len(at_p):
        contour_plot(at_p, 'power', 'n2_flow', 'T_gas_K',
                     'Power (W)', 'N$_2$ (%)', 'Gas temperature (K)',
                     f'Gas Temperature vs N$_2$ & Power ({p_sel} Torr)',
                     'contour_Tg_n2_power.png', cmap='hot', vmin=300, vmax=1000)
        contour_plot(at_p, 'power', 'n2_flow', 'N_s_cm3',
                     'Power (W)', 'N$_2$ (%)', 'Metastable density (cm$^{-3}$)',
                     f'Ar 1s$_5$ Density vs N$_2$ & Power ({p_sel} Torr)',
                     'contour_Ns_n2_power.png', cmap='plasma', fmt='%.1e')

    if len(at_w):
        contour_plot(at_w, 'n2_flow', 'pressure', 'N_s_cm3',
                     'N$_2$ (%)', 'Pressure (Torr)', 'Metastable density (cm$^{-3}$)',
                     f'Metastable Density vs Pressure & N$_2$ ({w_sel} W)',
                     'contour_Ns_pressure_n2.png', cmap='plasma', fmt='%.1e')

    if len(at_p):
        line_plot(at_p, 'power', 'N_s_cm3', 'n2_flow',
                  'Power (W)', 'Metastable density (cm$^{-3}$)',
                  f'Metastable Density vs Power ({p_sel} Torr)',
                  'line_Ns_vs_power.png',
                  yerr_col='N_s_cm3_err', yerr_stat_col='N_s_cm3_stat',
                  series_stride=2, series_unit=' %')

        line_plot(at_p, 'n2_flow', 'T_gas_K', 'power',
                  'N$_2$ (%)', 'Gas temperature (K)',
                  f'Gas Temperature vs N$_2$ ({p_sel} Torr)',
                  'line_Tg_vs_n2.png',
                  yerr_col='T_gas_K_err', yerr_stat_col='T_gas_K_stat',
                  series_stride=3, series_fmt="{:.0f}", series_unit=' W')

        line_plot(at_p, 'n2_flow', 'N_s_cm3', 'power',
                  'N$_2$ (%)', 'Metastable density (cm$^{-3}$)',
                  f'Metastable Density vs N$_2$ ({p_sel} Torr)',
                  'line_Ns_vs_n2.png',
                  yerr_col='N_s_cm3_err', yerr_stat_col='N_s_cm3_stat',
                  series_stride=3, series_fmt="{:.0f}", series_unit=' W')

    return p_sel, w_sel


def make_frequency_sweep_plots(summary, freq_col='freq'):
    """Plots specific to an excitation-frequency sweep."""
    if freq_col not in summary.columns or summary[freq_col].nunique() < 2:
        print("\nNo frequency variation; skipping frequency-sweep plots.\n")
        return

    print(f"\n{'=' * 60}\nFREQUENCY SWEEP PLOTS\n{'=' * 60}\n")

    contour_plot(summary, freq_col, 'power', 'T_gas_K',
                 'Excitation frequency (MHz)', 'Power (W)', 'Gas temperature (K)',
                 'Gas Temperature vs Frequency & Power',
                 'contour_Tg_freq_power.png', cmap='hot')
    contour_plot(summary, freq_col, 'power', 'N_s_cm3',
                 'Excitation frequency (MHz)', 'Power (W)',
                 'Metastable density (cm$^{-3}$)',
                 'Ar 1s$_5$ Density vs Frequency & Power',
                 'contour_Ns_freq_power.png', cmap='plasma', fmt='%.1e')

    line_plot(summary, freq_col, 'T_gas_K', 'power',
              'Excitation frequency (MHz)', 'Gas temperature (K)',
              'Gas Temperature vs Excitation Frequency',
              'line_Tg_vs_freq.png',
              yerr_col='T_gas_K_err', yerr_stat_col='T_gas_K_stat',
              series_stride=2, series_fmt="{:.0f}", series_unit=' W')
    line_plot(summary, freq_col, 'N_s_cm3', 'power',
              'Excitation frequency (MHz)', 'Metastable density (cm$^{-3}$)',
              'Metastable Density vs Excitation Frequency',
              'line_Ns_vs_freq.png',
              yerr_col='N_s_cm3_err', yerr_stat_col='N_s_cm3_stat',
              series_stride=2, series_fmt="{:.0f}", series_unit=' W')


def slope_analysis(summary, target_pressures=(1.0,)):
    """dN_s/dPower as a function of N2 admixture, at each target pressure."""
    print(f"\n{'=' * 60}\nSLOPE ANALYSIS: dN_s/dP_rf vs N2\n{'=' * 60}")

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']
    fig, ax = plt.subplots(figsize=(12, 7))
    results = {}

    for target, color, marker in zip(target_pressures, colors, markers):
        p_sel = nearest(summary['pressure'].dropna().unique(), target)
        d = summary[summary['pressure'] == p_sel]
        if len(d) == 0:
            continue

        print(f"\nPressure {p_sel} Torr")
        print(f"{'N2 (%)':>10} | {'slope (cm^-3/W)':>18} | {'std err':>14} | {'R^2':>8}")
        print("-" * 60)

        n2_vals, slopes, errs = [], [], []
        for n2 in sorted(d['n2_flow'].unique()):
            sub = (d[d['n2_flow'] == n2]
                   .groupby('power')['N_s_cm3'].mean()
                   .reset_index().sort_values('power'))
            if len(sub) < 2:
                continue
            sl, ic, r, p, se = linregress(sub['power'], sub['N_s_cm3'])
            n2_vals.append(n2); slopes.append(sl); errs.append(se)
            print(f"{n2:>10.2f} | {sl:>18.3e} | {se:>14.3e} | {r ** 2:>8.4f}")

        if slopes:
            ax.errorbar(n2_vals, slopes, yerr=errs, marker=marker, markersize=9,
                        linewidth=2.2, capsize=5, color=color, alpha=0.85,
                        label=f'{p_sel} Torr')
            results[p_sel] = dict(n2=n2_vals, slopes=slopes, errors=errs)

    ax.axhline(0, color='k', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('N$_2$ (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel(r'$dN_s/dP_{\rm rf}$ (cm$^{-3}$ W$^{-1}$)',
                  fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=11, title='Pressure')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.tight_layout()
    _savefig('slope_dNs_dPower_vs_n2.png')
    _show()

    for p in sorted(results):
        r = results[p]
        i_min, i_max = int(np.argmin(r['slopes'])), int(np.argmax(r['slopes']))
        print(f"\nAt {p} Torr:")
        print(f"  min slope {r['slopes'][i_min]:.3e} at N2 = {r['n2'][i_min]:.2f} %")
        print(f"  max slope {r['slopes'][i_max]:.3e} at N2 = {r['n2'][i_max]:.2f} %")
        print(f"  mean      {np.mean(r['slopes']):.3e}")

    return results


def diffusion_diagnostic(summary, target_pressures=(1.0,)):
    """Test whether N_s tracks T or T^1.5 (sanity check, not a hypothesis test)."""
    print(f"\n{'=' * 60}\nDIFFUSION DIAGNOSTIC: N_s vs T and T^1.5\n{'=' * 60}")

    d = summary[(summary['n2_flow'] == 0.0)
                & (summary['pressure'].isin(target_pressures))].copy()
    d = d.dropna(subset=['T_gas_K', 'N_s_cm3'])
    if len(d) < 3:
        print("Insufficient pure-argon data at the requested pressures.\n")
        return

    d['D_proxy'] = d['T_gas_K'] ** 1.5

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, xcol, xlabel in zip(axes, ['T_gas_K', 'D_proxy'],
                                ['Gas temperature (K)', r'$T^{3/2}$ (K$^{3/2}$)']):
        for pressure in sorted(d['pressure'].unique()):
            dp = d[d['pressure'] == pressure]
            ax.errorbar(dp[xcol], dp['N_s_cm3'],
                        yerr=dp['N_s_cm3_err'], xerr=None,
                        fmt='o', markersize=8, alpha=0.7, capsize=3,
                        label=f'{pressure} Torr')
            if len(dp) > 1:
                sl, ic, r, p, se = linregress(dp[xcol], dp['N_s_cm3'])
                xr = np.array([dp[xcol].min(), dp[xcol].max()])
                ax.plot(xr, sl * xr + ic, '--', linewidth=2, alpha=0.8)
                print(f"  {xlabel:28s} p={pressure} Torr: "
                      f"slope {sl:.3e}, R^2 {r ** 2:.4f}, p-value {p:.2e}")
        ax.set_xlabel(xlabel, fontweight='bold')
        ax.set_ylabel(r'$N_s$ (cm$^{-3}$)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    r_T = d[['T_gas_K', 'N_s_cm3']].corr().iloc[0, 1]
    r_D = d[['D_proxy', 'N_s_cm3']].corr().iloc[0, 1]
    axes[0].set_title(f'$N_s$ vs $T$  (r = {r_T:.3f})')
    axes[1].set_title(f'$N_s$ vs $T^{{3/2}}$  (r = {r_D:.3f})')

    plt.tight_layout()
    _savefig('diagnostic_diffusion_loss.png')
    _show()

    print(f"\n  r(N_s, T)      = {r_T:.4f}")
    print(f"  r(N_s, T^1.5)  = {r_D:.4f}")
    print("  Difference between these is expected to be small; do not over-read it.")


# =============================================================================
# STAGE 5: LOOKING AT ONE CONDITION
# =============================================================================

def pick(df, _verbose=True, **conditions):
    """Rows matching a condition, snapping each request to the nearest existing value."""
    d = df
    for col, want in conditions.items():
        if col not in d.columns:
            raise KeyError(f"no column '{col}'; have {list(d.columns)[:12]}...")
        vals = d[col].dropna().unique()
        if len(vals) == 0:
            print(f"  {col}: nothing left to match against")
            return d.iloc[0:0]
        best = min(vals, key=lambda v: abs(v - want))
        if _verbose and abs(best - want) > 1e-9:
            print(f"  {col}: {want} -> {best} (nearest available)")
        d = d[d[col] == best]
    if _verbose:
        print(f"  {len(d)} row(s) matched")
    return d


def inspect_point(index, filtered=None, show_periods=False, period=None,
                  **conditions):
    """Re-open one scope file and plot every sawtooth period's fit."""
    hit = pick(index, **conditions)
    if len(hit) == 0:
        print("No matching scope file.")
        return []
    if len(hit) > 1:
        print(f"{len(hit)} files matched; showing the first:\n"
              f"{hit['basename'].to_string(index=False)}")
    row = hit.iloc[0]
    print(f"\nFile: {row['basename']}\n  {row['path_on']}")

    mean_on = _load_scope(row['path_on'])
    mean_off = _load_scope(row['path_off'])
    t_ref, i_ref_full = _reference_intensity(row['ref_on'], row['ref_off'])

    t_meas = mean_on.index.to_numpy()
    i_off = (mean_off[DIODE_CHANNEL].to_numpy() if mean_on.index.equals(mean_off.index)
             else np.interp(t_meas, mean_off.index.to_numpy(),
                            mean_off[DIODE_CHANNEL].to_numpy()))
    i_ref = (i_ref_full if np.array_equal(t_meas, t_ref)
             else np.interp(t_meas, t_ref, i_ref_full))

    frequency, periods = find_relative_frequency(mean_on, plot=show_periods)

    i_m = mean_on[DIODE_CHANNEL].to_numpy() - i_off + BIAS_VOLTAGE
    with np.errstate(divide='ignore', invalid='ignore'):
        absorbance = np.log((i_ref + BIAS_VOLTAGE) / i_m)

    kept = set()
    if filtered is not None and 'row_uid' in filtered.columns:
        kept = set(filtered.loc[filtered['row_uid'] == row['row_uid'], 'period'])

    out = []
    for k, p in enumerate(periods):
        if p is None:
            print(f"  period {k}: no frequency calibration")
            continue
        sl = p['slice']
        f, y = frequency[sl], absorbance[sl]
        mask = np.isfinite(f) & np.isfinite(y)
        if mask.sum() < MIN_POINTS_PER_PERIOD:
            print(f"  period {k}: only {mask.sum()} usable points")
            continue
        try:
            res = analyze_period(f[mask], y[mask], estimate_noise(y[mask]))
        except RuntimeError as e:
            print(f"  period {k}: fit failed - {e}")
            continue
        partial = 'partial' in p['label'] and not INCLUDE_PARTIAL_PERIODS
        res['period'] = k
        res['kept'] = (not kept or k in kept) and not partial
        f_native = np.sort(f[mask]) / 1e9
        res['f_data'] = f_native - res['x0']
        res['y_data'] = y[mask][np.argsort(f[mask])]
        out.append(res)
        flag = 'KEPT' if res['kept'] else ('cut: partial' if partial else 'cut')
        print(f"  period {k} [{flag}]: FWHM {res['fwhm']:.3f} +/- "
              f"{res['fwhm_err']:.3f} GHz, area {res['area']:.4f}, "
              f"chi2 {res['Chi^2']:.2f}, inflate {res['inflate']:.1f}x")

    if out:
        _plot_data_vs_fit(out, row, only_period=period)

    return out


def _plot_data_vs_fit(fits, row, only_period=None):
    """Raw absorbance points with the fitted Gaussian over them, plus residuals."""
    if only_period is not None:
        fits = [r for r in fits if r['period'] == only_period]
        if not fits:
            print(f"  period {only_period} produced no fit")
            return

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(9.5, 7), sharex=True,
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.07})

    for res in fits:
        fd, yd = res['f_data'], res['y_data']
        dense = np.linspace(fd.min(), fd.max(), 800)
        curve = gaussian_lin(dense + res['x0'], res['A'], res['x0'],
                             res['sigma'], res['offset'], res['slope'])
        pts = ax.plot(fd, yd, '.', markersize=3, alpha=0.45)[0]
        cut = '' if (res['kept'] or only_period is not None) else ' (cut)'
        ax.plot(dense, curve, '-', linewidth=2, color=pts.get_color(),
                label=f"period {res['period']}{cut}: "
                      f"FWHM {res['fwhm']:.3f} GHz, "
                      f"$\\chi^2_\\nu$ {res['Chi^2']:.1f}")

        model = gaussian_lin(fd + res['x0'], res['A'], res['x0'],
                             res['sigma'], res['offset'], res['slope'])
        axr.plot(fd, yd - model, '.', markersize=3, alpha=0.5,
                 color=pts.get_color())

    axr.axhline(0, color='k', linewidth=0.8, alpha=0.6)
    axr.set_xlabel('Relative frequency (GHz)', fontweight='bold')
    axr.set_ylabel('residual', fontweight='bold')
    ax.set_ylabel('Absorbance', fontweight='bold')
    ax.set_title(f"{row['basename']}\n"
                 f"{row['power']:.0f} W, {row['pressure']:.2f} Torr, "
                 f"{row['n2_flow']:.2f} % N$_2$, {row['freq']:.2f} MHz",
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    axr.grid(True, alpha=0.3)
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

def _las_outputs(df_path):
    """Output files for one Laser_True dataframe (see the Output config block)."""
    t = OUTPUT_TAG
    return {
        'measurements': output_file(df_path, t),
        'final_table':  output_file(df_path, f'{t}_FinalTable'),
        'file_results': output_file(df_path, f'{t}_FileResults'),
        'period_fits':  output_file(df_path, f'{t}_PeriodFits'),
        'run_info':     output_file(df_path, f'{t}_RunInfo', '.json'),
        'log':          output_file(df_path, f'{t}_Log', '.txt'),
    }


def _by_source(frame):
    """Split a results table by the Laser_True dataframe each row came from."""
    return dict(tuple(frame.groupby('df_source', sort=False)))


def _code_version():
    """Git commit of this script, flagged if the file has local edits."""
    try:
        def git(*args):
            return subprocess.run(['git', *args], cwd=_HERE, capture_output=True,
                                  text=True, timeout=10).stdout.strip()
        commit = git('rev-parse', '--short', 'HEAD')
        if not commit:
            return 'unknown (not a git checkout)'
        edited = git('status', '--porcelain', '--', Path(__file__).name)
        return commit + (' + local edits' if edited else '')
    except Exception:
        return 'unknown'


def _write_run_info(df_path, df_false, outputs, index, raw, summary, started):
    """Record what went into one run's outputs, next to them."""
    uids = set(index.loc[index['df_source'] == str(df_path), 'row_uid'])
    mine = summary[summary['row_uid'].isin(uids)]
    info = {
        'created': datetime.now().isoformat(timespec='seconds'),
        'runtime_s': round((datetime.now() - started).total_seconds(), 1),
        'script': 'LASAnalysisv6.py',
        'code_version': _code_version(),
        'input_dataframe': str(df_path),
        'laser_off_dataframes': _as_path_list(df_false),
        'outputs': {k: str(v) for k, v in outputs.items()},
        'counts': {
            'scope_files': len(uids),
            'period_fits': int(raw['row_uid'].isin(uids).sum()),
            'scope_files_with_result': int(mine['T_gas_K'].notna().sum()),
        },
        'settings': {k: globals()[k] for k in RUN_INFO_SETTINGS if k in globals()},
        'warnings': list(RUN_WARNINGS),
    }
    outputs['run_info'].write_text(json.dumps(info, indent=2, default=str))
    print(f"Saved run info         -> {outputs['run_info']}")


def _master_list():
    """MasterList.py, loaded from its file (DataAnalysis/MasterList/)."""
    spec = importlib.util.spec_from_file_location(
        'MasterList', _HERE.parent / 'MasterList' / 'MasterList.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _clear_caches():
    """Forget the scope files and paths an earlier run loaded."""
    for func in (_load_scope, _reference_intensity, resolve_path):
        func.__defaults__[-1].clear()


def _warn(msg):
    print(f"\nWARNING: {msg}")
    RUN_WARNINGS.append(msg)


def _plot_safely(plot, *args, **kwargs):
    """One set of plots; if it fails, say so and carry on (results are saved)."""
    try:
        return plot(*args, **kwargs)
    except Exception:
        traceback.print_exc(file=sys.stdout)
        plt.close('all')
        _warn(f"plotting failed in {plot.__name__} (error above); "
              f"the results are saved")
        return None


class _Tee:
    """Write to the console and a log file at the same time."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, text):
        for s in self._streams:
            s.write(text)
        return len(text)

    def flush(self):
        for s in self._streams:
            s.flush()

    def __getattr__(self, name):          # anything else: behave like the console
        return getattr(self._streams[0], name)


def main(df_true=DF_TRUE, df_false=DF_FALSE, data_root=DATA_ROOT, add_master=None):
    """
    Full pipeline for one run: the Laser_True dataframe(s) in df_true and the
    Laser_False one(s) in df_false. Results go to Output/<run folder>/, and
    everything printed also goes to <dataframe>_LAS_Log.txt there.
    add_master: add the results to the main master list (default ADD_TO_MASTER).
    Returns the intermediate tables as a dict.
    """
    true_files = _as_path_list(df_true)
    if not true_files:
        raise FileNotFoundError(f"No Laser_True dataframe matched: {df_true}")
    if add_master is None:
        add_master = ADD_TO_MASTER

    log_path = _las_outputs(true_files[0])['log']
    with open(log_path, 'w', encoding='utf-8') as log, \
            contextlib.redirect_stdout(_Tee(sys.stdout, log)):
        false_files = _as_path_list(df_false) or [f'nothing matches {df_false}']
        print(f"LASAnalysisv6, {datetime.now():%Y-%m-%d %H:%M:%S}\n"
              f"  Laser_True:  {'; '.join(true_files)}\n"
              f"  Laser_False: {'; '.join(false_files)}")
        try:
            return _main(true_files, df_true, df_false, data_root, add_master)
        except BaseException:
            traceback.print_exc(file=log)     # the console gets it from the caller
            raise


def _main(true_files, df_true, df_false, data_root, add_master):
    global FIG_DIR
    started = datetime.now()
    RUN_WARNINGS.clear()
    _clear_caches()
    outputs = {f: _las_outputs(f) for f in true_files}
    FIG_DIR = output_dir(true_files[0]) / f'{OUTPUT_TAG}_figures'

    # --- stage 0 ---
    index = load_run_index(df_true, df_false, data_root)

    # --- stage 1 ---
    if RUN_RAW_PROCESSING:
        raw = process_index(index)
        if len(raw) == 0:
            raise SystemExit("Aborting: no raw results.")
        for src, part in _by_source(raw).items():
            part.to_csv(outputs[src]['period_fits'], index=False)
            print(f"\nSaved period fits      -> {outputs[src]['period_fits']}")
    else:
        cached = [outputs[f]['period_fits'] for f in true_files]
        missing = [str(p) for p in cached if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "RUN_RAW_PROCESSING = False, but these period fits don't exist "
                "yet:\n  " + "\n  ".join(missing)
                + "\nRun once with RUN_RAW_PROCESSING = True.")
        print("Loading saved period fits:\n  " + "\n  ".join(map(str, cached)))
        raw = pd.concat([pd.read_csv(p, dtype={RUN_ID_COL: str}) for p in cached],
                        ignore_index=True)

    # --- stage 2 ---
    filtered, summary, diag = filter_and_aggregate(raw, index=index)
    if len(summary) == 0:
        raise SystemExit("Aborting: aggregation produced no rows.")

    # --- stage 3 ---
    summary = add_physics(summary)
    for src, part in _by_source(summary).items():
        part.to_csv(outputs[src]['file_results'], index=False)
        build_final_table(part).to_csv(outputs[src]['final_table'], index=False)
        print(f"Saved per-file results -> {outputs[src]['file_results']}")
        print(f"Saved final table      -> {outputs[src]['final_table']}")
    final_table = build_final_table(summary)

    print("\nSample rows:")
    cols = [c for c in ['power', 'pressure', 'n2_flow', 'freq', 'n_fits',
                        'fwhm_mean', 'fwhm_err', 'fwhm_birge',
                        'T_gas_K', 'T_gas_K_stat', 'T_gas_K_err',
                        'N_s_cm3', 'N_s_cm3_stat', 'N_s_cm3_err']
            if c in summary.columns]
    print(summary[cols].head(10).to_string(index=False))

    # --- stage 3b: back onto the acquisition dataframe ---
    df_with_physics = attach_to_dataframe(summary, df_true)

    # --- master list ---
    if add_master:
        try:
            _master_list().add_to_master(
                [outputs[f]['measurements'] for f in true_files])
        except Exception:
            traceback.print_exc(file=sys.stdout)
            _warn("the master list was NOT updated (error above); "
                  "the run's own outputs are saved")

    # --- stages 3c and 4: plots ---
    _plot_safely(plot_representative_fits, index, filtered)
    error_comp = _plot_safely(plot_error_budget_aggregate, summary)
    _plot_safely(make_all_plots, summary)
    _plot_safely(make_frequency_sweep_plots, summary)
    _plot_safely(slope_analysis, summary, target_pressures=(1.0,))
    _plot_safely(diffusion_diagnostic, summary, target_pressures=(1.0,))

    for f in true_files:
        _write_run_info(f, df_false, outputs[f], index, raw, summary, started)

    print(f"\n{'=' * 60}\nDONE  ->  {output_dir(true_files[0])}\n{'=' * 60}")
    return dict(index=index, raw=raw, filtered=filtered, summary=summary,
                diag=diag, final_table=final_table,
                df_with_physics=df_with_physics, error_comp=error_comp)


# =============================================================================
# FOLDER OF RUNS
# =============================================================================

BATCH_COLUMNS = ['run_folder', 'status', 'run_ids', 'pressure_Torr',
                 'measurements', 'with_T_gas', 'runtime_min', 'error']


def _run_patterns(folder):
    """(Laser_True glob, Laser_False glob) for one run folder."""
    folder = glob.escape(str(folder))
    return os.path.join(folder, TrueName), os.path.join(folder, FalseName)


def find_run_folders(folder):
    """Every folder at or below `folder` that holds a Laser_True dataframe."""
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"No such folder: {folder}")
    return sorted(Path(d) for d, _, _ in os.walk(folder)
                  if _as_path_list(_run_patterns(d)[0]))


def _run_is_done(run):
    """True if every Laser_True dataframe of the run has a newer _LAS.csv."""
    dfs = _as_path_list(_run_patterns(run)[0])
    outs = [output_file(f, OUTPUT_TAG) for f in dfs]
    return bool(dfs) and all(o.exists() and o.stat().st_mtime >= os.path.getmtime(f)
                             for f, o in zip(dfs, outs))


def _describe_run(run):
    """run_ids, pressures and counts of a finished run, from its _LAS.csv files."""
    outs = [output_file(f, OUTPUT_TAG) for f in _as_path_list(_run_patterns(run)[0])]
    outs = [o for o in outs if o.exists()]
    if not outs:
        return {}, []
    t = pd.concat([pd.read_csv(o, dtype={RUN_ID_COL: str}) for o in outs])
    info = {
        'run_ids': ';'.join(t[RUN_ID_COL].astype(str).unique()),
        'pressure_Torr': ';'.join(f'{p:g}' for p in sorted(t['pressure'].dropna().unique())),
        'measurements': len(t),
        'with_T_gas': int(t['T_gas_K'].notna().sum()) if 'T_gas_K' in t else 0,
    }
    return info, outs


def run_batch(folder, runs=None):
    """
    Analyse every run folder under `folder`, one after the other. A run that
    fails is recorded and the rest carry on. Writes
    Output/Batch_<folder name>/Batch_Summary.csv and the master list chosen by
    BATCH_MASTER. Returns the summary table.
    """
    global SHOW_PLOTS
    folder = Path(folder)
    runs = find_run_folders(folder) if runs is None else runs
    batch_dir = OUTPUT_ROOT / f'Batch_{folder.resolve().name}'
    batch_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}\nFOLDER OF RUNS: {len(runs)} runs in {folder}\n{'=' * 70}")
    for i, run in enumerate(runs, 1):
        print(f"  {i:3d}. {run.relative_to(folder)}")
    names = pd.Series([r.name for r in runs])
    same = sorted(set(names[names.duplicated()]))
    if same:
        print(f"\nWARNING: more than one run folder is called {same}. Their results "
              f"share Output/<name>/: the tables stay apart (named after each "
              f"dataframe) but the figures overwrite each other.")

    rows, las_files, stopped = [], [], False
    batch_start = datetime.now()
    show_plots, SHOW_PLOTS = SHOW_PLOTS, False
    try:
        for i, run in enumerate(runs, 1):
            print(f"\n{'#' * 70}\n# RUN {i}/{len(runs)}: {run}\n{'#' * 70}")
            row = {'run_folder': str(run), 'status': 'ok', 'error': ''}
            t0 = datetime.now()
            try:
                if SKIP_DONE and _run_is_done(run):
                    row['status'] = 'skipped: already done'
                    print("Results are newer than the dataframes: skipped (SKIP_DONE)")
                else:
                    main(*_run_patterns(run), add_master=False)
            except KeyboardInterrupt:
                row.update(status='stopped', error='stopped by user')
                stopped = True
            except (Exception, SystemExit) as e:
                row.update(status='FAILED', error=f'{type(e).__name__}: {e}')
                print(f"\nRUN FAILED: {row['error']}\n  Full error in "
                      f"{OUTPUT_ROOT / run.name}/*_LAS_Log.txt (if the run got that far)")
            finally:
                plt.close('all')
            row['runtime_min'] = round((datetime.now() - t0).total_seconds() / 60, 2)
            if row['status'] in ('ok', 'skipped: already done'):
                info, outs = _describe_run(run)
                row.update(info)
                las_files += outs
            rows.append(row)
            if stopped:
                break
    finally:
        SHOW_PLOTS = show_plots

    table = pd.DataFrame(rows, columns=BATCH_COLUMNS)
    table[['measurements', 'with_T_gas']] = table[['measurements', 'with_T_gas']].astype('Int64')
    table.to_csv(batch_dir / 'Batch_Summary.csv', index=False)
    counts = table['status'].value_counts()
    minutes = (datetime.now() - batch_start).total_seconds() / 60
    print(f"\n{'=' * 70}\nFOLDER OF RUNS {'STOPPED' if stopped else 'DONE'} "
          f"after {minutes:.1f} min: "
          + ", ".join(f"{n} {s}" for s, n in counts.items())
          + (f", {len(runs) - len(rows)} not started" if len(rows) < len(runs) else '')
          + f"\n{'=' * 70}")
    show = table.assign(run_folder=[Path(r).name for r in table['run_folder']])
    print(show.drop(columns='error').to_string(index=False))
    for _, r in table[table['status'] == 'FAILED'].iterrows():
        print(f"  FAILED {Path(r['run_folder']).name}: {r['error']}")
    print(f"Summary -> {batch_dir / 'Batch_Summary.csv'}")
    if stopped:
        print("Stopped by user, so the master list was left as it was. To carry "
              "on without redoing the finished runs, set SKIP_DONE = True and "
              "run the folder again.")

    if BATCH_MASTER and las_files and not stopped:
        try:
            if BATCH_MASTER == 'separate':
                _master_list().rebuild_master(las_files, master_dir=batch_dir)
            elif BATCH_MASTER == 'main':
                _master_list().add_to_master(las_files)
            else:
                print(f"WARNING: BATCH_MASTER = {BATCH_MASTER!r}: use 'separate', "
                      f"'main' or None. No master list written.")
        except Exception:
            traceback.print_exc()
            print("WARNING: master list not written (error above); "
                  "every run's own results are saved.")
    return table


if __name__ == '__main__':
    # A run folder or a folder of runs, from the command line or Location.
    _results = None
    for target in (sys.argv[1:] or [Location]):
        runs_found = find_run_folders(target)
        if not runs_found:
            raise SystemExit(f"No {TrueName} in {target} or any folder below it.")
        if runs_found == [Path(target)]:
            _results = main(*_run_patterns(target))
        else:
            batch_summary = run_batch(target, runs_found)

    # plain variables (last single run) for Spyder's variable explorer
    if _results is not None:
        index, raw, filtered = _results['index'], _results['raw'], _results['filtered']
        summary, diag, final_table = (_results['summary'], _results['diag'],
                                      _results['final_table'])
        df_with_physics = _results['df_with_physics']
        error_comp = _results['error_comp']