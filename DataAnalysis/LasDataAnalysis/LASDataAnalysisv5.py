# -*- coding: utf-8 -*-
"""
Unified LAS analysis pipeline: argon plasma with N2 admixture.

DATAFRAME-DRIVEN VERSION
------------------------
Scope files and run parameters are taken from the acquisition dataframes
(LAS_DataFrame_<run_id>_Laser_True_*.csv / _Laser_False_*.csv) instead of being
scraped out of filenames with a regex. The old PATTERN / FOLDER_PATTERN /
build_index() path is gone; everything downstream is unchanged except that each
fit now carries a `row_uid` pointing back at the exact dataframe row it came
from.

At the end, T_gas, T_gas_err, N_s and N_s_err are merged back onto the
Laser_True dataframe (one row per scope file) and written out.

Pipeline stages:
    0. load_run_index()       -> paired True/False dataframe, resolved paths
    1. process_index()        -> raw per-fit results     (MasterResults_RawData)
    2. filter_and_aggregate() -> per-scope-file means    (MasterResults_Summary)
    3. add_physics()          -> T_gas, N_s with propagated errors
    4. attach_to_dataframe()  -> physics appended to the acquisition dataframe
    5. plotting section

@author: dptro
"""

from pathlib import Path, PurePosixPath
import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.interpolate import griddata
from scipy.stats import linregress
from scipy.spatial import cKDTree

# =============================================================================
# CONFIGURATION
# =============================================================================

# Root that the *relative* paths inside the dataframe hang off. The dataframe
# stores e.g. "Data\NafisaData\LAS_1.0Torr_...\OscopeData_Laser_True\Scope_...csv",
# so DATA_ROOT is the folder that contains "Data", not the sweep folder itself.
DATA_ROOT = r"D:\Data"

# Acquisition dataframes. Accepts a single path, a glob, or a list of either;
# multiple runs are concatenated. The False list is matched to the True list by
# run_id, so the order does not matter.
Location = r'D:\Data\NafisaData\LAS_1.0Torr_SmallAdmixture_1kHzLaserFreq_N1.3_very_fine'
TrueName = r'\LAS_DataFrame_*_Laser_True_*.csv'
FalseName = r'\LAS_DataFrame_*_Laser_False_*.csv'
DF_TRUE = Location + TrueName
DF_FALSE = Location + FalseName
#r"G:\My Drive\MiniChamberData\Data\NafisaData\LAS_1.5Torr_Freq_2420\LAS_DataFrame_*_Laser_False_*.csv"

# Set False to skip raw reprocessing and load cached CSVs instead.
RUN_RAW_PROCESSING = True

RAW_CSV      = 'MasterResults_RawData_FrequencySweep_veryFine.csv'
SUMMARY_CSV  = 'MasterResults_Summary_03_FrequencySweep_veryFine.csv'
PHYSICS_CSV  = 'PhysicsDataFrame1TorrOptimal'
FINAL_CSV = 'ReruningN21TorrTest'
# Where the augmented acquisition dataframe goes. None -> alongside the source
# file, with OUT_SUFFIX appended to the stem. Files carrying that suffix are
# skipped when expanding DF_TRUE/DF_FALSE, so re-running does not feed last
# run's output back in as an input.
OUT_SUFFIX = '_with_Tg_Ns'
DATAFRAME_OUT = None

# --- Dataframe column mapping ------------------------------------------------
# left = name used internally, right = column in the acquisition dataframe.
# Anything missing is filled with NaN and warned about.
COLUMN_MAP = {
    'power':    'power',
    'pressure': 'pressure',
    'n2_flow':  'Nitrogen_Percent',   # percent, matching the old "NitrogenFlow"
    'freq':     'frequency',
}

# Extra dataframe columns copied onto every fit row for later grouping/plotting.
CARRY_COLS = ['run_id', 'Measured Pressure', 'delivered_power',
              'delivered_power_error', 'gamma', 'gamma_error',
              'forward_power', 'forward_power_error',
              'reverse_power', 'reverse_power_error',
              'Nitrogen_Gas_Flow', 'Argon_Gas_Flow']

SCOPE_COL     = 'ScopeFileLocation'
SCOPE_OFF_COL = 'ScopeFileLocation_Off_Loop'   # plasma-off reference, per row
RUN_ID_COL    = 'run_id'

# Older runs (before per-condition plasma-off logging was added) used one
# shared plasma-off reference file for the whole run instead of logging it
# per row. Set the relevant path(s) below when SCOPE_OFF_COL is missing from
# a frame; leave None to require the column and fail loudly as before.
SINGLE_REF_ON_PATH = r'LAS_1.0Torr_SmallAdmixture_1kHzLaserFreq_N1.3_very_fine\OscopeData_Laser_True\Plasma Off Measuremnt.csv'   # laser-on, plasma-off  (fills Laser_True's column)
SINGLE_REF_OFF_PATH = r'LAS_1.0Torr_SmallAdmixture_1kHzLaserFreq_N1.3_very_fine\OscopeData_Laser_False\Plasma Off Measuremnt.csv'  # laser-off, plasma-off (fills Laser_False's column)


#SINGLE_REF_ON_PATH = r'D:\Data\NafisaData\LAS_1.0Torr_SmallAdmixture_1kHzLaserFreq_N1.3_very_fine_forward\OscopeData_Laser_False\Plasma Off Measuremnt.csv'   # laser-on, plasma-off  (fills Laser_True's column)
#SINGLE_REF_OFF_PATH = r'D:\Data\NafisaData\LAS_1.0Torr_SmallAdmixture_1kHzLaserFreq_N1.3_very_fine_forward\OscopeData_Laser_False\Plasma Off Measuremnt.csv'  # laser-off, plasma-off (fills Laser_False's column)

# --- Pairing laser-on to laser-off -------------------------------------------
# The acquisition program treats the two laser loops as separate runs, so
# run_id does NOT match between the True and False dataframes and cannot be a
# merge key. Strategies are tried in order until one pairs rows:
#
#   'path'      relative scope path with the True/False token removed. Exact,
#               and cannot collide across pressure folders because the run
#               folder is part of the key. Preferred.
#   'basename'  scope filename only. Use if the two loops write into
#               differently-named folders.
#   'params'    rounded (power, pressure, n2_flow, freq). Last resort: the
#               floats have to survive a round-trip through both CSVs.
#
# If several laser-off rows match one laser-on row, the first is taken and the
# rest are dropped with a count printed.
PAIR_STRATEGIES = ('path', 'basename', 'condition', 'params', 'fuzzy')
PAIR_PARAM_KEYS = ['power', 'pressure', 'n2_flow', 'freq']
# Decimal places used when matching on parameters. freq is stored to full float
# precision (2410.862068965517) but only 2 dp survive into the filename, so
# rounding is mandatory, not cosmetic.
PAIR_ROUND = {'power': 3, 'pressure': 3, 'n2_flow': 2, 'freq': 2}
# Tolerance for the 'fuzzy' pairing strategy (last resort): a True row matches
# a False row if EVERY one of these params is within its tolerance, using
# nearest-neighbor assignment so ties go to the closest pair first.
FUZZY_TOL = {'power': 0.05, 'pressure': 0.05, 'n2_flow': 0.05, 'freq': 0.05}
# Trial number is not a dataframe column; pull it out of the scope filename if
# it is there, otherwise 0.
TRIAL_RE = re.compile(r"TrialNumber_(?P<trial>\d+)")

# --- Oscilloscope channel mapping -------------------------------------------
FP_CHANNEL    = 'CH1'   # Fabry-Perot etalon fringes
RAMP_CHANNEL  = 'CH2'   # laser sweep sawtooth
DIODE_CHANNEL = 'CH3'   # transmitted intensity
BIAS_VOLTAGE  = 0.0     # replace with a measurement

# --- Filtering ---------------------------------------------------------------
# NSMALLEST_PER_GROUP now selects the N best-fitting sawtooth periods WITHIN a
# single scope file, which is what it was always meant to do.
# WARNING: selecting on goodness-of-fit biases toward high-amplitude periods and
# artificially suppresses the Birge ratio. Prefer CHI2_MAX now that chi2 is on a
# physically meaningful scale.
NSMALLEST_PER_GROUP = 3
CHI2_MAX = None          # e.g. 5.0 to reject genuinely bad fits instead

# One summary row per scope file, so the result maps 1:1 back onto the
# acquisition dataframe. Use aggregate_conditions() afterwards if you want to
# pool repeat runs of the same condition.
GROUP_KEYS = ['row_uid']

# --- Fitting -----------------------------------------------------------------
FSR = 1.5e9              # Fabry-Perot free spectral range, Hz
F_GRID = np.arange(-5, 5, 0.002)   # GHz, plotting grid only (not used for chi2)
MIN_POINTS_PER_PERIOD = 50

# --- Plotting ----------------------------------------------------------------
SAVE_FIGURES = True
FIG_DIR = 'figures'
PLOT_EVERY_N_FITS = None   # diagnostic spectrum plots; set None to disable


# =============================================================================
# SPECTROSCOPIC CONSTANTS
# =============================================================================
#
# N_s scales as g_lower / (g_upper * A_ki), so these choices move the answer by
# roughly an order of magnitude. Confirm the line identity and pull g and A_ki
# from the NIST ASD before trusting absolute densities.

LAMBDA_0 = 696.5431e-9      # m   <-- confirm: 696.5431e-9 for 1s5 -> 2p2?
L_PATH   = 0.04          # m   absorption path length (4 cm) -- SI, not cm
M_AR     = 40.0          # amu
G_LOWER  = 5             # <-- confirm: 5 for 1s5
G_UPPER  = 3             # <-- confirm: 3 for 2p2
A_KI     = 6.4e6         # s^-1  <-- confirm: 6.39e6 for 696.5431 nm

C = 2.9979e8             # m/s
DOPPLER_CONST = 7.16e-7  # Doppler FWHM coefficient, dlambda = K*lambda0*sqrt(T/M)

K_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))   # sigma -> FWHM


# =============================================================================
# MODEL FUNCTIONS
# =============================================================================

def gaussian_lin(x, A, x0, sig, b, m):
    """Gaussian on a linear background."""
    return A * np.exp(-(x - x0) ** 2 / (2 * sig ** 2)) + b + m * x


# =============================================================================
# ERROR COMBINATION
# =============================================================================

def birge_combine(values, errors):
    """
    Combine repeat measurements with individual uncertainties.

    Weighted mean with Birge-ratio inflation:

        w_j     = 1 / u_j^2
        xbar    = sum(w_j x_j) / sum(w_j)
        u_int   = 1 / sqrt(sum(w_j))                          (internal)
        u_ext   = sqrt( sum(w_j (x_j - xbar)^2) /
                        ((n-1) sum(w_j)) )                    (external)
        R       = u_ext / u_int                               (Birge ratio)
        u_final = u_int * max(1, R)

    R ~ 1 means the replicates scatter consistently with their own fit errors.
    R >> 1 means an unmodelled systematic (calibration drift, plasma wander)
    dominates, so the internal error is inflated to cover it. max(1, R) avoids
    deflating below the statistical floor when R < 1 (usually overfitting).

    Falls back to an unweighted mean with SEM if any per-point error is
    unusable.

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
    out = [f for f in out if not Path(f).stem.endswith(OUT_SUFFIX)]
    return sorted(dict.fromkeys(out))


def _normalise(rel):
    """Windows-style relative path from the dataframe -> platform-native Path."""
    return Path(str(rel).replace('\\', '/'))


def resolve_path(rel, data_root=DATA_ROOT, df_path=None, _cache={}):
    """
    Turn a dataframe path into something openable.

    Tries, in order:
      1. the path as given, if it is already absolute and exists
      2. DATA_ROOT / relpath
      3. every ancestor of the dataframe's own folder as a candidate root

    Step 3 is what saves you when the drive letter changes or the tree gets
    copied somewhere else. Returns None if nothing exists, so the caller can
    report the miss instead of dying on open().
    """
    rel = _normalise(rel)
    key = (str(rel), str(data_root), str(df_path))
    if key in _cache:
        return _cache[key]

    candidates = []
    if rel.is_absolute():
        candidates.append(rel)
    if data_root:
        candidates.append(Path(data_root) / rel)
    if df_path is not None:
        here = Path(df_path).resolve().parent
        candidates.extend(anc / rel for anc in [here, *here.parents])

    found = next((c for c in candidates if c.exists()), None)
    _cache[key] = found
    return found


def _laser_agnostic_path(p):
    """
    Relative scope path with the laser-loop token stripped, e.g.
    ...\\OscopeData_Laser_True\\Scope_...csv  ->  .../oscopedata_laser/scope_...csv
    so the laser-on and laser-off copies of one measurement collapse onto the
    same string.
    """
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
    """
    Nearest-neighbour pairing on (power, pressure, n2_flow, freq), independent
    per-column tolerance (Chebyshev/max-norm after scaling each column by its
    tolerance). Catches what exact-round strategies miss -- e.g. a repeating
    decimal like 1/3 % N2 landing as .33 in one loop's filename and .34 in the
    other's -- without matching anything that isn't obviously the same
    setpoint.

    One-to-one: the globally closest candidate pairs are assigned first and
    removed from the pool, so one False row can't be claimed by two different
    True rows.
    """
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
    # distance <= 1 after scaling == every column within its own tolerance
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
    """
    Merge laser-on to laser-off, cascading through strategies. Each strategy is
    applied only to the True rows still unmatched after the previous one, so a
    row that fails 'path' (e.g. a fractional n2_flow like 1/3 or 2/3 % that
    rounds into the filename differently between the two loops) still gets a
    shot at 'basename' or 'params' instead of being silently dropped.
    """
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
            # Broadcast join: the plasma-off reference doesn't depend on
            # excitation frequency (no plasma -> freq is meaningless), so the
            # acquisition program logs exactly one False row per (power,
            # pressure, n2_flow) and reuses it across every frequency in that
            # condition's True-loop sweep. Deliberately excludes 'freq' from
            # the key -- that's the whole point of this strategy.
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
            "  - 'path' needs the two loops to write into sibling folders whose\n"
            "    names differ only by Laser_True / Laser_False.\n"
            "  - 'basename' needs identical scope filenames in both loops.\n"
            "  - 'params' needs power/pressure/n2_flow/freq to agree once rounded\n"
            f"    to {PAIR_ROUND}.\n"
            "  Print t['_pathkey'].head() and f_['_pathkey'].head() to see how the\n"
            "  two sides actually differ.")

    pairs = pd.concat(all_pairs, ignore_index=True).drop(columns=['_true_idx'])
    print(f"  TOTAL paired: {len(pairs)} / {len(t)}  (by strategy: {counts})")

    if len(remaining):
        print(f"  {len(remaining)} rows never matched by ANY strategy, e.g.:")
        show = [c for c in ('_basename', '_p_n2_flow', '_p_freq') if c in remaining.columns]
        print(remaining[show].head(10).to_string(index=False))

    return pairs, counts, None
    raise RuntimeError(
        "No rows paired by any strategy in PAIR_STRATEGIES.\n"
        "  - 'path' needs the two loops to write into sibling folders whose\n"
        "    names differ only by Laser_True / Laser_False.\n"
        "  - 'basename' needs identical scope filenames in both loops.\n"
        "  - 'params' needs power/pressure/n2_flow/freq to agree once rounded\n"
        f"    to {PAIR_ROUND}.\n"
        "  Print t['_pathkey'].head() and f_['_pathkey'].head() to see how the\n"
        "  two sides actually differ.")


def _read_run_dataframe(path):
    df = pd.read_csv(path)
    df['_df_source'] = str(path)
    return df


def load_run_index(df_true=DF_TRUE, df_false=DF_FALSE, data_root=DATA_ROOT,
                   column_map=COLUMN_MAP, carry_cols=CARRY_COLS,
                   require_files=True):
    """
    Build the processing index from the acquisition dataframes.

    Replaces build_index() + the old filename regex. The Laser_True and
    Laser_False dataframes are paired on (run_id, scope basename), which is
    exact -- unlike merging on float columns such as `frequency`, where
    2410.862068965517 in one file and 2410.86 in the other would silently drop
    the row.

    Returns a dataframe with one row per scope file and these columns:
        row_uid, trial, power, pressure, n2_flow, freq,
        path_on, path_off, ref_on, ref_off, + carry_cols + the original columns.
    """
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

  ###############
    for frame, tag, fallback in (
          (t, 'Laser_True', SINGLE_REF_ON_PATH),
          (f_, 'Laser_False', SINGLE_REF_OFF_PATH)):
          if SCOPE_COL not in frame.columns:
              raise KeyError(f"{tag} dataframe is missing ['{SCOPE_COL}']")
          if SCOPE_OFF_COL not in frame.columns:
              if fallback is None:
                  raise KeyError(
                      f"{tag} dataframe is missing '{SCOPE_OFF_COL}' and no fallback "
                      f"path is set. This looks like an older run that used a single "
                      f"shared plasma-off file instead of logging it per row -- set "
                      f"SINGLE_REF_ON_PATH / SINGLE_REF_OFF_PATH to that file's path.")
              print(f"  NOTE: {tag} has no '{SCOPE_OFF_COL}' column; broadcasting "
                    f"single reference file to every row: {fallback}")
              frame[SCOPE_OFF_COL] = fallback
  
  
  
  ###############
    # --- pair the two loops ---
    # run_id is deliberately NOT a merge key: the acquisition program assigns a
    # fresh id to each laser loop, so the True and False dataframes never share
    # one.
    _prepare_keys(t, column_map)
    _prepare_keys(f_, column_map)

    print("\nPairing laser-on to laser-off:")
    pairs, strategy_counts, _ = _pair_frames(t, f_, column_map)
    print(f"Strategy breakdown: {strategy_counts}")
    print(f"  run_id on:  {sorted(t[RUN_ID_COL].unique())}")
    print(f"  run_id off: {sorted(f_[RUN_ID_COL].unique())}")

    # matched = set(map(tuple, pairs[keys].to_numpy()))
    # unmatched = t[~t[keys].apply(tuple, axis=1).isin(matched)]
    # if len(unmatched):
    #     print(f"WARNING: {len(unmatched)} laser-on rows have no laser-off "
    #           f"partner, e.g.\n{unmatched['_basename'].head().to_string(index=False)}")

    # --- internal parameter names ---
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

    # --- resolve every path once ---
    df_src = pairs['_df_source']
    idx['path_on']  = [resolve_path(p, data_root, s)
                       for p, s in zip(pairs[SCOPE_COL], df_src)]
    idx['path_off'] = [resolve_path(p, data_root, s)
                       for p, s in zip(pairs[f'{SCOPE_COL}_off'], df_src)]
    idx['ref_on']   = [resolve_path(p, data_root, s)
                       for p, s in zip(pairs[SCOPE_OFF_COL], df_src)]
    idx['ref_off']  = [resolve_path(p, data_root, s)
                       for p, s in zip(pairs[f'{SCOPE_OFF_COL}_off'], df_src)]

    idx['basename'] = pairs['_basename'].to_numpy()
    idx['df_source'] = df_src.to_numpy()
    idx['run_id_off'] = pairs[f'{RUN_ID_COL}_off'].to_numpy()
    # row_uid ties every downstream fit back to one row of the Laser_True file.
    idx['row_uid'] = (pairs[RUN_ID_COL].astype(str) + '|' + pairs['_basename'])

    if idx['row_uid'].duplicated().any():
        n = int(idx['row_uid'].duplicated().sum())
        print(f"WARNING: {n} duplicate row_uid values; results for those rows "
              f"will be pooled rather than kept separate")

    missing_cols = ['path_on', 'path_off', 'ref_on', 'ref_off']
    miss = idx[missing_cols].isna().any(axis=1)
    if miss.any():
        print(f"\nWARNING: {int(miss.sum())} rows have unresolvable file paths.")
        bad = pairs.loc[miss.to_numpy(), SCOPE_COL].head(3).tolist()
        for b in bad:
            print(f"   {b}")
        print(f"   DATA_ROOT is currently: {data_root}")
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
from scipy.interpolate import PchipInterpolator

def find_relative_frequency(mean_df, fsr=FSR, min_peaks=3, min_pts=1000,
                            prom=0.1, plot=False, method="pchip", poly_deg=2):
    """
    Build a relative frequency axis (Hz) from Fabry-Perot fringes.

    method: "linear" (original piecewise-linear), "pchip" (monotonic cubic
            through fringe points, default), or "poly" (global polynomial
            fit of degree poly_deg per period — more robust to noisy/missed
            fringes, less flexible for sharp local curvature).

    Returns (Frequency, periods). Frequency is NaN wherever calibration is
    invalid. periods is a list of dicts (or None) describing each sawtooth
    period, including leading/trailing partials.
    """
    time = mean_df.index.to_numpy()
    perot = mean_df[FP_CHANNEL].to_numpy()
    ramp = mean_df[RAMP_CHANNEL].to_numpy()

    # --- 1. sawtooth falling edges ---
    dy = np.diff(ramp)
    rough, _ = find_peaks(-dy, height=0.6 * np.max(-dy))
    if len(rough) < 2:
        raise RuntimeError("Could not find sawtooth edges")
    est_period = np.median(np.diff(rough))
    edges, _ = find_peaks(-dy, height=0.5 * np.max(-dy),
                          distance=int(0.8 * est_period))

    # --- 2. Fabry-Perot peaks, relative thresholds ---
    p = perot - np.median(perot)
    span = np.percentile(p, 99) - np.percentile(p, 1)
    peaks, _ = find_peaks(p, prominence=prom * span, distance=40, width=2)

    # --- 3. period list including partial ends ---
    period_ranges = []
    if edges[0] >= min_pts:
        period_ranges.append((0, edges[0], "leading partial"))
    for i in range(len(edges) - 1):
        period_ranges.append((edges[i], edges[i + 1], f"full {i}"))
    if (len(time) - edges[-1]) >= min_pts:
        period_ranges.append((edges[-1], len(time), "trailing partial"))

    # --- 4. calibrate each period ---
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
                        "method": method})

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

    chi2 is computed on the native data points, NOT on the interpolated
    F_GRID. Interpolating onto a +/-5 GHz grid produces NaN wherever the data
    does not span the full grid, which poisons np.sum and silently discards
    almost every fit; it also oversamples by ~10x and inflates chi2 by the
    oversampling factor.

    Parameter errors come from the curve_fit covariance matrix. Note that
    without sigma= passed to curve_fit, pcov is rescaled by the residual
    variance, giving an empirical error estimate. Pass
    sigma=<noise array>, absolute_sigma=True if you want errors tied to the
    independently measured detector noise instead.

    Area uncertainty includes the A-sigma covariance term:
        u^2(area) = 2*pi * [ sig^2 var(A) + A^2 var(sig) + 2 A sig cov(A,sig) ]
    Dropping the cross term overestimates the error; A and sigma are strongly
    anti-correlated in Gaussian fits.
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

    popt, pcov = curve_fit(gaussian_lin, f, y, p0=p0,
                           bounds=(lo, hi), maxfev=10000)
    A, x0, sig, b, m = popt
    sig = abs(sig)

    # --- chi2 on native points ---
    resid = y - gaussian_lin(f, A, x0, sig, b, m)
    dof = len(f) - len(popt)
    chi2_red = float(np.sum((resid / sigma_noise) ** 2) / dof) if dof > 0 else np.nan

    # --- parameter uncertainties ---
    pcov_ok = np.all(np.isfinite(pcov))
    if pcov_ok:
        perr = np.sqrt(np.diag(pcov))
        A_err, sig_err = perr[0], perr[2]
        dA, ds = sig * np.sqrt(2 * np.pi), A * np.sqrt(2 * np.pi)
        area_var = (dA ** 2 * pcov[0, 0] + ds ** 2 * pcov[2, 2]
                    + 2.0 * dA * ds * pcov[0, 2])
        area_err = float(np.sqrt(area_var)) if area_var > 0 else np.nan
    else:
        A_err = sig_err = area_err = np.nan

    # --- plotting arrays only ---
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
        "n_points": len(f),
    }


def _load_scope(path, chans=(FP_CHANNEL, RAMP_CHANNEL, DIODE_CHANNEL),
                cache=False, _cache={}):
    """
    Read a scope CSV and average repeat sweeps onto the time axis.

    cache=True only for the plasma-off references, which are reused by many
    rows. Caching the measurement files as well would hold every trace in the
    run in memory (~0.6 MB each, so ~200 MB per 330-file dataframe, and it
    grows with every extra pressure you add to DF_TRUE).
    """
    key = str(path)
    if cache and key in _cache:
        return _cache[key]
    d = pd.read_csv(path)
    out = d.groupby('time')[list(chans)].mean()
    if cache:
        _cache[key] = out
    return out


def _reference_intensity(ref_on_path, ref_off_path, _cache={}):
    """
    Laser-only reference trace: (plasma off, laser on) - (plasma off, laser off).

    Cached, because the same Plasma_Off file is reused by many rows. Returns
    (time, i_ref).
    """
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

    Replaces process_single_folder() / process_all_folders(). Each output row
    carries `row_uid` so the fits can be traced back to, and later merged onto,
    the acquisition dataframe.
    """
    print(f"\n{'=' * 60}\nRAW PROCESSING\n{'=' * 60}\n")

    meta_cols = [c for c in index.columns
                 if c not in ('path_on', 'path_off', 'ref_on', 'ref_off')]

    rows = []
    n_plotted = 0
    failures = []

    # to_dict('records') rather than itertuples: itertuples mangles column names
    # that are not valid identifiers ('Measured Pressure' -> _7), which silently
    # breaks the metadata carried onto every fit.
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

        # Align the measurement to the reference time base. The old code
        # subtracted two Series and relied on identical time grids; a mismatch
        # produced silent all-NaN absorbance.
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
        plasma_off = i_off

        meta = {c: row[c] for c in meta_cols}

        for k, p in enumerate(periods):
            if p is None:
                continue

            sl = p["slice"]
            f, y, z = frequency[sl], absorbance[sl], plasma_off[sl]
            mask = np.isfinite(f) & np.isfinite(y)
            if mask.sum() < MIN_POINTS_PER_PERIOD:
                continue

            sigma_noise = np.std(z, ddof=1)
            if not np.isfinite(sigma_noise) or sigma_noise <= 0:
                # dead or saturated channel for this slice; do not paper over it
                print(f"    period {k}: invalid sigma_noise ({sigma_noise}) - skipping")
                continue

            try:
                res = analyze_period(f[mask], y[mask], sigma_noise)
            except RuntimeError as e:
                print(f"    period {k}: fit failed - {e}")
                continue

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
                "sigma_noise": sigma_noise,
                "n_points": res["n_points"],
                "Fluctuations": float(np.mean(z)),
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
    if len(master):
        print(f"Files with >=1 fit: {master['row_uid'].nunique()}")
        for col in ('power', 'pressure', 'n2_flow', 'freq'):
            if col in master:
                print(f"  {col}: {np.sort(master[col].unique())[:8]}")
    return master


def _plot_fit_diagnostic(res, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(res['spec_x'], res['spec'], label='data', linewidth=2)
    plt.plot(res['spec_x'], res['Fit'], label='fit', linewidth=2)
    plt.xlabel('Relative frequency (GHz)')
    plt.ylabel('Absorbance')
    plt.title(filename, fontsize=9)
    plt.text(0.05, 0.95,
             f"$\\chi^2_\\nu$ = {res['Chi^2']:.2f}\n"
             f"FWHM = {res['fwhm']:.3f} $\\pm$ {res['fwhm_err']:.3f} GHz",
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
                         index=None):
    """
    Filter per-period fits and aggregate to one row per group (by default one
    row per scope file), carrying Birge-combined uncertainties on FWHM and area.

    Pass `index` to re-attach the acquisition-dataframe columns (power,
    pressure, n2_flow, freq, ...) to the summary.
    """
    print(f"\n{'=' * 60}\nFILTERING & AGGREGATION\n{'=' * 60}\n")

    diagnostics = {"n_raw": len(results_df), "stages": {}}

    stage = results_df.dropna(subset=['Chi^2']).copy()
    removed = len(results_df) - len(stage)
    diagnostics["stages"]["dropna_chi2"] = {"removed": removed, "remaining": len(stage)}
    print(f"After dropna(Chi^2):      removed {removed}, remaining {len(stage)}")

    if chi2_max is not None:
        before = len(stage)
        stage = stage[stage['Chi^2'] <= chi2_max]
        diagnostics["stages"]["chi2_max"] = {
            "threshold": chi2_max, "removed": before - len(stage), "remaining": len(stage)}
        print(f"After chi2 <= {chi2_max}:        removed {before - len(stage)}, "
              f"remaining {len(stage)}")

    if nsmallest is not None:
        # Keep the N best periods within each scope file. sort_values + head is
        # equivalent to per-group nsmallest but keeps every column;
        # groupby.apply(..., include_groups=False) silently DROPS the grouping
        # columns, which is what produced the old KeyError: 'trial'.
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

    # --- aggregate ---
    print(f"{'=' * 60}\nAGGREGATION\n{'=' * 60}\n")

    records = []
    for keys, g in filtered_df.groupby(group_keys, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = dict(zip(group_keys, keys))

        a = birge_combine(g['area'].to_numpy(), g['area_err'].to_numpy())
        w = birge_combine(g['fwhm'].to_numpy(), g['fwhm_err'].to_numpy())

        rec.update({
            'area_mean': a['mean'], 'area_err': a['err'],
            'area_err_int': a['err_int'], 'area_err_ext': a['err_ext'],
            'area_birge': a['birge'], 'area_method': a['method'],
            'area_std': g['area'].std(ddof=1) if len(g) > 1 else np.nan,
            'area_sem': (g['area'].std(ddof=1) / np.sqrt(len(g))) if len(g) > 1 else np.nan,

            'fwhm_mean': w['mean'], 'fwhm_err': w['err'],
            'fwhm_err_int': w['err_int'], 'fwhm_err_ext': w['err_ext'],
            'fwhm_birge': w['birge'], 'fwhm_method': w['method'],
            'fwhm_std': g['fwhm'].std(ddof=1) if len(g) > 1 else np.nan,
            'fwhm_sem': (g['fwhm'].std(ddof=1) / np.sqrt(len(g))) if len(g) > 1 else np.nan,

            'n_fits': len(g),
            'chi2_median': g['Chi^2'].median(),
        })
        records.append(rec)

    summary = pd.DataFrame(records)
    diagnostics["summary_groups"] = len(summary)

    # --- re-attach run parameters ---
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
    """
    Optional second pass: pool repeat scope files that share the same nominal
    condition. Only useful when a condition was measured more than once (several
    trials or run_ids); with one file per condition this is a no-op.
    """
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
# STAGE 3: PHYSICS
# =============================================================================

def add_physics(summary):
    """
    Convert fitted FWHM and area into gas temperature and metastable density,
    propagating uncertainties.

    Gas temperature (Doppler):
        dlambda = FWHM_nu * lambda0^2 / c
        T_g     = M * (dlambda / (K_D * lambda0))^2
      T is quadratic in FWHM, so the RELATIVE error doubles:
        u(T)/T = 2 * u(FWHM)/FWHM

    Metastable density (integrated absorption):
        N_s = 8*pi*g_lower / (lambda0^2 * g_upper * A_ki * L) * integral(OD dnu)
      N_s is linear in area, so:
        u(N_s)/N_s = u(area)/area

    CAVEAT: the fitted width is the TOTAL linewidth. Pressure broadening,
    laser linewidth, power broadening and etalon resolution all add to it, so
    T_g here is an upper bound. A Voigt deconvolution with an independently
    measured instrument function is the proper treatment.
    """
    s = summary.copy()

    # --- gas temperature ---
    fwhm_hz = s['fwhm_mean'] * 1e9          # GHz -> Hz
    fwhm_hz_err = s['fwhm_err'] * 1e9

    s['fwhm_wavelength_m'] = fwhm_hz * LAMBDA_0 ** 2 / C
    s['T_gas_K'] = M_AR * (s['fwhm_wavelength_m'] / (DOPPLER_CONST * LAMBDA_0)) ** 2

    rel_fwhm = fwhm_hz_err / fwhm_hz
    s['T_gas_K_err'] = 2.0 * s['T_gas_K'] * rel_fwhm

    # --- metastable density ---
    NS_CONST = 8.0 * np.pi * G_LOWER / (LAMBDA_0 ** 2 * G_UPPER * A_KI * L_PATH)

    s['N_s_m3'] = NS_CONST * s['area_mean'] * 1e9      # area GHz -> Hz
    s['N_s_m3_err'] = NS_CONST * s['area_err'] * 1e9
    s['N_s_cm3'] = s['N_s_m3'] * 1e-6
    s['N_s_cm3_err'] = s['N_s_m3_err'] * 1e-6

    print(f"\n{'=' * 60}\nDERIVED QUANTITIES\n{'=' * 60}")
    print(f"T_gas: {s['T_gas_K'].min():.0f} - {s['T_gas_K'].max():.0f} K "
          f"(mean {s['T_gas_K'].mean():.0f} K)")
    rel_T = (s['T_gas_K_err'] / s['T_gas_K']).replace([np.inf, -np.inf], np.nan).dropna()
    if len(rel_T):
        print(f"  median relative uncertainty: {rel_T.median() * 100:.1f}%")

    print(f"N_s:   {s['N_s_cm3'].min():.2e} - {s['N_s_cm3'].max():.2e} cm^-3 "
          f"(mean {s['N_s_cm3'].mean():.2e})")
    rel_N = (s['N_s_cm3_err'] / s['N_s_cm3']).replace([np.inf, -np.inf], np.nan).dropna()
    if len(rel_N):
        print(f"  median relative uncertainty: {rel_N.median() * 100:.1f}%")

    if not (1e8 <= s['N_s_cm3'].median() <= 1e12):
        print("\n  !! N_s median is outside the 1e8-1e12 cm^-3 range typical of\n"
              "     low-temperature plasmas. Re-check g_lower, g_upper, A_ki and\n"
              "     the line identity in the constants block.")
    return s

def build_final_table(summary):
    """
    Collapse the summary dataframe down to just the columns you actually want
    to look at: input setpoints (no error), then measured power/pressure/N2/
    flow, forward/reflected power, gamma, N_s, T_gas -- each with its error.

    Measured N2 percent isn't a logged column; it's computed from the two
    measured flows. No per-reading error exists for Nitrogen_Gas_Flow or
    Argon_Gas_Flow, so that computed percent has no propagated error either.
    """
    s = summary
    out = pd.DataFrame({
        # --- input / setpoint (no error) ---
        'Power_Input_W':       s['power'],
        'Pressure_Input_Torr': s['pressure'],
        'N2_Percent_Input':    s['n2_flow'],

        # --- measured ---
        'Power_Measured_W':       s['delivered_power'],
        'Power_Measured_W_err':   s['delivered_power_error'],
        'Pressure_Measured_Torr': s['Measured Pressure'],

        'N2_Percent_Measured':    100 * s['Nitrogen_Gas_Flow']
                                   / (s['Nitrogen_Gas_Flow'] + s['Argon_Gas_Flow']),
        'N2_Flow_Measured_sccm':  s['Nitrogen_Gas_Flow'],
        'Ar_Flow_Measured_sccm':  s['Argon_Gas_Flow'],

        'Forward_Power_W':      s['forward_power'],
        'Forward_Power_W_err':  s['forward_power_error'],
        'Reflected_Power_W':    s['reverse_power'],
        'Reflected_Power_W_err':s['reverse_power_error'],
        'Gamma':                s['gamma'],
        'Gamma_err':            s['gamma_error'],

        'N_s_cm3':     s['N_s_cm3'],
        'N_s_cm3_err': s['N_s_cm3_err'],
        'T_gas_K':     s['T_gas_K'],
        'T_gas_K_err': s['T_gas_K_err'],
    })
    print(f"\nFinal table: {len(out)} rows, {len(out.columns)} columns")
    return out

# =============================================================================
# STAGE 3b: WRITE RESULTS BACK ONTO THE ACQUISITION DATAFRAME
# =============================================================================

PHYSICS_OUT_COLS = {
    'T_gas_K':      'T_gas_K',
    'T_gas_K_err':  'T_gas_K_err',
    'N_s_cm3':      'N_s_cm3',
    'N_s_cm3_err':  'N_s_cm3_err',
    'fwhm_mean':    'fwhm_GHz',
    'fwhm_err':     'fwhm_GHz_err',
    'area_mean':    'area_GHz',
    'area_err':     'area_GHz_err',
    'n_fits':       'n_periods_used',
    'chi2_median':  'chi2_median',
    'fwhm_birge':   'fwhm_birge',
}


def attach_to_dataframe(summary, df_true=DF_TRUE, out_path=DATAFRAME_OUT,
                        cols=PHYSICS_OUT_COLS):
    """
    Append T_gas, T_gas_err, N_s, N_s_err (and the widths they came from) to the
    original Laser_True acquisition dataframe and write it out.

    Matching is on (run_id, scope basename) -- the same key used to build the
    index -- so every row lands on exactly the measurement it belongs to. Rows
    that produced no usable fit keep NaN rather than being dropped.
    """
    print(f"\n{'=' * 60}\nAPPENDING PHYSICS TO ACQUISITION DATAFRAME\n{'=' * 60}\n")

    if 'row_uid' not in summary.columns:
        raise KeyError("summary has no row_uid; run filter_and_aggregate with "
                       "GROUP_KEYS=['row_uid']")

    keep = ['row_uid'] + [c for c in cols if c in summary.columns]
    phys = summary[keep].rename(columns=cols).drop_duplicates('row_uid')

    outputs = []
    for path in _as_path_list(df_true):
        df = pd.read_csv(path)
        if RUN_ID_COL not in df.columns:
            df[RUN_ID_COL] = 'run0'
        basename = df[SCOPE_COL].map(
            lambda p: PurePosixPath(str(p).replace('\\', '/')).name)
        df['row_uid'] = df[RUN_ID_COL].astype(str) + '|' + basename

        merged = df.merge(phys, on='row_uid', how='left')
        n_hit = merged['T_gas_K'].notna().sum() if 'T_gas_K' in merged else 0
        print(f"{Path(path).name}: {n_hit}/{len(merged)} rows got results")

        if out_path is None:
            target = Path(path).with_name(Path(path).stem + OUT_SUFFIX + '.csv')
        else:
            target = Path(out_path)
        merged.to_csv(target, index=False)
        print(f"  -> {target}")
        outputs.append(merged)

    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


# =============================================================================
# STAGE 4: PLOTTING HELPERS
# =============================================================================

def _savefig(name):
    if not SAVE_FIGURES:
        return
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"  saved: {path}")


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
    plt.show()
    print()


def line_plot(data, x_col, y_col, series_col, xlabel, ylabel, title, fname,
              yerr_col=None, series_stride=1, series_fmt="{:.2f}",
              series_unit=""):
    """
    Plot y vs x with one line per unique value of series_col, with error bars.
    Replicates are averaged over any remaining grouping columns.
    """
    d = data.dropna(subset=[x_col, y_col])
    if len(d) == 0:
        print(f"  {title}: no data\n")
        return

    series_vals = sorted(d[series_col].unique())[::series_stride]
    colors = plt.cm.viridis(np.linspace(0, 0.92, max(len(series_vals), 1)))

    fig, ax = plt.subplots(figsize=(11, 7))
    for val, color in zip(series_vals, colors):
        sub = d[d[series_col] == val]
        agg_map = {y_col: 'mean'}
        if yerr_col is not None:
            # quadrature mean of the per-condition errors
            agg_map[yerr_col] = lambda e: np.sqrt(np.nansum(np.square(e))) / max(len(e), 1)
        agg = sub.groupby(x_col).agg(agg_map).reset_index().sort_values(x_col)

        yerr = agg[yerr_col] if yerr_col is not None else None
        ax.errorbar(agg[x_col], agg[y_col], yerr=yerr, marker='o', color=color,
                    linewidth=2.2, markersize=6, capsize=4, alpha=0.9,
                    label=f"{series_fmt.format(val)}{series_unit}")

    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.tight_layout()
    _savefig(fname)
    plt.show()
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

    # --- optional frequency isolation ---
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

    # --- contours ---
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

    # --- line plots ---
    if len(at_p):
        line_plot(at_p, 'power', 'N_s_cm3', 'n2_flow',
                  'Power (W)', 'Metastable density (cm$^{-3}$)',
                  f'Metastable Density vs Power ({p_sel} Torr)',
                  'line_Ns_vs_power.png',
                  yerr_col='N_s_cm3_err', series_stride=2, series_unit=' %')

        line_plot(at_p, 'n2_flow', 'T_gas_K', 'power',
                  'N$_2$ (%)', 'Gas temperature (K)',
                  f'Gas Temperature vs N$_2$ ({p_sel} Torr)',
                  'line_Tg_vs_n2.png',
                  yerr_col='T_gas_K_err', series_stride=3,
                  series_fmt="{:.0f}", series_unit=' W')

        line_plot(at_p, 'n2_flow', 'N_s_cm3', 'power',
                  'N$_2$ (%)', 'Metastable density (cm$^{-3}$)',
                  f'Metastable Density vs N$_2$ ({p_sel} Torr)',
                  'line_Ns_vs_n2.png',
                  yerr_col='N_s_cm3_err', series_stride=3,
                  series_fmt="{:.0f}", series_unit=' W')

    return p_sel, w_sel


def make_frequency_sweep_plots(summary, freq_col='freq'):
    """
    Plots specific to an excitation-frequency sweep, where `freq` varies and
    n2_flow / pressure are fixed. Skipped automatically if freq is constant.
    """
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
              yerr_col='T_gas_K_err', series_stride=2,
              series_fmt="{:.0f}", series_unit=' W')

    line_plot(summary, freq_col, 'N_s_cm3', 'power',
              'Excitation frequency (MHz)', 'Metastable density (cm$^{-3}$)',
              'Metastable Density vs Excitation Frequency',
              'line_Ns_vs_freq.png',
              yerr_col='N_s_cm3_err', series_stride=2,
              series_fmt="{:.0f}", series_unit=' W')


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
    plt.show()

    for p in sorted(results):
        r = results[p]
        i_min, i_max = int(np.argmin(r['slopes'])), int(np.argmax(r['slopes']))
        print(f"\nAt {p} Torr:")
        print(f"  min slope {r['slopes'][i_min]:.3e} at N2 = {r['n2'][i_min]:.2f} %")
        print(f"  max slope {r['slopes'][i_max]:.3e} at N2 = {r['n2'][i_max]:.2f} %")
        print(f"  mean      {np.mean(r['slopes']):.3e}")

    return results


def diffusion_diagnostic(summary, target_pressures=(1.0,)):
    """
    Test whether N_s tracks T or T^1.5.

    Ambipolar/neutral diffusion coefficient scales roughly as T^1.5/p, so if
    diffusion loss dominates the metastable balance, N_s should correlate more
    tightly with T^1.5 than with T itself.

    CAUTION: T and T^1.5 are monotonic transforms of one another over a narrow
    range, so their correlations with N_s will be very similar. A larger |r| for
    T^1.5 is weak evidence at best; it does not by itself establish diffusion
    loss. Treat this as a sanity check, not a hypothesis test.
    """
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
    plt.show()

    print(f"\n  r(N_s, T)      = {r_T:.4f}")
    print(f"  r(N_s, T^1.5)  = {r_D:.4f}")
    print("  Difference between these is expected to be small; do not over-read it.")


# =============================================================================
# STAGE 5: LOOKING AT ONE CONDITION
# =============================================================================
 
def pick(df, _verbose=True, **conditions):
    """
    Rows matching a condition, snapping each request to the nearest value that
    actually exists.
 
        pick(summary, n2_flow=2, power=45, freq=2430)
 
    Exact equality is useless on these columns: freq is stored as
    2429.3103448275865, so `summary[summary.freq == 2430]` is always empty.
    Each key is snapped independently, in the order given, and what it snapped
    to is printed so you never silently analyse the wrong point.
    """
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
    """
    Re-open one scope file and plot every sawtooth period's fit.
 
    Use it when a point looks wrong in the summary and you want to see whether
    the spectrum or the Gaussian is at fault:
 
        inspect_point(index, filtered, n2_flow=2, power=45, freq=2430)
 
    Returns the per-period fit dicts. `filtered` is optional; pass it to see
    which periods survived the chi2/nsmallest cut.
    """
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
        f, y, z = frequency[sl], absorbance[sl], i_off[sl]
        mask = np.isfinite(f) & np.isfinite(y)
        if mask.sum() < MIN_POINTS_PER_PERIOD:
            print(f"  period {k}: only {mask.sum()} usable points")
            continue
        try:
            res = analyze_period(f[mask], y[mask], np.std(z, ddof=1))
        except RuntimeError as e:
            print(f"  period {k}: fit failed - {e}")
            continue
        res['period'] = k
        res['kept'] = k in kept
        # native points, centred on the fitted line position, for plotting the
        # fit against the actual data rather than against the interpolated
        # F_GRID copy of it
        f_native = np.sort(f[mask]) / 1e9
        res['f_data'] = f_native - res['x0']
        res['y_data'] = y[mask][np.argsort(f[mask])]
        out.append(res)
        flag = 'KEPT' if (not kept or k in kept) else 'cut'
        print(f"  period {k} [{flag}]: FWHM {res['fwhm']:.3f} +/- "
              f"{res['fwhm_err']:.3f} GHz, area {res['area']:.4f}, "
              f"chi2 {res['Chi^2']:.2f}")
 
    if out:
        _plot_data_vs_fit(out, row, only_period=period)
 
    return out
 
 
def _plot_data_vs_fit(fits, row, only_period=None):
    """
    Raw absorbance points with the fitted Gaussian over them, plus residuals.
 
    The data are the NATIVE samples, not the F_GRID interpolation used for the
    averaged spectra: interpolation smooths exactly the wiggles you are looking
    for when a fit is suspect.
    """
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
    tag = '' if only_period is None else f'_period{only_period}'
    plt.show()
 
 


# =============================================================================
# MAIN
# =============================================================================

# --- stage 0 ---
index = load_run_index()

# --- stage 1 ---
if RUN_RAW_PROCESSING:
    raw = process_index(index)
    if len(raw) == 0:
        raise SystemExit("Aborting: no raw results.")
    raw.to_csv(RAW_CSV, index=False)
    print(f"\nSaved raw results -> {RAW_CSV}")
else:
    print(f"Loading cached raw results from {RAW_CSV}")
    raw = pd.read_csv(RAW_CSV)

# --- stage 2 ---
filtered, summary, diag = filter_and_aggregate(raw, index=index)
if len(summary) == 0:
    raise SystemExit("Aborting: aggregation produced no rows.")
summary.to_csv(SUMMARY_CSV, index=False)
print(f"Saved summary -> {SUMMARY_CSV}")

# --- stage 3 ---
summary = add_physics(summary)
summary.to_csv(PHYSICS_CSV, index=False)
print(f"Saved derived quantities -> {PHYSICS_CSV}")

final_table = build_final_table(summary)
final_table.to_csv(FINAL_CSV, index=False)
print(f"Saved final table -> {FINAL_CSV}")

print("\nSample rows:")
cols = [c for c in ['power', 'pressure', 'n2_flow', 'freq', 'n_fits',
                    'fwhm_mean', 'fwhm_err', 'fwhm_birge',
                    'T_gas_K', 'T_gas_K_err', 'N_s_cm3', 'N_s_cm3_err']
        if c in summary.columns]
print(summary[cols].head(10).to_string(index=False))

# --- stage 3b: back onto the acquisition dataframe ---
df_with_physics = attach_to_dataframe(summary)

# --- stage 4 ---
make_all_plots(summary)
make_frequency_sweep_plots(summary)
slope_analysis(summary, target_pressures=(1.0,))
diffusion_diagnostic(summary, target_pressures=(1.0,))

print(f"\n{'=' * 60}\nDONE\n{'=' * 60}")