# -*- coding: utf-8 -*-
"""
MASTER LIST: every measurement from every run, combined per condition.

Output/Master/ holds three tables:

  Master_AllMeasurements.csv  One row per measurement (a run_id + timestamp),
                              with every column of every file added for it.
  Master_Combined.csv         One row per condition, i.e. the same set-points
                              in KEY_COLUMNS (power, pressure, N2 %, total
                              flow, frequency). Repeat measurements are combined
                              and their errors updated. Every column.
  Master_Summary.csv          Master_Combined cut down to SUMMARY_COLUMNS:
                              set-points, measured inputs, gamma, T_gas, N_s.
                              NaN where a diagnostic hasn't been run yet.

What can go in
--------------
Any CSV with ONE ROW PER MEASUREMENT and the columns run_id, timestamp, power,
pressure, Nitrogen_Percent and frequency: the acquisition dataframes
(LAS_DataFrame_*.csv, Reflection_sweep_*.csv) and analysis outputs that keep
their input's rows (e.g. *_LAS.csv from LASAnalysisv6.py). Other files are
skipped with a message saying why.

A measurement that is already in the master is UPDATED, never added twice: the
columns the new file has replace the old values, other columns are kept. So
  - re-running an analysis and adding its output again replaces the old result
  - adding a raw dataframe and later its LAS output fills in the LAS columns of
    the same rows instead of creating new ones

How repeat measurements of one condition are combined
-----------------------------------------------------
  X with an error column (X_err or X_error)
      weighted mean. Error = the larger of the internal error 1/sqrt(sum 1/u^2)
      and the scatter between the measurements (Birge ratio, as in
      LASAnalysisv6). Errors that are zero or ~1e-16 (identical repeat readings)
      can't be used as weights; such groups fall back to mean and std/sqrt(n).
  X with X_stat and X_sys (T_gas_K, N_s_cm3)
      X_stat is combined as above. X_sys is a common scale error that doesn't
      average down, so it keeps its relative size. X_err = both in quadrature.
  X without an error
      plain mean, plus X_std for the measured inputs listed in MEASURED_INPUTS
  text
      the distinct values, ';'-separated
  X_n = number of measurements that had a value for X

Usage
-----
Spyder: set ACTION below and run (F5).
Terminal:
    python MasterList.py add FILE [FILE ...]
    python MasterList.py rebuild [FOLDER] [MASTER_FOLDER]
    python MasterList.py remove RUN_ID [RUN_ID ...]
LASAnalysisv6.py adds its output itself after every run (ADD_TO_MASTER).
The previous Master_AllMeasurements.csv is kept in Output/Master/backups.

@author: dptro
"""

import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    _HERE = Path(__file__).resolve().parent
except NameError:                        # pasted into a console
    _HERE = Path.cwd()
sys.path.insert(0, str(_HERE.parent))    # DataAnalysis/, for OutputPaths
from OutputPaths import OUTPUT_ROOT, MASTER_DIR

# =============================================================================
# CONFIGURATION
# =============================================================================

# What running this file does:
#   'add'      add FILES_TO_ADD to the master. With an empty list it just
#              rewrites the combined tables (e.g. after editing SUMMARY_COLUMNS).
#   'rebuild'  start over from every per-measurement CSV in REBUILD_FOLDER and
#              its sub-folders. Only what is in that folder ends up in the master.
#   'remove'   drop every measurement of the runs in RUN_IDS_TO_REMOVE.
ACTION = 'add'
FILES_TO_ADD = []
REBUILD_FOLDER = OUTPUT_ROOT
REBUILD_MASTER_FOLDER = None   # None: Output/Master. A folder: build a separate
                               # master there instead, leaving the main one alone.
RUN_IDS_TO_REMOVE = []

# Set-points that make two measurements "the same condition", and how many
# decimals each is rounded to before comparing.
KEY_COLUMNS = {
    'power':            2,
    'pressure':         3,
    'Nitrogen_Percent': 3,
    'Total_Flow_sccm':  0,
    'frequency':        2,
}

# The dataframes don't record the flow set-point, so Total_Flow_sccm comes from
# a set-point column if a file has one, otherwise from the measured Ar + N2 flow
# (within 0.05 sccm of the set-point in all data so far), rounded as above.
FLOW_SETPOINT_COLUMNS = ['Mass_Flow_Rate']
FLOW_MEASURED_COLUMNS = ['Argon_Gas_Flow', 'Nitrogen_Gas_Flow']

# A file needs these columns to go in.
REQUIRED_COLUMNS = ['run_id', 'timestamp', 'power', 'pressure',
                    'Nitrogen_Percent', 'frequency']

# Measured inputs: also get X_std, the spread between repeat measurements.
MEASURED_INPUTS = ['Measured Pressure', 'Argon_Gas_Flow', 'Nitrogen_Gas_Flow',
                   'N2_Percent_Measured', 'Temperaure3']

# Per-measurement file paths and notes: kept in Master_AllMeasurements only.
NOT_COMBINED = ['ScopeFileLocation', 'ScopeFileLocation_Off_Loop', 'row_uid',
                'OES_File_Location_OceanOptis', 'LAS_note']

# Master_Summary.csv: output column -> Master_Combined column.
# Add lines here as diagnostics are added; missing columns come out as NaN.
SUMMARY_COLUMNS = {
    # set-points (the condition)
    'Power_Input_W':              'power',
    'Pressure_Input_Torr':        'pressure',
    'N2_Percent_Input':           'Nitrogen_Percent',
    'Flow_Input_sccm':            'Total_Flow_sccm',
    'Frequency_Input_MHz':        'frequency',
    'n_measurements':             'n_measurements',
    # measured inputs
    'Power_Measured_W':           'delivered_power',
    'Power_Measured_W_err':       'delivered_power_error',
    'Forward_Power_W':            'forward_power',
    'Forward_Power_W_err':        'forward_power_error',
    'Reflected_Power_W':          'reverse_power',
    'Reflected_Power_W_err':      'reverse_power_error',
    'Pressure_Measured_Torr':     'Measured Pressure',
    'Pressure_Measured_Torr_std': 'Measured Pressure_std',
    'Ar_Flow_Measured_sccm':      'Argon_Gas_Flow',
    'N2_Flow_Measured_sccm':      'Nitrogen_Gas_Flow',
    'N2_Percent_Measured':        'N2_Percent_Measured',
    'N2_Percent_Measured_std':    'N2_Percent_Measured_std',
    'Temperature3':               'Temperaure3',
    # reflection coefficient
    'Gamma':                      'gamma',
    'Gamma_err':                  'gamma_error',
    'Gamma_n':                    'gamma_n',
    # LAS
    'T_gas_K':                    'T_gas_K',
    'T_gas_K_stat':               'T_gas_K_stat',
    'T_gas_K_sys':                'T_gas_K_sys',
    'T_gas_K_err':                'T_gas_K_err',
    'T_gas_K_n':                  'T_gas_K_n',
    'N_s_cm3':                    'N_s_cm3',
    'N_s_cm3_stat':               'N_s_cm3_stat',
    'N_s_cm3_sys':                'N_s_cm3_sys',
    'N_s_cm3_err':                'N_s_cm3_err',
    'N_s_cm3_n':                  'N_s_cm3_n',
    # where it came from
    'run_ids':                    'run_ids',
    'first_measured':             'first_timestamp',
    'last_measured':              'last_timestamp',
}

N_BACKUPS = 10      # copies of Master_AllMeasurements.csv kept in backups/


# =============================================================================
# READING AND MERGING MEASUREMENTS
# =============================================================================

_ID = '_measurement'            # run_id|timestamp, internal only
_SRC = 'source_files'
_ERR_SUFFIXES = ('_err', '_error')
_MISSING_TEXT = {'', '?', 'nan', 'NaN', 'None', 'none', 'NA', '<NA>', 'NaT'}


def _paths(master_dir=None):
    d = Path(master_dir) if master_dir else MASTER_DIR
    return {'dir': d,
            'ledger': d / 'Master_AllMeasurements.csv',
            'combined': d / 'Master_Combined.csv',
            'summary': d / 'Master_Summary.csv',
            'backups': d / 'backups'}


def _measurement_id(df):
    """run_id|timestamp, with the timestamp always written the same way."""
    raw = df['timestamp'].fillna('').astype(str)
    t = pd.to_datetime(raw, errors='coerce')
    ts = t.dt.strftime('%Y-%m-%d %H:%M:%S.%f').where(t.notna(), raw)
    return df['run_id'].fillna('').astype(str) + '|' + ts


def read_measurements(path):
    """A file as a table of measurements, or (None, reason) if it can't go in."""
    path = Path(path)
    try:
        header = pd.read_csv(path, nrows=0).columns
    except Exception as e:
        return None, f'not readable as CSV ({type(e).__name__})'
    missing = [c for c in REQUIRED_COLUMNS if c not in header]
    if missing:
        return None, 'no ' + ', '.join(missing) + ' column'

    df = pd.read_csv(path, dtype={'run_id': str}, low_memory=False)
    df = df.loc[:, ~df.columns.astype(str).str.startswith('Unnamed')]
    df[_ID] = _measurement_id(df)
    n_dup = int(df[_ID].duplicated().sum())
    if n_dup:
        return None, (f'{n_dup} rows repeat a run_id + timestamp '
                      f'(not one row per measurement)')
    if _SRC not in df.columns:          # another master keeps its own sources
        df[_SRC] = path.name
    return df, None


def _join(values):
    """Distinct non-empty values as 'a;b;c', in order of appearance."""
    out = []
    for v in values:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        for part in str(v).split(';'):
            part = part.strip()
            if part not in _MISSING_TEXT:
                out.append(part)
    return ';'.join(dict.fromkeys(out))


def _upsert(ledger, new):
    """Merge one file's measurements into the ledger -> (ledger, n_new, n_updated)."""
    if ledger.empty:
        return new.reset_index(drop=True), len(new), 0

    L = ledger.set_index(_ID)
    N = new.set_index(_ID)
    both = N.index.intersection(L.index)

    parts = [L.drop(index=both)]
    if len(both):
        old = L.loc[both]
        upd = pd.concat([old.drop(columns=[c for c in N.columns if c in old.columns]),
                         N.loc[both]], axis=1)
        upd[_SRC] = [_join([a, b]) for a, b in zip(old[_SRC], N.loc[both, _SRC])]
        parts.append(upd)
    parts.append(N.drop(index=both))

    order = list(dict.fromkeys([*L.columns, *N.columns]))
    merged = pd.concat([p for p in parts if len(p)])[order]
    return merged.reset_index(), len(N) - len(both), len(both)


def _find_files(folder, skip_dir):
    """
    Every CSV under folder (or folder itself if it is a file), oldest first.
    Skips master tables, their backups and SimpleMerge outputs (by name, and
    anything inside skip_dir or Output/Master): they only hold copies of other
    files' measurements, and an old copy would bring back removed runs. To
    merge in another master on purpose, use add_to_master on its file.
    """
    folder = Path(folder)
    if folder.is_file():
        return [folder]
    if not folder.is_dir():
        raise FileNotFoundError(f"No such folder: {folder}")
    skip = {Path(skip_dir).resolve(), MASTER_DIR.resolve()}
    files = [f for f in folder.rglob('*.csv')
             if not f.name.startswith(('Master_', 'SimpleMerge_'))
             and not skip & set(f.resolve().parents)]
    # oldest first, so where two files hold the same measurement the newer wins
    return sorted(files, key=lambda f: (f.stat().st_mtime, f.name))


def _add_files(ledger, files):
    report = []
    for f in files:
        new, why = read_measurements(f)
        if new is None:
            report.append((Path(f).name, None, why))
            continue
        ledger, n_new, n_upd = _upsert(ledger, new)
        report.append((Path(f).name, (n_new, n_upd), None))
    return ledger, report


def _print_report(report):
    skipped = {}
    for name, counts, why in report:
        if counts is None:
            skipped.setdefault(why, []).append(name)
        else:
            print(f"  + {name}: {counts[0]} new, {counts[1]} updated")
    for why, names in skipped.items():
        shown = ', '.join(names[:3]) + (' ...' if len(names) > 3 else '')
        print(f"  - skipped {len(names)} file(s), {why}: {shown}")


# =============================================================================
# COMBINING PER CONDITION
# =============================================================================

def _numeric(s):
    """s as floats if every non-empty entry is a number, else None (text)."""
    if pd.api.types.is_bool_dtype(s):
        return None
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    txt = s.dropna().astype(str).str.strip()
    num = pd.to_numeric(txt, errors='coerce')
    if (num.isna() & ~txt.isin(_MISSING_TEXT)).any():
        return None
    return pd.to_numeric(s, errors='coerce').astype(float)


def _birge(values, errors):
    """Weighted mean and Birge-inflated error of repeat measurements -> (mean, err, n)."""
    v = np.asarray(values, dtype=float)
    e = np.asarray(errors, dtype=float)
    ok = np.isfinite(v)
    v, e = v[ok], e[ok]
    n = len(v)
    if n == 0:
        return np.nan, np.nan, 0
    if n == 1:
        return v[0], (e[0] if np.isfinite(e[0]) else np.nan), 1
    usable = np.isfinite(e) & (e > 1e-6 * np.abs(v) + 1e-12)
    if not usable.all():
        return v.mean(), v.std(ddof=1) / np.sqrt(n), n
    w = 1.0 / e ** 2
    W = w.sum()
    mean = (w * v).sum() / W
    err_int = 1.0 / np.sqrt(W)
    err_ext = np.sqrt((w * (v - mean) ** 2).sum() / ((n - 1) * W))
    return mean, max(err_int, err_ext), n


def _with_derived(ledger):
    """Ledger plus Total_Flow_sccm, N2_Percent_Measured and rounded key columns."""
    df = ledger.copy()

    def num(col):
        return (pd.to_numeric(df[col], errors='coerce') if col in df.columns
                else pd.Series(np.nan, index=df.index))

    measured = sum(num(c) for c in FLOW_MEASURED_COLUMNS)
    setpoint = pd.Series(np.nan, index=df.index)
    for c in FLOW_SETPOINT_COLUMNS:
        setpoint = setpoint.fillna(num(c))
    df['Total_Flow_sccm'] = setpoint.fillna(measured)

    n2, ar = num('Nitrogen_Gas_Flow'), num('Argon_Gas_Flow')
    n2_pct = 100.0 * n2 / (ar + n2)
    if 'N2_Percent_Measured' in df.columns:
        df['N2_Percent_Measured'] = num('N2_Percent_Measured').fillna(n2_pct)
    else:
        at = (df.columns.get_loc('Nitrogen_Gas_Flow') + 1
              if 'Nitrogen_Gas_Flow' in df.columns else len(df.columns))
        df.insert(at, 'N2_Percent_Measured', n2_pct)

    for k, nd in KEY_COLUMNS.items():
        df[k] = num(k).round(nd)
    return df


def _plan(df, skip):
    """How each column is combined: [(column, kind, partner columns), ...]."""
    cols = [c for c in df.columns if c not in skip]
    numeric = {}
    for c in cols:
        s = _numeric(df[c])
        if s is not None:
            numeric[c] = s

    partners, used = {}, set()
    for c in numeric:
        if c.endswith(('_stat', '_sys') + _ERR_SUFFIXES):
            continue
        stat, sys_ = f'{c}_stat', f'{c}_sys'
        if stat in numeric and sys_ in numeric:
            err = f'{c}_err' if f'{c}_err' in numeric else None
            partners[c] = ('statsys', stat, sys_, err)
            used |= {stat, sys_, err} - {None}
            continue
        err = next((f'{c}{s}' for s in _ERR_SUFFIXES if f'{c}{s}' in numeric), None)
        partners[c] = ('err', err) if err else ('plain',)
        if err:
            used.add(err)

    plan = []
    for c in cols:
        if c in used:
            continue
        if c in partners:
            plan.append((c,) + partners[c])
        elif c in numeric:
            plan.append((c, 'plain'))
        else:
            plan.append((c, 'text'))
    return plan, numeric


def combine(ledger):
    """One row per condition (KEY_COLUMNS) from the table of measurements."""
    df = _with_derived(ledger).reset_index(drop=True)
    keys = list(KEY_COLUMNS)
    skip = set(keys) | {_ID, _SRC, 'run_id', 'timestamp'} | set(NOT_COMBINED)
    plan, numeric = _plan(df, skip)

    label = pd.Series('', index=df.index)
    for k in keys:
        label = label + '|' + df[k].map(lambda v: 'nan' if pd.isna(v) else f'{v:.10g}')
    groups = {}
    for i, g in enumerate(label):
        groups.setdefault(g, []).append(i)

    vals = {c: s.to_numpy() for c, s in numeric.items()}
    text = {c: df[c].to_numpy(dtype=object) for c, *_ in plan if c not in numeric}
    keyvals = df[keys].to_numpy(dtype=float)
    run_ids = df['run_id'].to_numpy(dtype=object)
    stamps = _measurement_id(df).str.split('|', n=1).str[1].to_numpy(dtype=object)
    sources = df[_SRC].to_numpy(dtype=object)

    columns = keys + ['n_measurements', 'run_ids', 'first_timestamp', 'last_timestamp']
    for c, kind, *p in plan:
        if kind == 'statsys':
            columns += [c, p[0], p[1], f'{c}_err', f'{c}_n']
        elif kind == 'err':
            columns += [c, p[0], f'{c}_n']
        elif kind == 'plain' and c in MEASURED_INPUTS:
            columns += [c, f'{c}_std']
        else:
            columns.append(c)
    columns.append(_SRC)

    rows = []
    for pos in groups.values():
        pos = np.asarray(pos)
        st = sorted(s for s in stamps[pos] if s)
        rec = dict(zip(keys, keyvals[pos[0]]))
        rec.update(n_measurements=len(pos), run_ids=_join(run_ids[pos]),
                   first_timestamp=st[0] if st else '',
                   last_timestamp=st[-1] if st else '')
        for c, kind, *p in plan:
            if kind == 'text':
                rec[c] = _join(text[c][pos])
            elif kind == 'plain':
                v = vals[c][pos]
                v = v[np.isfinite(v)]
                rec[c] = v.mean() if len(v) else np.nan
                if c in MEASURED_INPUTS:
                    rec[f'{c}_std'] = v.std(ddof=1) if len(v) > 1 else np.nan
            elif kind == 'err':
                rec[c], rec[p[0]], rec[f'{c}_n'] = _birge(vals[c][pos], vals[p[0]][pos])
            else:                                   # statsys
                stat_c, sys_c, err_c = p
                v, stat, sys_ = vals[c][pos], vals[stat_c][pos], vals[sys_c][pos]
                if err_c:       # older outputs with only a total error
                    stat = np.where(np.isfinite(stat), stat, vals[err_c][pos])
                mean, stat_err, n = _birge(v, stat)
                with np.errstate(divide='ignore', invalid='ignore'):
                    rel = sys_ / np.abs(v)
                rel = rel[np.isfinite(rel)]
                sys_err = np.median(rel) * abs(mean) if len(rel) else np.nan
                rec[c], rec[stat_c], rec[sys_c] = mean, stat_err, sys_err
                rec[f'{c}_err'] = (np.hypot(stat_err, sys_err)
                                   if np.isfinite(sys_err) else stat_err)
                rec[f'{c}_n'] = n
        rec[_SRC] = _join(sources[pos])
        rows.append(rec)

    out = pd.DataFrame(rows, columns=columns)
    return out.sort_values(keys, na_position='last', kind='stable').reset_index(drop=True)


def make_summary(combined):
    """Master_Combined cut down to SUMMARY_COLUMNS (NaN for missing ones)."""
    return pd.DataFrame({out: (combined[src] if src in combined.columns else np.nan)
                         for out, src in SUMMARY_COLUMNS.items()},
                        index=combined.index)


# =============================================================================
# SAVING AND THE THREE ACTIONS
# =============================================================================

def load_master(master_dir=None):
    """Master_AllMeasurements.csv (empty table if there is no master yet)."""
    p = _paths(master_dir)
    if not p['ledger'].exists():
        return pd.DataFrame()
    df = pd.read_csv(p['ledger'], dtype={'run_id': str}, low_memory=False)
    df[_ID] = _measurement_id(df)
    return df


def _save(ledger, master_dir=None, backup=True):
    p = _paths(master_dir)
    if ledger.empty:
        print("\nThe master list is empty: nothing written.")
        return None
    p['dir'].mkdir(parents=True, exist_ok=True)
    if backup and p['ledger'].exists():
        p['backups'].mkdir(exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        shutil.copy2(p['ledger'], p['backups'] / f'Master_AllMeasurements_{stamp}.csv')
        for old in sorted(p['backups'].glob('Master_AllMeasurements_*.csv'))[:-N_BACKUPS]:
            old.unlink()

    when = pd.to_datetime(ledger['timestamp'], errors='coerce')
    ledger = ledger.iloc[np.argsort(when.to_numpy(), kind='stable')]
    ledger.drop(columns=_ID).to_csv(p['ledger'], index=False)
    combined = combine(ledger)
    combined.to_csv(p['combined'], index=False)
    summary = make_summary(combined)
    summary.to_csv(p['summary'], index=False)

    print(f"\nMaster list: {len(ledger)} measurements, {len(combined)} conditions, "
          f"{ledger['run_id'].nunique()} runs")
    for k in ('ledger', 'combined', 'summary'):
        print(f"  {p[k]}")
    return ledger.drop(columns=_ID), combined, summary


def add_to_master(files, master_dir=None):
    """Add per-measurement CSV files to the master and rewrite its tables."""
    files = [files] if isinstance(files, (str, Path)) else list(files)
    print(f"\n{'=' * 60}\nMASTER LIST: adding {len(files)} file(s)\n{'=' * 60}")
    ledger, report = _add_files(load_master(master_dir), files)
    _print_report(report)
    changed = any(counts is not None for _, counts, _ in report)
    return _save(ledger, master_dir, backup=changed)


def rebuild_master(folder=REBUILD_FOLDER, master_dir=None):
    """Build the master from scratch from every per-measurement CSV under folder."""
    p = _paths(master_dir)
    files = _find_files(folder, skip_dir=p['dir'])
    print(f"\n{'=' * 60}\nMASTER LIST: rebuilding from {len(files)} CSV files in\n"
          f"  {folder}\n{'=' * 60}")
    ledger, report = _add_files(pd.DataFrame(), files)
    _print_report(report)
    return _save(ledger, master_dir)


def remove_from_master(run_ids, master_dir=None):
    """Drop every measurement of the given run(s)."""
    run_ids = {str(r) for r in ([run_ids] if isinstance(run_ids, str) else run_ids)}
    ledger = load_master(master_dir)
    if ledger.empty:
        print("The master list is empty.")
        return None
    gone = ledger['run_id'].astype(str).isin(run_ids)
    print(f"\nRemoving {int(gone.sum())} measurements of run(s) {sorted(run_ids)}")
    if not gone.any():
        return None
    return _save(ledger[~gone], master_dir)


if __name__ == '__main__':
    args = sys.argv[1:]
    action, rest = (args[0], args[1:]) if args else (ACTION, [])
    if action == 'add':
        result = add_to_master(rest or FILES_TO_ADD)
    elif action == 'rebuild':
        result = rebuild_master(rest[0] if rest else REBUILD_FOLDER,
                                rest[1] if len(rest) > 1 else REBUILD_MASTER_FOLDER)
    elif action == 'remove':
        result = remove_from_master(rest or RUN_IDS_TO_REMOVE)
    else:
        raise SystemExit(f"Unknown action {action!r}: use add, rebuild or remove.")

    # plain variables for Spyder's variable explorer
    if result is not None:
        measurements, combined, master_summary = result
