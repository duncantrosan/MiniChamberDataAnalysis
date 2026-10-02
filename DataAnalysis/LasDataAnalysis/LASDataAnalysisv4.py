# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 13:24:23 2026

@author: dptro
"""

# -*- coding: utf-8 -*-
"""
Unified LAS analysis pipeline: argon plasma with N2 admixture.

Combines raw oscilloscope processing, Fabry-Perot frequency calibration,
Gaussian fitting with full covariance-based error propagation, physical
conversion to gas temperature and metastable density, and plotting.

Pipeline stages:
    1. process_all_folders()  -> raw per-fit results  (MasterResults_RawData)
    2. filter_and_aggregate() -> per-condition means + errors (MasterResults_Summary)
    3. add_physics()          -> T_gas, N_s with propagated errors
    4. plotting section

@author: dptro
"""

from pathlib import Path
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.interpolate import griddata
from scipy.stats import linregress


# =============================================================================
# CONFIGURATION
# =============================================================================

PARENT_FOLDER = r"G:\My Drive\MiniChamberData\Data\NafisaData\LAS_0.6Torr_2Nitrogen_Freq_Sweep_4"

# Set False to skip raw reprocessing and load cached CSVs instead.
RUN_RAW_PROCESSING = True

RAW_CSV     = 'MasterResults_RawData_FrequencySweep.csv'
SUMMARY_CSV = 'MasterResults_Summary_03_FrequencySweep'
PHYSICS_CSV = 'MasterResults_Summary_with_Tg_Ns_freq_sweep.csv'

# --- Oscilloscope channel mapping -------------------------------------------
# NOTE: the original script defined Signal='CH1', Perot='CH2' but
# FindRelativeFrequency() actually reads CH1 as the Fabry-Perot etalon and CH2
# as the sawtooth ramp. Behaviour preserved; names made explicit here.
FP_CHANNEL    = 'CH1'   # Fabry-Perot etalon fringes
RAMP_CHANNEL  = 'CH2'   # laser sweep sawtooth
DIODE_CHANNEL = 'CH3'   # transmitted intensity
BIAS_VOLTAGE  = 0.0     # replace with a measurement

# --- Filtering ---------------------------------------------------------------
# NSMALLEST_PER_GROUP: keep N lowest-chi2 periods per condition, or None for all.
# WARNING: selecting on goodness-of-fit biases toward high-amplitude periods and
# artificially suppresses the Birge ratio. Prefer CHI2_MAX now that chi2 is on a
# physically meaningful scale.
NSMALLEST_PER_GROUP = 3
CHI2_MAX = None          # e.g. 5.0 to reject genuinely bad fits instead

GROUP_KEYS = ['trial', 'power', 'pressure', 'n2_flow']

# --- Fitting -----------------------------------------------------------------
FSR = 1.5e9              # Fabry-Perot free spectral range, Hz
F_GRID = np.arange(-5, 5, 0.002)   # GHz, plotting grid only (not used for chi2)
MIN_POINTS_PER_PERIOD = 50

# --- Plotting ----------------------------------------------------------------
SAVE_FIGURES = True
FIG_DIR = 'figures'
PLOT_EVERY_N_FITS = 12   # diagnostic spectrum plots; set None to disable


# =============================================================================
# SPECTROSCOPIC CONSTANTS
# =============================================================================
#
#
#
#
# N_s scales as g_lower / (g_upper * A_ki), so these choices move the answer by
# roughly an order of magnitude. Confirm the line identity and pull g and A_ki
# from the NIST ASD before trusting absolute densities.

LAMBDA_0 = 695.7e-9      # m   <-- confirm: 696.5431e-9 for 1s5 -> 2p2?
L_PATH   = 0.04          # m   absorption path length (4 cm) -- SI, not cm
M_AR     = 40.0          # amu
G_LOWER  = 5             # <-- confirm: 5 for 1s5
G_UPPER  = 3             # <-- confirm: 3 for 2p2
A_KI     = 6.4e6         # s^-1  <-- confirm: 6.39e6 for 696.5431 nm

C = 2.9979e8         # m/s
DOPPLER_CONST = 7.16e-7  # Doppler FWHM coefficient, dlambda = K*lambda0*sqrt(T/M)

K_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))   # sigma -> FWHM


# =============================================================================
# FILENAME PATTERNS
# =============================================================================

PATTERN = re.compile(
    r"Scope_TrialNumber_(?P<trial>\d+)_"
    r"(?P<power>[\d.]+)W_"
    r"[\d.]+Torr_"
    r"(?P<n2_flow>[\d.]+)NitrogenFlow_"
    r"(?P<freq>[\d.]+)MHz"
)
FOLDER_PATTERN = re.compile(r"LAS_(?P<pressure>[\d.]+)Torr")


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
# STAGE 1: RAW PROCESSING
# =============================================================================

def build_index(files, directory, label, pressure):
    """Build a dataframe index from CSV filenames."""
    rows = []
    for f in files:
        m = PATTERN.match(f)
        if not m:
            continue
        row = {k: float(v) for k, v in m.groupdict().items()}
        row["trial"] = int(row["trial"])
        row["filename"] = f
        row["path"] = str(Path(directory) / f)
        row["plasma"] = label
        row["pressure"] = float(pressure)
        rows.append(row)
    return pd.DataFrame(rows)


def find_relative_frequency(mean_df, fsr=FSR, min_peaks=3, min_pts=1000,
                            prom=0.1, plot=False):
    """
    Build a relative frequency axis (Hz) from Fabry-Perot fringes.

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
        frequency[sl] = np.interp(time[sl], peak_times, freq_peaks,
                                  left=np.nan, right=np.nan)
        periods.append({"slice": sl, "label": label,
                        "peak_times": peak_times, "peak_freqs": freq_peaks})

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


def process_single_folder(main_folder, folder_label=None):
    """Process one dataset folder into a dataframe of per-period fit results."""
    if folder_label is None:
        folder_label = Path(main_folder).name

    print(f"\n{'=' * 60}\nProcessing folder: {folder_label}\n{'=' * 60}\n")

    data_on_path = os.path.join(main_folder, 'OscopeData_Laser_True')
    data_off_path = os.path.join(main_folder, 'OscopeData_Laser_False')

    folder_match = FOLDER_PATTERN.search(main_folder)
    if not folder_match:
        print(f"WARNING: could not extract pressure from folder name: {main_folder}")
        return pd.DataFrame()
    pressure = folder_match.group('pressure')
    print(f"Pressure from folder name: {pressure} Torr")

    csv_on = [f for f in os.listdir(data_on_path) if f.endswith('.csv')]
    csv_off = [f for f in os.listdir(data_off_path) if f.endswith('.csv')]
    print(f"Found {len(csv_on)} / {len(csv_off)} files (laser True / False)")

    idx_on = build_index(csv_on, data_on_path, "on", pressure)
    idx_off = build_index(csv_off, data_off_path, "off", pressure)

    pairs = idx_on.merge(idx_off, on=["trial", "power", "n2_flow", "freq", "pressure"],
                         suffixes=("_on", "_off"), how="inner")
    print(f"Paired {len(pairs)} condition sets\n")

    only_off = idx_off.merge(idx_on, on=["trial", "power", "n2_flow"],
                             how="left", indicator=True, suffixes=("", "_x"))
    unmatched = only_off[only_off["_merge"] == "left_only"][["trial", "power", "n2_flow"]]
    if len(unmatched) > 0:
        print(f"WARNING: {len(unmatched)} unmatched files:\n{unmatched}\n")

    # --- reference (plasma-off) traces ---
    ref_name = 'Plasma Off Measuremnt.csv'
    data_true = pd.read_csv(os.path.join(data_on_path, ref_name))
    data_false = pd.read_csv(os.path.join(data_off_path, ref_name))

    chans = [FP_CHANNEL, RAMP_CHANNEL, DIODE_CHANNEL]
    on_mean = data_true.groupby('time')[chans].mean()
    off_mean = data_false.groupby('time')[chans].mean()

    i_ref = np.asarray(on_mean[DIODE_CHANNEL] - off_mean[DIODE_CHANNEL])

    rows = []
    n_plotted = 0

    for row_idx, row in enumerate(pairs.itertuples(), start=1):
        print(f"  [{row_idx}/{len(pairs)}] {row.filename_on}")

        d_on = pd.read_csv(os.path.join(data_on_path, row.filename_on))
        d_off = pd.read_csv(os.path.join(data_off_path, row.filename_off))

        mean_on = d_on.groupby('time')[chans].mean()
        mean_off = d_off.groupby('time')[chans].mean()

        try:
            frequency, periods = find_relative_frequency(mean_on)
        except RuntimeError as e:
            print(f"    WARNING: frequency calibration failed: {e}")
            continue

        i_m = (mean_on[DIODE_CHANNEL] - mean_off[DIODE_CHANNEL] + BIAS_VOLTAGE).to_numpy()
        with np.errstate(divide='ignore', invalid='ignore'):
            absorbance = np.log((i_ref + BIAS_VOLTAGE) / i_m)
        plasma_off = np.asarray(mean_off[DIODE_CHANNEL])

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

            rows.append({
                "folder_label": folder_label,
                "trial": row.trial,
                "power": row.power,
                "pressure": row.pressure,
                "n2_flow": row.n2_flow,
                "period": k,
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

            if PLOT_EVERY_N_FITS and len(rows) % PLOT_EVERY_N_FITS == 0:
                _plot_fit_diagnostic(res, folder_label, row.filename_on)
                n_plotted += 1

    results_df = pd.DataFrame(rows)
    print(f"\nFolder {folder_label}: {len(results_df)} raw fits "
          f"({n_plotted} diagnostic plots)\n")
    return results_df


def _plot_fit_diagnostic(res, folder_label, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(res['spec_x'], res['spec'], label='data', linewidth=2)
    plt.plot(res['spec_x'], res['Fit'], label='fit', linewidth=2)
    plt.xlabel('Relative frequency (GHz)')
    plt.ylabel('Absorbance')
    plt.title(f"{folder_label}\n{filename}", fontsize=9)
    plt.text(0.05, 0.95,
             f"$\\chi^2_\\nu$ = {res['Chi^2']:.2f}\n"
             f"FWHM = {res['fwhm']:.3f} $\\pm$ {res['fwhm_err']:.3f} GHz",
             transform=plt.gca().transAxes, va='top', ha='left',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.75))
    plt.legend()
    plt.tight_layout()
    plt.show()


def process_all_folders(parent_folder=PARENT_FOLDER):
    """Discover and process every subfolder; return concatenated raw results."""
    if not os.path.exists(parent_folder):
        print(f"ERROR: parent folder not found: {parent_folder}")
        return pd.DataFrame()

    subfolders = sorted(
        os.path.join(parent_folder, d) for d in os.listdir(parent_folder)
        if os.path.isdir(os.path.join(parent_folder, d))
    )
    if not subfolders:
        print(f"ERROR: no subfolders in {parent_folder}")
        return pd.DataFrame()

    print(f"Found {len(subfolders)} subfolders\n")

    all_results, failures = [], []
    for folder in subfolders:
        label = Path(folder).name
        try:
            df = process_single_folder(folder, label)
            if len(df) > 0:
                all_results.append(df)
        except Exception as e:
            print(f"\nERROR processing {label}: {e}")
            import traceback
            traceback.print_exc()
            failures.append(label)

    if failures:
        print(f"\nWARNING: {len(failures)} folders failed: {failures}")

    if not all_results:
        print("\nNo results generated from any folder.")
        return pd.DataFrame()

    master = pd.concat(all_results, ignore_index=True)
    print(f"\n{'=' * 60}\nRAW RESULTS\n{'=' * 60}")
    print(f"Total fits: {len(master)}")
    print(f"Folders:    {list(master['folder_label'].unique())}")
    print(f"Powers:     {sorted(master['power'].unique())}")
    print(f"Pressures:  {sorted(master['pressure'].unique())}")
    print(f"N2 flows:   {sorted(master['n2_flow'].unique())}")
    return master


# =============================================================================
# STAGE 2: FILTERING AND AGGREGATION
# =============================================================================

def filter_and_aggregate(results_df, nsmallest=NSMALLEST_PER_GROUP,
                         chi2_max=CHI2_MAX, group_keys=GROUP_KEYS):
    """
    Filter per-fit results and aggregate to one row per condition, carrying
    Birge-combined uncertainties on both FWHM and area.
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
        # sort_values + head is equivalent to per-group nsmallest but keeps every
        # column. groupby.apply(..., include_groups=False) silently DROPS the
        # grouping columns from the result, which is what produced the downstream
        # KeyError: 'trial'.
        before = len(stage)
        stage = (stage.sort_values('Chi^2')
                      .groupby(['trial', 'power', 'n2_flow'], sort=False)
                      .head(nsmallest)
                      .reset_index(drop=True))
        diagnostics["stages"]["nsmallest"] = {
            "nsmallest": nsmallest, "removed": before - len(stage), "remaining": len(stage)}
        print(f"After nsmallest({nsmallest}):        removed {before - len(stage)}, "
              f"remaining {len(stage)}")
    else:
        print("Skipped nsmallest filter (keeping all)")

    filtered_df = stage.copy()
    print(f"\nN2 flows present: {sorted(filtered_df['n2_flow'].unique())}")
    print(f"chi2 distribution:\n{filtered_df['Chi^2'].describe()}\n")

    # --- aggregate ---
    print(f"{'=' * 60}\nAGGREGATION\n{'=' * 60}\n")

    records = []
    for keys, g in filtered_df.groupby(group_keys, sort=True):
        rec = dict(zip(group_keys, keys if isinstance(keys, tuple) else (keys,)))

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

    print(f"Aggregated into {len(summary)} condition groups")
    for col in ('fwhm_birge', 'area_birge'):
        vals = summary[col].dropna()
        if len(vals):
            print(f"  {col}: median {vals.median():.2f}, "
                  f"{(vals > 2).sum()}/{len(vals)} groups above 2")
    print("  (Birge >> 1 means replicate scatter exceeds the fit errors:\n"
          "   an unmodelled systematic is present in those conditions.)")

    return filtered_df, summary, diagnostics


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
                 title, fname, cmap='viridis', fmt=None, method='cubic',vmin=None,vmax = None):
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

def make_all_plots(summary, target_pressure=1.0, target_power=40.0):
    print(f"\n{'=' * 60}\nPLOTS\n{'=' * 60}\n")

    pressures = summary['pressure'].unique()
    powers = summary['power'].unique()
    p_sel = nearest(pressures, target_pressure)
    w_sel = nearest(powers, target_power)
    print(f"Fixed pressure slice: {p_sel} Torr")
    print(f"Fixed power slice:    {w_sel} W\n")

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
                     'contour_Tg_n2_power.png', cmap='hot',vmin = 300,vmax = 1000) # cf = ax.contourf(Xi, Yi, Zi,


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


def slope_analysis(summary, target_pressures=(1.0,)):
    """dN_s/dPower as a function of N2 admixture, at each target pressure."""
    print(f"\n{'=' * 60}\nSLOPE ANALYSIS: dN_s/dP_rf vs N2\n{'=' * 60}")

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']
    fig, ax = plt.subplots(figsize=(12, 7))
    results = {}

    for target, color, marker in zip(target_pressures, colors, markers):
        p_sel = nearest(summary['pressure'].unique(), target)
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
# MAIN
# =============================================================================


# --- stage 1 ---
if RUN_RAW_PROCESSING:
    raw = process_all_folders()
    if len(raw) == 0:
        print("Aborting: no raw results.")
    raw.to_csv(RAW_CSV, index=False)
    print(f"\nSaved raw results -> {RAW_CSV}")
else:
    print(f"Loading cached raw results from {RAW_CSV}")
    raw = pd.read_csv(RAW_CSV)

# --- stage 2 ---
filtered, summary, diag = filter_and_aggregate(raw)
if len(summary) == 0:
    print("Aborting: aggregation produced no rows.")
summary.to_csv(SUMMARY_CSV, index=False)
print(f"Saved summary -> {SUMMARY_CSV}")

# --- stage 3 ---
summary = add_physics(summary)
#summary.to_csv(PHYSICS_CSV, index=False)
print(f"Saved derived quantities -> {PHYSICS_CSV}")

print("\nSample rows:")
cols = ['power', 'pressure', 'n2_flow', 'n_fits',
        'fwhm_mean', 'fwhm_err', 'fwhm_birge',
        'T_gas_K', 'T_gas_K_err', 'N_s_cm3', 'N_s_cm3_err']
print(summary[cols].head(10).to_string(index=False))

# --- stage 4 ---
make_all_plots(summary)
slope_analysis(summary, target_pressures=(1.0,))
diffusion_diagnostic(summary, target_pressures=(1.0,))

print(f"\n{'=' * 60}\nDONE\n{'=' * 60}")




