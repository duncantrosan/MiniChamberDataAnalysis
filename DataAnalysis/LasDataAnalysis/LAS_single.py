# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 17:19:16 2026

@author: dptro

Single-measurement LAS analysis: four scope files in, T_gas and N_s out.

Same physics and error analysis as the full pipeline, with the dataframe
indexing, run pairing, multi-file aggregation and sweep plotting removed.

Usage
-----
    python las_single.py on_laser.csv on_dark.csv off_laser.csv off_dark.csv

or edit the four paths in the CONFIG block and run with no arguments.

The four files are the two-by-two combination of plasma and laser state:

    path_on   plasma ON,  laser ON    the measurement
    path_off  plasma ON,  laser OFF   plasma emission background
    ref_on    plasma OFF, laser ON    laser reference, I_0
    ref_off   plasma OFF, laser OFF   detector offset

Each must be a CSV with columns: time, CH1, CH2, CH3.
"""

import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.interpolate import PchipInterpolator


# =============================================================================
# CONFIG
# =============================================================================

PATH_ON  = r"G:\My Drive\MiniChamberData\Data\NafisaData\LAS_1.5Torr_Freq_2420\OscopeData_Laser_True\Scope_TrialNumber_0_75.00W_1.50Torr_1.00NitrogenFlow_2420.00MHz.csv"
PATH_OFF = r'G:\My Drive\MiniChamberData\Data\NafisaData\LAS_1.5Torr_Freq_2420\OscopeData_Laser_False\Scope_TrialNumber_0_75.00W_1.50Torr_1.00NitrogenFlow_2420.00MHz.csv'
REF_ON   = r"G:\My Drive\MiniChamberData\Data\NafisaData\LAS_1.5Torr_Freq_2420\OscopeData_Laser_True\Plasma_Off_Measuremnt_1.0.csv"
REF_OFF  = r"G:\My Drive\MiniChamberData\Data\NafisaData\LAS_1.5Torr_Freq_2420\OscopeData_Laser_False\Plasma_Off_Measuremnt_1.0.csv"

FP_CHANNEL    = 'CH1'     # Fabry-Perot etalon fringes
RAMP_CHANNEL  = 'CH2'     # laser sweep sawtooth
DIODE_CHANNEL = 'CH3'     # transmitted intensity
BIAS_VOLTAGE  = 0.0

FSR = 1.5e9               # etalon free spectral range, Hz
MIN_POINTS_PER_PERIOD = 50
MIN_PEAKS_PER_PERIOD = 3

# Background polynomial order under the Gaussian. 1 = b + m*nu (the original).
# 2 adds curvature, which is the thing to try if residual etalon fringing or
# an emission-subtraction residual is tilting/bowing the baseline. Compare the
# fitted area between orders: if it moves by more than its own uncertainty,
# the lower order was biasing you.
BASELINE_ORDER = 1

CHI2_MAX = None           # e.g. 5.0 to reject bad periods; None keeps all

# Leading/trailing partial sawtooth periods (scope record cut mid-sweep) often
# miss most of the absorption line, yet fit with small errors and drag the
# Birge-weighted area down. Excluded unless this is True.
INCLUDE_PARTIAL_PERIODS = False
PLOT = True

# Width of the smoothing window used to detect correlated (non-white) residual
# structure. Should be well below the number of points in a period and well
# above the white-noise correlation length (i.e. 1 sample).
STRUCTURE_WINDOW = 401

# --- etalon fringe removal ---------------------------------------------------
# Periods (in GHz) of parasitic etalon fringes to fit out along with the line.
# [] disables it. Find them with find_fringe_periods() on the residuals of a
# first pass, or from a cavity length d via period = c / (2 n d).
#
# Each entry adds a sin and a cos term at that fixed period. Parametrising as
# sin+cos rather than amplitude+phase keeps the model LINEAR in the new
# parameters, so there is no phase degeneracy and no initial guess needed.
# The period itself is held fixed - letting it float makes it degenerate with
# the Gaussian and the fit will happily eat your line.
#
# Safe when the fringe period is well above the line FWHM (say 3x or more),
# since the two are then separated in Fourier space. Check that ratio before
# switching this on.
FRINGE_PERIODS = []

# --- systematic scale factors on N_s -----------------------------------------
# These are multiplicative and constant across a sweep, so they move the
# absolute density scale WITHOUT distorting trends in power, pressure or N2.
# Reported separately from the statistical error for exactly that reason.
L_PATH_REL_ERR = 0.10     # effective absorption length, 10 %
A_KI_REL_ERR   = 0.07     # Einstein coefficient; set from the NIST accuracy
                          # rating of the line you are actually using
FSR_REL_ERR    = 0.005    # etalon free spectral range; affects BOTH N_s
                          # (linearly) and T_gas (quadratically). Set to your
                          # actual etalon-spacing uncertainty - 0.005-0.01 for
                          # a spec'd etalon; 0.5 (50 %) was a placeholder.

# --- spectroscopic constants -------------------------------------------------
# N_s scales as g_lower / (g_upper * A_ki). CONFIRM these against the NIST ASD
# before quoting absolute densities; a misidentified upper level is an
# order-of-magnitude error, not a percentage one.
LAMBDA_0 = 696.5431e-9    # m    Ar I 1s5 -> 2p2
L_PATH   = 0.04           # m    absorption path length
M_AR     = 40.0           # amu
G_LOWER  = 5              # 1s5, J = 2
G_UPPER  = 3              # 2p2, J = 1
A_KI     = 6.39e6         # s^-1

C = 2.9979e8
DOPPLER_CONST = 7.16e-7   # dlambda_FWHM = K * lambda0 * sqrt(T/M), T in K, M in amu
K_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))


# =============================================================================
# MODEL
# =============================================================================

def gaussian_poly(x, A, x0, sig, *coef):
    """Gaussian plus a polynomial background. coef is (c0, c1, ...) ascending."""
    bg = np.polyval(np.asarray(coef)[::-1], x) if coef else 0.0
    return A * np.exp(-(x - x0) ** 2 / (2 * sig ** 2)) + bg


def make_model(order, fringe_periods):
    """
    Build the fit model: Gaussian + polynomial background + fixed-period
    sinusoids.

    Parameters are ordered (A, x0, sigma, c0..c_order, [s1, k1, s2, k2, ...])
    where each (s, k) pair is the sin and cos coefficient of one fringe. The
    fringe amplitude is sqrt(s^2 + k^2) and its phase atan2(k, s).
    """
    n_bg = order + 1

    def model(x, *p):
        A, x0, sig = p[0], p[1], p[2]
        out = A * np.exp(-(x - x0) ** 2 / (2 * sig ** 2))
        out = out + np.polyval(np.asarray(p[3:3 + n_bg])[::-1], x)
        q = 3 + n_bg
        for period in fringe_periods:
            w = 2.0 * np.pi / period
            out = out + p[q] * np.sin(w * x) + p[q + 1] * np.cos(w * x)
            q += 2
        return out

    return model


def find_fringe_periods(resid, f_ghz, n_peaks=2, min_period=2.0):
    """
    Locate dominant periodicities in the fit residuals.

    Returns periods in GHz, strongest first. Run this on a first pass with
    FRINGE_PERIODS = [], then put the answers back into FRINGE_PERIODS.

    min_period guards against picking up the line itself: anything with a
    period close to the linewidth is signal, not fringe.
    """
    r = np.asarray(resid, float)
    f = np.asarray(f_ghz, float)
    span = f.max() - f.min()
    if span <= 0 or len(r) < 16:
        return []

    grid = np.linspace(f.min(), f.max(), len(r))
    r = np.interp(grid, f, r)
    r = (r - r.mean()) * np.hanning(len(r))

    power = np.abs(np.fft.rfft(r)) ** 2
    freqs = np.fft.rfftfreq(len(r), d=span / (len(r) - 1))   # cycles per GHz

    ok = freqs > 0
    periods, power = 1.0 / freqs[ok], power[ok]
    keep = periods >= min_period
    periods, power = periods[keep], power[keep]
    if len(periods) == 0:
        return []
    return [float(periods[i]) for i in np.argsort(power)[::-1][:n_peaks]]


# =============================================================================
# NOISE AND RESIDUAL STRUCTURE
# =============================================================================

def estimate_noise(y):
    '''
    White-noise level of a trace, from the scatter of successive differences.

    sigma = 1.4826 * MAD(diff(y)) / sqrt(2)

    Two properties matter here. Differencing removes any smooth component, so
    a drifting baseline or the line profile itself does not inflate the
    estimate - unlike std() of the raw trace. And the MAD is robust, so spikes
    do not either. The sqrt(2) undoes the variance doubling from differencing.

    This replaces std() of the plasma-emission trace, which was (a) in volts
    rather than absorbance and (b) inflated by any real drift of the emission
    across the sweep, which deflated chi2 exactly when the plasma was
    brightest.
    '''
    d = np.diff(np.asarray(y, dtype=float))
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return np.nan
    mad = np.median(np.abs(d - np.median(d)))
    return float(1.4826 * mad / np.sqrt(2.0))


def excess_structure(resid, w=None):
    '''
    How much more residual survives smoothing than white noise would.

    Smoothing over w samples shrinks white noise by sqrt(w), so for white
    residuals std(smoothed)/std(raw) -> 1/sqrt(w) and this returns 1.
    Correlated residuals (etalon fringing, baseline curvature, a wrong line
    shape) survive smoothing and score above 1.

    N correlated points carry only N/tau independent pieces of information, so
    the 1/sqrt(N) averaging built into the fit covariance is optimistic by
    roughly sqrt(tau) - which is what this returns. Unlike chi2 it is
    scale-free in both signal and noise, so it is comparable between files.
    '''
    w = w or STRUCTURE_WINDOW
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
    Weighted mean of replicates with Birge-ratio inflation.

        w_j     = 1 / u_j^2
        xbar    = sum(w x) / sum(w)
        u_int   = 1 / sqrt(sum(w))
        u_ext   = sqrt( sum(w (x - xbar)^2) / ((n-1) sum(w)) )
        R       = u_ext / u_int = sqrt(chi2_reduced of the replicate set)
        u_final = u_int * max(1, R)

    R ~ 1 means the replicates scatter consistently with their own fit errors.
    R >> 1 means an unmodelled systematic dominates. Note that with n = 3 the
    sampling spread of R is about +/-0.5, so only clearly large values mean
    anything.
    """
    values = np.asarray(values, dtype=float)
    errors = np.asarray(errors, dtype=float)

    finite = np.isfinite(values)
    values, errors = values[finite], errors[finite]
    n = len(values)

    if n == 0:
        return dict(mean=np.nan, err=np.nan, err_int=np.nan, err_ext=np.nan,
                    birge=np.nan, n=0, method='none')
    if n == 1:
        e = errors[0] if np.isfinite(errors[0]) else np.nan
        return dict(mean=values[0], err=e, err_int=e, err_ext=np.nan,
                    birge=np.nan, n=1, method='single')

    usable = np.isfinite(errors) & (errors > 0)
    if not usable.all():
        sem = values.std(ddof=1) / np.sqrt(n)
        return dict(mean=values.mean(), err=sem, err_int=np.nan, err_ext=sem,
                    birge=np.nan, n=n, method='unweighted')

    w = 1.0 / errors ** 2
    W = w.sum()
    mean = float((w * values).sum() / W)
    err_int = float(1.0 / np.sqrt(W))
    err_ext = float(np.sqrt((w * (values - mean) ** 2).sum() / ((n - 1) * W)))
    birge = err_ext / err_int if err_int > 0 else np.nan
    err = err_int * max(1.0, birge) if np.isfinite(birge) else err_int

    return dict(mean=mean, err=err, err_int=err_int, err_ext=err_ext,
                birge=birge, n=n, method='birge')


# =============================================================================
# FREQUENCY AXIS
# =============================================================================

def find_relative_frequency(mean_df, fsr=FSR, min_peaks=MIN_PEAKS_PER_PERIOD,
                            min_pts=1000, prom=0.1, method='pchip', poly_deg=2):
    """
    Build a relative frequency axis (Hz) from the Fabry-Perot fringes.

    Returns (frequency, periods). frequency is NaN wherever calibration is
    invalid; periods is a list of dicts (or None) describing each sawtooth
    period, leading/trailing partials included.

    Each period dict also carries 'peak_times' and 'peak_spacings' (the
    cleaned FP peak sample-times and their successive differences, normalised
    to a single FSR). These are what frequency_axis_error() uses: the spacings
    are one FSR apart by construction, so their fractional scatter is the
    fractional uncertainty of the GHz-per-sample axis scale for that period.
    """
    time = mean_df.index.to_numpy()
    perot = mean_df[FP_CHANNEL].to_numpy()
    ramp = mean_df[RAMP_CHANNEL].to_numpy()

    # --- sawtooth falling edges, two passes so the minimum separation comes
    #     from the data rather than a guess ---
    dy = np.diff(ramp)
    rough, _ = find_peaks(-dy, height=0.6 * np.max(-dy))
    if len(rough) < 2:
        raise RuntimeError('could not find sawtooth edges')
    est_period = np.median(np.diff(rough))
    edges, _ = find_peaks(-dy, height=0.5 * np.max(-dy),
                          distance=int(0.8 * est_period))

    # --- etalon peaks, relative prominence threshold ---
    p = perot - np.median(perot)
    span = np.percentile(p, 99) - np.percentile(p, 1)
    peaks, _ = find_peaks(p, prominence=prom * span, distance=40, width=2)

    # --- period list, partial ends included ---
    period_ranges = []
    if edges[0] >= min_pts:
        period_ranges.append((0, edges[0], 'leading partial'))
    for i in range(len(edges) - 1):
        period_ranges.append((edges[i], edges[i + 1], f'full {i}'))
    if (len(time) - edges[-1]) >= min_pts:
        period_ranges.append((edges[-1], len(time), 'trailing partial'))

    if len(period_ranges) > 200:
        raise RuntimeError(
            f'{len(period_ranges)} sawtooth periods found in {len(time)} samples. '
            'The edge finder is firing on noise, not on flyback. Check that '
            f'{RAMP_CHANNEL} is the ramp and that it RISES then drops.')

    frequency = np.full(len(time), np.nan)
    periods = []
    n_skipped = 0

    for lo, hi, label in period_ranges:
        pk = np.sort(peaks[(peaks >= lo) & (peaks < hi)])
        if len(pk) < min_peaks:
            if n_skipped < 5:
                print(f'    period {label!r}: only {len(pk)} fringes - skipped')
            n_skipped += 1
            periods.append(None)
            continue

        peak_times = time[pk]

        # drop doublets
        d = np.diff(peak_times)
        keep = np.ones(len(peak_times), dtype=bool)
        keep[1:][d < 0.5 * np.median(d)] = False
        peak_times = peak_times[keep]
        if len(peak_times) < min_peaks:
            if n_skipped < 5:
                print(f'    period {label!r}: too few fringes after cleaning - skipped')
            n_skipped += 1
            periods.append(None)
            continue

        # integer FSR steps, so a missed fringe becomes a step of 2 rather
        # than a stretched step of 1
        d = np.diff(peak_times)
        steps = np.rint(d / np.median(d)).astype(int)
        steps[steps < 1] = 1
        freq_peaks = np.concatenate(([0.0], np.cumsum(steps))) * fsr

        sl = slice(lo, hi)
        t_sl = time[sl]

        if method == 'linear':
            frequency[sl] = np.interp(t_sl, peak_times, freq_peaks,
                                      left=np.nan, right=np.nan)
        elif method == 'pchip':
            frequency[sl] = PchipInterpolator(peak_times, freq_peaks,
                                              extrapolate=False)(t_sl)
        elif method == 'poly':
            coeffs = np.polyfit(peak_times, freq_peaks, poly_deg)
            f = np.polyval(coeffs, t_sl)
            f[(t_sl < peak_times[0]) | (t_sl > peak_times[-1])] = np.nan
            frequency[sl] = f
        else:
            raise ValueError(f'unknown method {method!r}')

        # 'steps' is the integer FSR multiple of each gap; dividing the
        # measured time-gap by steps normalises multi-FSR gaps (a missed
        # fringe) down to a single-FSR spacing, so the scatter reflects real
        # jitter rather than the occasional skipped peak.
        #
        # Only KEEP gaps whose length rounds cleanly to an integer number of
        # FSRs. A gap landing far from an integer multiple is a peak-finding
        # glitch (a doublet, a near-miss rint() rounded the wrong way), not
        # real spacing jitter - and with only ~5-8 gaps per period a single
        # such glitch dominates std/mean and inflates the axis-jitter estimate
        # by an order of magnitude. Rejecting them here is what stops
        # rel_axis coming out at ~4 % (glitch-driven) instead of the true
        # sub-percent FP stability.
        ratio = d / np.median(d)
        clean = np.abs(ratio - np.rint(ratio)) < 0.25    # within 1/4 FSR
        single_fsr_spacings = (d / steps)[clean]

        periods.append({'slice': sl, 'label': label,
                        'n_fringes': len(peak_times),
                        'peak_times': peak_times,
                        'peak_spacings': single_fsr_spacings})

    if n_skipped > 5:
        print(f'    ... and {n_skipped - 5} more periods skipped')
    if np.all(np.isnan(frequency)):
        raise RuntimeError(
            f'frequency calibration failed in all {len(period_ranges)} periods. '
            f'Check that {FP_CHANNEL} carries the etalon fringes and that at '
            f'least {min_peaks} are resolved per sweep.')

    return frequency, periods


# =============================================================================
# PER-PERIOD FIT
# =============================================================================

def analyze_period(f_hz, y, sigma_noise, order=BASELINE_ORDER):
    """
    Fit a Gaussian on a polynomial background to one sawtooth period.

    sigma_noise is on the ABSORBANCE scale, so chi2 is dimensionless and
    comparable between files, and it is passed to curve_fit with
    absolute_sigma=True. The covariance that comes back is therefore the pure
    statistical one, tied to the measured noise floor rather than to the
    residuals. Errors are then inflated explicitly:

        u -> u * max(1, sqrt(chi2_red)) * excess_structure

    The two factors are independent and do not double count:

      * sqrt(chi2_red) covers residuals LARGER in amplitude than the noise
        floor. This is what absolute_sigma=False used to do implicitly - but
        it also applied the factor when chi2_red < 1, deflating the error
        below what the measured noise supports. The max(1, .) clamp stops that.
      * excess_structure covers residuals that are CORRELATED. Correlated
        residuals break the 1/sqrt(N) averaging the covariance assumes, and
        chi2_red alone does not see this: residuals can sit exactly at the
        noise amplitude (chi2_red = 1) and still be a coherent fringe.

    Area uncertainty keeps the A-sigma covariance term:
        u^2(area) = 2*pi [ sig^2 var(A) + A^2 var(sig) + 2 A sig cov(A,sig) ]
    A and sigma are strongly anti-correlated, so dropping the cross term
    overestimates the area error, often badly.
    """
    idx = np.argsort(f_hz)
    f = f_hz[idx] / 1e9           # GHz
    y = y[idx]

    off0 = np.median(y)
    df = np.abs(np.median(np.diff(f)))
    if not np.isfinite(df) or df <= 0:
        df = (f.max() - f.min()) / max(len(f), 2)

    n_bg = order + 1
    n_fr = 2 * len(FRINGE_PERIODS)
    model = make_model(order, FRINGE_PERIODS)

    p0 = ([y.max() - off0, f[np.argmax(y)], 1.0, off0]
          + [0.0] * (n_bg - 1) + [0.0] * n_fr)
    lo = [0.0, f.min(), df] + [-np.inf] * (n_bg + n_fr)
    hi = [np.inf, f.max(), f.max() - f.min()] + [np.inf] * (n_bg + n_fr)

    popt, pcov = curve_fit(model, f, y, p0=p0, bounds=(lo, hi),
                           sigma=np.full(len(f), sigma_noise),
                           absolute_sigma=True, maxfev=20000)
    A, x0, sig = popt[0], popt[1], abs(popt[2])
    coef = popt[3:3 + n_bg]

    resid = y - model(f, *popt)
    dof = len(f) - len(popt)
    chi2_red = float(np.sum((resid / sigma_noise) ** 2) / dof) if dof > 0 else np.nan

    # --- error inflation ---
    k_chi2 = np.sqrt(max(1.0, chi2_red)) if np.isfinite(chi2_red) else 1.0
    k_struct = excess_structure(resid)
    inflate = k_chi2 * k_struct

    # flag fits sitting on a bound: the covariance is meaningless there, and
    # because the reported errors come out small such fits would otherwise
    # dominate the weighted mean
    at_bound = (A <= 1e-12) or (abs(sig - df) < 1e-9 * max(df, 1.0))

    if np.all(np.isfinite(pcov)):
        perr = np.sqrt(np.diag(pcov))
        A_err, sig_err = perr[0] * inflate, perr[2] * inflate
        dA, ds = sig * np.sqrt(2 * np.pi), A * np.sqrt(2 * np.pi)
        area_var = (dA ** 2 * pcov[0, 0] + ds ** 2 * pcov[2, 2]
                    + 2.0 * dA * ds * pcov[0, 2])
        area_err = float(np.sqrt(area_var)) * inflate if area_var > 0 else np.nan
    else:
        A_err = sig_err = area_err = np.nan

    return {
        'A': A, 'A_err': A_err,
        'x0': x0, 'sigma': sig, 'sigma_err': sig_err,
        'coef': coef,
        'fwhm': K_FWHM * sig, 'fwhm_err': K_FWHM * sig_err,
        'area': A * sig * np.sqrt(2 * np.pi), 'area_err': area_err,
        'chi2': chi2_red, 'n_points': len(f), 'at_bound': at_bound,
        'sigma_noise': sigma_noise,
        'k_chi2': k_chi2, 'k_struct': k_struct, 'inflate': inflate,
        'model': model, 'popt': popt, 'resid': resid,
        'fringe_amp': [float(np.hypot(popt[3 + n_bg + 2 * i],
                                      popt[3 + n_bg + 2 * i + 1]))
                       for i in range(len(FRINGE_PERIODS))],
        'f_data': f, 'y_data': y,
    }


# =============================================================================
# I/O AND ABSORBANCE
# =============================================================================

def load_scope(path):
    """Read a scope CSV and average repeat sweeps onto the time axis."""
    d = pd.read_csv(path)
    chans = [FP_CHANNEL, RAMP_CHANNEL, DIODE_CHANNEL]
    missing = [c for c in ['time'] + chans if c not in d.columns]
    if missing:
        raise KeyError(f'{path}: missing columns {missing}')
    return d.groupby('time')[chans].mean()


def _on_grid(target_t, source):
    """Diode channel of `source` sampled onto `target_t`."""
    t = source.index.to_numpy()
    v = source[DIODE_CHANNEL].to_numpy()
    if len(t) == len(target_t) and np.array_equal(t, target_t):
        return v
    return np.interp(target_t, t, v)


def build_absorbance(path_on, path_off, ref_on, ref_off):
    """
    Form the absorbance spectrum from the four traces.

        I_ref = (plasma off, laser on) - (plasma off, laser off)      = I_0
        I_m   = (plasma on,  laser on) - (plasma on,  laser off)      = I
        alpha = ln( (I_ref + Vbias) / I_m )                           = k * L

    Everything is put onto the plasma-on, laser-on time base. Subtracting the
    emission at the same plasma condition is the only correct choice, since
    emission scales with power and N2 fraction.

    Returns (mean_on, mean_off, alpha, i_m, i_emis, i_ref, r_on, r_off):
    mean_on and mean_off are the full (time, CH1, CH2, CH3) dataframes for
    the plasma-on laser-on and plasma-on laser-off traces; r_on and r_off are
    the same for the plasma-off reference traces (on their OWN time base -
    not yet interpolated). i_ref is the reference diode signal already
    interpolated onto mean_on's time base, i.e. the actual I_0 used in the
    division above.
    """
    mean_on = load_scope(path_on)
    mean_off = load_scope(path_off)
    r_on = load_scope(ref_on)
    r_off = load_scope(ref_off)

    t = mean_on.index.to_numpy()
    i_on = mean_on[DIODE_CHANNEL].to_numpy()
    i_emis = _on_grid(t, mean_off)
    i_ref = _on_grid(t, r_on) - _on_grid(t, r_off)

    i_m = i_on - i_emis + BIAS_VOLTAGE
    with np.errstate(divide='ignore', invalid='ignore'):
        alpha = np.log((i_ref + BIAS_VOLTAGE) / i_m)

    return mean_on, mean_off, alpha, i_m, i_emis, i_ref, r_on, r_off


# =============================================================================
# RIPPLE / LASER-RAMP PHASE DIAGNOSTIC
# =============================================================================
#
# The division alpha = ln(i_ref / i_m) only cancels the laser's own power
# ramp (and any parasitic interference ripple riding on it) if that ramp
# looks the SAME, at the SAME phase, in the plasma-off reference shot as in
# the plasma-on measurement. `_on_grid` brings the reference onto the
# on-shot's time base by plain time interpolation, which silently assumes
# the laser scan is perfectly reproducible shot to shot. The functions below
# check that assumption directly, instead of only inferring a problem from
# residual fringing on `alpha` after the Gaussian fit.

def find_ripple_period(f_ghz, y, bg_order=1, min_period=0.3):
    """
    Estimate the dominant ripple period (GHz) riding on top of a smooth
    power ramp: fit out a polynomial of `bg_order`, then FFT the residual.

    Meant to be run on the REFERENCE trace (plasma off), which has no
    absorption feature to confuse the ramp/ripple with. Returns None if no
    period could be found.
    """
    f = np.asarray(f_ghz, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(f) & np.isfinite(y)
    f, y = f[ok], y[ok]
    if len(f) < 16:
        return None
    coeffs = np.polyfit(f, y, bg_order)
    resid = y - np.polyval(coeffs, f)
    cand = find_fringe_periods(resid, f, n_peaks=1, min_period=min_period)
    return cand[0] if cand else None


def _ripple_residual_power(f_ghz, resid, period_ghz):
    """
    Fraction of `resid` variance explained by a sin+cos at `period_ghz`.
    A cheap, robust score for a 1-D period search: the sin/cos amplitudes
    come out of a 2x2 linear solve, no phase guess needed. Higher is better.
    """
    w = 2.0 * np.pi / period_ghz
    A = np.vstack([np.sin(w * f_ghz), np.cos(w * f_ghz)]).T
    coef, *_ = np.linalg.lstsq(A, resid, rcond=None)
    fitted = A @ coef
    return float(np.hypot(*coef))     # amplitude; monotone in explained power


def find_ripple_period_global(f_ghz_list, y_list, bg_order=1, min_period=0.3,
                              refine=True, refine_frac=0.5, n_grid=400):
    """
    Estimate ONE ripple period (GHz) from the whole reference scan.

    Each period's reference trace is detrended separately (its own polynomial
    of `bg_order`, so the per-period power ramp is removed) and the residuals
    are concatenated onto a common frequency axis. FFT-ing that combined
    residual gives a far tighter period estimate than one period alone,
    because you get many more ripple cycles into the transform.

    If `refine`, the FFT period is then polished by a bounded 1-D scan that
    maximises the sin+cos amplitude the period explains in the combined
    residual - this walks off the coarse FFT bin onto the true period. The
    scan spans +/- `refine_frac` of the FFT period over `n_grid` points.

    Parameters
    ----------
    f_ghz_list, y_list : lists of arrays
        Per-period frequency (GHz) and reference-diode arrays. NaNs allowed.

    Returns the period in GHz, or None if nothing usable was found.
    """
    fs, rs = [], []
    for f, y in zip(f_ghz_list, y_list):
        f = np.asarray(f, dtype=float)
        y = np.asarray(y, dtype=float)
        ok = np.isfinite(f) & np.isfinite(y)
        if ok.sum() < bg_order + 4:
            continue
        f, y = f[ok], y[ok]
        # detrend per period so each ramp is removed on its own terms; shift
        # frequency to start at 0 so periods stack on a common axis
        coeffs = np.polyfit(f, y, bg_order)
        rs.append(y - np.polyval(coeffs, f))
        fs.append(f - f.min())
    if not fs:
        return None

    f_all = np.concatenate(fs)
    r_all = np.concatenate(rs)
    order = np.argsort(f_all)
    f_all, r_all = f_all[order], r_all[order]

    coarse = find_fringe_periods(r_all, f_all, n_peaks=1, min_period=min_period)
    if not coarse:
        return None
    p0 = coarse[0]
    if not refine:
        return p0

    lo = max(min_period, p0 * (1.0 - refine_frac))
    hi = p0 * (1.0 + refine_frac)
    grid = np.linspace(lo, hi, n_grid)
    scores = [_ripple_residual_power(f_all, r_all, p) for p in grid]
    return float(grid[int(np.argmax(scores))])


def fit_ripple(f_ghz, y, period_ghz, bg_order=1):
    """
    Fit `y` vs `f_ghz` (GHz) to a polynomial background of order `bg_order`
    plus a sinusoid at the FIXED period `period_ghz`.

    Parametrised as sin+cos, same trick as `make_model`'s FRINGE_PERIODS, so
    the whole fit is a single linear least squares - no initial guess, no
    phase degeneracy. Returns background coefficients (ascending), the
    sin/cos coefficients, amplitude, phase (rad, atan2 convention matching
    `y_ripple = amp * sin(2*pi*f/period + phase)`), and the fitted curve.
    """
    f = np.asarray(f_ghz, dtype=float)
    y = np.asarray(y, dtype=float)
    w = 2.0 * np.pi / period_ghz
    cols = [f ** i for i in range(bg_order + 1)] + [np.sin(w * f), np.cos(w * f)]
    A = np.vstack(cols).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    bg = coef[:bg_order + 1]
    s, c = coef[bg_order + 1], coef[bg_order + 2]
    amp = float(np.hypot(s, c))
    phase = float(np.arctan2(c, s))
    return dict(bg=bg, s=float(s), c=float(c), amp=amp, phase=phase,
                period=period_ghz, curve=A @ coef)


def diagnose_ripple_phase(results, period_index=None, mask_fwhm_mult=3.0,
                          bg_order=1, ripple_period=None, plot=True):
    """
    Compare the ripple's phase in the plasma-off reference to its phase in
    the plasma-on measurement (line masked out), period by period.

    Parameters
    ----------
    results : dict
        The dict returned by `analyze()`.
    period_index : int or None
        Which sawtooth period (numbered as in the console output / `fits`)
        to check. None checks every period with a usable Gaussian fit and
        prints a summary; an int does one period and always plots it.
    mask_fwhm_mult : float
        Half-width, in multiples of the fitted FWHM, of the region excluded
        around the Gaussian center before fitting the on-shot ripple.
    ripple_period : float or None
        Ripple period in GHz. None estimates it once from the WHOLE reference
        scan (all target periods stacked and refined), then reuses that value
        for every period.
    plot : bool
        Plot the first (or the requested) period's traces and fits.

    Returns a list of per-period dicts: 'period', 'ripple_period', 'ref_fit',
    'on_fit', 'dphi' (wrapped to [-pi, pi]), plus the corrected reference and
    absorbance for that period ('i_ref_corrected', 'alpha_corrected') built
    by keeping the reference's own amplitude and background but swapping in
    the on-shot's fitted phase - a way to see whether the mismatch, if any,
    is actually what's distorting the absorbance baseline.
    """
    frequency = results['frequency']
    i_ref = results['i_ref']
    i_m = results['i_m']
    periods = results['periods']
    fits_by_period = {r['period']: r for r in results['fits']}

    targets = [period_index] if period_index is not None else sorted(fits_by_period)

    if ripple_period is None:
        # Estimate from the whole reference scan: stack every target period's
        # reference slice, detrend each, and refine. Many more ripple cycles
        # go into this than a single period, so the period lands far tighter.
        f_list, y_list = [], []
        for k in targets:
            if k not in fits_by_period:
                continue
            sl = periods[k]['slice']
            f_list.append(frequency[sl] / 1e9)
            y_list.append(i_ref[sl])
        ripple_period = find_ripple_period_global(f_list, y_list,
                                                  bg_order=bg_order)
        if ripple_period is None:
            raise RuntimeError('could not estimate a ripple period from the '
                               'reference scan - pass ripple_period explicitly')
        print(f'Estimated ripple period: {ripple_period:.3f} GHz '
              f'(whole reference scan, {len(f_list)} periods, refined)')

    out = []
    for k in targets:
        if k not in fits_by_period:
            print(f'  period {k}: no Gaussian fit available - skipped')
            continue
        r = fits_by_period[k]
        sl = periods[k]['slice']
        f = frequency[sl]
        yr, ym = i_ref[sl], i_m[sl]
        ok = np.isfinite(f) & np.isfinite(yr) & np.isfinite(ym)
        f_ghz, yr, ym = f[ok] / 1e9, yr[ok], ym[ok]

        ref_fit = fit_ripple(f_ghz, yr, ripple_period, bg_order=bg_order)

        mask = np.abs(f_ghz - r['x0']) > mask_fwhm_mult * r['fwhm']
        if mask.sum() < bg_order + 3:
            print(f'  period {k}: not enough points outside the line to fit '
                  f'the ripple - skipped')
            continue
        on_fit = fit_ripple(f_ghz[mask], ym[mask], ripple_period, bg_order=bg_order)

        dphi = (on_fit['phase'] - ref_fit['phase'] + np.pi) % (2 * np.pi) - np.pi
        print(f'  period {k}: ref phase {ref_fit["phase"]:+.2f} rad, '
              f'on phase {on_fit["phase"]:+.2f} rad, '
              f'diff {dphi:+.2f} rad ({np.degrees(dphi):+.0f} deg)   '
              f'amp ref {ref_fit["amp"]:.4g}, on {on_fit["amp"]:.4g}')

        # Phase-corrected reference: keep the reference's own background and
        # amplitude (that's the actual measured I_0 level), swap in the
        # on-shot's fitted phase for the ripple term only.
        w = 2.0 * np.pi / ripple_period
        i_ref_corrected = (np.polyval(ref_fit['bg'][::-1], f_ghz)
                          + ref_fit['amp'] * np.sin(w * f_ghz + on_fit['phase']))
        with np.errstate(divide='ignore', invalid='ignore'):
            alpha_corrected = np.log((i_ref_corrected + BIAS_VOLTAGE) / ym)

        out.append(dict(period=k, ripple_period=ripple_period,
                        ref_fit=ref_fit, on_fit=on_fit, dphi=dphi,
                        f_ghz=f_ghz, i_ref=yr, i_m=ym, mask=mask,
                        i_ref_corrected=i_ref_corrected,
                        alpha_corrected=alpha_corrected))

        if plot and (period_index is not None or k == targets[0]):
            _plot_ripple(f_ghz, yr, ym, mask, ref_fit, on_fit, k)

    dphis = [d['dphi'] for d in out]
    if dphis:
        print(f'\nMean |phase diff| over {len(dphis)} period(s): '
              f'{np.degrees(np.mean(np.abs(dphis))):.0f} deg   '
              f'(std {np.degrees(np.std(dphis)):.0f} deg)')
    return out


def _plot_ripple(f_ghz, i_ref, i_m, mask, ref_fit, on_fit, period_label):
    """Reference and on-shot traces, both ripple fits, and the masked region."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(f_ghz, i_ref, '.', color='0.6', markersize=3,
            label='reference (plasma off)')
    ax.plot(f_ghz, ref_fit['curve'], '-', color='k', linewidth=1.5,
            label=f'ref fit (phase {ref_fit["phase"]:+.2f} rad)')
    ax.plot(f_ghz, i_m, '.', color='C0', markersize=3, alpha=0.5,
            label='on-shot (plasma on)')
    ax.plot(f_ghz[mask], on_fit['curve'], '-', color='C3', linewidth=1.5,
            label=f'on-shot fit, line masked (phase {on_fit["phase"]:+.2f} rad)')
    if (~mask).any():
        ax.axvspan(f_ghz[~mask].min(), f_ghz[~mask].max(),
                  color='C3', alpha=0.08, label='masked (line region)')
    ax.set_xlabel('Relative frequency (GHz)', fontweight='bold')
    ax.set_ylabel('Diode signal (V)', fontweight='bold')
    ax.set_title(f'Ripple phase check - period {period_label}, '
                f'period {ref_fit["period"]:.2f} GHz')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.show()


# =============================================================================
# FREQUENCY-AXIS ERROR DECOMPOSITION
# =============================================================================

def frequency_axis_error(results, verbose=True):
    """
    Per-period frequency-axis (scale) uncertainty from the scatter of the
    Fabry-Perot peak spacings, propagated into each period's FWHM, then a
    quadrature decomposition of the total between-period FWHM scatter into a
    frequency-calibration piece and an unexplained remainder.

    Per period, the FP peaks are one FSR apart by construction, so

        rel_axis = std(spacings)/mean(spacings) / sqrt(n_gaps)

    is the fractional uncertainty of that period's GHz-per-sample scale (the
    sqrt(n_gaps) because the whole-axis scale is the mean of n_gaps spacings).
    FWHM is linear in the scale, so u_freq(FWHM) = rel_axis * FWHM.

    Decomposition (all sigma on FWHM, same axis, assumed independent):

        sigma_total   = between-period std of the fitted FWHM (all causes)
        sigma_freq    = quadrature-mean of u_freq(FWHM) over periods
        sigma_rem     = sqrt(max(0, sigma_total^2 - sigma_freq^2))   <- clamp

    sigma_rem is the scatter NOT explained by the frequency axis (fringe,
    drift, emission residual). Clamped at zero and reported as an upper bound,
    because with estimated variances sigma_total^2 - sigma_freq^2 can go
    slightly negative when the frequency term already covers the total.

    Falls back to reconstructing the per-period axis scale from the frequency
    array when a period dict lacks 'peak_spacings' (e.g. results built by an
    older find_relative_frequency). The FP-spacing version is preferred.

    Independence caveat: if the axis jitter and the fringe share a common
    cause (a laser scan that doesn't repeat drives both), the cross term is
    not zero and this splits them imperfectly. Treat the fractions as
    indicative, not exact.
    """
    periods = results['periods']
    frequency = results['frequency']
    kept    = [r for r in results['fits'] if r['keep']]

    rel_axis, ufreq_fwhm, fwhms = [], [], []
    for r in kept:
        pd_ = periods[r['period']]
        sp = np.asarray(pd_.get('peak_spacings', []), dtype=float)
        sp = sp[np.isfinite(sp) & (sp > 0)]

        if len(sp) >= 3:
            # robust fractional axis jitter: MAD (scaled to a sigma) instead
            # of std, so any single peak-finding glitch that survived the
            # clean-gap filter in find_relative_frequency can't dominate a
            # short (~5-8) per-period spacing set. /sqrt(n) because the axis
            # scale is the mean of n spacings.
            med = np.median(sp)
            mad = np.median(np.abs(sp - med)) * 1.4826
            rel = (mad / med) / np.sqrt(len(sp)) if med > 0 else np.nan
        elif len(sp) == 2:
            rel = np.std(sp, ddof=1) / np.mean(sp) / np.sqrt(2)
        else:
            # fallback: scale jitter straight off the axis for this period
            sl = pd_['slice']
            f = frequency[sl]
            f = f[np.isfinite(f)] / 1e9
            dg = np.diff(np.sort(f))
            dg = dg[dg > 0]
            if len(dg) < 2 or np.median(dg) <= 0:
                continue
            med = np.median(dg)
            mad = np.median(np.abs(dg - med)) * 1.4826
            rel = (mad / med) / np.sqrt(len(dg)) if med > 0 else np.nan

        if not np.isfinite(rel):
            continue

        rel_axis.append(rel)
        ufreq_fwhm.append(rel * r['fwhm'])
        fwhms.append(r['fwhm'])

    if len(fwhms) < 2:
        print('  frequency_axis_error: need >=2 kept periods with a usable '
              'axis scale - skipped')
        return None

    fwhms = np.asarray(fwhms)
    ufreq_fwhm = np.asarray(ufreq_fwhm)
    rel_axis = np.asarray(rel_axis)

    sigma_total = float(fwhms.std(ddof=1))
    sigma_freq  = float(np.sqrt(np.mean(ufreq_fwhm ** 2)))
    sigma_rem   = float(np.sqrt(max(0.0, sigma_total ** 2 - sigma_freq ** 2)))
    frac_freq   = (sigma_freq / sigma_total) ** 2 if sigma_total > 0 else np.nan

    Tstat_from_total = 2.0 * results['T_gas'] * sigma_total / fwhms.mean()
    Tstat_from_freq  = 2.0 * results['T_gas'] * sigma_freq  / fwhms.mean()

    if verbose:
        print('\n' + '=' * 58)
        print('FREQUENCY-AXIS ERROR DECOMPOSITION (on FWHM)')
        print('=' * 58)
        print(f'  kept periods used       {len(fwhms)}')
        print(f'  mean rel. axis error    {100*np.mean(rel_axis):.3f} %')
        print(f'  sigma_total (between)   {sigma_total:.4f} GHz   (all causes)')
        print(f'  sigma_freq  (forward)   {sigma_freq:.4f} GHz'
              f'   ({100*frac_freq:.0f} % of total variance)')
        print(f'  sigma_remain            <= {sigma_rem:.4f} GHz'
              f'   (fringe/drift/other)')
        print(f'\n  as T_gas scatter:  total {Tstat_from_total:.0f} K,'
              f'  frequency {Tstat_from_freq:.0f} K')
        if sigma_freq >= sigma_total:
            print('\n  NOTE: frequency calibration alone covers the full '
                  'between-period\n        scatter - remainder is an upper bound.')
        else:
            print(f'\n  Frequency axis explains ~{100*frac_freq:.0f} % of the '
                  f'width variance;\n  the rest is other systematics (fringe '
                  f'the prime suspect).')

    return dict(sigma_total=sigma_total, sigma_freq=sigma_freq,
                sigma_rem=sigma_rem, frac_freq=frac_freq,
                rel_axis=rel_axis, ufreq_fwhm=ufreq_fwhm, fwhms=fwhms,
                T_total=Tstat_from_total, T_freq=Tstat_from_freq)


# =============================================================================
# PHASE-CORRECTION AND ERROR-BUDGET PLOTS
# =============================================================================

def plot_absorbance_correction(check_entry):
    """
    Uncorrected vs phase-corrected absorbance for one period, overlaid.

    `check_entry` is one element of the list returned by
    diagnose_ripple_phase(). The uncorrected trace is rebuilt here from the
    same masked i_ref / i_m used in the diagnostic, so the two curves are
    strictly comparable (same points, same frequency axis).
    """
    f = check_entry['f_ghz']
    with np.errstate(divide='ignore', invalid='ignore'):
        alpha_uncorr = np.log((check_entry['i_ref'] + BIAS_VOLTAGE)
                              / check_entry['i_m'])
    alpha_corr = check_entry['alpha_corrected']

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(f, alpha_uncorr, '.', color='0.6', markersize=3, alpha=0.7,
            label='uncorrected')
    ax.plot(f, alpha_corr, '.', color='C0', markersize=3, alpha=0.7,
            label=f'phase corrected (dphi {np.degrees(check_entry["dphi"]):+.0f} deg)')
    ax.axhline(0, color='k', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Relative frequency (GHz)', fontweight='bold')
    ax.set_ylabel('Absorbance', fontweight='bold')
    ax.set_title(f'Absorbance with vs without phase correction - '
                 f'period {check_entry["period"]}')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.show()


def plot_corrected_with_fit(check_entry, order=BASELINE_ORDER):
    """
    Phase-corrected absorbance for one period with a fresh Gaussian-plus-
    background fit on top.

    Refits the corrected trace (rather than reusing the original fit, which
    was done on the uncorrected data) so the FWHM/area reported here reflect
    the correction. Returns the analyze_period result dict.
    """
    f = check_entry['f_ghz']
    y = check_entry['alpha_corrected']
    ok = np.isfinite(f) & np.isfinite(y)
    f, y = f[ok], y[ok]

    sigma_noise = estimate_noise(y)
    res = analyze_period(f * 1e9, y, sigma_noise, order=order)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(res['f_data'], res['y_data'], '.', color='C0', markersize=3,
            alpha=0.6, label='phase-corrected absorbance')
    dense = np.linspace(res['f_data'].min(), res['f_data'].max(), 2000)
    ax.plot(dense, res['model'](dense, *res['popt']), '-', color='C3',
            linewidth=2,
            label=f'Gaussian fit: FWHM {res["fwhm"]:.3f} GHz, '
                  f'area {res["area"]:.3f} GHz, chi2 {res["chi2"]:.1f}')
    ax.axhline(0, color='k', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Relative frequency (GHz)', fontweight='bold')
    ax.set_ylabel('Absorbance', fontweight='bold')
    ax.set_title(f'Phase-corrected absorbance + fit - '
                 f'period {check_entry["period"]}')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.show()
    return res


def _budget_components(results, freq_err=None):
    """
    Decompose the T_gas and N_s error into disjoint statistical and systematic
    pieces, all as relative % of the value, so they add IN QUADRATURE to the
    quoted total. Shared by plot_error_budget so the numbers and the bars come
    from one place.

    Statistical (red family - shifts trends, shrinks as periods are averaged):
      jitter  the MAD white-noise floor through the fit covariance. This is
              birge err_int: the error the fits predict from measured noise
              alone, before any replicate-scatter inflation.
      freq    frequency-axis (FP-spacing) jitter, from freq_err['sigma_freq'].
      fit     the 'everything else' in the full covariance+Birge statistical
              error - i.e. sqrt(full_birge^2 - jitter^2 - freq^2), clamped at
              zero. Carries the replicate scatter the fits did NOT predict
              (fringe, drift), which is what the Birge inflation is for. Kept
              disjoint from jitter/freq so the three don't double-count.

    Systematic (blue family - common-mode, moves absolute scale only):
      L, A_ki, FSR  the three multiplicative scale terms.

    Returns two dicts (t=..., n=...) each with keys
    jitter, freq, fit, L, A, fsr, stat_tot, sys_tot, tot   (all in %).

    Widths are in GHz; T_gas ~ width^2 so relative width errors DOUBLE into T
    (and FSR enters T quadratically, N_s linearly) - handled per quantity.
    """
    w, a = results['fwhm'], results['area']

    # --- relative statistical pieces on the WIDTH (fractional) ---
    # jitter floor = err_int (covariance from the MAD noise, no Birge inflation)
    rel_jit_w = (w['err_int'] / w['mean']) if (np.isfinite(w.get('err_int', np.nan))
                                               and w['mean']) else 0.0
    # full covariance+Birge statistical error on the width
    rel_full_w = (w['err'] / w['mean']) if w['mean'] else 0.0
    # frequency-axis jitter on the width
    if freq_err is not None and np.isfinite(freq_err.get('sigma_freq', np.nan)) \
            and w['mean']:
        rel_freq_w = freq_err['sigma_freq'] / w['mean']
    else:
        rel_freq_w = 0.0
    # 'fit / everything-else' = full minus jitter minus freq, in quadrature
    rel_fit_w = np.sqrt(max(0.0, rel_full_w**2 - rel_jit_w**2 - rel_freq_w**2))

    # --- same three on the AREA (for N_s) ---
    rel_jit_a = (a['err_int'] / a['mean']) if (np.isfinite(a.get('err_int', np.nan))
                                               and a['mean']) else 0.0
    rel_full_a = (a['err'] / a['mean']) if a['mean'] else 0.0
    # frequency jitter enters the area through the same axis scale as the width
    rel_freq_a = rel_freq_w
    rel_fit_a = np.sqrt(max(0.0, rel_full_a**2 - rel_jit_a**2 - rel_freq_a**2))

    # ---------------- T_gas: width errors DOUBLE (T ~ width^2) -------------
    t = {}
    t['jitter'] = 100 * 2.0 * rel_jit_w
    t['freq']   = 100 * 2.0 * rel_freq_w
    t['fit']    = 100 * 2.0 * rel_fit_w
    t['L']      = 0.0                       # L does not enter T_gas
    t['A']      = 0.0                       # A_ki does not enter T_gas
    t['fsr']    = 100 * 2.0 * FSR_REL_ERR   # FSR enters T quadratically
    t['stat_tot'] = np.sqrt(t['jitter']**2 + t['freq']**2 + t['fit']**2)
    t['sys_tot']  = np.sqrt(t['L']**2 + t['A']**2 + t['fsr']**2)
    t['tot']      = np.sqrt(t['stat_tot']**2 + t['sys_tot']**2)

    # ---------------- N_s: area errors pass through 1:1 -------------------
    n = {}
    n['jitter'] = 100 * rel_jit_a
    n['freq']   = 100 * rel_freq_a
    n['fit']    = 100 * rel_fit_a
    n['L']      = 100 * L_PATH_REL_ERR
    n['A']      = 100 * A_KI_REL_ERR
    n['fsr']    = 100 * FSR_REL_ERR
    n['stat_tot'] = np.sqrt(n['jitter']**2 + n['freq']**2 + n['fit']**2)
    n['sys_tot']  = np.sqrt(n['L']**2 + n['A']**2 + n['fsr']**2)
    n['tot']      = np.sqrt(n['stat_tot']**2 + n['sys_tot']**2)

    return dict(t=t, n=n)


def plot_error_budget(results, freq_err=None):
    """
    Stacked error budget, one bar per quantity, split into a red statistical
    family and a blue systematic family.

    Red gradient (statistical - affects TRENDS, averages down with periods):
        jitter (MAD floor)  ->  frequency jitter  ->  fit / chi2 / Birge
        light red to dark red, smallest to largest conceptual scale.

    Blue gradient (systematic - sets ABSOLUTE scale, common-mode across sweep):
        L path  ->  A_ki  ->  FSR
        light blue to dark blue.

    Segments are RELATIVE % and stack in QUADRATURE-consistent heights: each
    segment is drawn at its own % height, and the stack total equals the
    quadrature total (annotated), NOT the linear sum - so read the whole bar
    as the quoted error and each block as that source's contribution. A thin
    gap marks the statistical/systematic boundary. See _budget_components for
    how the three statistical pieces are kept disjoint (no double-counting of
    the frequency jitter, which otherwise lives inside the Birge term).

    T_gas gets no L/A_ki blocks (they don't enter the width); FSR appears in
    both, doubled in T_gas (quadratic) and linear in N_s.
    """
    comp = _budget_components(results, freq_err)
    w, a = results['fwhm'], results['area']

    # red (statistical) and blue (systematic) gradients, light -> dark
    red  = plt.cm.Reds(np.linspace(0.45, 0.85, 3))    # jitter, freq, fit
    blue = plt.cm.Blues(np.linspace(0.45, 0.85, 3))   # L, A, FSR

    def _draw(ax, c, title):
        # stack segments so the bar TOTAL is the quadrature total, not the
        # linear sum: scale each segment's drawn height by tot / linear_sum
        stat_parts = [('jitter\n(MAD floor)', c['jitter'], red[0]),
                      ('freq jitter\n(FP axis)', c['freq'], red[1]),
                      ('fit / chi2\n/ Birge',    c['fit'],  red[2])]
        sys_parts  = [('L path', c['L'], blue[0]),
                      ('A_ki',   c['A'], blue[1]),
                      ('FSR',    c['fsr'], blue[2])]
        parts = [p for p in stat_parts + sys_parts if p[1] > 1e-9]
        linsum = sum(p[1] for p in parts)
        scale = (c['tot'] / linsum) if linsum > 0 else 1.0

        bottom = 0.0
        for label, val, color in parts:
            h = val * scale
            ax.bar(0, h, bottom=bottom, width=0.6, color=color,
                   edgecolor='white', linewidth=1.2)
            if h > 0.03 * c['tot']:      # label only visible blocks
                ax.text(0, bottom + h / 2, f'{label}\n{val:.2f}%',
                        ha='center', va='center', fontsize=8,
                        color='white', fontweight='bold')
            bottom += h

        # boundary line between statistical and systematic families
        stat_h = c['stat_tot'] * scale if False else \
            sum(p[1] for p in stat_parts if p[1] > 1e-9) * scale
        if 0 < stat_h < bottom:
            ax.axhline(stat_h, color='k', linewidth=1.0, linestyle=':')

        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])
        ax.set_ylabel('Relative error (%)', fontweight='bold')
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis='y', alpha=0.3)
        # total annotation at the top
        ax.text(0, bottom, f'  total {c["tot"]:.2f}%\n'
                           f'  (stat {c["stat_tot"]:.2f} / sys {c["sys_tot"]:.2f})',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_ylim(0, bottom * 1.18)

    fig, (axt, axn) = plt.subplots(1, 2, figsize=(11, 6.5))

    _draw(axt, comp['t'],
          f'T_gas = {results["T_gas"]:.0f} K  [upper bound]\n'
          f'Birge R (FWHM) = {w["birge"]:.2f}')
    _draw(axn, comp['n'],
          f'N_s = {results["N_s"]:.2e} cm$^{{-3}}$\n'
          f'Birge R (area) = {a["birge"]:.2f}')

    # shared legend for the two families
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=red[1],  label='statistical  (trend-affecting: '
                                              'jitter, freq, fit/Birge)'),
               Patch(facecolor=blue[1], label='systematic  (absolute scale: '
                                              'L, A_ki, FSR)')]
    fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle('LAS error budget - red = statistical (trends), '
                 'blue = systematic (absolute)', fontweight='bold')
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    plt.show()
    return comp


def plot_absorbance_with_deviation(results, k_sigma=1.0):
    """
    Absorbance vs frequency for each kept period, with a shaded +/- k_sigma
    band from point-by-point propagated noise.

    sigma_A(nu) is built from the image equation:
        sigma_A^2 = sigma_ref^2 / I_ref^2 + sigma_m^2 / I_m^2
    with sigma_ref, sigma_m taken as the scalar white-noise floor of each
    trace (estimate_noise), broadcast across the period. This is the RANDOM
    part only - it is blind to the etalon fringe by construction, so the band
    is narrower than the true period-to-period scatter. Do not read it as the
    final error on T or N_s; that comes from the Birge combine.
    """
    frequency = results['frequency']
    i_m       = results['i_m']
    i_ref     = results['i_ref']

    # scalar noise floors, propagated point-by-point through the absorbance eq
    sig_m   = estimate_noise(i_m[np.isfinite(i_m)])
    sig_ref = estimate_noise(i_ref[np.isfinite(i_ref)])

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for r in results['fits']:
        if not r['keep']:
            continue
        sl = results['periods'][r['period']]['slice']
        f  = frequency[sl]
        Im, Iref = i_m[sl], i_ref[sl]
        ok = np.isfinite(f) & np.isfinite(Im) & np.isfinite(Iref) & (Im > 0)
        f, Im, Iref = f[ok] / 1e9, Im[ok], Iref[ok]

        alpha = np.log(Iref / Im)
        sig_A = np.sqrt(sig_ref**2 / Iref**2 + sig_m**2 / Im**2)

        order = np.argsort(f)
        f, alpha, sig_A = f[order], alpha[order], sig_A[order]
        fc = f - r['x0']    # re-centre on the line, matching plot_fits

        line = ax.plot(fc, alpha, '-', linewidth=1,
                       label=f'period {r["period"]}')[0]
        ax.fill_between(fc, alpha - k_sigma * sig_A, alpha + k_sigma * sig_A,
                        color=line.get_color(), alpha=0.25, linewidth=0)

    ax.axhline(0, color='k', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Relative frequency (GHz)', fontweight='bold')
    ax.set_ylabel('Absorbance', fontweight='bold')
    ax.set_title(f'Absorbance with +/- {k_sigma:g}sigma point-by-point band',
                 fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.show()


def sigma_A_propagated(path_on, path_off, ref_on, ref_off):
    """
    Per-frequency sigma_A(nu) by DIRECT error propagation - no MAD, no flat
    scalar. Each intensity's sigma comes from the std of its repeat sweeps at
    each time sample (what load_scope's .mean() throws away), propagated
    through:
        sigma_m^2   = sigma_on^2  + sigma_off^2
        sigma_ref^2 = sigma_refon^2 + sigma_refoff^2
        sigma_A^2   = sigma_ref^2 / I_ref^2 + sigma_m^2 / I_m^2
    Returns everything on the plasma-on time base.

    NOTE: only meaningful if each CSV holds MULTIPLE sweeps sharing the same
    `time` values, so groupby('time').std() is a real per-nu sample std. If a
    file is a single pre-averaged sweep, count == 1 everywhere and sig_A is
    NaN - fall back to the parametric (shot-noise / read-noise) model instead.
    """
    def mean_std_n(path):
        d = pd.read_csv(path)
        g = d.groupby('time')[DIODE_CHANNEL]
        return g.mean(), g.std(ddof=1), g.count()

    on_m,  on_s,  on_n  = mean_std_n(path_on)
    off_m, off_s, off_n = mean_std_n(path_off)
    ron_m, ron_s, ron_n = mean_std_n(ref_on)
    rof_m, rof_s, rof_n = mean_std_n(ref_off)

    t = on_m.index.to_numpy()
    def grid(series):
        return np.interp(t, series.index.to_numpy(), series.to_numpy())

    # std of the MEAN = std / sqrt(n), since I_on etc. are averaged sweeps
    I_on,  s_on  = on_m.to_numpy(),  (on_s  / np.sqrt(on_n)).to_numpy()
    I_off, s_off = grid(off_m),      grid(off_s  / np.sqrt(off_n))
    I_ron, s_ron = grid(ron_m),      grid(ron_s  / np.sqrt(ron_n))
    I_rof, s_rof = grid(rof_m),      grid(rof_s  / np.sqrt(rof_n))

    I_m   = I_on - I_off + BIAS_VOLTAGE
    I_ref = I_ron - I_rof + BIAS_VOLTAGE
    sig_m   = np.sqrt(s_on**2  + s_off**2)
    sig_ref = np.sqrt(s_ron**2 + s_rof**2)

    with np.errstate(divide='ignore', invalid='ignore'):
        alpha = np.log(I_ref / I_m)
        sig_A = np.sqrt(sig_ref**2 / I_ref**2 + sig_m**2 / I_m**2)

    return dict(t=t, alpha=alpha, sig_A=sig_A, I_m=I_m, I_ref=I_ref,
                sig_m=sig_m, sig_ref=sig_ref)


def plot_absorbance_propagated(results, path_on, path_off, ref_on, ref_off,
                               k_sigma=1.0):
    """
    Absorbance vs frequency, shaded with sigma_A(nu) from DIRECT propagation
    of the per-frequency repeat-sweep std (sigma_A_propagated), not the flat
    MAD floor. Overlays the per-period curves on the same frequency axis as
    plot_fits so it lines up with everything else.
    """
    prop = sigma_A_propagated(path_on, path_off, ref_on, ref_off)
    frequency = results['frequency']

    # prop arrays are on the same plasma-on time base as results['frequency']
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for r in results['fits']:
        if not r['keep']:
            continue
        sl = results['periods'][r['period']]['slice']
        f  = frequency[sl]
        a  = prop['alpha'][sl]
        sA = prop['sig_A'][sl]
        ok = np.isfinite(f) & np.isfinite(a) & np.isfinite(sA)
        f, a, sA = f[ok] / 1e9 - r['x0'], a[ok], sA[ok]
        o = np.argsort(f)
        f, a, sA = f[o], a[o], sA[o]

        line = ax.plot(f, a, '-', linewidth=1, label=f'period {r["period"]}')[0]
        ax.fill_between(f, a - k_sigma * sA, a + k_sigma * sA,
                        color=line.get_color(), alpha=0.25, linewidth=0)

    ax.axhline(0, color='k', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Relative frequency (GHz)', fontweight='bold')
    ax.set_ylabel('Absorbance', fontweight='bold')
    ax.set_title(f'Absorbance with +/- {k_sigma:g}sigma, '
                 f'DIRECT propagation of per-nu repeat std', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.show()
    return prop


# =============================================================================
# BOOTSTRAP
# =============================================================================

def bootstrap_point(results, n_boot=2000, seed=0, plot=True):
    """
    Naive point-level bootstrap in frequency space.

    Pools every kept period's (f, absorbance) points onto one axis, then for
    each draw resamples N points with replacement, fits ONE Gaussian+baseline,
    and converts to T_gas / N_s. No period structure, no Birge combine.

    WARNING: points within a period are correlated (etalon ripple, baseline,
    emission residual). Treating them as independent breaks the 1/sqrt(N)
    assumption, so this UNDERESTIMATES the error - compare against
    bootstrap_errors(mode='block'). The pooled single-Gaussian fit also gives
    a slightly different central value than analyze()'s Birge mean, because it
    weights by point density rather than per-period.
    """
    rng = np.random.default_rng(seed)
    kept = [r for r in results['fits'] if r['keep']]
    if not kept:
        raise RuntimeError('no kept periods to bootstrap')

    # pool all points onto one axis, each period re-centred on its own x0 so
    # the lines stack instead of smearing across the sweep
    f_all = np.concatenate([r['f_data'] - r['x0'] for r in kept])
    y_all = np.concatenate([r['y_data'] for r in kept])
    N = len(f_all)

    fwhm_s, area_s, T_s, N_s = [], [], [], []
    for _ in range(n_boot):
        pick = rng.integers(0, N, size=N)
        f, y = f_all[pick], y_all[pick]
        sig = estimate_noise(y[np.argsort(f)])
        if not np.isfinite(sig) or sig <= 0:
            continue
        try:
            rr = analyze_period(f * 1e9, y, sig)
        except RuntimeError:
            continue
        if rr['at_bound']:
            continue
        T, _, _ = gas_temperature(rr['fwhm'], rr['fwhm_err'])
        Ns, _, _ = metastable_density(rr['area'], rr['area_err'])
        fwhm_s.append(rr['fwhm']); area_s.append(rr['area'])
        T_s.append(T); N_s.append(Ns)

    out = {k: np.asarray(v) for k, v in
           (('fwhm', fwhm_s), ('area', area_s), ('T_gas', T_s), ('N_s', N_s))}

    def _summ(x):
        x = x[np.isfinite(x)]
        return dict(mean=float(np.mean(x)), std=float(np.std(x, ddof=1)),
                    lo=float(np.percentile(x, 2.5)),
                    hi=float(np.percentile(x, 97.5)), n=len(x))

    out['summary'] = {k: _summ(out[k]) for k in ('fwhm', 'area', 'T_gas', 'N_s')}

    print(f'\nPoint bootstrap ({n_boot} draws, {N} pooled points):')
    for k, unit in (('fwhm', 'GHz'), ('area', 'GHz'),
                    ('T_gas', 'K'), ('N_s', 'cm^-3')):
        s = out['summary'][k]
        fmt = '.3e' if k == 'N_s' else '.4g'
        print(f'  {k:6s} {s["mean"]:{fmt}} +/- {s["std"]:{fmt}} {unit}'
              f'   95% CI [{s["lo"]:{fmt}}, {s["hi"]:{fmt}}]')
    print('\n  Compare block bootstrap / Birge:')
    print(f'  fwhm   {results["fwhm"]["mean"]:.4g} +/- {results["fwhm"]["err"]:.4g} GHz')
    print(f'  area   {results["area"]["mean"]:.4g} +/- {results["area"]["err"]:.4g} GHz')

    if plot:
        _plot_bootstrap(out, results, 'point')
    return out


def bootstrap_errors(results, n_boot=12000, mode='block', seed=0, plot=True):
    """
    Bootstrap the statistical error on FWHM, area, T_gas and N_s.

    Two modes:

    'block'    Resample the KEPT PERIODS with replacement (n periods drawn
               from the n kept, with repeats), and rerun birge_combine on each
               draw. This treats each period as one exchangeable unit, so it
               respects the fact that residuals WITHIN a period are correlated
               (etalon fringing etc.) - you never break a period apart. This
               is the honest bootstrap for this data, and it's directly
               comparable to what birge_combine already reports.

    'residual' The textbook version: for each kept period, resample its
               fit residuals WITH REPLACEMENT, add them back to that period's
               fitted curve, refit, then birge_combine the refit results.
               This assumes residual points are independent, which etalon
               fringes violate - so it will generally look OVERCONFIDENT here.
               Included so you can see that gap, not because it's the one to
               quote.

    Returns a dict of arrays ('fwhm', 'area', 'T_gas', 'N_s'), each length
    n_boot, plus 'summary' with the mean/std/percentile CI of each. The scale
    terms (L, A_ki, FSR) are NOT bootstrapped - they're common-mode
    systematics, not statistical, so they stay as the separate band.
    """
    rng = np.random.default_rng(seed)
    kept = [r for r in results['fits'] if r['keep']]
    n = len(kept)
    if n == 0:
        raise RuntimeError('no kept periods to bootstrap')
    if mode == 'block' and n < 2:
        print('  WARNING: only one kept period - block bootstrap can only '
              'resample that one period, so it reports zero spread. Use '
              "mode='residual' for a single-period error.")
    if n > 2:
        print(f'Total number of periods n = {n}')
    fwhm_s, area_s, T_s, N_s = [], [], [], []

    for _ in range(n_boot):
        if mode == 'block':
            pick = rng.integers(0, n, size=n)
            draw = [kept[i] for i in pick]
            fw = [r['fwhm'] for r in draw]
            fe = [r['fwhm_err'] for r in draw]
            ar = [r['area'] for r in draw]
            ae = [r['area_err'] for r in draw]

        elif mode == 'residual':
            fw, fe, ar, ae = [], [], [], []
            for r in kept:
                f, y0 = r['f_data'], r['model'](r['f_data'], *r['popt'])
                res = r['resid']
                y_star = y0 + rng.choice(res, size=len(res), replace=True)
                try:
                    rr = analyze_period(f * 1e9, y_star, r['sigma_noise'])
                except RuntimeError:
                    continue
                fw.append(rr['fwhm']); fe.append(rr['fwhm_err'])
                ar.append(rr['area']); ae.append(rr['area_err'])
            if not fw:
                continue
        else:
            raise ValueError(f'unknown mode {mode!r}')

        w = birge_combine(fw, fe)
        a = birge_combine(ar, ae)
        T, _, _ = gas_temperature(w['mean'], w['err'])
        N, _, _ = metastable_density(a['mean'], a['err'])
        fwhm_s.append(w['mean']); area_s.append(a['mean'])
        T_s.append(T); N_s.append(N)

    out = {k: np.asarray(v) for k, v in
           (('fwhm', fwhm_s), ('area', area_s), ('T_gas', T_s), ('N_s', N_s))}

    def _summ(x):
        x = x[np.isfinite(x)]
        return dict(mean=float(np.mean(x)), std=float(np.std(x, ddof=1)),
                    lo=float(np.percentile(x, 2.5)),
                    hi=float(np.percentile(x, 97.5)), n=len(x))

    out['summary'] = {k: _summ(out[k]) for k in ('fwhm', 'area', 'T_gas', 'N_s')}

    print(f'\nBootstrap ({mode}, {n_boot} draws, {n} kept periods):')
    for k, unit in (('fwhm', 'GHz'), ('area', 'GHz'),
                    ('T_gas', 'K'), ('N_s', 'cm^-3')):
        s = out['summary'][k]
        fmt = '.3e' if k == 'N_s' else '.4g'
        print(f'  {k:6s} {s["mean"]:{fmt}} +/- {s["std"]:{fmt}} {unit}'
              f'   95% CI [{s["lo"]:{fmt}}, {s["hi"]:{fmt}}]')

    # compare to what birge/covariance already reported
    print('\n  For comparison, covariance+Birge gave:')
    print(f'  fwhm   {results["fwhm"]["mean"]:.4g} +/- {results["fwhm"]["err"]:.4g} GHz')
    print(f'  area   {results["area"]["mean"]:.4g} +/- {results["area"]["err"]:.4g} GHz')
    print(f'  T_gas  {results["T_gas"]:.4g} +/- {results["T_gas_stat"]:.4g} K (stat only)')
    print(f'  N_s    {results["N_s"]:.3e} +/- {results["N_s_stat"]:.3e} cm^-3 (stat only)')

    if plot:
        _plot_bootstrap(out, results, mode)
    return out


def _plot_bootstrap(out, results, mode):
    """Histograms of the four bootstrapped quantities."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    specs = [('fwhm', 'FWHM (GHz)', results['fwhm']['mean'], results['fwhm']['err']),
             ('area', 'Area (GHz)', results['area']['mean'], results['area']['err']),
             ('T_gas', 'T_gas (K)', results['T_gas'], results['T_gas_stat']),
             ('N_s', 'N_s (cm$^{-3}$)', results['N_s'], results['N_s_stat'])]
    for ax, (key, xlabel, ref, ref_err) in zip(axes.ravel(), specs):
        x = out[key][np.isfinite(out[key])]
        ax.hist(x, bins=40, color='C0', alpha=0.7, edgecolor='none')
        s = out['summary'][key]
        ax.axvline(s['mean'], color='C0', linewidth=1.5, label='bootstrap mean')
        ax.axvspan(s['lo'], s['hi'], color='C0', alpha=0.12, label='95% CI')
        ax.axvline(ref, color='C3', linewidth=1.5, linestyle='--',
                   label='covariance+Birge')
        ax.axvspan(ref - ref_err, ref + ref_err, color='C3', alpha=0.12)
        ax.set_xlabel(xlabel, fontweight='bold')
        ax.set_ylabel('count')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f'Bootstrap distributions ({mode})   '
                 f'blue = bootstrap, red dashed = covariance+Birge',
                 fontweight='bold')
    fig.tight_layout()
    plt.show()


def _plot_bootstrap_point(out, block=None, results=None):
    """
    Histograms of the four point-bootstrapped quantities.

    If `block` (the dict from bootstrap_errors(mode='block')) is passed, its
    95% CI is drawn as the reference band - the honest comparison for a point
    bootstrap, since the point CI ignores within-period correlation and the
    gap to the block band is exactly that missing error. Falls back to the
    covariance+Birge band from `results` if no block bootstrap is given.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    specs = [('fwhm', 'FWHM (GHz)'), ('area', 'Area (GHz)'),
             ('T_gas', 'T_gas (K)'), ('N_s', 'N_s (cm$^{-3}$)')]

    for ax, (key, xlabel) in zip(axes.ravel(), specs):
        x = out[key][np.isfinite(out[key])]
        ax.hist(x, bins=40, color='C0', alpha=0.7, edgecolor='none')
        s = out['summary'][key]
        ax.axvline(s['mean'], color='C0', linewidth=1.5, label='point mean')
        ax.axvspan(s['lo'], s['hi'], color='C0', alpha=0.12,
                   label='point 95% CI')

        if block is not None:
            b = block['summary'][key]
            ax.axvline(b['mean'], color='C3', linewidth=1.5, linestyle='--',
                       label='block mean')
            ax.axvspan(b['lo'], b['hi'], color='C3', alpha=0.12,
                       label='block 95% CI')
        elif results is not None:
            ref = {'fwhm': (results['fwhm']['mean'], results['fwhm']['err']),
                   'area': (results['area']['mean'], results['area']['err']),
                   'T_gas': (results['T_gas'], results['T_gas_stat']),
                   'N_s': (results['N_s'], results['N_s_stat'])}[key]
            ax.axvline(ref[0], color='C3', linewidth=1.5, linestyle='--',
                       label='covariance+Birge')
            ax.axvspan(ref[0] - ref[1], ref[0] + ref[1], color='C3', alpha=0.12)

        ax.set_xlabel(xlabel, fontweight='bold')
        ax.set_ylabel('count')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    ref_label = 'block bootstrap' if block is not None else 'covariance+Birge'
    fig.suptitle(f'Point bootstrap distributions   '
                 f'blue = point, red dashed = {ref_label}',
                 fontweight='bold')
    fig.tight_layout()
    plt.show()


# =============================================================================
# PHYSICS
# =============================================================================

def gas_temperature(fwhm_ghz, fwhm_ghz_err):
    """
    Doppler thermometry.

        dnu_D = (nu0/c) sqrt(8 ln2 kT / M)
        T     = M (dlambda / (K_D lambda0))^2,  dlambda = dnu lambda0^2 / c

    T is quadratic in the width, so the RELATIVE error doubles:
        u(T)/T = 2 u(dnu)/dnu
    The same factor 2 applies to any error in the frequency scale itself,
    which is why the etalon FSR uncertainty is returned as a separate term.

    The fitted width is the TOTAL width. Pressure, power and laser broadening
    and any frequency-axis error all add to it, so this is an UPPER BOUND on
    the true gas temperature.

    Returns (T, u_stat, u_sys) in K.
    """
    fwhm_hz = fwhm_ghz * 1e9
    dlam = fwhm_hz * LAMBDA_0 ** 2 / C
    T = M_AR * (dlam / (DOPPLER_CONST * LAMBDA_0)) ** 2
    T_stat = 2.0 * T * (fwhm_ghz_err / fwhm_ghz)
    T_sys = 2.0 * T * FSR_REL_ERR      # T is quadratic in the frequency scale
    return T, T_stat, T_sys


def metastable_density(area_ghz, area_ghz_err):
    """
    Integrated absorption to lower-level density.

        int k dnu = (g_u/g_l) A_ki lambda0^2 / (8 pi) * n_l
        N_s       = 8 pi g_l / (lambda0^2 g_u A_ki L) * int alpha dnu

    N_s is linear in the area and in 1/L and 1/A_ki, so all three relative
    uncertainties pass through one-for-one:

        u_r(N_s)|stat = u_r(area)
        u_r(N_s)|sys  = sqrt( u_r(L)^2 + u_r(A_ki)^2 + u_r(FSR)^2 )

    The systematic terms are multiplicative and constant across a sweep, so
    they shift the absolute scale without distorting trends in power, pressure
    or N2 fraction. They are returned separately so they can be drawn as a
    scale-factor band rather than as per-point error bars - putting them on
    every point would wrongly imply they are independent between points.

    Returns (N_s, u_stat, u_sys) in cm^-3.
    """
    const = 8.0 * np.pi * G_LOWER / (LAMBDA_0 ** 2 * G_UPPER * A_KI * L_PATH)
    n = const * area_ghz * 1e9 * 1e-6
    u_stat = const * area_ghz_err * 1e9 * 1e-6
    rel_sys = np.sqrt(L_PATH_REL_ERR ** 2 + A_KI_REL_ERR ** 2
                      + FSR_REL_ERR ** 2)
    return n, u_stat, n * rel_sys


# =============================================================================
# MAIN
# =============================================================================

def analyze(path_on, path_off, ref_on, ref_off, plot=PLOT):
    print('=' * 62)
    print('LAS SINGLE-MEASUREMENT ANALYSIS')
    print('=' * 62)
    for tag, p in [('plasma on,  laser on ', path_on),
                   ('plasma on,  laser off', path_off),
                   ('plasma off, laser on ', ref_on),
                   ('plasma off, laser off', ref_off)]:
        print(f'  {tag}  {p}')

    (mean_on, mean_off, alpha, i_m, i_emis, i_ref, r_on, r_off
     ) = build_absorbance(path_on, path_off, ref_on, ref_off)
    print(f'\nSamples: {len(alpha)}   '
          f'mean I_m = {np.nanmean(i_m):.4f} V   '
          f'mean emission = {np.nanmean(i_emis):.4f} V')

    frequency, periods = find_relative_frequency(mean_on)
    n_ok = sum(p is not None for p in periods)
    print(f'Periods: {len(periods)} found, {n_ok} calibrated')

    # Stitched fit curve, same length/order as `alpha` and `frequency`, so it
    # can be plotted directly against them. NaN wherever a period was
    # skipped, excluded, or never fit.
    alpha_fit = np.full(len(alpha), np.nan)

    print(f'\nFitting (baseline order {BASELINE_ORDER}):')
    fits = []
    for k, p in enumerate(periods):
        if p is None:
            continue
        sl = p['slice']
        f, y = frequency[sl], alpha[sl]
        mask = np.isfinite(f) & np.isfinite(y)
        if mask.sum() < MIN_POINTS_PER_PERIOD:
            print(f'  period {k}: only {mask.sum()} usable points - skipped')
            continue

        # Noise straight off the absorbance trace, so it is already
        # dimensionless and immune to emission drift (see estimate_noise).
        sigma_noise = estimate_noise(y[mask])
        if not np.isfinite(sigma_noise) or sigma_noise <= 0:
            print(f'  period {k}: invalid noise estimate - skipped')
            continue

        try:
            res = analyze_period(f[mask], y[mask], sigma_noise)
        except RuntimeError as e:
            print(f'  period {k}: fit failed - {e}')
            continue

        res['period'] = k
        res['label'] = p['label']
        res['keep'] = True
        if 'partial' in p['label'] and not INCLUDE_PARTIAL_PERIODS:
            res['keep'] = False
            print(f'  period {k}: {p["label"]} - excluded')
        elif res['at_bound']:
            res['keep'] = False
            print(f'  period {k}: fit on a parameter bound - excluded')
        elif CHI2_MAX is not None and res['chi2'] > CHI2_MAX:
            res['keep'] = False
            print(f'  period {k}: chi2 {res["chi2"]:.2f} > {CHI2_MAX} - excluded')
        else:
            print(f'  period {k} ({p["label"]:>16s}): '
                  f'FWHM {res["fwhm"]:.4f} +/- {res["fwhm_err"]:.4f} GHz   '
                  f'area {res["area"]:.4f} +/- {res["area_err"]:.4f} GHz\n'
                  f'{"":16s}chi2 {res["chi2"]:6.2f}   '
                  f'inflation {res["inflate"]:5.1f}x '
                  f'(chi2 {res["k_chi2"]:.1f}x, structure {res["k_struct"]:.1f}x)')
        fits.append(res)

        # Drop this period's fit back into the full-length alpha_fit array
        # (only for periods that passed the keep/exclude checks above).
        if res['keep']:
            idx_global = np.flatnonzero(mask) + sl.start
            idx_global = idx_global[np.argsort(f[mask])]
            alpha_fit[idx_global] = res['model'](res['f_data'], *res['popt'])

    kept = [r for r in fits if r['keep']]
    if not kept:
        raise SystemExit('No usable fits.')

    if not FRINGE_PERIODS:
        found = [find_fringe_periods(r['resid'], r['f_data']) for r in kept]
        if any(found):
            flat = np.array([p for sub in found for p in sub])
            hits = sorted(flat)[:4] if len(flat) else []
            if len(hits):
                print(f'\n  Residual periodicity at ~'
                      f'{", ".join(f"{h:.1f}" for h in np.unique(np.round(hits, 1)))}'
                      f' GHz.')
                print(f'  Line FWHM is {np.mean([r["fwhm"] for r in kept]):.2f} GHz; '
                      f'periods well above that are parasitic etalons.')
                print('  Put them in FRINGE_PERIODS to fit them out.')

    w = birge_combine([r['fwhm'] for r in kept], [r['fwhm_err'] for r in kept])
    a = birge_combine([r['area'] for r in kept], [r['area_err'] for r in kept])

    T, T_stat, T_sys = gas_temperature(w['mean'], w['err'])
    N, N_stat, N_sys = metastable_density(a['mean'], a['err'])
    T_tot = np.hypot(T_stat, T_sys)
    N_tot = np.hypot(N_stat, N_sys)

    print('\n' + '=' * 62)
    print('RESULTS')
    print('=' * 62)
    print(f'  periods used     {w["n"]}  ({w["method"]})')
    print(f'  FWHM             {w["mean"]:.4f} +/- {w["err"]:.4f} GHz'
          f'   (Birge R = {w["birge"]:.2f})' if np.isfinite(w['birge'])
          else f'  FWHM             {w["mean"]:.4f} +/- {w["err"]:.4f} GHz')
    print(f'  area             {a["mean"]:.4f} +/- {a["err"]:.4f} GHz'
          + (f'   (Birge R = {a["birge"]:.2f})' if np.isfinite(a['birge']) else ''))
    print(f'  mean inflation   {np.mean([r["inflate"] for r in kept]):.1f}x')
    print()
    print(f'  T_gas            {T:.0f} +/- {T_tot:.0f} K'
          f'   ({100 * T_tot / T:.1f} %)   [UPPER BOUND]')
    print(f'                   stat {T_stat:.0f} K'
          f'   scale {T_sys:.0f} K (FSR)')
    print(f'  N_s              {N:.3e} +/- {N_tot:.3e} cm^-3'
          f'   ({100 * N_tot / N:.1f} %)')
    print(f'                   stat {N_stat:.3e}  ({100 * N_stat / N:.1f} %)')
    print(f'                   scale {N_sys:.3e}  ({100 * N_sys / N:.1f} %)'
          f'   from L {100*L_PATH_REL_ERR:.0f} %, '
          f'A_ki {100*A_KI_REL_ERR:.0f} %, FSR {100*FSR_REL_ERR:.1f} %')
    print('\n  The scale terms are common to every point in a sweep. Show them')
    print('  as a band on the absolute axis, not as per-point error bars.')

    for name, comb in (('FWHM', w), ('area', a)):
        if np.isfinite(comb['birge']) and comb['birge'] > 2:
            print(f'\n  NOTE: Birge R = {comb["birge"]:.2f} on {name}. Replicate '
                  f'scatter exceeds the fit\n        errors - an unmodelled '
                  f'systematic is present in this measurement.')
    if not (1e8 <= N <= 1e12):
        print('\n  NOTE: N_s is outside the 1e8-1e12 cm^-3 range typical of\n'
              '        low-temperature plasmas. Re-check LAMBDA_0, G_LOWER,\n'
              '        G_UPPER and A_KI.')

    if plot:
        plot_fits(fits, T, N)

    return dict(fwhm=w, area=a,
                T_gas=T, T_gas_stat=T_stat, T_gas_sys=T_sys, T_gas_err=T_tot,
                N_s=N, N_s_stat=N_stat, N_s_sys=N_sys, N_s_err=N_tot,
                fits=fits, periods=periods,
                mean_on=mean_on, mean_off=mean_off,
                frequency=frequency, alpha=alpha, alpha_fit=alpha_fit,
                i_m=i_m, i_ref=i_ref,
                ReferenceLaser=r_on, ReferenceNoLaser=r_off)


def plot_fits(fits, T, N):
    """Absorbance with the fitted profiles, and the residuals underneath."""
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(9.5, 7), sharex=True,
                                  gridspec_kw={'height_ratios': [3, 1],
                                               'hspace': 0.07})
    for r in fits:
        fd = r['f_data'] - r['x0']
        pts = ax.plot(fd, r['y_data'], '.', markersize=3, alpha=0.45)[0]
        dense = np.linspace(fd.min(), fd.max(), 2000)
        curve = r['model'](dense + r['x0'], *r['popt'])
        tag = '' if r['keep'] else ' (excluded)'
        ax.plot(dense, curve, '-', linewidth=2, color=pts.get_color(),
                label=f'period {r["period"]}{tag}: FWHM {r["fwhm"]:.3f} GHz, '
                      f'chi2 {r["chi2"]:.1f}')
        axr.plot(fd, r['resid'], '.', markersize=3, alpha=0.5,
                 color=pts.get_color())

    axr.axhline(0, color='k', linewidth=0.8, alpha=0.6)
    axr.set_xlabel('Relative frequency (GHz)', fontweight='bold')
    axr.set_ylabel('residual', fontweight='bold')
    ax.set_ylabel('Absorbance', fontweight='bold')
    ax.set_title(f'T_gas = {T:.0f} K,  N_s = {N:.2e} cm$^{{-3}}$', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    axr.grid(True, alpha=0.3)
    plt.show()

def durbin_watson(resid):
    """
    DW = sum((r_i - r_{i-1})^2) / sum(r_i^2), residuals in FREQUENCY order.

    ~2 for white residuals, -> 0 for positively correlated ones (etalon
    fringing, baseline curvature, wrong line shape). Note this is scale-free,
    so it flags correlation even when chi2_red sits at 1.
    """
    r = np.asarray(resid, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return np.nan
    return float(np.sum(np.diff(r) ** 2) / np.sum(r ** 2))


def plot_dw_residuals(results, k_sigma=1.0, dw_min=1.0, kept_only=True):
    """
    Normalised residuals r/sigma vs relative frequency, one panel per period,
    with the Durbin-Watson statistic annotated. The +/- k_sigma band is the
    white-noise floor the fit assumed; residuals wandering coherently outside
    it while chi2_red stays near 1 is the fringing signature DW catches.

    dw_min tints a panel red when DW < dw_min (correlated residuals). None
    disables the flag. kept_only=False also shows excluded periods.
    """
    fits = [r for r in results['fits']
            if (r['keep'] or not kept_only)]
    if not fits:
        print('  plot_dw_residuals: no periods to plot')
        return {}

    n = len(fits)
    fig, axes = plt.subplots(n, 1, figsize=(9.5, 2.1 * n + 0.5),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]

    dw_by_period = {}
    for ax, r in zip(axes, fits):
        fc = r['f_data'] - r['x0']              # relative frequency, GHz
        rn = r['resid'] / r['sigma_noise']      # r / sigma
        dw = durbin_watson(r['resid'])
        dw_by_period[r['period']] = dw

        flagged = np.isfinite(dw) and dw_min is not None and dw < dw_min
        color = 'C3' if flagged else '0.35'

        ax.axhspan(-k_sigma, k_sigma, color='0.5', alpha=0.15, linewidth=0)
        ax.axhline(0, color='k', linewidth=0.8, alpha=0.6)
        ax.plot(fc, rn, '.', markersize=3, alpha=0.6, color=color)
        if flagged:
            ax.set_facecolor((1.0, 0.94, 0.94))

        tag = '' if r['keep'] else ' (excluded)'
        ax.text(0.015, 0.9,
                f'period {r["period"]}{tag}   DW = {dw:.2f}'
                + ('   <- correlated' if flagged else ''),
                transform=ax.transAxes, fontsize=9, va='top', fontweight='bold',
                color=('C3' if flagged else 'k'))
        ax.set_ylabel(r'$r/\sigma$', fontweight='bold')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Relative frequency (GHz)', fontweight='bold')
    axes[0].set_title('Residual autocorrelation (Durbin-Watson)  '
                      f'-  DW~2 white, ->0 correlated  (flag < {dw_min})',
                      fontweight='bold')
    fig.tight_layout()
    plt.show()
    return dw_by_period



if __name__ == '__main__':
    if len(sys.argv) == 5:
        paths = sys.argv[1:5]
    elif len(sys.argv) == 1:
        paths = [PATH_ON, PATH_OFF, REF_ON, REF_OFF]
    else:
        raise SystemExit(__doc__)
    results = analyze(*paths)
    dw = plot_dw_residuals(results, k_sigma=1.0, dw_min=1.0)
    # -------------------------------------------------------------------
    # Plain top-level variables for manual plotting (e.g. in Spyder's
    # variable explorer or a scratch cell) - no digging into `results`.
    # -------------------------------------------------------------------
    time_on   = results['mean_on'].index.to_numpy()
    ch1_on    = results['mean_on'][FP_CHANNEL].to_numpy()
    ch2_on    = results['mean_on'][RAMP_CHANNEL].to_numpy()
    ch3_on    = results['mean_on'][DIODE_CHANNEL].to_numpy()

    time_off  = results['mean_off'].index.to_numpy()
    ch1_off   = results['mean_off'][FP_CHANNEL].to_numpy()
    ch2_off   = results['mean_off'][RAMP_CHANNEL].to_numpy()
    ch3_off   = results['mean_off'][DIODE_CHANNEL].to_numpy()

    frequency      = results['frequency']       # Hz, aligned to time_on
    absorbance     = results['alpha']           # aligned to time_on / frequency
    absorbance_fit = results['alpha_fit']       # same alignment, NaN outside fitted periods
    i_m            = results['i_m']             # plasma-on transmitted intensity
    i_ref          = results['i_ref']           # reference I_0, on the on-shot time base

    T_gas, N_s = results['T_gas'], results['N_s']
    JustLaser = results['ReferenceLaser']

    # -------------------------------------------------------------------
    # Ripple / phase diagnostic - see the block above for what this checks.
    # -------------------------------------------------------------------
    ripple_check = diagnose_ripple_phase(results)

    # -------------------------------------------------------------------
    # Plots. The phase-correction plots are per-period; use the first
    # checked period here, or index ripple_check[k] for another.
    # -------------------------------------------------------------------
    if ripple_check:
        entry = ripple_check[0]
        plot_absorbance_correction(entry)      # corrected vs uncorrected
        corrected_fit = plot_corrected_with_fit(entry)  # corrected + Gaussian

    # Frequency-axis error decomposition, then the error budget with the
    # freq-axis split folded into the T_gas panel.
    freq_err = frequency_axis_error(results)
    plot_error_budget(results, freq_err)

    plot_absorbance_with_deviation(results, 1.0)
    prop = plot_absorbance_propagated(results, *paths)

    # -------------------------------------------------------------------
    # Bootstrap. 'block' resamples whole periods (honest for correlated
    # residuals); pass mode='residual' to see the overconfident textbook
    # version for comparison.
    # -------------------------------------------------------------------
    boot = bootstrap_errors(results, n_boot=2000, mode='block')
    boot2 = bootstrap_point(results, n_boot=50)
    _plot_bootstrap_point(boot2, block=None, results=results)