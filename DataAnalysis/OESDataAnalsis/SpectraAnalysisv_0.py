"""
SpectraAnalysis.py
-------------------
Tools for loading, sweeping, and analysing OES spectra.

Works with either:
  (A) A folder tree full of per-condition CSVs, one file per plasma
      condition, named like:
          Spectra_30_00W_1_10Torr_0_00NitrogenFlow_2420_00MHz.csv
      each containing columns [repeat, wavelength, intensity, subtracted_intensity]
      -> use load_folder()
  (B) An already-merged CSV (old workflow)
      -> use load_data()

WHAT CHANGED vs the old version
--------------------------------
1. Baseline-subtracted read-in
   Every loader now takes the 'subtracted_intensity' column from each file
   and uses THAT as the working 'intensity' column that every downstream
   function (mean_spectrum, peak_intensity, integrated_intensity, ...)
   operates on. The original un-subtracted values are kept alongside as
   'raw_intensity' in case you ever need them.
   -> To go back to raw values, pass intensity_col="intensity" to
      load_folder()/load_data(), or change INTENSITY_COL below.

2. Folder sweep loader (load_folder)
   Point it at a top-level directory; it recursively finds every file
   matching the naming pattern, parses power / pressure / N2 flow /
   frequency straight out of the filename, and stitches everything into
   one tidy DataFrame -- no manual merged_data.csv assembly needed.

   ASSUMPTION: the "...NitrogenFlow..." token in the filename is stored in
   the 'Nitrogen_Percent' column, to match the rest of this pipeline
   (TrialforPeakIntensity.py etc. all key off 'Nitrogen_Percent'). If flow
   and percent are actually different quantities for you, rename that one
   line in parse_filename().

3. Peak-location comparison tools
   find_peaks_spectrum() / peak_table() / track_peak() use
   scipy.signal.find_peaks to find where peaks ACTUALLY sit in a spectrum,
   rather than just averaging intensity in a fixed window. track_peak()
   in particular follows one peak across a sweep (e.g. vs power or N2%)
   and reports how far it has shifted -- handy for line-shift / overlap
   checks instead of assuming a peak stays at a fixed wavelength.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

# ── CONFIG ────────────────────────────────────────────────────────────────
# Column in the per-file CSVs that every downstream function should treat
# as "the" intensity. Set to "intensity" to analyse raw (non-subtracted)
# data instead.
INTENSITY_COL = "subtracted_intensity"

# Only used by load_data() for the old single merged-file workflow.
MERGED_CSV = r"G:\My Drive\MiniChamberData\Data\OES_20260521_150026\ProcessedData\merged_data.csv"

# Filename pattern, e.g. Spectra_30_00W_1_10Torr_0_00NitrogenFlow_2420_00MHz.csv
FILENAME_RE = re.compile(
    r"Spectra_(?P<power>\d+_\d+)W_"
    r"(?P<pressure>\d+_\d+)Torr_"
    r"(?P<n2flow>\d+_\d+)NitrogenFlow_"
    r"(?P<freq>\d+_\d+)MHz",
    re.IGNORECASE,
)
# ── END CONFIG ───────────────────────────────────────────────────────────


# =============================================================================
# 1.  LOAD  (folder sweep + legacy single-file)
# =============================================================================

def _num(token: str) -> float:
    """'30_00' -> 30.00"""
    return float(token.replace("_", "."))


def parse_filename(path) -> dict:
    """
    Pull power / pressure / nitrogen flow / frequency out of a filename like
    'Spectra_30_00W_1_10Torr_0_00NitrogenFlow_2420_00MHz.csv'.
    Raises ValueError if the filename doesn't match the expected pattern.
    """
    name = Path(path).name
    m = FILENAME_RE.search(name)
    if not m:
        raise ValueError(f"Filename does not match expected pattern: {name}")
    return {
        "power": _num(m["power"]),
        "pressure": _num(m["pressure"]),
        "Nitrogen_Percent": _num(m["n2flow"]),
        "frequency": _num(m["freq"]),
    }


def load_one_file(path, intensity_col: str = INTENSITY_COL) -> pd.DataFrame:
    """
    Load a single per-condition CSV and tag every row with the plasma
    condition parsed from its filename.

    Expects columns: repeat, wavelength, intensity, subtracted_intensity.
    """
    path = Path(path)
    raw = pd.read_csv(path)

    col = intensity_col
    if col not in raw.columns:
        print(f"[WARNING] '{col}' not found in {path.name}, falling back to 'intensity'")
        col = "intensity"

    meta = parse_filename(path)

    out = pd.DataFrame({
        "repeat": raw["repeat"],
        "wavelength": raw["wavelength"],
        "intensity": raw[col],                 # working column used by everything downstream
        "raw_intensity": raw["intensity"],
        "power": meta["power"],
        "pressure": meta["pressure"],
        "Nitrogen_Percent": meta["Nitrogen_Percent"],
        "frequency": meta["frequency"],
        "source_file": path.name,
    })
    return out


def load_folder(root_dir, pattern: str = "Spectra_*.csv",
                 intensity_col: str = INTENSITY_COL,
                 recursive: bool = True) -> pd.DataFrame:
    """
    Walk `root_dir` (and every subfolder, if recursive=True) for files
    matching `pattern`, parse each filename for its plasma condition, and
    concatenate everything into one tidy DataFrame -- same shape as the
    old merged_data.csv, built straight from disk.

    Parameters
    ----------
    root_dir      : top-level folder to search
    pattern       : glob pattern for the per-condition CSVs
    intensity_col : which column to use as the working 'intensity'
                    (default: 'subtracted_intensity')
    recursive     : search subfolders too (default True)
    """
    root = Path(root_dir)
    files = sorted(root.rglob(pattern) if recursive else root.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' found under {root}")

    frames, skipped = [], []
    for f in files:
        try:
            frames.append(load_one_file(f, intensity_col=intensity_col))
        except ValueError as e:
            skipped.append(str(e))

    if skipped:
        print(f"[WARNING] Skipped {len(skipped)} file(s) that didn't match the naming pattern:")
        for s in skipped:
            print(f"    {s}")

    if not frames:
        raise ValueError("No files could be parsed -- check FILENAME_RE against your filenames.")

    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(df):,} rows from {len(frames)} file(s) under '{root}' "
          f"(using '{intensity_col}' as intensity)")
    print(f"Powers   : {sorted(df['power'].unique())}")
    print(f"Pressures: {sorted(df['pressure'].unique())}")
    print(f"N2 %     : {sorted(df['Nitrogen_Percent'].unique())}")
    print(f"Freqs    : {sorted(df['frequency'].unique())}")
    print(f"Repeats  : {sorted(df['repeat'].unique())}")
    return df


def load_data(path: str = MERGED_CSV, intensity_col: str = INTENSITY_COL) -> pd.DataFrame:
    """
    Backwards-compatible loader for an already-merged CSV (old workflow).
    If the file has an `intensity_col` column (default 'subtracted_intensity'),
    it is swapped in as the working 'intensity' column and the original is
    kept as 'raw_intensity'.
    """
    df = pd.read_csv(path)
    if intensity_col in df.columns and intensity_col != "intensity":
        df = df.copy()
        df["raw_intensity"] = df["intensity"]
        df["intensity"] = df[intensity_col]
    print(f"Loaded {len(df):,} rows from '{path}'")
    print(f"Columns : {df.columns.tolist()}")
    if "power" in df.columns:
        print(f"Powers  : {sorted(df['power'].unique())}")
    if "pressure" in df.columns:
        print(f"Pressures: {sorted(df['pressure'].unique())}")
    if "Nitrogen_Percent" in df.columns:
        print(f"N2 %    : {sorted(df['Nitrogen_Percent'].unique())}")
    if "repeat" in df.columns:
        print(f"Repeats : {sorted(df['repeat'].unique())}")
    return df


# =============================================================================
# 2.  QUERY – get the raw rows for one condition
# =============================================================================

def get_spectra(df: pd.DataFrame,
                power: float,
                pressure: float,
                nitrogen_pct: float,
                frequency: float = 2420,
                repeat: int | None = None) -> pd.DataFrame:
    """
    Return the spectra rows matching the given plasma condition.

    Parameters
    ----------
    df           : the full merged DataFrame
    power        : set-point power in Watts  (e.g. 30)
    pressure     : pressure in Torr          (e.g. 1.3)
    nitrogen_pct : nitrogen percentage       (e.g. 0.0)
    frequency    : MHz                       (default 2420)
    repeat       : 0, 1, or 2 – omit to get all three repeats

    Returns
    -------
    DataFrame with columns [repeat, wavelength, intensity, ...]
    sorted by repeat then wavelength.
    """
    mask = (
        (df["power"]            == power)        &
        (df["pressure"]         == pressure)      &
        (df["Nitrogen_Percent"] == nitrogen_pct)  &
        (df["frequency"]        == frequency)
    )
    if repeat is not None:
        mask &= (df["repeat"] == repeat)

    result = df[mask].sort_values(["repeat", "wavelength"]).reset_index(drop=True)

    if result.empty:
        print(f"[WARNING] No data found for "
              f"power={power}W, pressure={pressure}Torr, "
              f"N2={nitrogen_pct}%, freq={frequency}MHz, repeat={repeat}")
    else:
        print(f"Found {len(result):,} rows  "
              f"({result['repeat'].nunique()} repeat(s), "
              f"{result['wavelength'].nunique()} wavelength points each)")
    return result


# =============================================================================
# 3.  MEAN SPECTRUM – average intensity across the three repeats
# =============================================================================

def mean_spectrum(df: pd.DataFrame,
                  power: float,
                  pressure: float,
                  nitrogen_pct: float,
                  frequency: float = 2420) -> pd.DataFrame:
    """
    Return a single mean spectrum (one row per wavelength) for the given
    condition, with columns:
        wavelength | intensity_mean | intensity_std | intensity_sem

    Parameters
    ----------
    Same as get_spectra (no repeat argument – averages all repeats).
    """
    raw = get_spectra(df, power, pressure, nitrogen_pct, frequency)
    if raw.empty:
        return raw

    agg = (
        raw.groupby("wavelength", sort=True)["intensity"]
        .agg(intensity_mean="mean",
             intensity_std="std",
             intensity_sem=lambda x: x.std() / np.sqrt(len(x)))
        .reset_index()
    )
    return agg


# =============================================================================
# 4.  COMPARE – build a tidy table of mean spectra for a sweep
# =============================================================================

def sweep_spectra(df: pd.DataFrame,
                  vary: str,
                  fixed: dict) -> pd.DataFrame:
    """
    Build a DataFrame of mean spectra across a sweep of one parameter.

    Parameters
    ----------
    vary  : column to sweep, e.g. 'power', 'Nitrogen_Percent', 'pressure'
    fixed : dict of fixed parameters, e.g.
            {'pressure': 1.3, 'Nitrogen_Percent': 0.0, 'frequency': 2420}

    Returns
    -------
    DataFrame with columns [<vary>, wavelength, intensity_mean, intensity_std]
    """
    results = []
    for val in sorted(df[vary].unique()):
        kwargs = {**fixed, vary: val}
        spec = mean_spectrum(
            df,
            power         = kwargs.get("power", fixed.get("power")),
            pressure      = kwargs.get("pressure", fixed.get("pressure")),
            nitrogen_pct  = kwargs.get("Nitrogen_Percent",
                                       fixed.get("Nitrogen_Percent", 0.0)),
            frequency     = kwargs.get("frequency", fixed.get("frequency", 2420)),
        )
        if not spec.empty:
            spec[vary] = val
            results.append(spec)

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


# BaselineCorrect Spectra (only needed now if you want a SECOND, different
# baseline on top of the file's own subtracted_intensity)
def baseline_correct_spectra(df, baseline_wavelengths):
    """
    Apply baseline correction to spectra in a dataframe.

    Parameters:
        df (pd.DataFrame): DataFrame with columns including 'wavelength' and 'intensity'
                           and 'spectra_file'.
        baseline_wavelengths (list): Wavelength values to use as baseline anchor points.

    Returns:
        pd.DataFrame: New dataframe with baseline-corrected intensity values.
                      Original df is not modified.
    """
    df_corrected = df.copy()

    wavelength_col = 'wavelength'
    intensity_col = 'intensity'

    def correct_group(group):
        wavelengths = group[wavelength_col].values
        intensities = group[intensity_col].values

        spectrum_interp = interp1d(
            wavelengths, intensities,
            kind='linear',
            bounds_error=False,
            fill_value='extrapolate'
        )
        baseline_intensities = spectrum_interp(baseline_wavelengths)

        baseline_fn = interp1d(
            sorted(baseline_wavelengths),
            [baseline_intensities[i] for i in np.argsort(baseline_wavelengths)],
            kind='linear',
            bounds_error=False,
            fill_value='extrapolate'
        )

        baseline = baseline_fn(wavelengths)
        group = group.copy()
        group[intensity_col] = intensities - baseline
        return group

    group_cols = ['run_id', 'spectra_file']  # adjust if needed
    df_corrected = (
        df_corrected
        .groupby(group_cols, group_keys=False)
        .apply(correct_group)
    )

    return df_corrected


# =============================================================================
# 5.  PEAK INTENSITY – track one emission line across conditions (fixed window)
# =============================================================================

def peak_intensity(df, wavelength_nm, tolerance_nm=1.0):
    window = df[df["wavelength"].between(wavelength_nm - tolerance_nm,
                                         wavelength_nm + tolerance_nm)]

    per_repeat = (
        window
        .groupby(["power", "pressure", "Nitrogen_Percent", "frequency", "repeat"],
                 sort=True)["intensity"]
        .mean()
        .reset_index(name="intensity")
    )

    agg = (
        per_repeat
        .groupby(["power", "pressure", "Nitrogen_Percent", "frequency"], sort=True)
        ["intensity"]
        .agg(intensity_mean="mean", intensity_std="std", n_repeats="count")
        .reset_index()
    )
    return agg


def integrated_intensity(
    df: pd.DataFrame,
    line: str | None = None,
    wl_min: float | None = None,
    wl_max: float | None = None,
    subtract_baseline: bool = True,
) -> pd.DataFrame:
    """
    For every unique plasma condition, return the integrated (trapezoid) intensity
    in a wavelength window.

    Call with EITHER a named line OR explicit bounds — not both.

    Parameters
    ----------
    df                : full merged DataFrame from load_data()/load_folder()
    line              : named emission line string from EMISSION_LINES, e.g. "Ar_750"
    wl_min, wl_max    : manual integration bounds in nm (used if line is None)
    subtract_baseline : if True, subtract a linear baseline drawn between the
                        mean intensity at the two edges of the window before
                        integrating (simple but effective for continuum removal)

    Returns
    -------
    DataFrame with one row per plasma condition:
        power | pressure | Nitrogen_Percent | frequency |
        integrated_intensity | integrated_std | n_repeats |
        wl_min | wl_max | line_name

    Examples
    --------
    ar750 = integrated_intensity(df, line="Ar_750")
    custom = integrated_intensity(df, wl_min=748.0, wl_max=753.0)
    raw = integrated_intensity(df, line="N2_337", subtract_baseline=False)
    """

    EMISSION_LINES = {
        # Argon
        "Ar_696"  : (695.0,  697.5),
        "Ar_706"  : (705.0,  707.5),
        "Ar_727"  : (726.0,  728.5),
        "Ar_738"  : (737.0,  739.5),
        "Ar_750"  : (748.5,  752.0),
        "Ar_751"  : (750.5,  753.0),
        "Ar_763"  : (762.0,  765.0),
        "Ar_772"  : (771.0,  773.5),
        "Ar_794"  : (793.0,  795.5),
        "Ar_801"  : (800.0,  802.5),
        "Ar_811"  : (810.0,  812.5),
        "Ar_826"  : (825.0,  827.5),
        "Ar_840"  : (839.0,  841.5),
        "Ar_842"  : (841.0,  843.5),
        "Ar_912"  : (911.0,  913.5),

        # Nitrogen
        "N2_337"  : (335.5,  338.5),
        "N2_357"  : (356.0,  358.5),
        "N2_380"  : (379.0,  381.5),
        "N2p_391" : (390.0,  392.5),

        # Other common plasma lines
        "H_alpha" : (655.5,  657.5),
        "H_beta"  : (485.5,  487.0),
        "O_777"   : (776.0,  778.5),
        "O_844"   : (843.0,  845.5),
    }

    if line is not None and (wl_min is not None or wl_max is not None):
        raise ValueError("Provide either `line` OR `wl_min`/`wl_max`, not both.")

    if line is not None:
        if line not in EMISSION_LINES:
            raise KeyError(
                f"Unknown line '{line}'.\n"
                f"Available lines: {list(EMISSION_LINES.keys())}"
            )
        wl_min, wl_max = EMISSION_LINES[line]
        line_name = line
    else:
        if wl_min is None or wl_max is None:
            raise ValueError("Provide both `wl_min` and `wl_max`, or a `line` name.")
        line_name = f"{wl_min:.1f}-{wl_max:.1f}nm"

    window = df[df["wavelength"].between(wl_min, wl_max)].copy()

    if window.empty:
        print(f"[WARNING] No data in wavelength window [{wl_min}, {wl_max}] nm")
        return pd.DataFrame()

    group_cols = ["power", "pressure", "Nitrogen_Percent", "frequency", "repeat"]

    def trapz_integrate(grp: pd.DataFrame) -> float:
        grp = grp.sort_values("wavelength")
        wl  = grp["wavelength"].values
        I   = grp["intensity"].values

        if subtract_baseline and len(wl) >= 2:
            baseline = np.interp(wl, [wl[0], wl[-1]], [I[0], I[-1]])
            I = I - baseline
            I = np.clip(I, 0, None)

        return float(np.trapz(I, wl))

    per_repeat = (
        window
        .groupby(group_cols, sort=True)
        .apply(trapz_integrate, include_groups=False)
        .reset_index(name="integrated_intensity")
    )

    condition_cols = ["power", "pressure", "Nitrogen_Percent", "frequency"]
    agg = (
        per_repeat
        .groupby(condition_cols, sort=True)["integrated_intensity"]
        .agg(
            integrated_intensity = "mean",
            integrated_std       = "std",
            n_repeats            = "count",
        )
        .reset_index()
    )

    agg["wl_min"]     = wl_min
    agg["wl_max"]     = wl_max
    agg["line_name"]  = line_name

    return agg


# =============================================================================
# 6.  PEAK LOCATION TOOLS – find & compare where peaks actually sit
# =============================================================================

def find_peaks_spectrum(spec: pd.DataFrame,
                         prominence: float | None = None,
                         height: float | None = None,
                         distance: int | None = None,
                         wl_range: tuple | None = None) -> pd.DataFrame:
    """
    Run scipy.signal.find_peaks on a mean spectrum (the output of
    mean_spectrum()) and return the ACTUAL peak locations found, instead of
    just the intensity averaged in a fixed window.

    Parameters
    ----------
    spec       : output of mean_spectrum() -- needs 'wavelength' and
                 'intensity_mean' columns
    prominence : minimum prominence for a peak to count (recommended --
                 start around 5-10% of your spectrum's max intensity)
    height     : minimum absolute intensity for a peak to count
    distance   : minimum number of samples between neighbouring peaks
    wl_range   : optional (wl_min, wl_max) to restrict the search window

    Returns
    -------
    DataFrame: wavelength | intensity | prominence | height
    (one row per detected peak, sorted by wavelength)
    """
    s = spec.copy()
    if wl_range is not None:
        s = s[s["wavelength"].between(wl_range[0], wl_range[1])]
    s = s.sort_values("wavelength").reset_index(drop=True)

    if s.empty:
        return pd.DataFrame(columns=["wavelength", "intensity", "prominence", "height"])

    idx, props = find_peaks(s["intensity_mean"].values,
                             prominence=prominence, height=height, distance=distance)

    result = pd.DataFrame({
        "wavelength": s["wavelength"].values[idx],
        "intensity": s["intensity_mean"].values[idx],
    })
    result["prominence"] = props.get("prominences", np.full(len(idx), np.nan))
    result["height"] = props.get("peak_heights", np.full(len(idx), np.nan))
    return result.sort_values("wavelength").reset_index(drop=True)


def peak_table(df: pd.DataFrame, vary: str, fixed: dict,
                prominence: float | None = None,
                height: float | None = None,
                distance: int | None = None,
                wl_range: tuple | None = None) -> pd.DataFrame:
    """
    Detect peaks in the mean spectrum at every value of `vary` (other
    conditions held at `fixed`), and stack the results into one long table.
    Useful for eyeballing how many peaks show up, and exactly where they
    sit, as a parameter is swept.

    Returns
    -------
    DataFrame: <vary> | wavelength | intensity | prominence | height
    """
    rows = []
    for val in sorted(df[vary].unique()):
        kwargs = {**fixed, vary: val}
        spec = mean_spectrum(
            df,
            power        = kwargs.get("power", fixed.get("power")),
            pressure     = kwargs.get("pressure", fixed.get("pressure")),
            nitrogen_pct = kwargs.get("Nitrogen_Percent", fixed.get("Nitrogen_Percent", 0.0)),
            frequency    = kwargs.get("frequency", fixed.get("frequency", 2420)),
        )
        if spec.empty:
            continue
        pk = find_peaks_spectrum(spec, prominence=prominence, height=height,
                                  distance=distance, wl_range=wl_range)
        if pk.empty:
            continue
        pk[vary] = val
        rows.append(pk)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def track_peak(df: pd.DataFrame, approx_wavelength: float, vary: str, fixed: dict,
               tolerance: float = 2.0,
               prominence: float | None = None,
               height: float | None = None,
               distance: int | None = None) -> pd.DataFrame:
    """
    Follow ONE specific peak (the detected peak nearest `approx_wavelength`)
    across a sweep of `vary`, reporting where it ACTUALLY sits each time.
    Handy for checking peak shift / drift / overlap instead of assuming a
    line stays at a fixed wavelength.

    Returns
    -------
    DataFrame: <vary> | wavelength | intensity | prominence | shift_nm
    (shift_nm = measured wavelength - approx_wavelength; empty rows for
    conditions where nothing was found within `tolerance` are dropped)
    """
    search_pad = tolerance + 3.0
    wl_range = (approx_wavelength - search_pad, approx_wavelength + search_pad)
    pk = peak_table(df, vary, fixed, prominence=prominence, height=height,
                     distance=distance, wl_range=wl_range)
    if pk.empty:
        print(f"[WARNING] No peaks found near {approx_wavelength} nm anywhere in this sweep.")
        return pk

    rows = []
    for val, grp in pk.groupby(vary):
        grp = grp.copy()
        grp["dist"] = (grp["wavelength"] - approx_wavelength).abs()
        nearest = grp[grp["dist"] <= tolerance].sort_values("dist")
        if not nearest.empty:
            rows.append(nearest.iloc[0])

    if not rows:
        print(f"[WARNING] Peaks were found nearby but none within tolerance={tolerance} nm.")
        return pd.DataFrame()

    out = pd.DataFrame(rows).drop(columns="dist").reset_index(drop=True)
    out["shift_nm"] = out["wavelength"] - approx_wavelength
    return out.sort_values(vary).reset_index(drop=True)


def plot_peak_shift(track_df: pd.DataFrame, vary: str, ax=None):
    """Plot a tracked peak's measured position vs the swept parameter."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(track_df[vary], track_df["wavelength"], "o-")
    ax.set_xlabel(vary)
    ax.set_ylabel("Peak wavelength / nm")
    ax.set_title(f"Peak position vs {vary}")
    return ax


# =============================================================================
# 7.  PLOTTING HELPERS
# =============================================================================

def plot_spectrum(spec: pd.DataFrame,
                  label: str = "",
                  ax=None,
                  show_std: bool = True,
                  mark_peaks: bool = False,
                  peak_kwargs: dict | None = None):
    """
    Plot a mean spectrum (output of mean_spectrum()).
    Shaded band shows +/- 1 std if show_std=True.
    If mark_peaks=True, also runs find_peaks_spectrum() and marks the
    detected peaks (pass find_peaks kwargs via peak_kwargs, e.g.
    {'prominence': 500}).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(spec["wavelength"], spec["intensity_mean"], label=label, lw=1)
    if show_std and "intensity_std" in spec.columns:
        ax.fill_between(
            spec["wavelength"],
            spec["intensity_mean"] - spec["intensity_std"],
            spec["intensity_mean"] + spec["intensity_std"],
            alpha=0.2
        )
    if mark_peaks:
        pk = find_peaks_spectrum(spec, **(peak_kwargs or {}))
        ax.plot(pk["wavelength"], pk["intensity"], "rx", ms=8, mew=2, label="_nolegend_")
        for _, row in pk.iterrows():
            ax.annotate(f"{row['wavelength']:.1f}",
                        (row["wavelength"], row["intensity"]),
                        textcoords="offset points", xytext=(0, 6), fontsize=7, ha="center")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Intensity (counts)")
    ax.legend(fontsize=8)
    return ax


def plot_sweep(sweep_df: pd.DataFrame, vary: str, ax=None):
    """
    Plot a stack of mean spectra from sweep_spectra(), one line per value
    of `vary`.  Colour-coded by the sweep variable.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 5))

    vals = sorted(sweep_df[vary].unique())
    cmap = plt.get_cmap("viridis", len(vals))

    for i, val in enumerate(vals):
        sub = sweep_df[sweep_df[vary] == val]
        ax.plot(sub["wavelength"], sub["intensity_mean"],
                label=f"{vary}={val}", color=cmap(i), lw=0.9)

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Mean intensity (counts)")
    ax.set_title(f"Spectra vs {vary}")
    ax.legend(fontsize=7, ncol=2)
    return ax


# =============================================================================
# 8.  EXAMPLES  (uncomment to run)
# =============================================================================

if __name__ == "__main__":

    # ── A. Sweep-load every Spectra_*.csv under a top folder ───────────────
    df = load_folder(r'G:\My Drive\MiniChamberData\Data\OES_Pressure[1.0]_20260626_183407')

    # ── B. Mean spectrum for one condition (now built from subtracted_intensity) ─
    mean = mean_spectrum(df, power=30, pressure=1.3, nitrogen_pct=0.0)
    print("\nMean spectrum (first 5 wavelengths):")
    print(mean.head())

    # ── C. Find actual peak locations in that spectrum ─────────────────────
    peaks = find_peaks_spectrum(mean, prominence=200)
    print("\nDetected peaks:")
    print(peaks)

    # ── D. Track the Ar ~811 nm peak's real position across a power sweep ──
    tracked = track_peak(df, approx_wavelength=811.0, vary="power",
                          fixed={"pressure": 1.3, "Nitrogen_Percent": 0.0},
                          tolerance=1.5, prominence=200)
    print("\nTracked peak position vs power:")
    print(tracked)

    ax = plot_spectrum(mean, label="30 W, 0% N2", mark_peaks=True, peak_kwargs={"prominence": 200})
    ax.set_title("Mean OES spectrum (subtracted_intensity) - 30 W, 1.3 Torr, 0% N2")
    plt.tight_layout()
    plt.show()
    
   