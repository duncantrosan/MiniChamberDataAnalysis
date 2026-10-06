# -*- coding: utf-8 -*-
"""
Our metastable density on top of a pure-argon reference.

Draws the reference Ar(1s5) metastable density against pressure (line with
dots) and puts our measured N_s from a master list over it as stars, at the
pressures we have. Our line is the 696.5 nm Ar 1s5 -> 2p2 line, so the 1s5
column of the reference is the one that matches.

Reference file: columns 'Pressure (mTorr)' and 'Ar(1s5) metastable (m^-3)'
(and the other 1s levels). Densities are converted from m^-3 to cm^-3 here,
the unit of the master list.

Which of our points are plotted: the reference is pure argon at 85 W delivered
power, so by default only master-list conditions with N2 <= N2_MAX_PERCENT
and delivered power within POWER_TOL_W of POWER_W. Set N2_MAX_PERCENT = None
to plot every N2 % (stars are then coloured by N2 %), and POWER_W = None to
plot every power.

Output: next to the master list, named after it:
    Output/Master/Master_Summary_vs_PureAr_1s5.png

Usage: set the paths below and run, or
    python PlotNsVsReference.py [MASTER_SUMMARY.csv] [REFERENCE.csv]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

try:
    _HERE = Path(__file__).resolve().parent
except NameError:                        # pasted into a console
    _HERE = Path.cwd()
sys.path.insert(0, str(_HERE.parent))    # DataAnalysis/, for OutputPaths
from OutputPaths import MASTER_DIR, output_file

# =============================================================================
# CONFIGURATION
# =============================================================================

# Master_Summary.csv from MasterList.py. Use any other master list (e.g. one
# made by a folder-of-runs analysis, Output/Batch_<folder>/Master_Summary.csv)
# by changing this or giving it on the command line.
MASTER_CSV = MASTER_DIR / 'Master_Summary.csv'

# The reference data, kept outside the repository: put the file here, or set
# the full path.
REFERENCE_CSV = _HERE.parents[1] / 'ReferenceData' / '1s_densities_microwave_nc.csv'
REFERENCE_COLUMN = 'Ar(1s5) metastable (m^-3)'
REFERENCE_LABEL = 'Pure Ar, 85 W (reference)'

# Which of our points go on the plot
POWER_W = 85            # delivered power, W (None: every power)
POWER_TOL_W = 5
N2_MAX_PERCENT = 0.05   # 0.05 = pure argon (None: every N2 %, coloured by N2 %)

# Master list column names
PRESSURE_COL = 'Pressure_Input_Torr'
POWER_COLS = ('Power_Measured_W', 'Power_Input_W')   # first one present is used
N2_COL = 'N2_Percent_Input'
NS_COL = 'N_s_cm3'
NS_ERR_COL = 'N_s_cm3_err'          # total error: statistical + systematic

M3_TO_CM3 = 1e-6

REF_COLOR = '#2a78d6'               # reference line
OUR_COLOR = '#eb6834'               # our stars
SEQUENTIAL = LinearSegmentedColormap.from_list(   # N2 %, light to dark, one hue
    'orange', ['#f6b99d', '#eb6834', '#8f2f0c'])


# =============================================================================
# DATA
# =============================================================================

def load_reference(path=REFERENCE_CSV, column=REFERENCE_COLUMN):
    """Reference as pressure_mTorr and density_cm3, sorted by pressure."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Reference file not found: {path}\n"
            f"Put it there, or set REFERENCE_CSV at the top of this script "
            f"(or pass its path as the second argument).")
    df = pd.read_csv(path)
    pcol = next((c for c in df.columns if c.lower().startswith('pressure')), None)
    if pcol is None or column not in df.columns:
        raise KeyError(f"{path.name} needs a 'Pressure ...' column and "
                       f"'{column}'; it has {list(df.columns)}")
    ref = pd.DataFrame({'pressure_mTorr': pd.to_numeric(df[pcol], errors='coerce'),
                        'density_cm3': pd.to_numeric(df[column], errors='coerce') * M3_TO_CM3})
    return ref.dropna().sort_values('pressure_mTorr').reset_index(drop=True)


def select_ours(master):
    """
    Master-list conditions to plot (POWER_W, POWER_TOL_W, N2_MAX_PERCENT):
    pressure_mTorr, power_W, n2_percent, N_s, N_s_err.
    """
    power, tol, n2_max = POWER_W, POWER_TOL_W, N2_MAX_PERCENT
    for col in (PRESSURE_COL, N2_COL, NS_COL):
        if col not in master.columns:
            raise KeyError(f"master list has no '{col}' column; it has "
                           f"{list(master.columns)[:15]} ...")
    pcol = next((c for c in POWER_COLS if c in master.columns), None)
    ours = pd.DataFrame({
        'pressure_mTorr': master[PRESSURE_COL] * 1000.0,
        'power_W': master[pcol] if pcol else np.nan,
        'n2_percent': master[N2_COL],
        'N_s': master[NS_COL],
        'N_s_err': master[NS_ERR_COL] if NS_ERR_COL in master.columns else np.nan,
    })
    keep = ours['N_s'].notna() & ours['pressure_mTorr'].notna()
    if n2_max is not None:
        keep &= ours['n2_percent'] <= n2_max
    if power is not None:
        keep &= (ours['power_W'] - power).abs() <= tol
    return ours[keep].sort_values(['pressure_mTorr', 'n2_percent', 'power_W']).reset_index(drop=True)


def describe_master(master):
    """What the master list holds, for when nothing matches the selection."""
    pcol = next((c for c in POWER_COLS if c in master.columns), None)
    lines = [f"  pressures (Torr): {sorted(master[PRESSURE_COL].dropna().unique())}",
             f"  N2 (%):           {sorted(master[N2_COL].dropna().unique())}"]
    if pcol:
        lines.append(f"  power ({pcol}): {master[pcol].min():.0f} - {master[pcol].max():.0f} W")
    return "\n".join(lines)


# =============================================================================
# PLOT
# =============================================================================

def plot_vs_reference(master_csv=MASTER_CSV, reference_csv=REFERENCE_CSV):
    """Draw the figure, save it next to the master list, return (figure, table)."""
    master = pd.read_csv(master_csv)
    ref = load_reference(reference_csv)
    ours = select_ours(master)

    what = ("N2 <= %g %%" % N2_MAX_PERCENT if N2_MAX_PERCENT is not None else "any N2")
    what += (", %g +/- %g W" % (POWER_W, POWER_TOL_W) if POWER_W is not None else ", any power")
    pure = (N2_MAX_PERCENT is not None and N2_MAX_PERCENT <= 0.05)
    label_what = (("pure Ar" if pure else "N$_2$ $\\leq$ %g %%" % N2_MAX_PERCENT)
                  if N2_MAX_PERCENT is not None else "any N$_2$")
    label_what += (", %g $\\pm$ %g W" % (POWER_W, POWER_TOL_W)
                   if POWER_W is not None else ", any power")
    if ours.empty:
        raise SystemExit(
            f"Nothing in {Path(master_csv).name} matches {what}.\n"
            f"The master list has:\n{describe_master(master)}\n"
            f"Change POWER_W / POWER_TOL_W / N2_MAX_PERCENT at the top of this script.")

    ours['reference_cm3'] = np.interp(ours['pressure_mTorr'], ref['pressure_mTorr'],
                                      ref['density_cm3'], left=np.nan, right=np.nan)
    ours['ratio_to_reference'] = ours['N_s'] / ours['reference_cm3']

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(ref['pressure_mTorr'], ref['density_cm3'], '-o', color=REF_COLOR,
            lw=2, ms=5, label=REFERENCE_LABEL, zorder=2)

    mixed = ours['n2_percent'].nunique() > 1
    if ours['N_s_err'].notna().any():
        ax.errorbar(ours['pressure_mTorr'], ours['N_s'], yerr=ours['N_s_err'],
                    fmt='none', ecolor='0.55', elinewidth=1.0, capsize=2.5, zorder=3)
    if mixed:
        norm = Normalize(vmin=ours['n2_percent'].min(), vmax=ours['n2_percent'].max())
        stars = ax.scatter(ours['pressure_mTorr'], ours['N_s'], marker='*', s=190,
                           c=ours['n2_percent'], cmap=SEQUENTIAL, norm=norm,
                           edgecolors='white', linewidths=0.8, zorder=4,
                           label=f"This work, {label_what} (n = {len(ours)}), colour = N$_2$ %")
        cbar = fig.colorbar(stars, ax=ax, pad=0.02)
        cbar.set_label('N$_2$ (%)')
        cbar.outline.set_visible(False)
    else:
        ax.scatter(ours['pressure_mTorr'], ours['N_s'], marker='*', s=190,
                   color=OUR_COLOR, edgecolors='white', linewidths=0.8, zorder=4,
                   label=f"This work, {label_what} (n = {len(ours)})")

    ax.set_xlabel('Pressure (mTorr)')
    ax.set_ylabel('Ar 1s$_5$ metastable density (cm$^{-3}$)')
    ax.set_ylim(bottom=0)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.grid(True, alpha=0.2, lw=0.6)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc='lower left', bbox_to_anchor=(0, 1.01), fontsize=9.5)
    fig.tight_layout()

    out = output_file(master_csv, 'vs_PureAr_1s5', '.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f"Saved {out}")
    return fig, ours


if __name__ == '__main__':
    args = sys.argv[1:]
    figure, table = plot_vs_reference(args[0] if args else MASTER_CSV,
                                      args[1] if len(args) > 1 else REFERENCE_CSV)
    show = table[['pressure_mTorr', 'power_W', 'n2_percent', 'N_s', 'N_s_err',
                  'reference_cm3', 'ratio_to_reference']]
    print("\nStars on the plot (reference interpolated to the same pressure):")
    print(show.to_string(index=False, float_format=lambda x: f'{x:.4g}'))
    plt.show()
