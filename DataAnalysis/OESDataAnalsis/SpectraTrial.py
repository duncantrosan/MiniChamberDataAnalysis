# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 12:32:16 2026

@author: dptro
"""

# -*- coding: utf-8 -*-
"""
example_sweep_and_peaks.py
---------------------------
Example of the new workflow:
  1. Point load_folder() at the top-level folder that contains all your
     Spectra_*.csv files (any number of subfolders deep) and it reads them
     all in, using each file's `subtracted_intensity` column.
  2. Use peak_table() / track_peak() to compare where peaks actually sit
     across a sweep, instead of just reading intensity at a fixed
     wavelength.

This mirrors TrialforPeakIntensity.py but on the new loader/tools.
"""

import SpectraAnalysis as SpecA
import matplotlib.pyplot as plt
import os
# ── 1. Load every Spectra_*.csv found anywhere under this folder ───────────
ROOT = r"G:\My Drive\MiniChamberData\Data\OES_baseline_files"        # <- change to your top folder
DataFolders = os.listdir(ROOT)
ReadInFolders=[]
for n in DataFolders:
    Temp = os.path.join(n,'Spectra')
    ReadInFolders.append(Temp)


df = SpecA.load_folder(ROOT)                       # recursive by default



n2_pcts = sorted(df["Nitrogen_Percent"].unique())
powers  = sorted(df["power"].unique())

# ── 2. Fixed-window line ratio, same as before (706.7 / 811 nm) ────────────
NumeratorLine, DenominatorLine = 706.7, 811.0

num = SpecA.peak_intensity(df, NumeratorLine)
den = SpecA.peak_intensity(df, DenominatorLine)

mean_num = num.pivot_table(index="power", columns="Nitrogen_Percent",
                            values="intensity_mean", aggfunc="mean").reindex(index=powers, columns=n2_pcts)
mean_den = den.pivot_table(index="power", columns="Nitrogen_Percent",
                            values="intensity_mean", aggfunc="mean").reindex(index=powers, columns=n2_pcts)
std_num = num.pivot_table(index="power", columns="Nitrogen_Percent",
                           values="intensity_std", aggfunc="mean").reindex(index=powers, columns=n2_pcts)
std_den = den.pivot_table(index="power", columns="Nitrogen_Percent",
                           values="intensity_std", aggfunc="mean").reindex(index=powers, columns=n2_pcts)

ratio = mean_num / mean_den
ratio_err = ratio.abs() * ((std_num / mean_num) ** 2 + (std_den / mean_den) ** 2) ** 0.5

fig, ax = plt.subplots(figsize=(10, 5))
for pwr in powers:
    y, yerr, x = ratio.loc[pwr].values, ratio_err.loc[pwr].values, ratio.columns.values
    ax.errorbar(x, y, yerr=yerr, marker='o', capsize=4, label=f"{pwr} W")
ax.set_title(f"{NumeratorLine}/{DenominatorLine} nm Line Ratio")
ax.set_xlabel("Nitrogen %")
ax.set_ylabel("Intensity Ratio / a.u.")
ax.legend(title="Power / W")
plt.tight_layout()

# ── 3. NEW: compare actual peak LOCATIONS across the N2% sweep ─────────────
# instead of assuming the Ar line sits exactly at 811.0 nm, find where its
# peak actually is at every N2%, at a fixed power/pressure.
tracked_811 = SpecA.track_peak(
    df,
    approx_wavelength=811.0,
    vary="Nitrogen_Percent",
    fixed={"power": powers[-1], "pressure": sorted(df["pressure"].unique())[0]},
    tolerance=1.5,
    prominence=200,     # tune this to your noise floor
)
print("\n811 nm peak position vs Nitrogen %:")
print(tracked_811)

SpecA.plot_peak_shift(tracked_811, vary="Nitrogen_Percent")
plt.tight_layout()

# ── 4. NEW: full peak inventory for one condition, with locations marked ───
mean_spec = SpecA.mean_spectrum(df, power=powers[-1],
                                 pressure=sorted(df["pressure"].unique())[0],
                                 nitrogen_pct=n2_pcts[0])
ax = SpecA.plot_spectrum(mean_spec, label=f"{powers[-1]} W, {n2_pcts[0]}% N2",
                          mark_peaks=True, peak_kwargs={"prominence": 200})
ax.set_xlim(600, 950)
plt.tight_layout()

plt.show()