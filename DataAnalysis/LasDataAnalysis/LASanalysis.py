# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 10:12:39 2026

@author: dptro
"""

from pathlib import Path
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import numpy as np
import random
import re
from scipy.optimize import curve_fit
############################## Change for each data set
<<<<<<< HEAD:DataAnalysis/LASanalysis.py
MainFolder = r"C:\Users\scsha\Documents\Duncan\MiniChamberControlCode\Data\LAS_0.6Torr_SmallAdmixture_1kHzLaserFreq_CapCoupling"
=======
MainFolder = r"G:\My Drive\MiniChamberData\Data\LAS_Gas_3%"
>>>>>>> abdc6268ab6e5a1dabb291f74c3d86c839bde985:DataAnalysis/LasDataAnalysis/LASanalysis.py
#######################################################
######################################################## IMPORTANT PLEASE READ
# Code Assumes data was taken in certain channels change this if you did not 
# take the data on these channels 
Signal = 'CH1'
Perot = 'CH2'
Diode = 'CH3'
## Help Parse data structure in build_index
PATTERN = re.compile(
    r"Scope_TrialNumber_(?P<trial>\d+)_"
    r"(?P<power>[\d.]+)W_"
    r"(?P<pressure>[\d.]+)Torr_"
    r"(?P<n2_flow>[\d.]+)NitrogenFlow_"
    r"(?P<freq>[\d.]+)MHz\.csv"
)

BiasVoltage = 5 # Replace later with a measurment 


## 
def build_index(files, directory, label):
    rows = []
    for f in files:
        m = PATTERN.match(f)
        if not m:   # skips 'Plasma Off Measuremnt.csv' etc.
            continue
        row = {k: float(v) for k, v in m.groupdict().items()}
        row["trial"] = int(row["trial"])
        row["filename"] = f
        row["path"] = str(Path(directory) / f)
        row["plasma"] = label
        rows.append(row)
    return pd.DataFrame(rows)

# Compare data and subtract of background files 
def compare(row):
    df_on  = pd.read_csv(row["path_on"])
    df_off = pd.read_csv(row["path_off"])
    # whatever your actual metric is, e.g. mean voltage difference:
    return df_on["Voltage"].mean() - df_off["Voltage"].mean()

# pairs["metric"] = pairs.apply(compare, axis=1)
# results = pairs[["trial", "power", "n2_flow", "metric"]]


# Unfinished Bootstrapping methodology
def bootstrap_ci(vals, n_boot=10_000, ci=95):
    rng = np.random.default_rng(0)
    vals = np.asarray(vals)
    boots = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [(100-ci)/2, 100-(100-ci)/2])
    return pd.Series({"mean": vals.mean(), "ci_lo": lo, "ci_hi": hi, "n_trials": len(vals)})

#summary = results.groupby(["power", "n2_flow"])["metric"].apply(bootstrap_ci).unstack()

def FindRelativeFrequency(Mean, FSR=1.5e9, min_peaks=3, min_pts=1000, prom = 0.1 , plot=False):
    """
    Returns a frequency axis (Hz) the same length as the trace,
    NaN wherever calibration isn't valid, plus per-period slices.

    Now includes partial periods at the start and end of the trace,
    provided they contain at least `min_pts` sample points and
    at least `min_peaks` Fabry-Perot peaks.
    """
    Time    = Mean.index.to_numpy()
    Perot   = Mean['CH1'].to_numpy()
    Voltage = Mean['CH2'].to_numpy()

    # ---------- 1. Sawtooth falling edges ----------
    dy = np.diff(Voltage)
    rough, _ = find_peaks(-dy, height=0.6* np.max(-dy))
    if len(rough) < 2:
        raise RuntimeError("Could not find sawtooth edges")
    est_period = np.median(np.diff(rough))
    edges, _ = find_peaks(-dy, height=0.5 * np.max(-dy),
                          distance=int(0.8 * est_period))
    print(f'Rough Period Guess {rough}')
    print(f'The edges are {edges}')
    # ---------- 2. Fabry-Perot peaks (relative thresholds) ----------
    p    = Perot - np.median(Perot)
    span = np.percentile(p, 99) - np.percentile(p, 1)
    peaks, _ = find_peaks(p, prominence=prom * span, distance=40, width=2)

    # ---------- 3. Build period list, including partial ends ----------
    # Each entry is (lo, hi, label) where lo/hi are sample indices.
    # "Partial" periods bookend the full periods found between edges.
    n_full = len(edges) - 1
    period_ranges = []

    # Leading partial period (index 0 → first edge)
    if edges[0] >= min_pts:
        period_ranges.append((0, edges[0], "leading partial"))

    # Full periods
    for i in range(n_full):
        period_ranges.append((edges[i], edges[i + 1], f"full {i}"))

    # Trailing partial period (last edge → end of trace)
    if (len(Time) - edges[-1]) >= min_pts:
        period_ranges.append((edges[-1], len(Time), "trailing partial"))

    # ---------- 4. Calibrate each period ----------
    Frequency = np.full(len(Time), np.nan)
    periods   = []

    for lo, hi, label in period_ranges:
        pk = peaks[(peaks >= lo) & (peaks < hi)]
        pk = np.sort(pk)

        if len(pk) < min_peaks:
            print(f"Period '{label}': only {len(pk)} peaks — skipping")
            periods.append(None)
            continue

        PeakTimes = Time[pk]

        # Remove doublets (peaks closer than half the median spacing)
        d   = np.diff(PeakTimes)
        med = np.median(d)
        keep = np.ones(len(PeakTimes), dtype=bool)
        keep[1:][d < 0.5 * med] = False
        PeakTimes = PeakTimes[keep]

        if len(PeakTimes) < min_peaks:
            print(f"Period '{label}': too few peaks after cleaning — skipping")
            periods.append(None)
            continue

        # Integer FSR steps (tolerates missed peaks)
        d     = np.diff(PeakTimes)
        med   = np.median(d)
        steps = np.rint(d / med).astype(int)
        steps[steps < 1] = 1
        FreqPeaks = np.concatenate(([0.0], np.cumsum(steps))) * FSR

        # Interpolate only inside this period; NaN outside the peak span
        sl = slice(lo, hi)
        Frequency[sl] = np.interp(Time[sl], PeakTimes, FreqPeaks,
                                  left=np.nan, right=np.nan)

        periods.append({"slice": sl, "label": label,
                        "peak_times": PeakTimes, "peak_freqs": FreqPeaks})

    if np.all(np.isnan(Frequency)):
        raise RuntimeError("Frequency calibration failed in every period")

    # ---------- 5. Optional plots ----------
    if plot:
        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[0].plot(Time, Perot)
        ax[0].plot(Time[peaks], Perot[peaks], 'rx')
        for e in edges:
            ax[0].axvline(Time[e], color='k', alpha=0.3)
        ax[1].plot(Time, Frequency)
        ax[1].set_ylabel("Rel. frequency (Hz)")
        plt.tight_layout()
        plt.show()

        plt.figure()
        plt.plot(Time, Voltage)
        plt.xlabel('Time / s')
        plt.ylabel('Voltage / V')
        plt.tight_layout()
        plt.show()

    return Frequency, periods
    
####
# Define Guassian to use in absorption 
def gaussian(x, A, x0, sigma, offset):
    return A * np.exp(-(x - x0)**2 / (2 * sigma**2)) + offset

def gaussian_lin(x, A, x0, sig, b, m):
    return A * np.exp(-(x - x0)**2 / (2*sig**2)) + b + m*x


# Analyze and break apart periods 
def analyze_period(f, y,sigma_noise):
    """Fit Gaussian, return fit results + spectrum recentered on F_GRID."""
    f = f/(1E9)
    #### Pure Guassian Fit
    # off0 = np.median(y)
    # half = off0 + (y.max() - off0) / 2
    # above = f[y > half]
    # fwhm0 = above.max() - above.min() if len(above) > 1 else 1
    # sig0 = fwhm0 / 2.355
    # sig0 = 2
    # p0 = [y.max() - off0, f[np.argmax(y)], sig0, off0]
    
    #popt, pcov = curve_fit(gaussian, f, y, p0=p0, maxfev=5000)
    # popt, pcov = curve_fit(gaussian, f, y, p0, maxfev=5000)
    
    
    # A, x0, sig, off = popt
    # sig = abs(sig)
    
    ##### Lin + guassian fit
    # f = f/1e9 # Converts to gHz
    off0 = np.median(y)
    half = off0 + (y.max() - off0) / 2
    above = f[y > half]
    fwhm0 = above.max() - above.min() if len(above) > 1 else 1e9
    #sig0 = fwhm0 / 2.355
    sig0 = 1
    p0 = [y.max() - off0, f[np.argmax(y)], sig0, off0, 0.0]
    lo = [0,      f.min(), (f[1]-f[0]),       -np.inf, -np.inf]
    hi = [np.inf, f.max(), (f.max()-f.min()),  np.inf,  np.inf]
    
    popt, pcov = curve_fit(gaussian_lin, f, y, p0=p0, bounds=(lo, hi), maxfev=10000)
    A, x0, sig, b, m = popt
    sig = abs(sig)


    spec = np.interp(F_GRID, f - x0, y, left=np.nan, right=np.nan)
    # # Find Chi squared for goodness of fit 
    # y_fit = gaussian(F_GRID, *popt)
    
    y_fit = gaussian_lin(F_GRID, A, 0.0, sig, b,m)   # peak at 0, matches spec
    #y_model = gaussian(f, *popt)          # on original grid, no NaNs
    #res = y - y_model
    #chi2_red = np.sum(res**2) / (len(y) - len(popt))
    n_params = 5
    res = spec - y_fit
    chi2_reduced = np.sum(((spec - y_fit) / sigma_noise)**2) / (len(spec) - n_params)
    #chi2_red = np.sum(res**2) / (len(spec) - len(popt))
    
    # return {
    #     "spec": spec,
    #     'spec_x':F_GRID,
    #     "A": A, "x0": x0, "sigma": sig, "offset": off,
    #     "fwhm": 2*np.sqrt(2*np.log(2)) * sig,
    #     "area": A * sig * np.sqrt(2*np.pi),
    #     "pcov": pcov,
    #     "Fit":y_fit,
    #     "Chi^2":chi2_red
        
    

    print("off0:", off0, "half:", half)
    print("n above half:", len(above), "fwhm0:", fwhm0, "sig0:", sig0)
    print("p0:", p0)
    print("f spacing f[1]-f[0]:", f[1]-f[0])
    
    return {
        "spec": spec,
        'spec_x':F_GRID,
        "A": A, "x0": x0, "sigma": sig, "offset":m,
        "fwhm": 2*np.sqrt(2*np.log(2)) * sig,
        "area": A * sig * np.sqrt(2*np.pi),
        "pcov": pcov,
        "Fit":y_fit,
        "Chi^2":chi2_reduced
        
    }

    

##########################################################
#########Main Data file
DataOnPath = os.path.join(MainFolder,'OscopeData_Laser_True')
DataOffPath = os.path.join(MainFolder,'OscopeData_Laser_False')

# Find all the data 
csvFilesOn = [f for f in os.listdir(DataOnPath) if f.endswith('.csv')]
print('List of all found files in the True directory')
print(csvFilesOn)
# Find all the data 
csvFilesOff = [f for f in os.listdir(DataOffPath) if f.endswith('.csv')]
print('List of all found files in the False directory')
print(csvFilesOff)

############ Create merged directory to handle files 
idx_on  = build_index(csvFilesOn,  DataOnPath,  "on")
idx_off = build_index(csvFilesOff, DataOffPath, "off")

Pairs = idx_on.merge(
    idx_off,
    on=["trial", "power", "pressure", "n2_flow", "freq"],
    suffixes=("_on", "_off"),
    how="inner",          # only conditions present in both
)

# Check how many files did not merge and got left out
only_off = idx_off.merge(idx_on, on=["trial","power","n2_flow"],
                         how="left", indicator=True, suffixes=("","_x"))
print(only_off[only_off["_merge"] == "left_only"][["trial","power","n2_flow"]])



# Read in Data and Save to data frame 
PlasmaOffMeasurment = 'Plasma Off Measuremnt.csv'
DataTrue = pd.read_csv(os.path.join(DataOnPath,PlasmaOffMeasurment))
DataFalse = pd.read_csv(os.path.join(DataOffPath,PlasmaOffMeasurment))
OffMeanDf = DataFalse.groupby('time')[['CH1','CH2','CH3']].mean()
OffstdDf = DataFalse.groupby('time')[['CH1','CH2','CH3']].std()
OnstdDf = DataTrue.groupby('time')[['CH1','CH2','CH3']].std()
OnMeanDf = DataTrue.groupby('time')[['CH1','CH2','CH3']].mean()
Frequency = FindRelativeFrequency(OnMeanDf)
# Find Reference Data For Measurment
I_ref_arr = np.array(OnMeanDf[Diode] - OffMeanDf[Diode])
n_on  = DataTrue.groupby('time')[Diode].count()
n_off = DataFalse.groupby('time')[Diode].count()
I_ref_err = np.array(np.sqrt(OnstdDf[Diode]**2 / n_on + OffstdDf[Diode]**2 / n_off))

# shared axis: span it generously, trim later. Resolution ~ your etalon spacing/50
F_GRID = np.arange(-5, 5, 0.002)

i = 0
Data1 = []
# Create a random 3 graphs to plot 
PlotIndex = []
for nn in range(3): 
    PlotIndex.append(random.randint(0,len(Pairs['filename_on'])))
nn = 0


# Trial for the main loop and data handeling 
rows = []   # one entry per (file, trace... if you keep CH separate, period)
SaveFluc = []
Compare = 0
for row in Pairs.itertuples():
    DataOn  = pd.read_csv(os.path.join(DataOnPath,  row.filename_on))
    DataOff = pd.read_csv(os.path.join(DataOffPath, row.filename_off))
    MeanLaserOn  = DataOn.groupby('time')[['CH1','CH2','CH3']].mean()
    MeanLaserOff = DataOff.groupby('time')[['CH1','CH2','CH3']].mean()

    Time = MeanLaserOn.index.to_numpy()
    Frequency, periods = FindRelativeFrequency(MeanLaserOn)   # new version

    I_m = (MeanLaserOn[Diode] - MeanLaserOff[Diode] + BiasVoltage ).to_numpy()
    Abs = np.log((I_ref_arr+BiasVoltage) / I_m)        # I_ref_arr precomputed once, same time base
    PlasmaOff = np.array(MeanLaserOff[Diode])
    Fluc1 = np.std(PlasmaOff)
    if Fluc1 > Compare :
        Compare = Fluc1
        SaveFluc = PlasmaOff
        SaveTime = MeanLaserOff.index

    
    for k, p in enumerate(periods):
        if p is None:
            continue
        sl = p["slice"]
        f, y ,z  = Frequency[sl], Abs[sl] , PlasmaOff[sl]
        m = np.isfinite(f) & np.isfinite(y)
        if m.sum() < 50:
            continue
        try:
            SigmaNoise = np.std(z , ddof = 1)
            res = analyze_period(f[m], y[m],SigmaNoise)
            Fluc = np.mean(z)

        except RuntimeError:
            print(f"{row.filename_on} period {k}: fit failed, skipping")
            continue
        rows.append({
            "trial": row.trial, "power": row.power, "n2_flow": row.n2_flow,
            "period": k,
            "fwhm": res["fwhm"], "area": res["area"],
            "A": res["A"], "sigma": res["sigma"], "offset": res["offset"],
            "spec": res["spec"],"Fit": res["Fit"],"Chi^2": res['Chi^2'],"Fluctuations":Fluc
        })
        if len(rows) % 4 == 0:
            plt.figure()
            plt.plot(res['spec_x'],res['spec'],label = 'data')
            plt.plot(res['spec_x'],res['Fit'],label = 'Fit')
            plt.xlabel('Relative Frequency / GHz')
            plt.ylabel('Absorption / unitless')
            plt.title(row.filename_on)
            plt.text(
                0.05, 0.95,
                f"χ² = {res['Chi^2']:.2e}",
                transform=plt.gca().transAxes,
                va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
            )
            plt.legend()
            plt.show()

Results = pd.DataFrame(rows)

Results = pd.DataFrame(rows).dropna(subset=['Chi^2'])
Results = (
    Results.groupby(['power', 'n2_flow'], group_keys=False)
    .apply(lambda g: g.nsmallest(3, 'Chi^2'))
    .reset_index(drop=True)
)

B = [row['Chi^2'] for row in rows]

A = [row['Fluctuations'] for row in rows]

#### Plotting 

# ---------- 0. Filter by Chi^2 ----------
Filtered = Results[Results['Chi^2'] < 10]


# ---------- 1. Average replicates ----------
grouped = Filtered.groupby(['trial', 'power', 'n2_flow'])

summary = grouped['area'].agg(
    area_mean='mean',
    area_std='std',          # std of replicates
    area_sem=lambda x: x.std() / np.sqrt(len(x))   # SEM
).reset_index()

# ---------- 2. Plot ----------
powers = sorted(summary['power'].unique())
cmap   = plt.cm.viridis
colors = {p: cmap(i / (len(powers) - 1)) for i, p in enumerate(powers)}

fig, ax = plt.subplots(figsize=(7, 5))

for power, grp in summary.groupby('power'):
    grp = grp.sort_values('n2_flow')
    ax.errorbar(
        grp['n2_flow'],
        grp['area_mean'],
        yerr=grp['area_std'],      # swap for area_sem if preferred
        label=f'{int(power)} mW',
        color=colors[power],
        marker='o',
        capsize=4,
        linewidth=1.5,
        markersize=5,
    )

ax.set_xlabel('N₂ flow (sccm)')
ax.set_ylabel('Area (Hz)')
ax.set_title('Spectral area vs N₂ flow by power')
ax.legend(title='Power')
ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
plt.tight_layout()
plt.show()



## Plot guassians  ------------------------------------

# ---------- 1. Average replicates ----------
# grouped2 = Results.groupby(['trial', 'power', 'n2_flow'])

# summary2 = grouped['fit'].agg(
#     fit_mean='mean',
#     fit_std='std',          # std of replicates
# ).reset_index()

# fig, ax = plt.subplots(figsize=(7, 5))

# # Plot Nitrogen Percent 
# for power, grp in summary.groupby('n2_flow'):
#     grp = grp.sort_values('power')
#     ax.errorbar(
#         grp['n2_flow'],
#         grp['area_mean'],
#         yerr=grp['area_std'],      # swap for area_sem if preferred
#         label=f'{int(power)} mW',
#         color=colors[power],
#         marker='o',
#         capsize=4,
#         linewidth=1.5,
#         markersize=5,
#     )

# ax.set_xlabel('N₂ flow (sccm)')
# ax.set_ylabel('Area (Hz)')
# ax.set_title('Spectral area vs N₂ flow by power')
# ax.legend(title='Power')
# ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
# plt.tight_layout()
# plt.show()


# __________________________________________

plt.figure()
mask = ~np.isnan(B)
A = np.array(A)[mask]
B = np.array(B)[mask]
mask = A<0.002
A = A[mask]
B = B[mask]


slope, intercept = np.polyfit(A, B, 1)
fit = slope * A + intercept

# R^2
ss_res = np.sum((B - fit) ** 2)
ss_tot = np.sum((B - np.mean(B)) ** 2)
r2 = 1 - ss_res / ss_tot

plt.scatter(A, B, label='Data')

x_line = np.linspace(A.min(), A.max(), 100)
plt.plot(x_line, slope * x_line + intercept, color='red',
         label=f'Fit: y={slope:.3f}x+{intercept:.3f}\n$R^2$={r2:.3f}')

plt.xlabel('Fluctuations')
plt.ylabel('Chi^2')
plt.legend()
plt.show()

# for row in Pairs.itertuples():
#     file = row.filename_on
#     Data = pd.read_csv(os.path.join(DataOnPath,file))
#     MeanLaserOn = Data.groupby('time')[['CH1','CH2','CH3']].mean()
#     stdLaserOn = Data.groupby('time')[['CH1','CH2','CH3']].std()
#     Freq = FindRelativeFrequency(MeanLaserOn)
#     Data = pd.read_csv(os.path.join(DataOffPath,file))
#     MeanLaserOff = Data.groupby('time')[['CH1','CH2','CH3']].mean()
#     stdLaserOff = Data.groupby('time')[['CH1','CH2','CH3']].std()
#     I_m = np.array(MeanLaserOn[Diode] - MeanLaserOff[Diode])
#     Abs = np.log(I_ref/I_m) # Natural log form used when using absoption coefficient 
#     # alpha instead of molar form epsilon 
    
#     # Fit a Guassian to Absorption Data
#     x = np.array(Frequency)        # your frequency axis (same length as Abs)
#     y = np.array(Abs)
    
#     # Initial guesses — curve_fit is sensitive to these
#     offset0 = np.median(y)
#     A0      = y.max() - offset0
#     x00     = x[np.argmax(y)]
#     sigma0  = (x.max() - x.min()) / 10
    
#     popt, pcov = curve_fit(gaussian, x, y, p0=[A0, x00, sigma0, offset0])
#     perr = np.sqrt(np.diag(pcov))          # 1-sigma uncertainties on parameters
    
#     A, x0, sigma, offset = popt
#     sigma = abs(sigma)                     # sign is degenerate in the model
#     FWHM = 2 * np.sqrt(2 * np.log(2)) * sigma        # ≈ 2.3548 * sigma
#     FWHM_err = 2 * np.sqrt(2 * np.log(2)) * perr[2]
#     # Find area under the generated guassian 
#     area = A * sigma * np.sqrt(2 * np.pi)
    
#     # area = A*sigma*sqrt(2pi); propagate with the covariance term
#     dA, dsig = sigma * np.sqrt(2*np.pi), A * np.sqrt(2*np.pi)
#     area_err = np.sqrt(dA**2 * pcov[0,0] + dsig**2 * pcov[2,2] + 2*dA*dsig*pcov[0,2])
    
    
#     # Take Guassian data and merge into data frame 
    
#     if nn in PlotIndex:
#         plt.figure()
#         plt.plot(MeanLaserOn.index,MeanLaserOn['CH1'],label = 'Ch1')
#         plt.plot(MeanLaserOn.index,MeanLaserOn['CH2'] ,label = 'Ch2')
#         plt.plot(MeanLaserOn.index,MeanLaserOn['CH3'] ,label = 'Ch3')
#         plt.legend()
#         plt.xlabel('Time / s')
#         plt.ylabel('Voltage / V')
#         plt.show()

#         plt.figure()
#         plt.plot(Freq,Abs,label = 'Raw Data')
#         plt.plot(Frequency, gaussian(Frequency, *popt), label='Guassian fit')
#         plt.legend
#         plt.xlabel('Frequency / Hz')
#         plt.ylabel('Absorbance / unitless')
#         plt.show()

        
#         Data1 = Data['CH3']
#     nn = nn  + 1


