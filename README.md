# MiniChamberDataAnalysis

Data analysis codes for the mini chamber: laser absorption spectroscopy (LAS),
optical emission (OES) and reflection coefficient (gamma).
Needs Python 3 with numpy, pandas, scipy and matplotlib.

## Folders

| Folder | Contents |
|---|---|
| `DataAnalysis/LasDataAnalysis/` | `LASAnalysisv6.py`: full LAS analysis of a run (T_gas, N_s). `LAS_single.py`: one measurement. `plotter.py`, `AnimationTrial.py`: plots |
| `DataAnalysis/MasterList/` | `MasterList.py`: the master list. `SimpleMerge.py`: stack every dataframe in a folder |
| `DataAnalysis/OESDataAnalsis/` | emission spectra |
| `DataAnalysis/ReflectionCoefficient/` | reflection-coefficient sweeps |
| `DataAnalysis/OutputPaths.py` | where outputs go and what they are called |
| `TestData/` | example run: 0.7 Torr, 30–110 W, 1–11 % N2 |
| `Output/` | everything the scripts write |

## Output naming

Every script writes to `Output/<folder of the input file>/` and names each file
after its input file plus a tag, so each output points back to its input:

    input   D:\Data\NafisaData\LAS_0.7Torr\LAS_DataFrame_33a310dd_Laser_True_20260922_151050.csv
    output  Output\LAS_0.7Torr\LAS_DataFrame_33a310dd_Laser_True_20260922_151050_LAS.csv

`LASAnalysisv6.py` writes, for each Laser_True dataframe:

| File | Contents |
|---|---|
| `<dataframe>_LAS.csv` | the input dataframe, same rows, plus T_gas / N_s columns. This is what goes into the master list |
| `<dataframe>_LAS_FinalTable.csv` | clean table of inputs and results |
| `<dataframe>_LAS_FileResults.csv` | one row per scope file: fit summary, Birge ratios, stat/sys errors |
| `<dataframe>_LAS_PeriodFits.csv` | one row per sawtooth period (the raw fits; reused when `RUN_RAW_PROCESSING = False`) |
| `<dataframe>_LAS_RunInfo.json` | input files, settings, code version, warnings |
| `<dataframe>_LAS_Log.txt` | everything the run printed, including any error |
| `LAS_figures/` | plots, including `representative_fits.png` |

`LAS_figures/representative_fits.png` shows `N_REPRESENTATIVE_FITS` (10) random fits that went into the
results: the absorbance data, the fitted Gaussian and the residuals, labelled with power, N2 %,
sawtooth period, chi2 and FWHM. It is saved for every run, folder of runs included, and also shown on
screen when you analyse a single run. `FIT_PLOT_SEED = 0` always picks the same fits; set it to `None`
for different ones each run.

Run it from Spyder after setting `Location`, or from a terminal:
`python LASAnalysisv6.py "D:\Data\NafisaData\<run folder>" ...`

### A folder of runs

`Location` (or the folder given on the command line) can also be a folder of
runs, e.g.

    D:\Data\LAS\
        LAS_1.0Torr_...\     <- a run: its LAS_DataFrame_*.csv files + OscopeData_Laser_True/False
        LAS_0.9Torr_...\
        Older\LAS_0.6Torr_...\

Every run folder in it, at any depth, is analysed in turn:

- each run's results go to `Output/<run folder>/` as usual
- a run that fails is reported and the rest carry on; the error is in its `_LAS_Log.txt`
- `Output/Batch_<folder>/Batch_Summary.csv` lists every run: status, run_id,
  pressure, number of measurements, time, error
- `BATCH_MASTER = 'separate'` (default) writes a master list for just these runs to
  `Output/Batch_<folder>/`. Merge it into the main one with
  `python MasterList.py add Output/Batch_<folder>/Master_AllMeasurements.csv`.
  `'main'` adds the runs straight to `Output/Master/`
- `SKIP_DONE = True` skips runs already analysed, to carry on after stopping
  (Ctrl+C) or after adding new run folders. Leave it `False` after changing
  analysis settings
- figures are saved but not shown

To put `Output` somewhere else on a computer (e.g. a shared Google Drive
folder), set the environment variable `MINICHAMBER_OUTPUT` to that folder, or
edit the `OUTPUT_ROOT` line in `DataAnalysis/OutputPaths.py`.

## Master list

`Output/Master/` holds:

| File | Contents |
|---|---|
| `Master_AllMeasurements.csv` | one row per measurement (run_id + timestamp), every column |
| `Master_Combined.csv` | one row per condition (power, pressure, N2 %, total flow, frequency), repeat measurements combined, every column |
| `Master_Summary.csv` | set-points, measured inputs, gamma, T_gas, N_s, with errors. NaN where a diagnostic hasn't been run yet |

Workflow:

1. Run `LASAnalysisv6.py` on a run. Its `_LAS.csv` is added to the master
   automatically (`ADD_TO_MASTER = True`).
2. Add other dataframes (a reflection sweep, a run not yet analysed with LAS):
   `python MasterList.py add FILE ...`, or set `ACTION = 'add'` and
   `FILES_TO_ADD` in the script and run it.
3. Start over from everything in a folder: `python MasterList.py rebuild FOLDER`
   (default: the whole `Output` folder).
4. Take a run out: `python MasterList.py remove RUN_ID`.

Adding a measurement that is already there (e.g. after re-running an analysis)
updates it instead of adding a second row. Repeat measurements of the same
condition are combined: weighted mean, with the error inflated to the scatter
between measurements when that is larger (Birge ratio). For T_gas and N_s only
the statistical part averages down; the systematic part (path length, A_ki,
FSR) keeps its relative size. Details are at the top of `MasterList.py`.

To add a column to the summary, add a line to `SUMMARY_COLUMNS` in
`MasterList.py` and run it with no files: the tables are rewritten.
A new diagnostic fits in if its output keeps the input dataframe's rows
(run_id, timestamp and set-point columns) and adds its own columns, the way
`_LAS.csv` does.

`SimpleMerge.py` only stacks every dataframe in a folder into
`Output/Master/SimpleMerge_<folder>.csv`, with a `source_file` column.

## Compare with a reference

`DataAnalysis/LasDataAnalysis/PlotNsVsReference.py` plots the Ar 1s5 metastable density
of a reference (e.g. a pure-argon take at 85 W) against pressure, and puts our N_s from a
master list on top as stars. By default only our pure-argon points within 5 W of 85 W
delivered power are plotted (`N2_MAX_PERCENT`, `POWER_W`, `POWER_TOL_W` at the top of the
script). The reference file is not part of the repository: put it in `ReferenceData/`
(`1s_densities_microwave_nc.csv`) or set `REFERENCE_CSV`. Run it from Spyder, or
`python PlotNsVsReference.py [MASTER_SUMMARY.csv] [REFERENCE.csv]`. The figure is saved next to
the master list as `Master_Summary_vs_PureAr_1s5.png`, and the table of plotted points, with the
ratio to the reference at the same pressure, is printed.

## Older files

`LASanalysis.py`, `LASAnalysisV2.py`, `LASAnalysisV3.py`,
`LASDataAnalysisv4.py`, `LASDataAnalysisv5.py` and
`CalculateMetastableDensity.py` are earlier versions of the LAS pipeline and
still write into `DataAnalysis/LasDataAnalysis/`, where the old result files
also are (`MasterResults_*.csv`, `PhysicsDataFrame1TorrOptimal`,
`ReruningN21TorrTest`, ...). Re-running `LASAnalysisv6.py` on a run regenerates
its results in `Output/`.
