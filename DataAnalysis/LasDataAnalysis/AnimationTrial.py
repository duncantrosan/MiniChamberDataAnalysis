# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 13:57:21 2026

@author: dptro

Animate contour plots (N2% vs. Frequency) sweeping through Power, saved as a GIF.
 
Usage:
    python n2_freq_power_sweep.py
 
Edit the CONFIG section below to point at your CSV and choose which column
to plot as the contour color.
"""
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
 
# ------------------------- CONFIG -------------------------
CSV_PATH = r'C:\Users\dptro\Documents\Work\Python\MiniChamberControlCode\DataAnalysis\LasDataAnalysis\MasterResults_Summary_with_Tg_Ns_freq_sweep.csv'         # path to your dataframe, or set df = ... directly
Z_COL = "gamma"                 # column to use as the contour color
                                     # other good options: "fwhm_mean", "T_gas_K",
                                     # "N_s_cm3", "power" (delivered), "chi2_median"
PRESSURE_COL = "pressure"           # set to None if you don't want to fix pressure
FIXED_PRESSURE = 0.6                # only used if PRESSURE_COL is not None
OUTPUT_GIF = "Gamma_n2_freq_power_sweep.gif"
FPS = 1.5                           # frames per second in the gif
N_LEVELS = 20
CMAP = "viridis"
# ------------------------------------------------------------
 
 
def load_data():
    df = pd.read_csv(CSV_PATH)
 
    # Fix pressure if requested (your dataset sweeps power/freq/N2 at fixed 0.6 Torr)
    if PRESSURE_COL is not None and PRESSURE_COL in df.columns:
        df = df[np.isclose(df[PRESSURE_COL], FIXED_PRESSURE, atol=0.05)]
 
    # Compute true N2 percent from the flows (more reliable than any derived column)
    df = df.copy()
    df["n2_percent"] = (
        100
        * df["Nitrogen_Gas_Flow"]
        / (df["Nitrogen_Gas_Flow"] + df["Argon_Gas_Flow"])
    )
    df["n2_percent"]= df["n2_percent"].round(1)
 
    return df
 
 
def get_grid(sub, z_col):
    """Pivot one power's worth of data onto a (freq x n2_percent) grid."""
    pivot = sub.pivot_table(index="n2_flow", columns="freq", values=z_col, aggfunc="mean")
    pivot = pivot.sort_index().sort_index(axis=1)
    X, Y = np.meshgrid(pivot.columns.values, pivot.index.values)
    
    Z = pivot.values
    return X, Y, Z
 
 
def main():
    df = load_data()
 
    powers = sorted(df["power"].unique())
    vmin, vmax = df[Z_COL].min(), df[Z_COL].max()
    levels = np.linspace(vmin, vmax, N_LEVELS)
 
    fig, ax = plt.subplots(figsize=(7, 6))
    cbar = {"obj": None}  # mutable holder so update() can set it once
 
    def update(frame_idx):
        ax.clear()
        p = powers[frame_idx]
        sub = df[df["power"] == p]
        X, Y, Z = get_grid(sub, Z_COL)
        y = Y/80*100
        cf = ax.contourf(X, y, Z, levels=levels, cmap=CMAP, extend="both")
        ax.set_xlabel("Frequency / MHz")
        ax.set_ylabel("N2 percentage / %")
        ax.set_title(f"{Z_COL}  —  Power set point = {p} W")
 
        if cbar["obj"] is None:
            cbar["obj"] = fig.colorbar(cf, ax=ax, label=Z_COL)
        else:
            cbar["obj"].update_normal(cf)
 
        return cf
 
    ani = animation.FuncAnimation(
        fig, update, frames=len(powers), interval=1000 / FPS, blit=False
    )
 
    ani.save(OUTPUT_GIF, writer=animation.PillowWriter(fps=FPS))
    print(f"Saved {OUTPUT_GIF} ({len(powers)} frames, one per power level: {powers})")
 
 
if __name__ == "__main__":
    main()