# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 14:19:44 2026

@author: dptro
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

df = pd.read_csv("ReruningN21TorrTest")

# best-matched (lowest gamma) frequency for each pressure/power/N2% condition Group data
best = df.dropna(subset=['Gamma']).loc[
    df.groupby(['Pressure_Input_Torr', 'Power_Input_W', 'N2_Percent_Input'])['Gamma'].idxmin()]




# Gas Temperature plot
ERR_MAX_K = 96  # points with T_gas_K_err above this are treated as unreliable
MinTemp = 100
# Fin
for p in sorted(best['Pressure_Input_Torr'].dropna().unique()):
    d = best[best['Pressure_Input_Torr'] == p].dropna(subset=['Power_Input_W', 'N2_Percent_Input', 'N_s_cm3'])
    if len(d) < 4:
        continue
    xi = np.linspace(d['Power_Input_W'].min(), d['Power_Input_W'].max(), 80)
    yi = np.linspace(d['N2_Percent_Input'].min(), d['N2_Percent_Input'].max(), 80)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((d['Power_Input_W'], d['N2_Percent_Input']), d['N_s_cm3'], (Xi, Yi), method='cubic')

    fig, ax = plt.subplots(figsize=(9, 6))
    cf = ax.contourf(Xi, Yi, Zi, levels=15, cmap='plasma')
    ax.scatter(d['Power_Input_W'], d['N2_Percent_Input'], c='k', s=15, marker='x')
    plt.colorbar(cf, ax=ax, label='Metastable density (cm$^{-3}$)')
    ax.set_xlabel('Power (W)')
    ax.set_ylabel('N$_2$ (%)')
    ax.set_title(f'N$_s$ at 2420 MHz, {p} Torr')
    plt.tight_layout()
    plt.savefig(f'contour_Ns_bestgamma_{p}Torr.png', dpi=300)
    plt.show()
    
# Gamma Plot 
for p in sorted(best['Pressure_Input_Torr'].dropna().unique()):
    d = best[best['Pressure_Input_Torr'] == p].dropna(subset=['Power_Input_W', 'N2_Percent_Input', 'Gamma'])
    if len(d) < 4:
        continue

    xi = np.linspace(d['Power_Input_W'].min(), d['Power_Input_W'].max(), 80)
    yi = np.linspace(d['N2_Percent_Input'].min(), d['N2_Percent_Input'].max(), 80)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((d['Power_Input_W'], d['N2_Percent_Input']), d['Gamma'], (Xi, Yi), method='cubic')

    fig, ax = plt.subplots(figsize=(9, 6))
    cf = ax.contourf(Xi, Yi, Zi, levels=15, cmap='plasma')
    ax.scatter(d['Power_Input_W'], d['N2_Percent_Input'], c='k', s=15, marker='x')
    plt.colorbar(cf, ax=ax, label='Gamma / a.u.')
    ax.set_xlabel('Power (W)')
    ax.set_ylabel('N$_2$ (%)')
    ax.set_title(f'N$_s$ at 2420 MHz, {p} Torr')
    plt.tight_layout()
    plt.show()
    
    

for p in sorted(best['Pressure_Input_Torr'].dropna().unique()):
    d = best[best['Pressure_Input_Torr'] == p].dropna(
        subset=['Power_Input_W', 'N2_Percent_Input', 'T_gas_K']).copy()
    if len(d) < 4:
        continue

    # mask out high-uncertainty points rather than dropping the row outright,
    # so N_s or other columns on that row stay usable elsewhere
    d.loc[d['T_gas_K_err'] > ERR_MAX_K, 'T_gas_K'] = np.nan
    d.loc[d['T_gas_K'] < MinTemp, 'T_gas_K'] = np.nan
    d = d.dropna(subset=['T_gas_K'])
    if len(d) < 4:
        print(f"{p} Torr: fewer than 4 points survive the {ERR_MAX_K} K error cut, skipping")
        continue

    xi = np.linspace(d['Power_Input_W'].min(), d['Power_Input_W'].max(), 80)
    yi = np.linspace(d['N2_Percent_Input'].min(), d['N2_Percent_Input'].max(), 80)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((d['Power_Input_W'], d['N2_Percent_Input']), d['T_gas_K'], (Xi, Yi), method='cubic')

    fig, ax = plt.subplots(figsize=(9, 6))
    cf = ax.contourf(Xi, Yi, Zi, levels=15, cmap='plasma')
    ax.scatter(d['Power_Input_W'], d['N2_Percent_Input'], c='k', s=15, marker='x')
    plt.colorbar(cf, ax=ax, label='Gas Temperature / K')
    ax.set_xlabel('Power (W)')
    ax.set_ylabel('N$_2$ (%)')
    ax.set_title(f'Gas Temperature at 2420 MHz, {p} Torr')
    plt.tight_layout()
    plt.show()
    
# Reverse Power
for p in sorted(best['Pressure_Input_Torr'].dropna().unique()):
    d = best[best['Pressure_Input_Torr'] == p].dropna(subset=['Power_Input_W', 'N2_Percent_Input', 'Reflected_Power_W'])
    if len(d) < 4:
        continue

    xi = np.linspace(d['Power_Input_W'].min(), d['Power_Input_W'].max(), 80)
    yi = np.linspace(d['N2_Percent_Input'].min(), d['N2_Percent_Input'].max(), 80)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((d['Power_Input_W'], d['N2_Percent_Input']), d['Reflected_Power_W'], (Xi, Yi), method='cubic')

    fig, ax = plt.subplots(figsize=(9, 6))
    cf = ax.contourf(Xi, Yi, Zi, levels=15, cmap='plasma')
    ax.scatter(d['Power_Input_W'], d['N2_Percent_Input'], c='k', s=15, marker='x')
    plt.colorbar(cf, ax=ax, label='Reverse Power / W')
    ax.set_xlabel('Power (W)')
    ax.set_ylabel('N$_2$ (%)')
    ax.set_title(f'Reverse Power at 2420 MHz')
    plt.tight_layout()
    plt.show()
    
    
    
from scipy.stats import pearsonr, spearmanr

import numpy as np
from scipy.stats import pearsonr

def ols_residuals(y, X_cols):
    """Residuals of y after regressing on X_cols (list of 1D arrays), with intercept."""
    X = np.column_stack([np.ones(len(y))] + list(X_cols))
    coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ coefs
    return y - fitted

d = best.dropna(subset=['Power_Measured_W', 'N2_Percent_Input', 'Gamma', 'N_s_cm3']).copy()

resid_Ns    = ols_residuals(d['N_s_cm3'].to_numpy(),
                            [d['Power_Measured_W'].to_numpy(), d['N2_Percent_Input'].to_numpy()])
resid_gamma = ols_residuals(d['Gamma'].to_numpy(),
                            [d['Power_Measured_W'].to_numpy(), d['N2_Percent_Input'].to_numpy()])

r_partial, p_partial = pearsonr(resid_gamma, resid_Ns)
print(f"Partial correlation (controlling Power, N2%): r = {r_partial:.3f}, p = {p_partial:.2e}")

plt.figure(figsize=(6, 5))
plt.scatter(resid_gamma, resid_Ns, s=20, alpha=0.7)
plt.axhline(0, color='k', linewidth=0.6, alpha=0.5)
plt.axvline(0, color='k', linewidth=0.6, alpha=0.5)
plt.xlabel('$\\Gamma$ residual (after removing Power, N$_2$ trend)')
plt.ylabel('$N_s$ residual (after removing Power, N$_2$ trend)')
plt.title(f'Partial correlation r = {r_partial:.2f}')
plt.tight_layout()
plt.show()

P  = d['Power_Measured_W'].to_numpy()
N2 = d['N2_Percent_Input'].to_numpy()
X_poly = [P, N2, P**2, N2**2, P*N2]

resid_Ns_poly    = ols_residuals(d['N_s_cm3'].to_numpy(), X_poly)
resid_gamma_poly = ols_residuals(d['Gamma'].to_numpy(), X_poly)

r_poly, p_poly = pearsonr(resid_gamma_poly, resid_Ns_poly)
print(f"Partial correlation, quadratic control: r = {r_poly:.3f}, p = {p_poly:.2e}")
