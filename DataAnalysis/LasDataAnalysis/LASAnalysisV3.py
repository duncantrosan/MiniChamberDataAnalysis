# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 10:43:54 2026

@author: dptro
"""

"""
Improved contour plots from master summary data
Handles sparse data gracefully with griddata interpolation
"""
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

 





# Load the summary data
summary = pd.read_csv('MasterResults_Summary_03_Fine.csv')
 
print("Data summary:")
print(f"  Power levels: {sorted(summary['power'].unique())}")
print(f"  Pressures: {sorted(summary['pressure'].unique())}")
print(f"  N2 flows: {sorted(summary['n2_flow'].unique())}")
print(f"  Total rows: {len(summary)}\n")
 
# ============================================================
# HELPER: Check data density
# ============================================================
def check_data_density(data, x_col, y_col, z_col, title=""):
    """Check if data is dense enough for contour plotting"""
    n_unique_x = data[x_col].nunique()
    n_unique_y = data[y_col].nunique()
    n_data = len(data)
    
    print(f"{title}")
    print(f"  Unique {x_col}: {n_unique_x}")
    print(f"  Unique {y_col}: {n_unique_y}")
    print(f"  Total points: {n_data}")
    print(f"  Minimum for contour: {n_unique_x} × {n_unique_y} ≥ 2×2 = 4")
    
    is_dense = n_unique_x >= 2 and n_unique_y >= 2
    print(f"  Density check: {'✓ PASS' if is_dense else '✗ FAIL'}\n")
    
    return is_dense
 
# ============================================================
# PLOT 1: Area vs Power & Pressure (Pure Argon, n2_flow=0)
# ============================================================
print("Creating Plot 1: Area vs Power & Pressure (Pure Argon)")
 
data_argon = summary[summary['n2_flow'] == 0.0].copy()
if len(data_argon) > 0:
    is_dense = check_data_density(data_argon, 'power', 'pressure', 'area_mean', 
                                   "  Plot 1 (Argon)")
    
    if is_dense:
        # Use griddata for interpolation (handles sparse data)
        xi = np.linspace(data_argon['power'].min(), data_argon['power'].max(), 50)
        yi = np.linspace(data_argon['pressure'].min(), data_argon['pressure'].max(), 50)
        Xi, Yi = np.meshgrid(xi, yi)
        
        Zi = griddata(
            (data_argon['power'], data_argon['pressure']),
            data_argon['area_mean'],
            (Xi, Yi),
            method='cubic'
        )
        
        fig, ax = plt.subplots(figsize=(10, 6))
        contour = ax.contourf(Xi, Yi, Zi, levels=15, cmap='viridis')
        contour_lines = ax.contour(Xi, Yi, Zi, levels=10, colors='black', alpha=0.3, linewidths=0.5)
        ax.clabel(contour_lines, inline=True, fontsize=8)
        
        # Overlay actual data points
        scatter = ax.scatter(data_argon['power'], data_argon['pressure'], 
                           c='red', s=20, marker='x', alpha=0.5, label='Data points')
        
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Area (Hz)', fontsize=11)
        
        ax.set_xlabel('Power (W)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Pressure (Torr)', fontsize=12, fontweight='bold')
        ax.set_title('Spectral Area vs Power & Pressure\n(Pure Argon)', 
                    fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('contour_01_argon_power_pressure.png', dpi=300, bbox_inches='tight')
        print("  ✓ Saved: contour_01_argon_power_pressure.png\n")
        plt.show()
    else:
        print("  ✗ Skipped: Insufficient data density for contour plot\n")
else:
    print("  ✗ No data for pure Argon (n2_flow=0)\n")
 
# ============================================================
# PLOT 2: Area vs Power & Pressure (4% Nitrogen)
# ============================================================
print("Creating Plot 2: Area vs Power & Pressure (with Nitrogen)")
 
# Find the best n2_flow value for this plot
# Prefer higher n2_flow values with good coverage
n2_candidates = summary['n2_flow'].unique()
n2_candidates = sorted([x for x in n2_candidates if x > 0])  # Exclude pure argon
 
best_n2 = None
best_score = 0
 
for n2_val in n2_candidates:
    data_temp = summary[summary['n2_flow'] == n2_val]
    score = len(data_temp)  # More data = better
    if score > best_score:
        best_score = score
        best_n2 = n2_val
 
if best_n2 is not None:
    print(f"  Using n2_flow = {best_n2} sccm (best coverage: {best_score} points)")
    
    data_nitrogen = summary[summary['n2_flow'] == best_n2].copy()
    is_dense = check_data_density(data_nitrogen, 'power', 'pressure', 'area_mean',
                                   f"  Plot 2 (N2={best_n2})")
    
    if is_dense:
        # Interpolate sparse data
        xi = np.linspace(data_nitrogen['power'].min(), data_nitrogen['power'].max(), 50)
        yi = np.linspace(data_nitrogen['pressure'].min(), data_nitrogen['pressure'].max(), 50)
        Xi, Yi = np.meshgrid(xi, yi)
        
        Zi = griddata(
            (data_nitrogen['power'], data_nitrogen['pressure']),
            data_nitrogen['area_mean'],
            (Xi, Yi),
            method='cubic'
        )
        
        fig, ax = plt.subplots(figsize=(10, 6))
        contour = ax.contourf(Xi, Yi, Zi, levels=15, cmap='viridis')
        contour_lines = ax.contour(Xi, Yi, Zi, levels=10, colors='black', alpha=0.3, linewidths=0.5)
        ax.clabel(contour_lines, inline=True, fontsize=8)
        
        # Overlay actual data points
        ax.scatter(data_nitrogen['power'], data_nitrogen['pressure'], 
                  c='red', s=20, marker='x', alpha=0.5, label='Data points')
        
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Area (Hz)', fontsize=11)
        
        ax.set_xlabel('Power (W)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Pressure (Torr)', fontsize=12, fontweight='bold')
        ax.set_title(f'Spectral Area vs Power & Pressure\n(N₂ Flow = {best_n2} sccm)', 
                    fontsize=13, fontweight='bold')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('contour_02_nitrogen_power_pressure.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: contour_02_nitrogen_power_pressure.png\n")
        plt.show()
    else:
        print("  ✗ Skipped: Insufficient data density for contour plot\n")
else:
    print("  ✗ No nitrogen data found\n")
 
# ============================================================
# PLOT 3: Area vs Nitrogen & Power (at fixed pressure)
# ============================================================
print("Creating Plot 3: Area vs Nitrogen & Power (at fixed pressure)")
 
target_pressure = 1.2
pressures = summary['pressure'].unique()
closest_pressure = min(pressures, key=lambda x: abs(x - target_pressure))
print(f"  Using pressure = {closest_pressure} Torr (closest to {target_pressure})")
 
data_pressure = summary[summary['pressure'] == closest_pressure].copy()
is_dense = check_data_density(data_pressure, 'power', 'n2_flow', 'area_mean',
                               f"  Plot 3 (P={closest_pressure})")
 
if is_dense:
    xi = np.linspace(data_pressure['power'].min(), data_pressure['power'].max(), 50)
    yi = np.linspace(data_pressure['n2_flow'].min(), data_pressure['n2_flow'].max(), 50)
    Xi, Yi = np.meshgrid(xi, yi)
    
    Zi = griddata(
        (data_pressure['power'], data_pressure['n2_flow']),
        data_pressure['area_mean'],
        (Xi, Yi),
        method='cubic'
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    contour = ax.contourf(Xi, Yi, Zi, levels=15, cmap='viridis')
    contour_lines = ax.contour(Xi, Yi, Zi, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    ax.clabel(contour_lines, inline=True, fontsize=8)
    
    ax.scatter(data_pressure['power'], data_pressure['n2_flow'], 
              c='red', s=20, marker='x', alpha=0.5, label='Data points')
    
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Area (Hz)', fontsize=11)
    
    ax.set_xlabel('Power (W)', fontsize=12, fontweight='bold')
    ax.set_ylabel('N₂ %', fontsize=12, fontweight='bold')
    ax.set_title(f'Spectral Area vs Nitrogen & Power\n(at {closest_pressure} Torr)', 
                fontsize=13, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('contour_03_nitrogen_power_at_pressure.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: contour_03_nitrogen_power_at_pressure.png\n")
    plt.show()
else:
    print("  ✗ Skipped: Insufficient data density for contour plot\n")
 
# ============================================================
# PLOT 4: Area vs Pressure & Nitrogen (at fixed power)
# ============================================================
print("Creating Plot 4: Area vs Pressure & Nitrogen (at fixed power)")
 
target_power = 40.0
powers = summary['power'].unique()
closest_power = min(powers, key=lambda x: abs(x - target_power))
print(f"  Using power = {closest_power} W (closest to {target_power})")
 
data_power = summary[summary['power'] == closest_power].copy()
is_dense = check_data_density(data_power, 'n2_flow', 'pressure', 'area_mean',
                               f"  Plot 4 (Power={closest_power})")
 
if is_dense:
    xi = np.linspace(data_power['n2_flow'].min(), data_power['n2_flow'].max(), 50)
    yi = np.linspace(data_power['pressure'].min(), data_power['pressure'].max(), 50)
    Xi, Yi = np.meshgrid(xi, yi)
    
    Zi = griddata(
        (data_power['n2_flow'], data_power['pressure']),
        data_power['area_mean'],
        (Xi, Yi),
        method='cubic'
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    contour = ax.contourf(Xi, Yi, Zi, levels=15, cmap='viridis')
    contour_lines = ax.contour(Xi, Yi, Zi, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    ax.clabel(contour_lines, inline=True, fontsize=8)
    
    ax.scatter(data_power['n2_flow'], data_power['pressure'], 
              c='red', s=20, marker='x', alpha=0.5, label='Data points')
    
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Area (Hz)', fontsize=11)
    
    ax.set_xlabel('N₂ %', fontsize=12, fontweight='bold')
    ax.set_ylabel('Pressure (Torr)', fontsize=12, fontweight='bold')
    ax.set_title(f'Spectral Area vs Pressure & Nitrogen\n(at {closest_power} W)', 
                fontsize=13, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('contour_04_pressure_nitrogen_at_power.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: contour_04_pressure_nitrogen_at_power.png\n")
    plt.show()
else:
    print("  ✗ Skipped: Insufficient data density for contour plot\n")
 
print("="*60)
print("Contour plotting complete!")
print("="*60)
print("\nGenerated files:")
print("  - contour_01_argon_power_pressure.png")
print("  - contour_02_nitrogen_power_pressure.png (best N2 coverage)")
print("  - contour_03_nitrogen_power_at_pressure.png")
print("  - contour_04_pressure_nitrogen_at_power.png")
 
print("\n" + "="*60)
print("DATA REFERENCE")
print("="*60)
print(f"Power levels available: {sorted(summary['power'].unique())}")
print(f"Pressures available: {sorted(summary['pressure'].unique())}")
print(f"N2 flows available: {sorted(summary['n2_flow'].unique())}")
 
print("\nNOTE: Sparse n2_flow values (0.2, 0.4, 0.6, etc.) are interpolated")
print("using cubic spline. Red 'x' marks show actual data points.")




#### Calculate Metastable density 

# ============================================================
# SPECTROSCOPIC CONSTANTS FOR Ar I 695.7 nm (5s line)
# ============================================================

# Wavelength
LAMBDA_0 = 695.7e-9  # meters (695.7 nm)

L = 4/100# Length of chamber in cm

# Atomic mass of Argon
M_AR = 40  # amu

# Statistical weights
# 5s line: 2p1 → 1s5
G_I = 2    # Statistical weight of excited state (2p1, J=1/2)
G_K = 4    # Statistical weight of metastable state (1s5, J=1/2→3/2)

# Einstein coefficient for Ar I 695.7 nm (5s line)
# From NIST: A_ki ≈ 2.5 × 10^7 s^-1
A_KI = 2.5e7  # s^-1

# Speed of light
C = 3e8  # m/s

# Doppler broadening constant
DOPPLER_CONST = 7.16e-7  # (in SI units)

print("="*60)
print("SPECTROSCOPIC CONSTANTS - Ar I 695.7 nm")
print("="*60)
print(f"Wavelength (λ₀): {LAMBDA_0*1e9:.1f} nm")
print(f"Atomic mass (M): {M_AR} amu")
print(f"Statistical weight excited (gᵢ): {G_I}")
print(f"Statistical weight metastable (gₖ): {G_K}")
print(f"Einstein coefficient (Aₖᵢ): {A_KI:.2e} s⁻¹")
print(f"Speed of light (c): {C:.2e} m/s")
print("="*60 + "\n")

# ============================================================
# LOAD DATA
# ============================================================
print("Loading MasterResults_Summary.csv...")
summary = pd.read_csv('MasterResults_Summary_03_Fine.csv')

print(f"  Loaded {len(summary)} rows\n")

# ============================================================
# CALCULATE GAS TEMPERATURE FROM FWHM
# ============================================================
print("Calculating gas temperature from FWHM...")

# Load raw data to get FWHM (need to reconstruct from raw data)
print("  Note: Loading raw data for FWHM values...")
raw_data = pd.read_csv('MasterResults_RawData_03_fine.csv')

# Group by (trial, power, pressure, n2_flow) and get mean FWHM
fwhm_grouped = raw_data.groupby(['trial', 'power', 'pressure', 'n2_flow'])['fwhm'].mean().reset_index()
fwhm_grouped = fwhm_grouped.rename(columns={'fwhm': 'fwhm_mean'})

# Merge FWHM back into summary
summary = summary.merge(fwhm_grouped, on=['trial', 'power', 'pressure', 'n2_flow'], how='left')

# Calculate Doppler broadening in wavelength space (convert from frequency to wavelength)
# FWHM_freq in Hz → FWHM_wavelength in m
# FWHM_lambda = FWHM_freq × (λ₀²/c)
summary['fwhm_wavelength'] = summary['fwhm_mean']* 10**9 * (LAMBDA_0**2 / C)

# Calculate gas temperature
# T_g = M × (Δλ_g / (7.16 × 10^(-7) × λ_0))^2
numerator = summary['fwhm_wavelength']
denominator = DOPPLER_CONST * LAMBDA_0
summary['T_gas_K'] = M_AR * (numerator / denominator)**2

print(f"  Temperature range: {summary['T_gas_K'].min():.0f} - {summary['T_gas_K'].max():.0f} K")
print(f"  Temperature mean: {summary['T_gas_K'].mean():.0f} K\n")




# ============================================================
# CALCULATE METASTABLE DENSITY FROM AREA (OD INTEGRAL)
# ============================================================
print("Calculating metastable density from spectral area...")

# The area in our analysis is ∫OD(λ)dλ in frequency space
# Need to convert to wavelength space: ∫OD(λ)dλ_wavelength = ∫OD(λ)dλ_freq × (λ₀²/c)
od_integral_wavelength = summary['area_mean'] * 10**9
Freq = (C/LAMBDA_0 )
# N_s = (8π g_i c) / (λ_0 g_k A_ki) × ∫OD(λ)dλ
numerator_ns = 8 * np.pi * G_I * Freq**2 
denominator_ns = C**2 * G_K * A_KI * L

summary['N_s_m3'] = (numerator_ns / denominator_ns) * od_integral_wavelength 

# Convert to more useful units (cm^-3)
summary['N_s_cm3'] = summary['N_s_m3'] * 1e-6

print(f"  Density range: {summary['N_s_cm3'].min():.2e} - {summary['N_s_cm3'].max():.2e} cm⁻³")
print(f"  Density mean: {summary['N_s_cm3'].mean():.2e} cm⁻³\n")

# ============================================================
# SAVE ENHANCED DATAFRAME
# ============================================================
print("Saving enhanced dataframe...")
summary.to_csv('MasterResults_Summary_with_Tg_Ns_03.csv', index=False)
print("  ✓ Saved: MasterResults_Summary_with_Tg_Ns.csv\n")

print("="*60)
print("CALCULATED VALUES")
print("="*60)
print("\nNew columns added:")
print("  - fwhm_wavelength: Doppler broadening in wavelength (m)")
print("  - T_gas_K: Gas temperature (Kelvin)")
print("  - N_s_m3: Metastable density (m⁻³)")
print("  - N_s_cm3: Metastable density (cm⁻³)")

print("\nSample data:")
print(summary[['power', 'pressure', 'n2_flow', 'area_mean', 'fwhm_mean', 'T_gas_K', 'N_s_cm3']].head(10))

print("\n" + "="*60)
print("CREATING CONTOUR PLOTS")
print("="*60 + "\n")

# ============================================================
# HELPER: Check data density
# ============================================================
def check_data_density(data, x_col, y_col, z_col, title=""):
    """Check if data is dense enough for contour plotting"""
    n_unique_x = data[x_col].nunique()
    n_unique_y = data[y_col].nunique()
    n_data = len(data)
    
    print(f"{title}")
    print(f"  Unique {x_col}: {n_unique_x}")
    print(f"  Unique {y_col}: {n_unique_y}")
    print(f"  Total points: {n_data}")
    print(f"  Minimum for contour: {n_unique_x} × {n_unique_y} ≥ 2×2 = 4")
    
    is_dense = n_unique_x >= 2 and n_unique_y >= 2
    print(f"  Density check: {'✓ PASS' if is_dense else '✗ FAIL'}\n")
    
    return is_dense





# ============================================================
# PROPAGATE ERROR FROM AREA TO METASTABLE DENSITY
# ============================================================
print("Propagating uncertainty from area to metastable density...")

# Since N_s = (constant) × area, the error propagates as:
# N_s_std = (constant) × area_std

od_integral_wavelength_std = summary['area_std'] * 10**9

# Propagate through the calculation
summary['N_s_m3_std'] = (numerator_ns / denominator_ns) * od_integral_wavelength_std
summary['N_s_cm3_std'] = summary['N_s_m3_std'] * 1e-6

print(f"  Relative uncertainty: {(summary['N_s_cm3_std'] / summary['N_s_cm3']).mean() * 100:.1f}%")
print(f"  N_s range: {summary['N_s_cm3'].mean():.2e} ± {summary['N_s_cm3_std'].mean():.2e} cm⁻³\n")


# ============================================================
# PLOT 1: T_g vs Power & Pressure (Pure Argon)
# ============================================================
print("Creating Plot 1: Temperature vs Power & Pressure (Pure Argon)")

data_argon = summary[summary['n2_flow'] == 0.0].copy()
if len(data_argon) > 0:
    is_dense = check_data_density(data_argon, 'power', 'pressure', 'T_gas_K', 
                                   "  Plot 1 (Argon)")
    
    if is_dense:
        xi = np.linspace(data_argon['power'].min(), data_argon['power'].max(), 50)
        yi = np.linspace(data_argon['pressure'].min(), data_argon['pressure'].max(), 50)
        Xi, Yi = np.meshgrid(xi, yi)
        
        Zi = griddata(
            (data_argon['power'], data_argon['pressure']),
            data_argon['T_gas_K'],
            (Xi, Yi),
            method='cubic'
        )
        
        fig, ax = plt.subplots(figsize=(10, 6))
        contour = ax.contourf(Xi, Yi, Zi, levels=15, cmap='hot')
        contour_lines = ax.contour(Xi, Yi, Zi, levels=10, colors='black', alpha=0.3, linewidths=0.5)
        ax.clabel(contour_lines, inline=True, fontsize=8)
        
        ax.scatter(data_argon['power'], data_argon['pressure'], 
                  c='blue', s=20, marker='x', alpha=0.5, label='Data points')
        
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Gas Temperature (K)', fontsize=11)
        
        ax.set_xlabel('Power (W)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Pressure (Torr)', fontsize=12, fontweight='bold')
        ax.set_title('Gas Temperature vs Power & Pressure\n(Pure Argon)', 
                    fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('contour_01_Tg_power_pressure.png', dpi=300, bbox_inches='tight')
        print("  ✓ Saved: contour_01_Tg_power_pressure.png\n")
        plt.show()
    else:
        print("  ✗ Skipped: Insufficient data density\n")
else:
    print("  ✗ No data for pure Argon\n")

# ============================================================
# PLOT 2: N_s vs Power & Pressure (Best nitrogen coverage)
# ============================================================
print("Creating Plot 2: Metastable Density vs Power & Pressure (with Nitrogen)")

n2_candidates = summary['n2_flow'].unique()
n2_candidates = sorted([x for x in n2_candidates if x > 0])

best_n2 = None
best_score = 0

for n2_val in n2_candidates:
    data_temp = summary[summary['n2_flow'] == n2_val]
    score = len(data_temp)
    if score > best_score:
        best_score = score
        best_n2 = n2_val

if best_n2 is not None:
    print(f"  Using n2_flow = {best_n2} sccm (coverage: {best_score} points)")
    
    data_nitrogen = summary[summary['n2_flow'] == best_n2].copy()
    is_dense = check_data_density(data_nitrogen, 'power', 'pressure', 'N_s_cm3',
                                   f"  Plot 2 (N2={best_n2})")
    
    if is_dense:
        xi = np.linspace(data_nitrogen['power'].min(), data_nitrogen['power'].max(), 50)
        yi = np.linspace(data_nitrogen['pressure'].min(), data_nitrogen['pressure'].max(), 50)
        Xi, Yi = np.meshgrid(xi, yi)
        
        Zi = griddata(
            (data_nitrogen['power'], data_nitrogen['pressure']),
            data_nitrogen['N_s_cm3'],
            (Xi, Yi),
            method='cubic'
        )
        
        fig, ax = plt.subplots(figsize=(10, 6))
        contour = ax.contourf(Xi, Yi, Zi, levels=15, cmap='plasma')
        contour_lines = ax.contour(Xi, Yi, Zi, levels=10, colors='black', alpha=0.3, linewidths=0.5)
        ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1e')
        
        ax.scatter(data_nitrogen['power'], data_nitrogen['pressure'], 
                  c='blue', s=20, marker='x', alpha=0.5, label='Data points')
        
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Metastable Density (cm⁻³)', fontsize=11)
        
        ax.set_xlabel('Power (W)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Pressure (Torr)', fontsize=12, fontweight='bold')
        ax.set_title(f'Metastable Density vs Power & Pressure\n(N₂ Flow = {best_n2} sccm)', 
                    fontsize=13, fontweight='bold')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('contour_02_Ns_power_pressure.png', dpi=300, bbox_inches='tight')
        print("  ✓ Saved: contour_02_Ns_power_pressure.png\n")
        plt.show()
    else:
        print("  ✗ Skipped: Insufficient data density\n")
else:
    print("  ✗ No nitrogen data\n")

# ============================================================
# PLOT 3: T_g vs Nitrogen & Power (fixed pressure)
# ============================================================
print("Creating Plot 3: Temperature vs Nitrogen & Power (fixed pressure)")

target_pressure = 1.0
pressures = summary['pressure'].unique()
closest_pressure = min(pressures, key=lambda x: abs(x - target_pressure))
print(f"  Using pressure = {closest_pressure} Torr")

data_pressure = summary[summary['pressure'] == closest_pressure].copy()
is_dense = check_data_density(data_pressure, 'power', 'n2_flow', 'T_gas_K',
                               f"  Plot 3 (P={closest_pressure})")

if is_dense:
    xi = np.linspace(data_pressure['power'].min(), data_pressure['power'].max(), 50)
    yi = np.linspace(data_pressure['n2_flow'].min(), data_pressure['n2_flow'].max(), 50)
    Xi, Yi = np.meshgrid(xi, yi)
    
    Zi = griddata(
        (data_pressure['power'], data_pressure['n2_flow']),
        data_pressure['T_gas_K'],
        (Xi, Yi),
        method='cubic'
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    contour = ax.contourf(Xi, Yi, Zi, levels=15, cmap='hot')
    contour_lines = ax.contour(Xi, Yi, Zi, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    ax.clabel(contour_lines, inline=True, fontsize=8)
    
    ax.scatter(data_pressure['power'], data_pressure['n2_flow'], 
              c='blue', s=20, marker='x', alpha=0.5, label='Data points')
    
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Gas Temperature (K)', fontsize=11)
    
    ax.set_xlabel('Power (W)', fontsize=12, fontweight='bold')
    ax.set_ylabel('N₂ %', fontsize=12, fontweight='bold')
    ax.set_title(f'Gas Temperature vs Nitrogen & Power\n(at {closest_pressure} Torr)', 
                fontsize=13, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('contour_03_Tg_nitrogen_power.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: contour_03_Tg_nitrogen_power.png\n")
    plt.show()
else:
    print("  ✗ Skipped: Insufficient data density\n")





# ============================================================
# PLOT 6: Ni vs Nitrogen & Power (fixed pressure)
# ============================================================
print("Creating Plot 6: MetastableDensity vs Nitrogen & Power (fixed pressure)")

target_pressure = 1.2
pressures = summary['pressure'].unique()
closest_pressure = min(pressures, key=lambda x: abs(x - target_pressure))
print(f"  Using pressure = {closest_pressure} Torr")

data_pressure = summary[summary['pressure'] == closest_pressure].copy()
is_dense = check_data_density(data_pressure, 'power', 'n2_flow', 'T_gas_K',
                               f"  Plot 3 (P={closest_pressure})")

if is_dense:
    xi = np.linspace(data_pressure['power'].min(), data_pressure['power'].max(), 50)
    yi = np.linspace(data_pressure['n2_flow'].min(), data_pressure['n2_flow'].max(), 50)
    Xi, Yi = np.meshgrid(xi, yi)
    
    Zi = griddata(
        (data_pressure['power'], data_pressure['n2_flow']),
        data_pressure['N_s_cm3'],
        (Xi, Yi),
        method='cubic'
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    contour = ax.contourf(Xi, Yi, Zi, levels=15, cmap='hot')
    contour_lines = ax.contour(Xi, Yi, Zi, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    ax.clabel(contour_lines, inline=True, fontsize=8)
    
    ax.scatter(data_pressure['power'], data_pressure['n2_flow'], 
              c='blue', s=20, marker='x', alpha=0.5, label='Data points')
    
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Metastable Density (cm^-3)', fontsize=11)
    
    ax.set_xlabel('Power (W)', fontsize=12, fontweight='bold')
    ax.set_ylabel('N₂ Percentage', fontsize=12, fontweight='bold')
    ax.set_title(f'Argon (s_5) Density vs Nitrogen & Power\n(at {closest_pressure} Torr)', 
                fontsize=13, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    print("  ✓ Saved: contour_03_Tg_nitrogen_power.png\n")
    plt.show()
else:
    print("  ✗ Skipped: Insufficient data density\n")

#%%
# ============================================================
# PLOT 4: N_s vs Pressure & Nitrogen (fixed power)
# ============================================================
print("Creating Plot 4: Metastable Density vs Pressure & Nitrogen (fixed power)")

target_power = 40.0
powers = summary['power'].unique()
closest_power = min(powers, key=lambda x: abs(x - target_power))
print(f"  Using power = {closest_power} W")

data_power = summary[summary['power'] == closest_power].copy()
is_dense = check_data_density(data_power, 'n2_flow', 'pressure', 'N_s_cm3',
                               f"  Plot 4 (Power={closest_power})")

if is_dense:
    xi = np.linspace(data_power['n2_flow'].min(), data_power['n2_flow'].max(), 50)
    yi = np.linspace(data_power['pressure'].min(), data_power['pressure'].max(), 50)
    Xi, Yi = np.meshgrid(xi, yi)
    
    Zi = griddata(
        (data_power['n2_flow'], data_power['pressure']),
        data_power['N_s_cm3'],
        (Xi, Yi),
        method='cubic'
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    contour = ax.contourf(Xi, Yi, Zi, levels=15, cmap='plasma')
    contour_lines = ax.contour(Xi, Yi, Zi, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1e')
    
    ax.scatter(data_power['n2_flow'], data_power['pressure'], 
              c='blue', s=20, marker='x', alpha=0.5, label='Data points')
    
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Metastable Density (cm⁻³)', fontsize=11)
    
    ax.set_xlabel('N₂ Percentage', fontsize=12, fontweight='bold')
    ax.set_ylabel('Pressure (Torr)', fontsize=12, fontweight='bold')
    ax.set_title(f'Metastable Density vs Pressure & Nitrogen\n(at {closest_power} W)', 
                fontsize=13, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('contour_04_Ns_pressure_nitrogen.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: contour_04_Ns_pressure_nitrogen.png\n")
    plt.show()
else:
    print("  ✗ Skipped: Insufficient data density\n")


# Select data at target pressure
target_pressure = 1
pressures = summary['pressure'].unique()
closest_pressure = min(pressures, key=lambda x: abs(x - target_pressure))
print(f"Using pressure = {closest_pressure} Torr")

data_pressure = summary[summary['pressure'] == closest_pressure].copy()
 
if len(data_pressure) > 0:
    fig, ax = plt.subplots(figsize=(11, 7))
    
     # Get all N2 flows, but plot only every other one (or every 3rd, etc.)
    n2_flows = sorted(data_pressure['n2_flow'].unique())
    n2_flows = n2_flows[::2]  # Every 3rd value
    n2_flows.pop()
    # n2_flows = n2_flows[::3]  # Every 3rd value
    # n2_flows = n2_flows[[0, -1]]  # Only first and last
    colors = plt.cm.viridis(np.linspace(0, 1, len(n2_flows)))
    
    print(f"\nN2 flows at {closest_pressure} Torr: {n2_flows}")
    
    for n2, color in zip(n2_flows, colors):
        data = data_pressure[data_pressure['n2_flow'] == n2].groupby('power').agg({
            'N_s_cm3': 'mean',
            'N_s_cm3_std': 'mean'  # Use the propagated error
        }).reset_index()
        data = data.sort_values('power')
        
        if len(data) > 0:
            ax.errorbar(data['power'], data['N_s_cm3'], yerr=data['N_s_cm3_std'], 
                       marker='o', label=f'N₂ = {n2} sccm', 
                       color=color, linewidth=2.5, markersize=7, capsize=5, alpha=0.8)
    
    ax.set_xlabel('Power (W)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Metastable Density (cm⁻³)', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("\nPlot Summary:")
    print(f"  X-axis: Power (W)")
    print(f"  Y-axis: Metastable Density (cm⁻³)")
    print(f"  Fixed: Pressure = {closest_pressure} Torr")
    print(f"  Lines: Different N₂ flow values")
    print(f"  Error bars: ± propagated uncertainty from area")
    
else:
    print(f"✗ No data for pressure = {closest_pressure} Torr")
    

# Gas Temperature
# Select data at target pressure
target_pressure = 1
pressures = summary['pressure'].unique()
closest_pressure = min(pressures, key=lambda x: abs(x - target_pressure))
print(f"Using pressure = {closest_pressure} Torr")
data_pressure = summary[summary['pressure'] == closest_pressure].copy()
 
if len(data_pressure) > 0:
    fig, ax = plt.subplots(figsize=(11, 7))
    
    # Get all N2 flows, but plot only every 3rd one
    n2_flows = sorted(data_pressure['n2_flow'].unique())
    powers = sorted(data_pressure['power'].unique())
    n2_flows = n2_flows[::2]  # Every 3rd value
    PowerSort = powers[::3]
    n2_flows.pop()
    colors = plt.cm.viridis(np.linspace(0, 1, len(powers)))
    
    print(f"\nN2 flows at {closest_pressure} Torr: {n2_flows}")
    
    for p, color in zip(PowerSort, colors):
        data_T = data_pressure[data_pressure['power'] == p].groupby('n2_flow').agg({
            'T_gas_K': 'mean',      # ← CHANGED from N_s_cm3
        }).reset_index()
       # data_T = data_T.sort_values('n2_flow')
        
        if len(data_T) > 0:
            ax.plot(data_T['n2_flow'], data_T['T_gas_K'],
                       marker='o', label=f'{p} W', 
                       color=color, linewidth=2.5, markersize=7)
    
    ax.set_xlabel('Nitrogen Percent / %', fontsize=12, fontweight='bold')
    ax.set_ylabel('Gas Temperature (K)', fontsize=12, fontweight='bold')  # ← CHANGED
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("\nPlot Summary:")
    print(f"  X-axis: Power (W)")
    print(f"  Y-axis: Gas Temperature (K)")  # ← CHANGED
    print(f"  Fixed: Pressure = {closest_pressure} Torr")
    print(f"  Lines: Different N₂ flow values")
    print(f"  Error bars: ± propagated uncertainty from FWHM")
    
else:
    print(f"✗ No data for pressure = {closest_pressure} Torr")

    


## Metastable density vs Nitgoren percent 

# Gas Temperature
# Select data at target pressure
target_pressure = 1
pressures = summary['pressure'].unique()
closest_pressure = min(pressures, key=lambda x: abs(x - target_pressure))
print(f"Using pressure = {closest_pressure} Torr")
data_pressure = summary[summary['pressure'] == closest_pressure].copy()
 
if len(data_pressure) > 0:
    fig, ax = plt.subplots(figsize=(11, 7))
    
    # Get all N2 flows, but plot only every 3rd one
    n2_flows = sorted(data_pressure['n2_flow'].unique())
    powers = sorted(data_pressure['power'].unique())
    n2_flows = n2_flows[::2]  # Every 3rd value
    PowerSort = powers[::3]
    n2_flows.pop()
    colors = plt.cm.viridis(np.linspace(0, 1, len(powers)))
    
    print(f"\nN2 flows at {closest_pressure} Torr: {n2_flows}")
    
    for p, color in zip(PowerSort, colors):
        data_N = data_pressure[data_pressure['power'] == p].groupby('n2_flow').agg({
            'N_s_cm3': 'mean',      # ← CHANGED from N_s_cm3
        }).reset_index()
       # data_T = data_T.sort_values('n2_flow')
        
        if len(data_T) > 0:
            ax.plot(data_N['n2_flow'], data_N['N_s_cm3'],
                       marker='o', label=f'{p} W', 
                       color=color, linewidth=2.5, markersize=7)
    
    ax.set_xlabel('Nitrogen Percent / %', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'Metstable Density / cm⁻³', fontsize=12, fontweight='bold')  # ← CHANGED
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("\nPlot Summary:")
    print(f"  X-axis: Power (W)")
    print(f"  Y-axis: Gas Temperature (K)")  # ← CHANGED
    print(f"  Fixed: Pressure = {closest_pressure} Torr")
    print(f"  Lines: Different N₂ flow values")
    print(f"  Error bars: ± propagated uncertainty from FWHM")
    
else:
    print(f"✗ No data for pressure = {closest_pressure} Torr")

    

    
#%%
# ============================================================
# ANALYZE SLOPES: dN_s/dPower vs Nitrogen Flow (Multiple Pressures)
# ============================================================
from scipy.stats import linregress

print("="*60)
print("SLOPE ANALYSIS: Multiple Pressures")
print("="*60)

# Target pressures to analyze
target_pressures = [1]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
markers = ['o', 's', '^']

fig, ax = plt.subplots(figsize=(13, 8))

all_results = {}

for target_pressure, color, marker in zip(target_pressures, colors, markers):
    pressures = summary['pressure'].unique()
    closest_pressure = min(pressures, key=lambda x: abs(x - target_pressure))
    
    print(f"\nAnalyzing slopes at pressure = {closest_pressure} Torr")
    
    data_pressure = summary[summary['pressure'] == closest_pressure].copy()
    
    if len(data_pressure) > 0:
        # Get all N2 flows at this pressure
        n2_flows = sorted(data_pressure['n2_flow'].unique())
        
        slopes = []
        n2_values = []
        slope_errors = []
        
        print(f"N2 flows at {closest_pressure} Torr: {n2_flows}")
        print(f"{'N₂ %':>15} | {'Slope (cm⁻³/W)':>20} | {'Std Error':>15} | {'R²':>10}")
        print("-" * 80)
        
        for n2 in n2_flows:
            data = data_pressure[data_pressure['n2_flow'] == n2].groupby('power').agg({
                'N_s_cm3': 'mean',
                'N_s_cm3_std': 'mean'
            }).reset_index()
            data = data.sort_values('power')
            
            if len(data) > 1:  # Need at least 2 points to fit a line
                slope, intercept, r_value, p_value, std_err = linregress(data['power'], data['N_s_cm3'])
                
                slopes.append(slope)
                n2_values.append(n2)
                slope_errors.append(std_err)
                
                print(f"{n2:>15.2f} | {slope:>20.3e} | {std_err:>15.3e} | {r_value**2:>10.4f}")
        
        print("-" * 80)
        
        # Plot this pressure's data
        if len(slopes) > 0:
            ax.errorbar(n2_values, slopes, yerr=slope_errors, 
                       marker=marker, markersize=10, linestyle='-', linewidth=2.5,
                       capsize=6, capthick=2, color=color, ecolor=color, 
                       alpha=0.8, label=f'{closest_pressure} Torr')
            
            all_results[closest_pressure] = {
                'n2_flows': n2_values,
                'slopes': slopes,
                'errors': slope_errors
            }

ax.set_xlabel('N₂ %', fontsize=13, fontweight='bold')
ax.set_ylabel('Slope of Metstable Density vs Power (cm⁻³/W)', fontsize=13, fontweight='bold')
ax.legend(loc='best', fontsize=12, title='Pressure', title_fontsize=12)
ax.grid(True, alpha=0.3, linestyle='--')
ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

plt.tight_layout()
plt.show()

# Summary statistics
print("\n" + "="*60)
print("COMPARISON ACROSS PRESSURES")
print("="*60)

for pressure in sorted(all_results.keys()):
    data = all_results[pressure]
    slopes = data['slopes']
    n2_flows = data['n2_flows']
    
    print(f"\nAt {pressure} Torr:")
    print(f"  Min slope:  {min(slopes):.3e} cm⁻³/W (at N₂ = {n2_flows[slopes.index(min(slopes))]:.2f} sccm)")
    print(f"  Max slope:  {max(slopes):.3e} cm⁻³/W (at N₂ = {n2_flows[slopes.index(max(slopes))]:.2f} sccm)")
    print(f"  Mean slope: {np.mean(slopes):.3e} cm⁻³/W")
#%%
#%%
# ============================================================
# DIAGNOSTIC: Metastable Density vs Gas Temperature (Pure Argon)
# ============================================================
from scipy.stats import linregress

print("="*60)
print("DIAGNOSTIC: N_s vs T_gas for Pure Argon")
print("="*60)

# Get pure argon data
data_argon = summary[summary['n2_flow'] == 0.0].copy()

# Filter to only 1.2, 1.4, 1.5 Torr
target_pressures = [ 1.4, 1.5]
data_argon = data_argon[data_argon['pressure'].isin(target_pressures)]

if len(data_argon) > 0:
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get unique pressures in pure argon data
    pressures = sorted(data_argon['pressure'].unique())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    
    print(f"Pure Argon data at: {pressures}\n")
    
    # Plot each pressure separately
    for pressure, color in zip(pressures, colors):
        data_p = data_argon[data_argon['pressure'] == pressure].sort_values('T_gas_K')
        
        if len(data_p) > 0:
            ax.scatter(data_p['T_gas_K'], data_p['N_s_cm3'], 
                      s=100, alpha=0.7, color=color, 
                      label=f'{pressure} Torr', edgecolors='black', linewidth=0.5)
            
            # Linear fit for this pressure
            if len(data_p) > 1:
                slope, intercept, r_value, p_value, std_err = linregress(data_p['T_gas_K'], data_p['N_s_cm3'])
                
                # Plot fit line
                T_range = np.array([data_p['T_gas_K'].min(), data_p['T_gas_K'].max()])
                N_fit = slope * T_range + intercept
                ax.plot(T_range, N_fit, color=color, linestyle='--', linewidth=2, alpha=0.7)
                
                print(f"At {pressure} Torr:")
                print(f"  Slope: {slope:.3e} cm⁻³/K")
                print(f"  R²: {r_value**2:.4f}")
                print(f"  Correlation: {'STRONG' if abs(r_value) > 0.7 else 'WEAK'}\n")
    
    ax.set_xlabel('Gas Temperature (K)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Metastable Density N_s (cm⁻³)', fontsize=13, fontweight='bold')
    ax.set_title('Metastable Density vs Gas Temperature\n(Pure Argon)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11, title='Pressure')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    
    plt.tight_layout()
    plt.savefig('diagnostic_Ns_vs_Tgas_pure_ar.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: diagnostic_Ns_vs_Tgas_pure_ar.png")
    plt.show()
    
    # Overall correlation
    print("\n" + "="*60)
    print("OVERALL CORRELATION (Pure Ar, 1.2-1.5 Torr)")
    print("="*60)
    
    corr_coefficient = data_argon[['T_gas_K', 'N_s_cm3']].corr().iloc[0, 1]
    slope_overall, intercept_overall, r_value_overall, p_value_overall, std_err_overall = linregress(data_argon['T_gas_K'], data_argon['N_s_cm3'])
    
    print(f"Correlation coefficient: {corr_coefficient:.4f}")
    print(f"Slope (dN_s/dT): {slope_overall:.3e} cm⁻³/K")
    print(f"R² value: {r_value_overall**2:.4f}")
    print(f"P-value: {p_value_overall:.2e}")
    
    if p_value_overall < 0.05:
        print("✓ Statistically significant relationship (p < 0.05)")
    else:
        print("✗ NOT statistically significant (p > 0.05)")
    
    print("\nInterpretation:")
    if abs(corr_coefficient) > 0.7:
        if slope_overall > 0:
            print("  Strong POSITIVE correlation: Higher T → Higher N_s")
            print("  Temperature increase explains metastable density increase")
        else:
            print("  Strong NEGATIVE correlation: Higher T → Lower N_s")
            print("  → Supports ionization loss mechanism at higher T")
    elif abs(corr_coefficient) > 0.4:
        print("  Moderate correlation: Temperature partially explains N_s behavior")
    else:
        print("  Weak correlation: Temperature alone does NOT explain N_s changes")
        print("  → Other mechanisms must be at play (pressure, electron density, etc.)")

else:
    print("✗ No pure argon data found at 1.2, 1.4, 1.5 Torr")




#%%
#%%
# ============================================================
# DIFFUSION LOSS HYPOTHESIS TEST
# ============================================================

print("="*60)
print("TESTING DIFFUSION LOSS HYPOTHESIS")
print("="*60)

# Check if FWHM (related to T) correlates with N_s drop
data_argon = summary[summary['n2_flow'] == 0.0].copy()
data_argon = data_argon[data_argon['pressure'].isin([ 1.4, 1.5])].sort_values('power')

# Calculate diffusion-related quantity: assume D ∝ T^1.5
data_argon['D_proxy'] = data_argon['T_gas_K'] ** 1.5

# Correlate with N_s
corr_T = data_argon[['T_gas_K', 'N_s_cm3']].corr().iloc[0, 1]
corr_D = data_argon[['D_proxy', 'N_s_cm3']].corr().iloc[0, 1]

print(f"\nCorrelation with Temperature (T): {corr_T:.4f}")
print(f"Correlation with Diffusion (T^1.5): {corr_D:.4f}")

if abs(corr_D) > abs(corr_T):
    print("→ Diffusion loss is STRONGER correlate than temperature alone")
    print("→ SUPPORTS diffusion loss hypothesis!")
else:
    print("→ Temperature correlation is stronger")
    print("→ May indicate other temperature-dependent mechanisms")

# Plot N_s vs diffusion proxy
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Temperature
for pressure in sorted(data_argon['pressure'].unique()):
    data_p = data_argon[data_argon['pressure'] == pressure]
    ax1.scatter(data_p['T_gas_K'], data_p['N_s_cm3'], s=100, alpha=0.7, label=f'{pressure} Torr')
ax1.set_xlabel('Temperature (K)', fontweight='bold')
ax1.set_ylabel('N_s (cm⁻³)', fontweight='bold')
ax1.set_title(f'N_s vs T (r = {corr_T:.3f})')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

# Diffusion proxy
for pressure in sorted(data_argon['pressure'].unique()):
    data_p = data_argon[data_argon['pressure'] == pressure]
    ax2.scatter(data_p['D_proxy'], data_p['N_s_cm3'], s=100, alpha=0.7, label=f'{pressure} Torr')
ax2.set_xlabel('Diffusion Proxy (T^1.5)', fontweight='bold')
ax2.set_ylabel('N_s (cm⁻³)', fontweight='bold')
ax2.set_title(f'N_s vs D (r = {corr_D:.3f})')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

plt.tight_layout()
plt.savefig('hypothesis_diffusion_loss.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("NITROGEN EFFECT ON DIFFUSION")
print("="*60)

# Compare slopes: how does N_s respond to power (which drives T)?
for n2_val in [0.0, max(summary[summary['n2_flow'] > 0]['n2_flow'].unique())]:
    data_n2 = summary[summary['n2_flow'] == n2_val].copy()
    data_n2 = data_n2[data_n2['pressure'] == 1.2].sort_values('power')
    
    if len(data_n2) > 1:
        slope_T, _, _, _, _ = linregress(data_n2['T_gas_K'], data_n2['N_s_cm3'])
        label = "Pure Ar" if n2_val == 0.0 else f"With N₂ ({n2_val} sccm)"
        sign = "↓ Decreases" if slope_T < 0 else "↑ Increases"
        print(f"{label}: dN_s/dT = {slope_T:.2e} {sign}")

#%%
#%%
# ============================================================
# DIAGNOSTIC: Metastable Density vs Gas Temperature (Pure Argon)
# WITH R² AND BEST FIT LINES
# ============================================================
from scipy.stats import linregress

print("="*60)
print("DIAGNOSTIC: N_s vs T_gas for Pure Argon (with fit lines)")
print("="*60)

# Get pure argon data
data_argon = summary[summary['n2_flow'] == 0.0].copy()

# Filter to only 1.2, 1.4, 1.5 Torr
target_pressures = [1]
data_argon = data_argon[data_argon['pressure'].isin(target_pressures)]

if len(data_argon) > 0:
    fig, ax = plt.subplots(figsize=(13, 8))

    # Get unique pressures in pure argon data
    pressures = sorted(data_argon['pressure'].unique())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    
    print(f"Pure Argon data at: {pressures}\n")
    print("Regression Results:")
    print("-" * 90)
    print(f"{'Pressure':>12} | {'Slope':>18} | {'Intercept':>15} | {'R²':>8} | {'P-value':>12}")
    print("-" * 90)
    
    # Plot each pressure separately
    for pressure, color in zip(pressures, colors):
        data_p = data_argon[data_argon['pressure'] == pressure].sort_values('T_gas_K')
        DiffProxy = data_p['T_gas_K']**(3/2)
        if len(data_p) > 0:
            ax.scatter(DiffProxy, data_p['N_s_cm3'], 
                      s=120, alpha=0.7, color=color, 
                      edgecolors='black', linewidth=1.5, zorder=3)
            
            # Linear fit for this pressure
            if len(data_p) > 1:
                slope, intercept, r_value, p_value, std_err = linregress(DiffProxy, data_p['N_s_cm3'])
                r_squared = r_value ** 2
                
                # Plot fit line
                T_range = np.array([DiffProxy.min() - 50, DiffProxy.max() + 50])
                N_fit = slope * T_range + intercept
                ax.plot(T_range, N_fit, color=color, linestyle='-', linewidth=2.5, alpha=0.8, zorder=2,
                       label=f'{pressure} Torr (R² = {r_squared:.4f})')
                
                print(f"{pressure:>12.1f} | {slope:>18.3e} | {intercept:>15.3e} | {r_squared:>8.4f} | {p_value:>12.2e}")
    
    print("-" * 90)
        
    ax.set_xlabel(r'$T^{3/2}$ (K$^{3/2}$)', fontsize=13, fontweight='bold')
    ax.set_ylabel(r'Metastable Density $N_s$ (cm$^{-3}$)', fontsize=13, fontweight='bold')

    ax.legend(loc='best', fontsize=12, title='Pressure (R² values)', title_fontsize=12, 
             framealpha=0.95, edgecolor='black', fancybox=True)
    ax.grid(True, alpha=0.3, linestyle='--', zorder=0)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    

#%%
# ============================================================
# DIAGNOSTIC: Metastable Density vs Gas Temperature (Pure Argon)
# WITH RAW DATA POINTS, R² AND BEST FIT LINES
# ============================================================

print("="*60)
print("DIAGNOSTIC: N_s vs T_gas for Pure Argon (RAW DATA)")
print("="*60)

# Load raw data
raw_data = pd.read_csv('MasterResults_RawData_03_Fine.csv')

# Calculate T_gas_K for each raw point
raw_data['fwhm_hz'] = raw_data['fwhm'] * 1e9
raw_data['fwhm_wavelength'] = raw_data['fwhm_hz'] * (LAMBDA_0**2 / C)
raw_data['T_gas_K'] = M_AR * (raw_data['fwhm_wavelength'] / (DOPPLER_CONST * LAMBDA_0))**2

# Calculate N_s for each raw point
area_hz = raw_data['area'] * 1e9
od_integral_wavelength = area_hz  # GHz → Hz
Freq = C / LAMBDA_0
numerator_ns = 8 * np.pi * G_I * Freq**2 
denominator_ns = C**2 * G_K * A_KI * L
raw_data['N_s_m3'] = (numerator_ns / denominator_ns) * od_integral_wavelength
raw_data['N_s_cm3'] = raw_data['N_s_m3'] * 1e-6

# Filter to pure argon and target pressures
target_pressures = [1]
data_argon_raw = raw_data[raw_data['n2_flow'] == 0.0].copy()
data_argon_raw = data_argon_raw[data_argon_raw['pressure'].isin(target_pressures)]

if len(data_argon_raw) > 0:
    # Remove NaN values
    #data_argon_raw = data_argon_raw.dropna(subset=['T_gas_K', 'N_s_cm3'])
    
    fig, ax = plt.subplots(figsize=(13, 8))
    
    pressures = sorted(data_argon_raw['pressure'].unique())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    
    print(f"Pure Argon raw data at: {pressures}\n")
    print("Regression Results (on raw individual measurements):")
    print("-" * 100)
    print(f"{'Pressure':>12} | {'Slope':>18} | {'Intercept':>15} | {'R²':>8} | {'P-value':>12} | {'N points':>8}")
    print("-" * 100)
    
    # Plot each pressure separately
    for pressure, color in zip(pressures, colors):
        data_p = data_argon_raw[data_argon_raw['pressure'] == pressure].copy()
        
        if len(data_p) > 0:
            DiffProxy = data_p['T_gas_K']**(3/2)
            
            # Plot ALL individual raw data points
            ax.scatter(DiffProxy, data_p['N_s_cm3'], 
                      s=80, alpha=0.5, color=color, 
                      edgecolors='black', linewidth=0.8, zorder=3)
            
            # Linear fit on raw data
            if len(data_p) > 1:
                slope, intercept, r_value, p_value, std_err = linregress(DiffProxy, data_p['N_s_cm3'])
                r_squared = r_value ** 2
                
                # Plot fit line
                T_range = np.array([DiffProxy.min() - 100, DiffProxy.max() + 100])
                N_fit = slope * T_range + intercept
                ax.plot(T_range, N_fit, color=color, linestyle='-', linewidth=2.5, alpha=0.9, zorder=2,
                       label=f'{pressure} Torr (R² = {r_squared:.4f}, n={len(data_p)} points)')
                
                print(f"{pressure:>12.1f} | {slope:>18.3e} | {intercept:>15.3e} | {r_squared:>8.4f} | {p_value:>12.2e} | {len(data_p):>8}")
    
    print("-" * 100)
    print(f"\nTotal raw data points plotted: {len(data_argon_raw)}")
    
    ax.set_xlabel(r'$T^{3/2}$ (K$^{3/2}$)', fontsize=13, fontweight='bold')
    ax.set_ylabel(r'Metastable Density $N_s$ (cm$^{-3}$)', fontsize=13, fontweight='bold')
    ax.set_title('Metastable Density vs Gas Temperature (Pure Argon, Raw Data)', 
                fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='best', fontsize=11, title='Pressure (R² values)', title_fontsize=11, 
             framealpha=0.95, edgecolor='black', fancybox=True)
    ax.grid(True, alpha=0.3, linestyle='--', zorder=0)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    
    plt.tight_layout()
    plt.show()
else:
    print("✗ No raw data for pure argon at target pressures")


#%%

print("="*60)
print("CONTOUR PLOTTING COMPLETE")
print("="*60)

print("\nGenerated files:")
print("  - MasterResults_Summary_with_Tg_Ns.csv (enhanced dataframe)")
print("  - contour_01_Tg_power_pressure.png")
print("  - contour_02_Ns_power_pressure.png")
print("  - contour_03_Tg_nitrogen_power.png")
print("  - contour_04_Ns_pressure_nitrogen.png")

print("\n" + "="*60)
print("PHYSICAL QUANTITIES")
print("="*60)
print("\nT_gas (Gas Temperature):")
print("  - From Doppler broadening of Ar I 695.7 nm line")
print("  - Units: Kelvin (K)")
print("  - Typical range: 300-5000 K for low-temperature plasmas")

print("\nN_s (Metastable Density):")
print("  - Density of Ar metastable atoms (1s₅ state)")
print("  - Units: cm⁻³")
print("  - Typical range: 10⁸ - 10¹¹ cm⁻³ for LTP")

print("\nConstants used:")
print(f"  λ₀ = {LAMBDA_0*1e9:.1f} nm")
print(f"  M = {M_AR} amu (Ar mass)")
print(f"  gᵢ/gₖ = {G_I}/{G_K} (statistical weights)")
print(f"  Aₖᵢ = {A_KI:.2e} s⁻¹ (Einstein coefficient)")
