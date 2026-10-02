# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 15:29:45 2026

@author: dptro
"""

"""
Calculate gas temperature and metastable density from Gaussian fit parameters
Ar I 5s line at 695.7 nm

Equations:
  T_g = M × (Δλ_g / (7.16 × 10^(-7) × λ_0))^2
  N_s = (8π g_i c) / (λ_0 g_k A_ki) × ∫OD(λ)dλ
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

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
summary = pd.read_csv('MasterResults_Summary.csv')

print(f"  Loaded {len(summary)} rows\n")

# ============================================================
# CALCULATE GAS TEMPERATURE FROM FWHM
# ============================================================
print("Calculating gas temperature from FWHM...")

# Load raw data to get FWHM (need to reconstruct from raw data)
print("  Note: Loading raw data for FWHM values...")
raw_data = pd.read_csv('MasterResults_RawData.csv')

# Group by (trial, power, pressure, n2_flow) and get mean FWHM
fwhm_grouped = raw_data.groupby(['trial', 'power', 'pressure', 'n2_flow'])['fwhm'].mean().reset_index()
fwhm_grouped = fwhm_grouped.rename(columns={'fwhm': 'fwhm_mean'})

# Merge FWHM back into summary
summary = summary.merge(fwhm_grouped, on=['trial', 'power', 'pressure', 'n2_flow'], how='left')

# Calculate Doppler broadening in wavelength space (convert from frequency to wavelength)
# FWHM_freq in Hz → FWHM_wavelength in m
# FWHM_lambda = FWHM_freq × (λ₀²/c)
summary['fwhm_wavelength'] = summary['fwhm_mean'] * (LAMBDA_0**2 / C)

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
od_integral_wavelength = summary['area_mean'] 
Freq = (LAMBDA_0 / C) *10**9
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
summary.to_csv('MasterResults_Summary_with_Tg_Ns.csv', index=False)
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
    ax.set_ylabel('N₂ Flow (sccm)', fontsize=12, fontweight='bold')
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
    
    ax.set_xlabel('N₂ Flow (sccm)', fontsize=12, fontweight='bold')
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