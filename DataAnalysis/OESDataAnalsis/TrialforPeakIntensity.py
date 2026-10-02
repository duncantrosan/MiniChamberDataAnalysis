# -*- coding: utf-8 -*-
"""
Created on Sun May 31 12:09:56 2026

@author: dptro
"""

import SpectraAnalysis as SpecA
import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import interp1d


path = "G:\My Drive\MiniChamberData\Data\OES_baseline_files\OES_Pressure[1.45]_20260701_143057" 

df = SpecA.load_data(path)
BaseLineNumbers = [691.4,836.2,898.9,805.2,947.4]
#df = SpecA.baseline_correct_spectra(df,BaseLineNumbers)
n2_pcts  = sorted(df["Nitrogen_Percent"].unique())

Spectra0 = SpecA.mean_spectrum(df,65,1.3,n2_pcts[0])
Spec0 = Spectra0['intensity_mean']
#Spec0 = Spec0/np.max(Spec0)
Spectra1 = SpecA.mean_spectrum(df,65,1.3,n2_pcts[1])
Spec1 = Spectra1['intensity_mean']
#Spec1 = Spec1/np.max(Spec1)
Spectra2 = SpecA.mean_spectrum(df,65,1.3,n2_pcts[8])
Spec2 = Spectra2['intensity_mean']
#Spec2 = Spec2/np.max(Spec2)
Spectra3 = SpecA.mean_spectrum(df,65,1.3,n2_pcts[15])
Spec3 = Spectra3['intensity_mean']
#Spec3 = Spec3/np.max(Spec3)
WL = df['wavelength'].unique()




dfInten = SpecA.peak_intensity(df, 706.7)

powers   = sorted(df["power"].unique())


# Pivot to matrix: rows = power, cols = N2%
pivot = dfInten.pivot_table(index="power", columns="Nitrogen_Percent",
                       values='intensity_mean', aggfunc="mean")
pivot750 = pivot.reindex(index=powers, columns=n2_pcts)
R1 = pivot750.values          # shape (n_powers, n_n2)

dfInten = SpecA.peak_intensity(df, 811)
# Pivot to matrix: rows = power, cols = N2%
pivot = dfInten.pivot_table(index="power", columns="Nitrogen_Percent",
                       values='intensity_mean', aggfunc="mean")
pivot811 = pivot.reindex(index=powers, columns=n2_pcts)



R2 = pivot811.values          # shape (n_powers, n_n2)
Z = R1/R2

RatioTable = pivot750/pivot811
X,Y = np.meshgrid(n2_pcts, powers)


NumeratorLine = 821.6
DenominatorLine = 811
# Get std for both lines
dfInten_706 = SpecA.peak_intensity(df, NumeratorLine)
dfInten_826 = SpecA.peak_intensity(df,DenominatorLine)

# Pivot mean AND std for each
mean_706 = dfInten_706.pivot_table(index="power", columns="Nitrogen_Percent",
                                    values='intensity_mean', aggfunc="mean").reindex(index=powers, columns=n2_pcts)

std_706  = dfInten_706.pivot_table(index="power", columns="Nitrogen_Percent",
                                    values='intensity_std', aggfunc="mean").reindex(index=powers, columns=n2_pcts)

mean_826 = dfInten_826.pivot_table(index="power", columns="Nitrogen_Percent",
                                    values='intensity_mean', aggfunc="mean").reindex(index=powers, columns=n2_pcts)

std_826  = dfInten_826.pivot_table(index="power", columns="Nitrogen_Percent",
                                    values='intensity_std', aggfunc="mean").reindex(index=powers, columns=n2_pcts)


# Ratio and propagated error  (σ_R/R = sqrt((σ_a/a)² + (σ_b/b)²))
RatioTable = mean_706 / mean_826
RatioError = np.abs(RatioTable) * np.sqrt((std_706 / mean_706)**2 + (std_826 / mean_826)**2)



plt.figure()
plt.contourf(X,Y,Z,levels = 50)
plt.xlabel('Nitrogen Perrcent / %')
plt.ylabel('Powers / W')
plt.title('706.7/811 nm Line Ratio')
plt.colorbar()

powers_to_plot = [30,35, 40, 45, 50, 55, 60, 65]

# Ratio Plot
fig, ax = plt.subplots(figsize=(10, 5))

for pwr in powers_to_plot:
    y    = RatioTable.loc[pwr].values
    y = y - np.min(y)
    yerr = RatioError.loc[pwr].values
    x    = RatioTable.columns.values  # Nitrogen %

    ax.errorbar(x, y, yerr=yerr,
                label=f'{pwr} W',
                marker='o',
                capsize=4,          # little horizontal caps on error bars
                capthick=1.5,
                linewidth=1.5)

ax.set_title(f'{NumeratorLine}/{DenominatorLine} nm Line Ratio')
ax.set_xlabel('Nitrogen %')
ax.set_ylabel('Intensity Ratio / a.u.')
ax.legend(title='Power /W')
plt.tight_layout()
plt.show()

# Denominator Plot
fig, ax = plt.subplots(figsize=(10, 5))

for pwr in powers_to_plot:

    y    = mean_826.loc[pwr].values
    # y = y - np.min(y)
    yerr = std_826.loc[pwr].values
    x    = mean_826.columns.values  # Nitrogen %



    ax.errorbar(x, y, yerr=yerr,
                label=f'{pwr} W',
                marker='o',
                capsize=4,          # little horizontal caps on error bars
                capthick=1.5,
                linewidth=1.5)

ax.set_title(f'{DenominatorLine} nm Line Intensity')
ax.set_xlabel('Nitrogen %')
ax.set_ylabel('Intensity / a.u.')
ax.legend(title='Power /W')
plt.tight_layout()
plt.show()


# Numirator Plot
fig, ax = plt.subplots(figsize=(10, 5))

for pwr in powers_to_plot:
    y    = mean_706.loc[pwr].values
    y = y - 1800
    yerr = std_706.loc[pwr].values
    x    = mean_706.columns.values  # Nitrogen %

    ax.errorbar(x, y, yerr=yerr,
                label=f'{pwr} W',
                marker='o',
                capsize=4,          # little horizontal caps on error bars
                capthick=1.5,
                linewidth=1.5)

ax.set_title(f'{NumeratorLine} nm Line Intensity')
ax.set_xlabel('Nitrogen %')
ax.set_ylabel('Intensity / a.u.')
ax.legend(title='Power /W')
plt.tight_layout()
plt.show()



plt.figure()
plt.plot(WL,Spec0,label='0% $N_2$')
plt.plot(WL,Spec1,label=f'{n2_pcts[1]:.3g}% $N_2$')
plt.plot(WL,Spec2,label=f'{n2_pcts[8]:.3g}% $N_2$')
plt.plot(WL,Spec3,label=f'{n2_pcts[15]:.3g}% $N_2$')
plt.xlabel('Wavelength / nm')
plt.ylabel('Normilized Intensity / a.u.')
plt.xlim([600,950])
plt.legend()


