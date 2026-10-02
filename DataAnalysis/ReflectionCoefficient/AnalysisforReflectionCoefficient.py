# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 14:03:32 2026

@author: dptro
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv(r"C:\Users\dptro\Documents\Work\Python\MiniChamberControlCode\Data\ReflectionCoefficientWithNitrogen_20260519_202354\Reflection_sweep_537be2de_20260519_202407.csv")

NitPer = df['Nitrogen_Percent'].unique()
Powers = df['power'].unique()
# Build pivot tables for both gamma and gamma_error
pivots_gamma = {}
pivots_gamma_error = {}

for i in NitPer:
    df_n = df[df['Nitrogen_Percent'] == i]
    pivots_gamma[i] = pd.pivot_table(df_n, columns='frequency', index='power',
                                      values='gamma', aggfunc='mean')
    pivots_gamma_error[i] = pd.pivot_table(df_n, columns='frequency', index='power',
                                            values='gamma_error', aggfunc='mean')

# Add new columns to original dataframe
df['min_gamma'] = None
df['min_gamma_freq'] = None
df['min_gamma_error'] = None
df['min_gamma_error_freq'] = None

for n_val in NitPer:
    mask = df['Nitrogen_Percent'] == n_val

    # Min gamma
    min_gamma = pivots_gamma[n_val].min(axis=1)
    min_gamma_freq = pivots_gamma[n_val].idxmin(axis=1)
    df.loc[mask, 'min_gamma'] = df.loc[mask, 'power'].map(min_gamma)
    df.loc[mask, 'min_gamma_freq'] = df.loc[mask, 'power'].map(min_gamma_freq)

    # Min gamma_error
    min_gamma_error = pivots_gamma_error[n_val].min(axis=1)
    min_gamma_error_freq = pivots_gamma_error[n_val].idxmin(axis=1)
    df.loc[mask, 'min_gamma_error'] = df.loc[mask, 'power'].map(min_gamma_error)
    df.loc[mask, 'min_gamma_error_freq'] = df.loc[mask, 'power'].map(min_gamma_error_freq)

print(df[['power', 'Nitrogen_Percent', 'gamma', 'min_gamma', 'min_gamma_freq',
          'gamma_error', 'min_gamma_error', 'min_gamma_error_freq']])

# Build a summary dataframe: for each (Nitrogen_Percent, power), get the min_gamma
summary = df.drop_duplicates(subset=['Nitrogen_Percent', 'power'])[
    ['Nitrogen_Percent', 'power', 'min_gamma', 'min_gamma_error']
].sort_values(['power', 'Nitrogen_Percent']).copy()

summary['min_gamma'] = summary['min_gamma'].astype(float)
summary['min_gamma_error'] = summary['min_gamma_error'].astype(float)


summary_5 = summary[summary['Nitrogen_Percent'] == 6].copy()

fig, ax = plt.subplots(figsize=(8, 6))

ax.errorbar(summary_5['power'], summary_5['min_gamma'],
            yerr=summary_5['min_gamma_error'],
            marker='o', capsize=3, color='steelblue')

ax.set_xlabel('Power (W)')
ax.set_ylabel('Optimum Gamma / a.u.')
ax.set_title('Optimum Gamma at 5% Nitrogen')
ax.grid(True)
plt.tight_layout()
plt.show()


# Build summary dataframe - cast to float to fix the TypeError
summary = df.drop_duplicates(subset=['Nitrogen_Percent', 'power'])[
    ['Nitrogen_Percent', 'power', 'min_gamma']
].sort_values(['power', 'Nitrogen_Percent']).copy()

summary['min_gamma'] = summary['min_gamma'].astype(float)  # ← fixes the error

# Pivot into 2D grid
grid = summary.pivot(index='power', columns='Nitrogen_Percent', values='min_gamma')

X = grid.columns.values.astype(float)
Y = grid.index.values.astype(float)
Z = grid.values.astype(float)

# Plot
fig, ax = plt.subplots(figsize=(8, 6))

contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis')
cbar = fig.colorbar(contour, ax=ax)
cbar.set_label('Optimum Gamma', rotation=270, labelpad=15)

ax.set_xlabel('Nitrogen Percent / %')
ax.set_ylabel('Power (W)')
ax.set_title('Optimum Gamma / a.u.')

plt.tight_layout()
plt.show()


summary = df.drop_duplicates(subset=['Nitrogen_Percent', 'power'])[
    ['Nitrogen_Percent', 'power', 'min_gamma', 'min_gamma_error']
].sort_values(['power', 'Nitrogen_Percent']).copy()

summary['min_gamma'] = summary['min_gamma'].astype(float)
summary['min_gamma_error'] = summary['min_gamma_error'].astype(float)

fig, ax = plt.subplots(figsize=(10, 6))

for p in Powers:
    df_p = summary[summary['power'] == p]
    ax.errorbar(df_p['Nitrogen_Percent'], df_p['min_gamma'],
                yerr=df_p['min_gamma_error'],
                marker='o', capsize=3, label=f'{p} W')

ax.set_xlabel('Nitrogen Percent (%)')
ax.set_ylabel('Min Gamma')
ax.set_title('Minimum Gamma vs Nitrogen Percent for Each Power')
ax.legend(title='Power', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True)
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt

# Filter to 6% nitrogen
df_6 = df[np.isclose(df['Nitrogen_Percent'], 6, atol=0.1)]

fig, ax = plt.subplots(figsize=(10, 6))

for p in sorted(df_6['power'].unique()):
    df_p = df_6[df_6['power'] == p]
    grp = df_p.groupby('frequency')[['gamma', 'gamma_error']].mean().reset_index()
    ax.errorbar(grp['frequency'], grp['gamma'],
                yerr=grp['gamma_error'],
                marker='o', markersize=3, capsize=3, label=f'{p} W')

ax.set_xlabel('Frequency (MHz)')
ax.set_ylabel('Gamma (reflection coefficient)')
ax.set_title('Reflection vs Frequency — Nitrogen 6.0%')
ax.legend(title='Power', loc='lower right')

plt.show()