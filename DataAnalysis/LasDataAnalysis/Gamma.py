# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 14:16:45 2026

@author: dptro
"""

import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("G:\My Drive\MiniChamberData\Data\LAS_3%_ManyPoints\LAS_1.0Torr_SmallAdmixture_1kHzLaserFreq_CapCoupling_3%_N2_Trial\LAS_DataFrame_a404ea41_Laser_True_20260723_143719.csv")

grouped = df.groupby(['Nitrogen_Percent','power'])['gamma'].mean()
print(grouped)
matrix = grouped.unstack()
print(matrix)
plt.figure()
x = matrix.columns
y = matrix.index
c = plt.contourf(x,y,matrix,levels = 20)
#plt.axvline(x=1.3, color='black', linestyle='--')
plt.scatter(df['power'], df['Nitrogen_Percent'], marker='x', color='red')
plt.xlabel('Power / W')
plt.ylabel('Nitrogen Percent / %')
plt.colorbar(c,label='Reflection Coefficient / a.u.')
plt.show



  




"""
Normalize Ar(1s5) metastable density by its known loss channels to recover
an apparent source (production) rate.
 
In steady state:   dn_m/dt = 0  =>  S = n_m * nu_loss
with
    nu_loss = nu_diff + k_q * n_N2 + k_e * n_e   (+ two-body/three-body Ar terms)
 
We can compute the first two terms from measured quantities.  The k_e*n_e
term (electron mixing 1s5 -> 2p, k_e ~ 1e-7 cm3/s) is NOT included because
n_e is not measured -- see the docstring of metastable_source_rate for why
that matters for interpretation.
 
All internal arithmetic is CGS: cm^-3, cm^3/s, cm^2/s, cm, s^-1.
"""
 
import numpy as np
import matplotlib.pyplot as plt
 
# ---------------------------------------------------------------- constants
# n_tot [cm^-3] = TORR_K / T[K] * p[Torr]
#   1 Torr = 133.322 Pa; n = P/(k_B T)
TORR_K = 133.322 / 1.380649e-23 / 1e6   # = 9.657e18  (cm^-3 K Torr^-1)
 
 
def loss_rates(p_torr, T_K, f_n2,
               radius_cm, length_cm,
               k_q_300=3.6e-11, k_q_texp=0.5,
               Dp_300=60.0):
    """
    Return (nu_diff, nu_quench, n_N2) in s^-1, s^-1, cm^-3.
 
    Parameters
    ----------
    f_n2 : N2 mole *fraction* (not percent).
    k_q_300 : Ar(1s5) + N2 -> Ar + N2(C) rate coefficient at 300 K [cm3/s].
        Setser-group value ~3.6e-11.  Treat as +/-30%.
    k_q_texp : exponent in k_q ~ T^n.  0.5 = constant cross section
        (pure thermal-velocity scaling).  This is the weakly constrained
        knob; the reaction is near-resonant excitation transfer, so a
        constant-sigma assumption is defensible but not measured.
    Dp_300 : D*p for Ar(1s5) in Ar at 300 K [Torr cm2/s].  ~60.
        D scales as T^1.5 / p at fixed pressure.
    """
    T_K = np.asarray(T_K, dtype=float)
 
    n_tot = TORR_K * p_torr / T_K          # cm^-3
    n_N2 = f_n2 * n_tot                    # cm^-3
 
    k_q = k_q_300 * (T_K / 300.0) ** k_q_texp
    nu_quench = k_q * n_N2
 
    # fundamental-mode diffusion length for a finite cylinder
    inv_L2 = (2.405 / radius_cm) ** 2 + (np.pi / length_cm) ** 2
    D = (Dp_300 / p_torr) * (T_K / 300.0) ** 1.5
    nu_diff = D * inv_L2
 
    return nu_diff, nu_quench, n_N2
 
 
def metastable_source_rate(df, radius_cm, length_cm,
                           n2_is_percent=True,
                           T_override=None,
                           **kw):
    """
    Add columns nu_diff, nu_quench, nu_partial, S_app, S_app_err to a copy
    of df and return it.
 
    S_app = n_m * (nu_diff + k_q n_N2)   [cm^-3 s^-1]
 
    INTERPRETATION WARNING
    ----------------------
    Because electron mixing (k_e n_e) is omitted,
        S_app = S_true * (nu_diff + nu_quench) / (nu_diff + nu_quench + k_e n_e)
    At n_e ~ 1e11 the omitted term is ~1e4 s^-1, i.e. LARGER than quenching
    at 1% N2 (~5e3 s^-1).  So S_app is suppressed wherever n_e is high --
    which is wherever power is high.  That means this normalization can
    *manufacture* an anticorrelation between S_app and Gamma.  S_app is a
    lower bound on the true source rate, most reliable at low power.
 
    T_override : optional array of gas temperatures to use instead of
        df['T_gas_K'] (e.g. a version smoothed across the 1.3% notch), so
        you can tell which structure comes from the normalization itself.
    """
    out = df.copy()
 
    f_n2 = out['n2_flow'].to_numpy(dtype=float)
    if n2_is_percent:
        f_n2 = f_n2 / 100.0
 
    p = out['pressure'].to_numpy(dtype=float)
    T = out['T_gas_K'].to_numpy(dtype=float) if T_override is None \
        else np.asarray(T_override, dtype=float)
    dT = out['T_gas_K_err'].to_numpy(dtype=float)
 
    n_m = out['N_s_cm3'].to_numpy(dtype=float)
    dn_m = out['N_s_cm3_err'].to_numpy(dtype=float)
 
    nu_d, nu_q, n_N2 = loss_rates(p, T, f_n2, radius_cm, length_cm, **kw)
    nu_tot = nu_d + nu_q
    S = n_m * nu_tot
 
    # numerical propagation of the T uncertainty through nu(T); avoids
    # hand-differentiating three different T powers
    nu_hi = np.sum(loss_rates(p, T + dT, f_n2, radius_cm, length_cm, **kw)[:2], axis=0)
    nu_lo = np.sum(loss_rates(p, np.maximum(T - dT, 1.0), f_n2,
                              radius_cm, length_cm, **kw)[:2], axis=0)
    dnu = 0.5 * np.abs(nu_hi - nu_lo)
 
    # independent relative errors in n_m and nu
    rel = np.hypot(dn_m / n_m, dnu / nu_tot)
 
    out['n_N2_cm3'] = n_N2
    out['nu_diff'] = nu_d
    out['nu_quench'] = nu_q
    out['nu_partial'] = nu_tot
    out['S_app'] = S
    out['S_app_err'] = S * rel
    return out
 
 
def plot_normalization(df, radius_cm, length_cm, pressure=1.0,
                       n2_is_percent=True, T_override=None, **kw):
    """
    Three panels:
      (a) raw metastable density        -- what you already have
      (b) S_app = n_m * nu_partial      -- quenching divided out
      (c) S_app vs power, one trace per N2 setpoint
 
    Panel (c) is the one that answers the Gamma question: if the traces are
    flat, all the power dependence of n_m was loss-driven.  If they still
    slope down with power, something beyond quenching+diffusion is at work
    (electron mixing being the prime suspect).
 
    Uses tricontourf on the raw points rather than griddata onto a mesh, so
    no interpolation is invented between setpoints.
    """
    d = df[np.isclose(df['pressure'], pressure)]
    d = metastable_source_rate(d, radius_cm, length_cm,
                               n2_is_percent=n2_is_percent,
                               T_override=T_override, **kw)
 
    x = d['power'].to_numpy(dtype=float)
    y = d['n2_flow'].to_numpy(dtype=float)
 
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
 
    for a, key, lab, cmap in (
        (ax[0], 'N_s_cm3', r'$n_{\rm 1s_5}$ / cm$^{-3}$', 'plasma'),
        (ax[1], 'S_app',   r'$S_{\rm app}$ / cm$^{-3}$s$^{-1}$', 'viridis'),
    ):
        z = d[key].to_numpy(dtype=float)
        c = a.tricontourf(x, y, z, levels=24, cmap=cmap)
        a.plot(x, y, 'rx', ms=4, lw=0.6, alpha=0.55)
        fig.colorbar(c, ax=a, label=lab)
        a.set_xlabel('Power / W')
        a.set_ylabel(r'N$_2$ / %')
 
    ax[0].set_title('measured metastable density')
    ax[1].set_title(r'normalized by $\nu_{\rm diff}+k_q n_{\rm N_2}$')
 
    # panel (c): collapse test.  S_app spans ~2 decades across the N2 axis
    # (because nu_quench >> nu_diff above ~0.1% N2), which would bury the
    # power dependence.  So each trace is divided by its own lowest-power
    # value -- this isolates the *shape* in power, which is the quantity
    # that should be flat if quenching explained everything.
    setpts = np.unique(np.round(y, 2))
    norm = plt.Normalize(setpts.min(), setpts.max())
    cm = plt.get_cmap('coolwarm')
    S = d['S_app'].to_numpy(dtype=float)
    dS = d['S_app_err'].to_numpy(dtype=float)
    for s in setpts:
        m = np.isclose(np.round(y, 2), s)
        o = np.argsort(x[m])
        ref = S[m][o][0]
        if not np.isfinite(ref) or ref == 0:
            continue
        ax[2].errorbar(x[m][o], S[m][o] / ref, yerr=dS[m][o] / ref,
                       color=cm(norm(s)), marker='o', ms=3, lw=1,
                       capsize=1.5, alpha=0.85)
    ax[2].axhline(1.0, color='k', lw=0.8, ls='--')
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cm); sm.set_array([])
    fig.colorbar(sm, ax=ax[2], label=r'N$_2$ / %')
    ax[2].set_xlabel('Power / W')
    ax[2].set_ylabel(r'$S_{\rm app}(P)\,/\,S_{\rm app}(P_{\min})$')
    ax[2].set_title('flat = all power dependence was loss')
 
    fig.tight_layout()
    return fig, d


df1 = pd.read_csv(r"C:\Users\dptro\Documents\Work\Python\MiniChamberControlCode\DataAnalysis\LasDataAnalysis\MasterResults_Summary_with_Tg_Ns_03.csv")  
radius = 2
length = 4
plot_normalization(df1,radius,length)