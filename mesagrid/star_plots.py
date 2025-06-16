import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib
import os
from datetime import datetime
print(f'Updated starplots.py {datetime.now()}')


plt.rcParams.update({'axes.linewidth' : 1,
                     'ytick.major.width' : 1,
                     'ytick.minor.width' : 1,
                     'xtick.major.width' : 1,
                     'xtick.minor.width' : 1,
                     'xtick.labelsize': 12, 
                     'ytick.labelsize': 12,
                     'axes.labelsize': 14,
                     'font.family': 'Serif',
                     'figure.figsize': (6, 4),
                     'mathtext.fontset': 'custom',
                     'mathtext.rm': 'Serif',
                     'mathtext.it': 'Serif:italic',
                     'mathtext.bf': 'Serif:bold'
                    })

red = "#CA0020"
orange = "#F97100" 
blue = "#0571b0"

# labels 
density = r'Density $\rho$ [g cm$^{-3}$]'
frac_radius = r'Fractional Radius [R$_\odot$]'
frac_mass = r'Fractional Mass [M$_\odot$]'
Teff = 'Effective Temperature [K]'
luminosity = r'Luminosity [L$_\odot$]'
frequency = r'Frequency [$\mu$Hz]'
numodDnu = r'$\nu$ % $\Delta$\nu$'

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def plot_colors(ax=None, zorder=-100, alpha=0.5):
    if ax is None:
        ax = plt.gca()

    y = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1])
    xlim = ax.get_xlim()

    star_colors = pd.read_table(os.path.join(project_root, './docs/bbr_color.txt'), skiprows=19, header=None, sep=r'\s+').iloc[1::2, :]
    star_colors.columns = ['temperature', 'unit', 'deg', 'x', 'y', 'power', 'R', 'G', 'B', 'r', 'g', 'b', 'hex']

    for temp, hex in zip(star_colors.temperature, star_colors.hex):
        ax.fill_between(np.linspace(float(temp), float(temp)+100), y[0], y[-1], color=hex, zorder=zorder)

    ax.add_patch(plt.Rectangle((xlim[0], y[0]), xlim[1]-xlim[0], y[-1]-y[0], facecolor='white', edgecolor='none', alpha=1-alpha, zorder=zorder+1))

    ax.set_xlim(xlim)
    ax.set_ylim(y[0], y[-1])



def plot_hr(track, profile_number=-1, show_profiles=False, solar_symbol=False, ax=None):
    if ax is None:
        ax = plt.gca()

    hist = track.history
    ax.plot(10**hist['log_Teff'], 
             10**hist['log_L'], lw=1, c='k', zorder=-9)

    if show_profiles:
        for prof_num in track.index.profile_number:
            hist = track.get_history(prof_num)
            ax.plot(10**hist['log_Teff'], 
                     10**hist['log_L'], '.', c='k', ms=3, mfc='none', mew=2, zorder=-9)

    if profile_number > 0:
        hist = track.get_history(profile_number)
        ax.plot(10**hist['log_Teff'], 
                 10**hist['log_L'], 'o', c=red, ms=15, mfc='none', mew=2, zorder=-9)

    if solar_symbol:
        ax.plot(5772.003429098915, 1, 'k.')
        ax.plot(5772.003429098915, 1, 'ko', mfc='none', ms=10)

    ax.invert_xaxis()
    ax.set_yscale('log')

    
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_yaxis().set_minor_formatter(matplotlib.ticker.ScalarFormatter())

    ax.set_xlabel(Teff)
    ax.set_ylabel(luminosity)

def plot_composition(track, profile_number):
    profs = track.profiles
    ZAMS_X = profs[0].x_mass_fraction_H.values[0]
    ZAMS_Y = profs[0].y_mass_fraction_He.values[0]
    Y_p = 0.2463
    
    prof = track.profiles[profile_number-1]
    x = 10**prof.logR / np.max(10**prof.logR)
    plt.plot(x, prof.x_mass_fraction_H, lw=5, label='Hydrogen', c='k')
    plt.plot(x, prof.y_mass_fraction_He, lw=5, label='Helium', c='b')
    plt.axhline(ZAMS_X, c='k', ls='--', zorder=-99)
    plt.axhline(ZAMS_Y, c='k', ls='--', zorder=-99)
    plt.axhline(Y_p, c='lightgray', ls='--', zorder=-99)
    
    plt.ylim([0, 1])
    
    plt.xlabel(frac_radius)
    plt.ylabel(r'mass fraction')
    plt.legend()

def plot_propagation(track, profile_number):
    hist = track.get_history(profile_number)
    gyre = track.gyres[profile_number-1]
    
    muHz = 10**6/(2*np.pi)

    x = gyre.x
    
    brunt = gyre.N * muHz
    brunt[brunt<0] = 0
    brunt[np.isnan(brunt)] = 0
    
    r = gyre.r 
    r[r <= 0] = 1e-99
    lamb = np.sqrt(1*(1+1))*gyre.cs / r * muHz
    
    plt.plot(x, brunt, lw=3, label='Buoyancy')
    plt.plot(x, lamb, lw=3, label='Lamb')
    
    gmodes = np.minimum(brunt, lamb)
    pmodes = np.maximum(brunt, lamb)
    plt.fill_between(x, 
                     np.zeros(len(gmodes)), 
                     gmodes, 
                     color='blue', alpha=0.1, zorder=-99)
    plt.fill_between(x, 
                     1e99*np.ones(len(pmodes)), 
                     pmodes, 
                     color='orange', alpha=0.1, zorder=-99)
    
    nu_max   = hist.nu_max.values[0]
    Delta_nu = hist.delta_nu.values[0]
    plt.axhline(nu_max, ls='--', c='k', label=r'$\nu_\max$', zorder=100)
    plt.fill_between([0, 1], 
                     nu_max-5*Delta_nu, 
                     nu_max+5*Delta_nu, 
                     color='#aaaaaa', zorder=-98)
    
    plt.semilogy()
    plt.ylim([1e1, 1e4])
    plt.xlim([0,1])
    plt.ylabel(frequency)
    plt.xlabel(frac_radius)
    plt.legend()

def plot_echelle(track, profile_number, sph_deg=-1, rad_ord=-1):
    ell_label = {0: 'radial', 1: 'dipole', 2: 'quadrupole', 3: 'octupole'}
    
    hist = track.get_history(profile_number)
    profs = track.profiles
    freqs = track.freqs
    
    prof = profs[profile_number-1] if profile_number < len(profs) else profs[0]
    freq = freqs[profile_number-1] if profile_number < len(freqs) else freqs[0]
    
    radial = freq[freq.l == 0]
    Dnu = np.median(np.diff(radial['Re(freq)']))

    nu_max = hist.nu_max.values[0]
    #Dnu = hist.delta_nu.values[0]
    
    if np.isnan(Dnu):
        plt.ylabel(frequency)
        plt.xlabel(numodDnu)
        return 
    
    colors = ('k', red, blue, orange)
    markers = ('s', 'D', 'o', '^')
    for ell in np.unique(freq.l.values):
        nus = freq[freq.l == ell]
        for ii in [0, 1]:
            plt.plot(nus['Re(freq)'] % Dnu + Dnu*ii,
                 nus['Re(freq)'], marker=markers[ell], ls='none',#'.', 
                 mfc=colors[ell], mec='white', alpha=0.85,
                 ms=8, mew=1,  
                 label=str(ell) + ' (' + ell_label[ell] + ')')
    
    if sph_deg >= 0 and rad_ord >= 0:
        freq = freq[np.logical_and(freq.l == sph_deg, freq.n_pg == rad_ord)]
        if len(freq) > 0:
            plt.plot(freq['Re(freq)'] % Dnu, freq['Re(freq)'], 'o', zorder=-99, mec='k', ms=10, mfc='w')
    
    plt.axvline(Dnu, ls='--', c='darkgray', zorder=-99)
    plt.axhline(nu_max, ls='--', c='darkgray', zorder=-99)
    
    plt.ylim([0, nu_max*5/3*1.1])
    plt.xlim([0, Dnu*1.5])
    
    plt.ylabel(frequency)
    plt.xlabel(numodDnu)

def plot_panels(track, profile_number):
    fig = plt.figure(figsize=(12,10))
    
    plt.subplot(2,2,1)
    plot_hr(track, profile_number)
    
    plt.subplot(2,2,2)
    plot_composition(track, profile_number)
    
    plt.subplot(2,2,3)
    plot_echelle(track, profile_number)
    
    plt.subplot(2,2,4)
    plot_propagation(track, profile_number)
    
    plt.tight_layout()

def star_interact(track):
    from ipywidgets import interact, IntSlider
    interact(lambda profile_number: 
             plot_panels(track, profile_number), 
             profile_number=IntSlider(min=1, max=np.max(track.index.profile_number)));
