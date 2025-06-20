import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import scipy
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
                     'axes.labelsize': 16,
                     'font.family': 'Serif',
                     'figure.figsize': (6, 4),
                     'mathtext.fontset': 'custom',
                     'mathtext.rm': 'Serif',
                     'mathtext.it': 'Serif:italic',
                     'mathtext.bf': 'Serif:bold',
                     'axes.labelpad' : 10,
                     'legend.fontsize' : 14
                    })

red = "#CA0020"
orange = "#F97100" 
blue = "#0571b0"
color1 = '#287271'
color2 = '#E76F51'


# labels 
density = r'Density $\rho$ [g cm$^{-3}$]'
frac_radius = r'Fractional Radius [R$_\odot$]'
frac_mass = r'Fractional Mass [M$_\odot$]'
Teff = 'Effective Temperature [K]'
luminosity = r'Luminosity [L$_\odot$]'
frequency = r'Frequency [$\mu$Hz]'
numodDnu = r'$\nu$ % $\Delta\nu$'

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def plot_colors(ax=None, zorder=-100, alpha=0.5):
    if ax is None:
        ax = plt.gca()

    y = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1])
    xlim = ax.get_xlim()

    star_colors = pd.read_table(os.path.join(project_root, 'mesagrid/bbr_color.txt'), skiprows=19, header=None, sep=r'\s+').iloc[1::2, :]
    star_colors.columns = ['temperature', 'unit', 'deg', 'x', 'y', 'power', 'R', 'G', 'B', 'r', 'g', 'b', 'hex']

    for temp, hex in zip(star_colors.temperature, star_colors.hex):
        ax.fill_between(np.linspace(float(temp), float(temp)+100), y[0], y[-1], color=hex, zorder=zorder)

    ax.add_patch(plt.Rectangle((xlim[0], y[0]), xlim[1]-xlim[0], y[-1]-y[0], facecolor='white', edgecolor='none', alpha=1-alpha, zorder=zorder+1))

    ax.set_xlim(xlim)
    ax.set_ylim(y[0], y[-1])


def plot_colors_interp(track, x, ax=None):
    if ax is None:
        ax = plt.gca()

    y = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1])
    xlim = ax.get_xlim()

    temp_to_x = scipy.interpolate.interp1d(10**track.history['log_Teff'], x, fill_value='extrapolate')

    star_colors = pd.read_table('bbr_color.txt', skiprows=19, header=None, sep=r'\s+').iloc[1::2, :]
    star_colors.columns = ['temperature', 'unit', 'deg', 'x', 'y', 'power', 'R', 'G', 'B', 'r', 'g', 'b', 'hex']
    for temp, hex in zip(star_colors.temperature, star_colors.hex):
        ax.fill_between(np.linspace(temp_to_x(float(temp)), temp_to_x(float(temp)+100)), y[0], y[-1], color=hex, zorder=-99)

    
    ax.set_xlim(xlim)
    ax.set_ylim(y[0], y[-1])




def plot_hr(track, profile_number=-1, show_profiles=False, solar_symbol=False, ax=None, color='k', alpha=1, alpha_colors=0.5, label=None):
    if ax is None:
        ax = plt.gca()
    if label is None:
        label = track.name
        
    hist = track.history
    ax.plot(10**hist['log_Teff'], 
             10**hist['log_L'], lw=2, alpha=alpha, c=color, zorder=-9, label=label)

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

    plot_colors(ax=ax, alpha=alpha_colors)

    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_yaxis().set_minor_formatter(matplotlib.ticker.ScalarFormatter())

    ax.set_xlabel(Teff)
    ax.set_ylabel(luminosity)


def plot_composition_old(track, profile_number):
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


def plot_composition(track, profile_number, mass=True, ax=None, title=None):
    if ax is None:
        ax = plt.gca()
    if title is None:
        title = track.name

    if track._profiles is not None:
        prof = track.profiles[profile_number-1]
    else:
        prof = track.load_profile(profile_number)

    x = prof.mass

    # Get Convection Zones in Mass Coordinate
    c1 = track.history['mass_conv_core'][profile_number]
    c2 = track.history['conv_mx1_bot'][profile_number]*max(x)
    c3 = track.history['conv_mx1_top'][profile_number]*max(x)
    c4 = track.history['conv_mx2_bot'][profile_number]*max(x)
    c5 = track.history['conv_mx2_top'][profile_number]*max(x)
    
    # Get Hydrogen and Helium Burning Regions in Mass Coordinate
    ppcnomin = np.min(prof.mass[(prof.pp+prof.cno) > 0.001])
    ppcnomax = np.max(prof.mass[(prof.pp+prof.cno) > 0.001]) 

    try:
        triamin = np.min(prof.mass[(prof.tri_alpha) > 0.001]) 
        triamax = np.max(prof.mass[(prof.tri_alpha) > 0.001])
    except:
        triamin = np.min(prof.mass[(prof.tri_alfa) > 0.001]) 
        triamax = np.max(prof.mass[(prof.tri_alfa) > 0.001])
    
    ax.set_xlabel(frac_mass)

    if not mass:
        x = 10**prof.logR / max(10**prof.logR)

        mass_to_radius = scipy.interpolate.interp1d(prof.mass, x, fill_value='extrapolate')

        # Convert Convection Zones to Radius Coordinate
        c1 = mass_to_radius(c1)
        c2 = mass_to_radius(c2)
        c3 = mass_to_radius(c3)
        c4 = mass_to_radius(c4)
        c5 = mass_to_radius(c5)

        # Convert Hydrogen and Helium Burning Regions to Radius Coordinate
        ppcnomin = mass_to_radius(ppcnomin)
        ppcnomax = mass_to_radius(ppcnomax)
        triamin = mass_to_radius(triamin)
        triamax = mass_to_radius(triamax)

        ax.set_xlabel(frac_radius)

    # Plot Convection Zones
    ax.fill_betweenx(np.linspace(0,1), 0,  c1, color='darkgray', zorder=-9999)
    ax.fill_betweenx(np.linspace(0,1), c2, c3, color='darkgray', zorder=-9999)
    ax.fill_betweenx(np.linspace(0,1), c4, c5, color='darkgray', zorder=-9999)

    # Plot Hydrogen + Helium Burning Zones
    ax.fill_betweenx(np.linspace(0, 1), ppcnomin, ppcnomax, hatch='\\\\', ec='k', fc='none', alpha=0.5, lw=0, zorder=-999)
    ax.fill_betweenx(np.linspace(0, 1), triamin,  triamax,  hatch='////', ec='k', fc='none', alpha=0.5, lw=0, zorder=-999)

    # Plot Abundances
    ax.fill_between(x, 0, prof.x_mass_fraction_H, color=color1, alpha=0.3)
    ax.fill_between(x, 0, prof.y_mass_fraction_He, color=color2, alpha=0.3)
    ax.plot(x, prof.x_mass_fraction_H, lw=3, label='H', c=color1)
    ax.plot(x, prof.y_mass_fraction_He, lw=3, label='He', c=color2)

    ax.set_xlim(0, max(x))
    ax.set_ylim(0, 1)
    ax.set_ylabel('Mass Fraction')
    ax.tick_params(axis='both', which='major')
    ax.tick_params(axis='both', which='minor')   


def plot_propagation(track, profile_number, ax=None, mass=False):
    if ax is None:
        ax = plt.gca()

    hist = track.get_history(profile_number)

    if track._gyres is not None:
        gyre = track.gyres[profile_number-1]
    else:
        gyre = track.load_gyre(profile_number)
    
    muHz = 10**6/(2*np.pi)

    x = gyre.x
    ax.set_xlabel(frac_radius)

    if mass:
        x = gyre.m/gyre.M
        ax.set_xlabel(frac_mass)

    
    brunt = gyre.N * muHz
    brunt[brunt<0] = 0
    brunt[np.isnan(brunt)] = 0
    
    r = gyre.r 
    r[r <= 0] = 1e-99
    lamb = np.sqrt(1*(1+1))*gyre.cs / r * muHz
    
    ax.plot(x, brunt, lw=3, color=color1, label='Buoyancy')
    ax.plot(x, lamb, lw=3, color=color2, label='Lamb')
    
    gmodes = np.minimum(brunt, lamb)
    pmodes = np.maximum(brunt, lamb)
    ax.fill_between(x, 
                     np.zeros(len(gmodes)), 
                     gmodes, 
                     color=color1, alpha=0.1, zorder=-99)
    ax.fill_between(x, 
                     1e99*np.ones(len(pmodes)), 
                     pmodes, 
                     color=color2, alpha=0.1, zorder=-99)
    
    nu_max   = hist.nu_max.values[0]
    Delta_nu = hist.delta_nu.values[0]
    ax.axhline(nu_max, ls='--', c='k', label=r'$\nu_\max$', zorder=100)
    ax.fill_between([0, 1], 
                     nu_max-2*Delta_nu, 
                     nu_max+2*Delta_nu, 
                     color='#aaaaaa', alpha=0.5, zorder=-98)
    
    ax.set_yscale('log')
    ax.set_ylim([1e1, 1e5])
    ax.set_xlim([0,1])
    ax.set_ylabel(frequency)


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
             profile_number=IntSlider(min=1, max=np.max(track.index.profile_number)))
    

def plot_kippenhahn_extras(fig):
    fig.supxlabel('Age [Gyr]')
    fig.supylabel(r'Fractional Mass [m / M$_\odot$]', x=0)

    norm = matplotlib.colors.Normalize(vmin=-4, vmax=0)
    cmap = matplotlib.cm.ScalarMappable(norm=norm, cmap='YlOrRd')
    vmin = int(norm.vmin)
    vmax = int(norm.vmax)

    cbar_ax = fig.add_axes([1, 0.2, 0.02, 0.6])
    cb = fig.colorbar(cmap, label=r'Buoyancy frequency $\mathrm{log~N}$/Hz',
                        boundaries=np.array(range(vmin, vmax+2, 1))-0.5,
                        ticks=np.array(range(vmin, vmax+1, 1)), cax=cbar_ax)
    cb.ax.minorticks_off()
    cb.ax.set_yticklabels([r'$10^{-4}$',r'$10^{-3}$',r'$10^{-2}$',r'$10^{-1}$',r'$10^{0}$'])
    cb.set_label(label=r'Buoyancy frequency [Hz]', labelpad=10)

    fig.tight_layout()


def plot_kippenhahn(track, ax=None, plot_extras=False, title=None):
    if ax is None:
        fig = plt.figure(figsize=(7,6.5))
        ax = fig.gca()
    if title is None:
        title = track.name
        
    # Get axes values
    mass_max = np.min(np.array([np.max(prof.mass) for prof in track.profiles])) # (mass_max sets axes limits)
    xm = np.linspace(0, mass_max, 10000)
    ages = np.array([track.get_history(prof_num).star_age.values[0]/1e9 
                     for prof_num in track.index.profile_number])
    masses = np.array([track.get_history(prof_num).star_mass.values[0]
                     for prof_num in track.index.profile_number])
    

    # Plot Buoyancy Frequency and Convection
    X, Y = np.meshgrid(xm, ages)
    Z = np.array([scipy.interpolate.interp1d(g.m/(1.989e33), np.log10(g.N *  10**6/(2*np.pi)), 
                                            fill_value=np.nan, bounds_error=0)(xm) 
                                            for g in track.gyres])
    conv = np.array([scipy.interpolate.interp1d(g.m/(1.989e33), g.N<0, 
                fill_value=np.nan, bounds_error=0)(xm) 
            for g in track.gyres])
    
    norm = matplotlib.colors.Normalize(vmin=-4, vmax=0)
    cmap = matplotlib.cm.ScalarMappable(norm=norm, cmap='YlOrRd')
    vmin = int(norm.vmin)
    vmax = int(norm.vmax)

    ax.contourf(Y, X, conv, levels=[0,3], vmin=-1, vmax=3, cmap='Greys', zorder=-99999)
    ax.contourf(Y, X, Z, levels=np.arange(-4, 1, 0.2), vmin=vmin, vmax=vmax, cmap='YlOrRd', zorder=-99999)
    ax.set_rasterization_zorder(-1)

    # Plot Hydrogen and Helium Burning Zones
    ppcnomin = np.array([np.min(p.mass[(p.pp+p.cno) > 0.001]) for p in track.profiles])
    ppcnomax = np.array([np.max(p.mass[(p.pp+p.cno) > 0.001]) for p in track.profiles])
    try:
        triamin = np.array([np.min(p.mass[(p.tri_alpha) > 0.001]) for p in track.profiles])
        triamax = np.array([np.max(p.mass[(p.tri_alpha) > 0.001]) for p in track.profiles])
    except:
        triamin = np.array([np.min(p.mass[(p.tri_alfa) > 0.001]) for p in track.profiles])
        triamax = np.array([np.max(p.mass[(p.tri_alfa) > 0.001]) for p in track.profiles])

    ax.fill_between(ages, ppcnomin, ppcnomax, hatch='\\\\', ec='k', fc='none', alpha=0.8, lw=0, zorder=-9999)
    ax.fill_between(ages, triamin,  triamax,  hatch='////', ec='k', fc='none', alpha=1,   lw=0, zorder=-9999)
    
    # Plot Spectral Type Line
    star_colors = pd.read_table(os.path.join(project_root, 'mesagrid/bbr_color.txt'), skiprows=19, header=None, sep=r'\s+').iloc[1::2, :]
    star_colors.columns = ['temperature', 'unit', 'deg', 'x', 'y', 'power', 'R', 'G', 'B', 'r', 'g', 'b', 'hex']

    rgbs_in_teff = scipy.interpolate.interp1d(
        [float(t) for t in star_colors['temperature']], (star_colors['R'], star_colors['G'], star_colors['B'])
        )(10**track.history.log_Teff)
    
    ax.plot(ages, masses, c='w', lw=10, zorder=-999)
    for i in range(len(ages)):
        ax.plot(ages, masses, c=rgbs_in_teff[:, i], lw=8, zorder=-99)
        

    ax.set_xlabel('Age [Gyr]')
    ax.set_ylabel(frac_mass)
    ax.set_ylim(0, mass_max * 1.1)         
    ax.tick_params(axis='both', which='major')
    ax.tick_params(axis='both', which='minor')

    if plot_extras:
        plot_kippenhahn_extras(fig)


def plot_structure(track, axs=None):
    from ipywidgets import interact, IntSlider
    from IPython.display import display

    fig, axs = plt.subplots(1, 2, figsize=(12,6))
    plot_kippenhahn(track, ax=axs[0])
    line = [None]
    
    def change_profile(profile_num):
        axs[1].cla()

        if line[0]:
            line[0].remove()
            line[0] = False

        line[0] = axs[0].axvline(float(10**-9 * track.get_history(profile_num).star_age), color='black', linestyle='dashed')
        plot_composition(track, profile_number=profile_num, ax=axs[1])

    interact(lambda profile_number: change_profile(profile_number), profile_number=IntSlider(min=1, max=np.max(track.index.profile_number)))


def plot_temperature_gradients(track, profile_number, mass=True, ax=None, c1=color1, c2=color2, label=None):
    if ax is None:
        ax = plt.gca()
    if label is None:
        label=r'$\nabla_\mathrm{rad}$'
    
    if track._gyres is not None:
        gyre = track.gyres[profile_number-1]
    else:
        gyre = track.load_gyre(profile_number)
    
    x = gyre.m/gyre.M
    ax.set_xlabel(frac_mass)

    if not mass:
        x = gyre.r/gyre.R
        ax.set_xlabel(frac_radius)

    ax.plot(x, gyre.grad_a, color=c1, lw=3, label=r'$\nabla_\mathrm{ad}$')
    ax.plot(x, gyre.grad_r, color=c2, lw=3, label=label)

    ax.set_ylabel('Temperature Gradient')
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.legend()


def plot_deltapi_vs_x(track, x, ax=None, min=0, max=-1, color=color1, label=None):
    if ax is None:
        ax = plt.gca()
    if label is None:
        label = track.name

    if 'delta_Pg' in track.history.columns:
        pg = track.history['delta_Pg'].iloc[min:max]
    else:
        print('period spacing might be noisy')
        pg = [2*np.pi**2/np.sqrt(2)/ scipy.integrate.trapezoid(gyre.N[gyre.N>0]/gyre.x[gyre.N>0], gyre.x[gyre.N>0]) for gyre in track.gyres[min:max]]
    
    if isinstance(x, str):
        x = track.history[x].iloc[min:max]

    ax.plot(x, pg, color=color, lw=3, label=label)
    ax.set_ylabel(r'Period Spacing $\Delta\Pi$ [s]')
    

def plot_beta(track, ax=None, color=color1, label=None, min=0, max=-1):
    if ax is None:
        ax = plt.gca()
    if label is None:
        label = track.name

    x = track.history['Fundamental Period'].iloc[min:max]

    ax.plot(x[1:], np.diff(x)/np.diff(track.history['star_age'].iloc[min:max]) * 1e6/24, color=color, lw=3, label=label)
    
    ax.set_ylabel(r'Period Change $\beta$ [d Myr$^{-1}]$')
    ax.set_xlabel('Fundamental Period [h]')


def plot_exact_deltapi(track, profile_num, tag='', ell=1, ax=None, color=color1):
    if ax is None:
        ax = plt.gca()
    frequencies = track.load_freq(profile_num, tag=tag)

    # get g-mode radial orders:
    frequencies.n_g = [int(x) for x in frequencies.n_g]

    # get only g-modes/mixed modes:
    dipole_g = frequencies[np.logical_and(frequencies.l == ell, frequencies.n_g > 0)] 
    dipole_g['P']  = 1/(dipole_g['Re(freq)'] * 10**-6) # seconds
    
    for mode in dipole_g.iterrows():
        # get sequential modes
        n_pg = mode[1]['n_pg']
        n_pg2 = dipole_g[dipole_g['n_pg'] == n_pg-1]

        if not n_pg2.empty:
            dP = (n_pg2['P'].values[0] - mode[1]['P']) # seconds
            dipole_g.loc[mode[0], 'dP'] = dP

    if 'delta_Pg' in track.history.columns:
        pg = track.history['delta_Pg'][profile_num]
    else:
        gyre = track.load_gyre(profile_num)
        pg = 2*np.pi**2/np.sqrt(2)/ scipy.integrate.trapezoid(gyre.N[gyre.N>0]/gyre.x[gyre.N>0], gyre.x[gyre.N>0])
    ax.scatter(dipole_g['P']/3600, dipole_g['dP'], s=10, color=color)
    
    ax.plot(dipole_g['P'].iloc[1:]/3600, dipole_g['dP'].iloc[1:], alpha=0.5, linestyle='dashed', lw=1, color=color)
    ax.axhline(pg, ls='--', c='k', label='asymptotic')

    ax.set_xlabel('Period [h]')
    ax.set_ylabel('Exact Period Spacing [s]')