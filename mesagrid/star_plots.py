import numpy as np
from ipywidgets import interact, FloatSlider, IntSlider
import matplotlib as mpl
import matplotlib.pyplot as plt

red = "#CA0020"
orange = "#F97100" 
blue = "#0571b0"

# labels 
density = r'density $\mathbf{\rho~/~[g/cm^3]}$'
frac_radius = r'fractional radius $\mathbf{r/R_\odot}$'
frac_mass = r'fractional mass $\mathbf{m/M_\odot}$'
Teff = r'effective temperature $\mathbf{T_{eff}/K}$'
luminosity = r'luminosity $\mathbf{L/L_\odot}$'
frequency = r'frequency $\mathbf{\nu/\mu Hz}$'
numodDnu = r'$\mathbf{\nu}$ mod $\mathbf{\Delta\nu/\mu Hz}$'

def plot_hr(track, profile_number=-1):
    hist = track.history
    plt.plot(10**hist['log_Teff'], 
             10**hist['log_L'], lw=1, c='k', zorder=-9999)

    if profile_number > 0:
        hist = track.get_history(profile_number)
        plt.plot(10**hist['log_Teff'], 
                 10**hist['log_L'], 'o', c=red, ms=15, mfc='none', mew=2, zorder=-9)

    plt.gca().invert_xaxis()
    plt.xlabel(Teff)
    plt.ylabel(luminosity)

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
    
    x = gyre.x
    brunt = gyre.N/(2*np.pi)*1e6
    brunt[brunt<0]=1e-10
    brunt[np.isnan(brunt)]=1e-10
    
    lamb = np.sqrt(1*(1+1))*gyre.cs / gyre.r * 1e6
    
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
    
    nu_max = hist.nu_max.values[0]
    Dnu = hist.Dnu0.values[0]
    
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
    interact(lambda profile_number: 
             plot_panels(track, profile_number), 
             profile_number=IntSlider(min=1, max=np.max(track.index.profile_number)));
