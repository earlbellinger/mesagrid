"""Main module."""

import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tomso import gyre
from concurrent.futures import ThreadPoolExecutor

import warnings
from tqdm import TqdmExperimentalWarning
warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
from tqdm.autonotebook import tqdm
from datetime import datetime

print(f'Updated star.py {datetime.now()}')


class Track:
    def __init__(self, dir, parameters=None, 
                 load_history_extras=None, 
                 usecols_profiles=None,
                 usecols_history=None, 
                 cpus=None,
                 parallel=True,
                 history_name='history.data',
                 freqdir='',
                 classical_pulsator=None,
                 name=None,
                 color='k',
                 freq_tag=''):
        self.dir = dir
        self.parameters = parameters
        self.cpus = os.cpu_count() if cpus is None else cpus
        self.parallel=parallel
        self.name=name
        self.color = color
        if name is None:
            self.name = self.dir
        self.freq_tag = freq_tag
        
        
        self.usecols_profiles = usecols_profiles
        self.usecols_history = usecols_history
        
        self.history_name = history_name
        self.load_history_extras = load_history_extras

        if classical_pulsator is None:
            if any(re.search('-freqs', f) for f in os.listdir(self.dir) if os.path.isfile(os.path.join(self.dir, f))):
                if self.load_history_extras is not None:     
                    self.load_history_extras.append(load_fundamentals) 
                if self.load_history_extras is None:
                    self.load_history_extras = load_fundamentals  

        self.loaded = False

        self.freqdir = freqdir
        
        self._history  = None
        self._index    = None
        self._profiles = None
        self._gyres    = None
        self._freqs    = None

        self._profile_cache = {}  
        self._gyre_cache = {}  
        self._freq_cache = {}  


    
    #def __repr__(self):
    #    return repr(self.history)
    def __repr__(self):
        return f"Track in directory: {self.dir}\ncall track.history, track.profiles, track.freqs, track.index"

    #def __repr__(self):
    #    display(self.history)
    #    return ''
    #def __repr__(self):
    #    return f"Track with {len(self.history)} models\nParameters: " + str(self.parameters)
    
    ### Lazy-load getters and setters for the data 
    @property
    def history(self):
        if self._history is None:
            self._history = self.load_history_file()
        return self._history
    
    @property
    def index(self):
        if self._index is None:
            self._index = self.get_index()
        return self._index

    @property
    def profiles(self):
        if self._profiles is None:
            self._profiles = self.get_profiles()
        return self._profiles
    
    @property
    def gyres(self):
        if self._gyres is None:
            self._gyres = self.get_gyres()
        return self._gyres
    
    @property
    def freqs(self):
        if self._freqs is None:
            self._freqs = self.get_freqs()
        return self._freqs
    
    
    @property
    def nad_freqs(self):
        if self._nadfreqs is None:
            self._nadfreqs = self.get_nad_freqs()
        return self._nadfreqs
    

    ### HISTORY
    def load_history_file(self):
        if self.parameters is not None:
            print("Parsing", self.parameters)
        DF_ = pd.read_table(os.path.join(self.dir, self.history_name), 
                            skiprows=5, sep=r'\s+', 
                            usecols=self.usecols_history)
        
        if self.load_history_extras is not None and not self.loaded:
            if isinstance(self.load_history_extras, list):
                for load_extras in self.load_history_extras:
                    DF_ = load_extras(self, DF_)
            else:
                DF_ = self.load_history_extras(self, DF_)

        
        self.loaded = True
        return DF_
    
    def get_history(self, profile_number):
        model_number = self.index[self.index.profile_number == profile_number].model_number.values[0]
        return self.history[self.history.model_number == model_number]
    
    ### INDEX
    def get_index(self):
        return pd.read_table(os.path.join(self.dir, 'profiles.index'), 
                             names=['model_number', 'priority', 'profile_number'], 
                             skiprows=1, sep=r'\s+')
    
    ### PROFILES
    def load_profile(self, profile_number):
        if profile_number in self._profile_cache:
            return self._profile_cache[profile_number]

        prof = pd.read_table(
            os.path.join(self.dir, 'profile' + str(profile_number) + '.data'), 
            skiprows=5, sep=r'\s+',
            usecols=self.usecols_profiles)
        
        self._profile_cache[profile_number] = prof
        return prof
    
    def get_profile_list(self):
        if not self.parallel:
            return [self.load_profile(profile_number) 
                for profile_number in tqdm(self.index.profile_number, desc='Loading Profiles: ')]
        with ThreadPoolExecutor(max_workers=self.cpus) as executor:
            return list(tqdm(executor.map(self.load_profile, self.index.profile_number),
                                 total=len(self.index.profile_number), desc='Loading Profiles: '))
        
    def get_profiles(self):
        profiles =  self.get_profile_list()
        return pd.DataFrame({'profile_number' : self.index.profile_number,'profiles' : profiles}).set_index('profile_number').profiles
    

    ### GYRE FILES
    def load_gyre(self, profile_number):
        if profile_number in self._gyre_cache:
            return self._gyre_cache[profile_number]
        
        if os.path.exists(os.path.join(self.dir, 'profile' + str(profile_number) + '.data.GYRE')) is False:
            raise Exception('tried to load .GYRE files but no .GYRE files found in directory (or improperly named)')

        prof = gyre.load_gyre(
            os.path.join(self.dir, 'profile' + str(profile_number) + '.data.GYRE'))
        
        self._gyre_cache[profile_number] = prof
        return prof
    
    def get_gyre_list(self):
        if not self.parallel:
            return [self.load_gyre(profile_number) 
                    for profile_number in tqdm(self.index.profile_number, desc='Loading GYRE Files: ')]
        with ThreadPoolExecutor(max_workers=self.cpus) as executor:
            return list(tqdm(executor.map(self.load_gyre, self.index.profile_number),
                                 total=len(self.index.profile_number), desc='Loading GYRE Files: '))
    
    def get_gyres(self):
        gyres =  self.get_gyre_list()
        return pd.DataFrame({'profile_number' : self.index.profile_number,'gyres' : gyres}).set_index('profile_number').gyres
    

    ### FREQUENCIES
    def load_freq(self, profile_number):
        if profile_number in self._freq_cache:
            return self._freq_cache[profile_number]
        
        try:
            freq = pd.read_table(
                os.path.join(self.dir, self.freqdir, 'profile' + str(profile_number) + self.freq_tag + '-freqs.dat'), 
                sep=r'\s+', skiprows=5)
            
            self._freq_cache[profile_number] = freq
        except:
            return None
        return freq


    def get_freq_list(self):
        if not self.parallel:
            return [self.load_freq(profile_number) 
                    for profile_number in tqdm(self.index.profile_number, desc='Loading Frequencies: ')]
        with ThreadPoolExecutor(max_workers=self.cpus) as executor:
            return list(tqdm(executor.map(self.load_freq, self.index.profile_number),
                                 total=len(self.index.profile_number), desc='Loading Frequencies: '))


    def get_freqs(self):
        freqs =  self.get_freq_list()
        return pd.DataFrame({'profile_number' : self.index.profile_number,'freqs' : freqs}).set_index('profile_number').freqs


    def get_nad_freqs(self, min=1, max=None):
        if max is None:
            max = len(self.freqs) + 1

        freqs = [pd.read_table(os.path.join(self.dir, self.freqdir, f'profile{p}'+ self.freq_tag + '-nad' +'-freqs.dat'), skiprows=5, sep=r'\s+')
                                for p in tqdm(np.arange(min, max), desc='Loading Frequencies')]
        
        df = self.history.copy()
        for i in range(10):
            df[f'radial {i} freq'] = pd.Series([f.iloc[i]['Re(freq)'] if len(f) > i else np.nan for f in freqs],
                index=range(min, min + len(freqs)))
            df[f'radial {i} im freq'] = pd.Series([f.iloc[i]['Im(freq)'] if len(f) > i else np.nan for f in freqs],
                index=range(min, min + len(freqs)))
            df[f'radial {i} eta'] = pd.Series([f.iloc[i]['eta'] if len(f) > i else np.nan for f in freqs],
                index=range(min, min + len(freqs)))
            df[f'radial {i} work'] = pd.Series([f.iloc[i]['W'] if len(f) > i else np.nan for f in freqs],
                index=range(min, min + len(freqs)))
        
        return df


class Grid:
    def __init__(self, dir, load_history_extras=None, cpus=None, usecols_profiles=None, usecols_history=None,
                parse_filename=None, cmap='plasma', logs_dir='', freq_label=''):
        self.dir = dir
        self._df = None
        self.load_history_extras = load_history_extras
        self.cpus = os.cpu_count() if cpus is None else cpus
        self.usecols_profiles = usecols_profiles
        self.usecols_history = usecols_history
        self.parse_filename = parse_filename if parse_filename is not None else self.parse_filename
        self.logs_dir = logs_dir
        self.colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(os.listdir(self.dir))+1))
        self.freq_label = freq_label
    
    def __repr__(self):
        return f"Grid with {len(self.df)} tracks\nColumns: " + ', '.join(list(self.df.columns)) + "\n"+\
            str(self.df.select_dtypes(include=np.number).apply(pd.Series.unique))
    
    def __getitem__(self, item):
         return self._df.iloc[item]
    
    @property
    def df(self):
        if self._df is None:
            self._df = self.load()
        return self._df
    
    def parse_filename(self, filename):
        file = filename.split('/')[0]
        parts = file.split('-')
        parameters = {p.split('_')[0]: float(p.split('_')[1]) 
                      for p in parts}
        return parameters
    
    def process_directory(self, d, load_history_extras, color):
        if os.path.isdir(os.path.join(self.dir, d)):
            parameters = self.parse_filename(d)
            track = Track(os.path.join(self.dir, d), 
                          parameters=parameters, 
                          load_history_extras=load_history_extras,
                          usecols_profiles=self.usecols_profiles, 
                          usecols_history=self.usecols_history, 
                          cpus=self.cpus,
                          color=color,
                          name=d.replace(self.logs_dir, ''),
                          freq_tag = self.freq_label) 
            parameters['Track'] = track
            return parameters
    
    """
    def load(self):
        with ThreadPoolExecutor(max_workers=self.cpus) as executor:
            data = list(tqdm(executor.map(lambda d: self.process_directory(d, self.load_history_extras), os.listdir(self.dir)),
                            total=len(os.listdir(self.dir))))
        data = [d for d in data if d is not None]  # Filter out any None results
        self._df = pd.DataFrame(data)
        for col in self._df.columns:
            if col not in ['Track']:
                self._df[col] = self._df[col].astype(float)
        return self._df
    """
    
    def load(self):
        self._df = self.process_directories(os.listdir(self.dir))
        return self._df
    
    def process_directories(self, dirs):
        #with ThreadPoolExecutor(max_workers=self.cpus) as executor:
        #    data = list(tqdm(executor.map(lambda d: self.process_directory(d, self.load_history_extras), dirs),
        #                    total=len(dirs)))
        #data = [d for d in data if d is not None]  # Filter out any None results
        data = [self.process_directory(os.path.join(d, self.logs_dir), self.load_history_extras, self.colors[i]) for i, d in enumerate(tqdm(dirs))]
        df = pd.DataFrame(data)
        for col in df.columns:
            if col not in ['Track']:
                df[col] = df[col].astype(float)

        return df

    def update(self):
        existing_dirs = set([os.path.dirname(track.dir) for track in self.df['Track']])
        all_dirs = set(os.listdir(self.dir))

        # Find new directories
        new_dirs = all_dirs - existing_dirs
        
        if new_dirs:
            new_df = self.process_directories(new_dirs)
            # Append to existing dataframe
            self._df = self.df.append(new_df, ignore_index=True)
        else:
            print("No new tracks found.")
    
    def filter(self, params):
        condition = True
        for param, values in params.items():
            if not isinstance(values, list):
                values = [values]
            condition = condition & self.df[param].isin(values)
        return self.df[condition]


### Helpful load_history_extras 
def uhz_to_h(f):
    return 1/(f*10**-6)/3600

def gaussian_weight(x, center, fwhm):
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return np.exp(-0.5 * ((x - center) / sigma)**2)

def load_fundamentals(self, DF_):
    DF_['Fundamental Period'] = pd.Series([uhz_to_h(float(f.iloc[0]['Re(freq)'])) if f is not None else np.nan for f in self.freqs])
    return DF_

def hr_velocity(self, DF_):
    v_hr = np.sqrt(np.diff(DF_['log_L'])**2 + np.diff(DF_['log_Teff'])**2) / np.diff(DF_['star_age'])
    DF_['v_hr'] = np.concatenate(([np.nan], v_hr))

    return DF_

def load_history_seismology(self, DF_):
    import ratios
    
    M = DF_.star_mass
    R = 10**DF_.log_R
    T = (10**DF_.log_Teff)/5772
    DF_['nu_max']   = M        * R**-2     * T**(-1/2) * 3090  # muHz
    DF_['delta_nu'] = M**(1/2) * R**(-3/2)             * 135.1 # muHz
    
    # small and large frequency separation
    Dnu0_dict  = {}
    dnu02_dict = {}
    r02_dict   = {}
    freqs = self.get_freqs()
    for i, freq in enumerate(freqs):
        if freq is None:
            continue
        profile_number = self.index.profile_number[i]
        model_number   = self.index.loc[self.index.profile_number == profile_number, 'model_number'].values[0]
        idx            = np.where(DF_.model_number == model_number)[0][0]
        
        nu_max = DF_.nu_max.values[idx]
        fwhm = 0.66 * nu_max ** 0.88
        
        Dnu0 = np.nan
        dnu02 = np.nan
        r02 = np.nan
        
        freq = freq[np.logical_and(freq.n_p <= 50, freq.n_g == 0)].reset_index()
        
        l0 = freq[np.logical_and(freq.l == 0, freq.n_p > 10)].copy()
        l2 = freq[np.logical_and(freq.l == 2, freq.n_p > 10)].copy()
        
        if len(l0) > 10:
            l0_diff = np.diff(l0['Re(freq)'].values)
            weights0 = gaussian_weight(l0['Re(freq)'].values[:-1], nu_max, fwhm)
            if np.any(weights0):
                Dnu0 = np.average(l0_diff, weights=weights0)
        
        l2.n_p += 1 
        freq_merge = pd.merge(l0, l2, on='n_p')
        if len(freq_merge) > 10:
            freq_merge['d02'] = freq_merge['Re(freq)_x'] - freq_merge['Re(freq)_y']
            weights2 = gaussian_weight(freq_merge['Re(freq)_x'].values, nu_max, fwhm)
            if np.any(weights2):
                dnu02 = np.average(freq_merge['d02'].values, weights=weights2)
        
        r02s = ratios.r02(freq)
        if r02s['ratios'] is not None:
            weights = gaussian_weight(r02s['freqs'], nu_max, fwhm)
            if np.any(weights):
                r02 = np.average(r02s['ratios'], weights=weights)
        
        Dnu0_dict[model_number] = Dnu0
        dnu02_dict[model_number] = dnu02
        r02_dict[model_number] = r02
    DF_['Dnu0']  = DF_['model_number'].map(Dnu0_dict)
    DF_['dnu02'] = DF_['model_number'].map(dnu02_dict)
    DF_['r02']   = DF_['model_number'].map(r02_dict)
    self._profiles = None
    self._freqs = None
    
    return DF_