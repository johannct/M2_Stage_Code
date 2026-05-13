"""A package containing classes to manipulate spectra."""

## Package importation:
import fitsio
import numpy as np
import healpy as hp
import pandas as pd
from astropy.table import Table
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter

try:
    from simulMap import *
except:
    from Simulation.simulMap import *



### Constants definition:
LSST_Bands = pd.DataFrame({"u": (304.30, 403.50), "g": (385.60, 566.30), "r": (533.70, 705.70),
              "i": (669.90, 837.80), "z": (799.30, 939.20), "y": (907.50, 1100.00)})
    



### Classes definition:
class Spectrum():
    """A class to read and manipulate spectra."""
    _settingsPlot = {"xname": "Wave Length", "xunit": "Angström", #default settings to use in self.plot()
        "yname": "Flux", "yunit": "erg/s/cm^2/A",
        "xscale": "linear", "yscale": "linear"}
    
    def __init__(self, flux, wave):
        self.flux, self.wavelength = flux, wave
        self._instance_settingsPlot = {} #to create new default settings to use in self.plot(), specific to the instance.

    
    def _set_instance_settingsPlot(self, **kwargs):
        """Allow to set default settings specific to the instance self, in order to be used in self.plot()."""
        self._instance_settingsPlot = self._instance_settingsPlot | kwargs  #take values in kwargs if their exist, else take values in self._instance_settingsPlot

    def _select_settingsPlot(self, **kwargs):
        title = kwargs.pop('title', "Spectrum")
        xscale = kwargs.pop('xscale')
        yscale = kwargs.pop('yscale')
        loglog = kwargs.pop('loglog', False)
        if loglog: xscale, yscale = 'log', 'log'
        
        xname = kwargs.pop('xname')
        yname = kwargs.pop('yname')
        xunit = kwargs.pop('xunit')
        yunit = kwargs.pop('yunit')
        xlabel = kwargs.pop('xlabel', f"{xname} [${xunit}$]")
        ylabel = kwargs.pop('ylabel', f"{yname} [${yunit}$]")
        return kwargs, title, xscale, yscale, xlabel, ylabel
        

    def _select_useFlux(self, suffix='', band=None):
        """Retun the attribut flux represented by suffix. Especially used to choose which attribut flux a method has to act on.
        Example of values for suffix:
        - "" : return self.flux (default)
        - "Filter" : return self. self.flux_Filter"""
        if band is not None: suffix = f"_{band}{suffix}"
        flux = self.__dict__["flux" + suffix].copy()
        return flux

    
    def _select_useBand(self, band, use_flux=''):
        """Retun the attribut wave and flux for the given band.
        Parameter band: name of the band; must be a string."""
        if band is not None: wave = self.__dict__["wave_" + band].copy()
        else: wave = self.wavelength.copy()
        flux = self._select_useFlux(suffix=use_flux, band=band)
        return wave, flux

    
    @classmethod
    def read_sed(cls, filename, yname="SED", yunit="erg/s/cm^2/Hz"):
        with open(filename, "r") as sed:
            wave, flux = np.loadtxt(sed, dtype='float', usecols=(0,1), unpack=True)
        spectum = cls(flux=flux, wave=wave)
        spectum._set_instance_settingsPlot(yname=yname, yunit=yunit)
        return spectum


    def apply_gaussianFilter(self, sigma):
        self.fluxFilter = gaussian_filter(self.flux, sigma)

    
    def apply_savgolFilter(self, window_length, polyorder):
        self.fluxFilter = savgol_filter(self.flux, window, poly)

    
    def get_bands(self, band=LSST_Bands, nm2angström=True):
        """Add band attributs to the instance self from a dictionnary or a DataFrame band defining the name of the band and the min and max waveleangths."""
        bands = band.copy()
        bands = pd.DataFrame(bands)
        if nm2angström: bands = bands*10
        self.bands_name = list(bands.columns)
        for k, (wmin, wmax) in bands.items():
            mask = (self.wavelength >= wmin) & (self.wavelength <= wmax)
            wave_cut = self.wavelength[mask]
            flux_cut = self.flux[mask]
            self.__dict__[f"wave_{k}"] = wave_cut
            self.__dict__[f"flux_{k}"] = flux_cut


    def plot(self, use_flux='', band=None, **kwargs):
        settings = self._settingsPlot | self._instance_settingsPlot | kwargs
        # flux =  self._select_useFlux(use_flux)
        # wave =  self.wavelength.copy()
        wave, flux = self._select_useBand(band=band, use_flux=use_flux)
        z_renorm = settings.pop('z_renorm', None)
        if z_renorm is not None: wave /= (1+z_renorm)
        
        if "figax" in settings.keys(): fig, ax = settings.pop("figax") #figax have to be tuple (fig, ax).
        else: fig, ax = plt.subplots(figsize=(12, 6))
        settings, title, xscale, yscale, xlabel, ylabel = self._select_settingsPlot(**settings)
        
        ax.plot(wave, flux, **settings)
        ax.set_title(title)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if 'label' in settings.keys(): ax.legend()
        return fig, ax

    
    def get_surrounding(self, spectr):
        wave_1, wave_2 = spectr.wavelength.copy(), self.wavelength.copy()
        flux_1 = spectr.flux.copy()
        idx_max = np.searchsorted(wave_1, wave_2)
        idx_min = idx_max - 1
        
        #Avoiding wavelengths out of range:
        mask_min = (idx_min >= 0) & (idx_min < len(wave_1))
        mask_max = (idx_max >= 0) & (idx_max < len(wave_1))
        idx_min, idx_max = idx_min[mask_min], idx_max[mask_max]
        
        wave_min, flux_min = wave_2.copy(), np.zeros_like(wave_2)
        wave_max, flux_max = wave_min.copy(), flux_min.copy()
        
        wave_min[mask_min] = wave_1[idx_min]
        wave_max[mask_max] = wave_1[idx_max]
        wave_new =  np.row_stack([wave_min, wave_max])
        
        flux_min[mask_min] = flux_1[idx_min]
        flux_max[mask_max] = flux_1[idx_max]
        flux_new =  np.row_stack([flux_min, flux_max])
        #return self.__class__(wave=wave_new, flux=flux_new)
        return wave_new, flux_new


    def integrate(self, response=1):
        product = self.flux*response
        return np.trapz(product, x=self.wavelength)


    def ratio_integrate(self, response):
        return self.integrate(response) / self.integrate()
        



class Spectrum3bands():
    """A class to read and manipulate spectra with 3 bands (B, R, Z)S."""
    
    def __init__(self, flux_b, flux_r, flux_z, wave_b, wave_r, wave_z, target_ID=None):
        self.target_ID = target_ID
        self.flux_b, self.flux_r, self.flux_z = flux_b, flux_r, flux_z
        self.wave_b, self.wave_r, self.wave_z = wave_b, wave_r, wave_z

    
    def _select_useFlux(self, suffix=''):
        """Retun the attribut flux represented by suffix. Especially used to choose which attribut flux a method has to act on.
        Example of values for suffix:
        - "" : return self.flux_b, self.flux_r, self.flux_z (default)
        - "Filter" : return self. self.flux_bFilter, self.flux_rFilter, self.flux_zFilter"""
        flux_b = self.__dict__["flux_b" + suffix].copy()
        flux_r = self.__dict__["flux_r" + suffix].copy()
        flux_z = self.__dict__["flux_z" + suffix].copy()
        return flux_b, flux_r, flux_z

    
    @classmethod
    def from_fits(cls, data, target_ID, fiberName='FIBERMAP', idName='TARGETID'):
        idx = np.where(data[fiberName][idName].read() == target_ID)[0][0]
        
        flux_b = data['B_FLUX'].read()
        flux_r = data['R_FLUX'].read()
        flux_z = data['Z_FLUX'].read()
        flux_b, flux_r, flux_z = flux_b[idx], flux_r[idx], flux_z[idx]
        
        wave_b = data['B_WAVELENGTH'].read()
        wave_r = data['R_WAVELENGTH'].read()
        wave_z = data['Z_WAVELENGTH'].read()
        
        return cls(flux_b, flux_r, flux_z, wave_b, wave_r, wave_z, target_ID)

    
    def plot_spectrum(self, use_flux='', cat='', **kwargs):
        flux_b, flux_r, flux_z =  self._select_useFlux(use_flux)
        wave_b, wave_r, wave_z =  self.wave_b.copy(),  self.wave_r.copy(),  self.wave_z.copy()
        z_renorm = kwargs.get('z_renorm', None)
        if z_renorm is not None:
            wave_b /= (1+z_renorm)
            wave_r /= (1+z_renorm)
            wave_z /= (1+z_renorm)
        
        if "figax" in kwargs.keys(): fig, ax = kwargs["figax"] #figax have to be tuple (fig, ax).
        else: fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(wave_b, flux_b, label='Blue', color='blue', alpha=0.7)
        ax.plot(wave_r, flux_r, label='Red', color='red', alpha=0.7)
        ax.plot(wave_z, flux_z, label='IR', color='darkred', alpha=0.7)
    
        title = kwargs.get('title', f"Spectrum for {cat} Target ID: {self.target_ID}")
        ax.set_title(title)
        ax.set_xlabel("Wave Length [Angström]")
        ax.set_ylabel("Flux $[10^{-17} erg/s/cm^2/A]$")
        ax.legend()
        return fig, ax

    def apply_gaussianFilter(self, sigma):
        if np.array(sigma).shape == (): sigmaB, sigmaR, sigmaZ = sigma, sigma, sigma
        else: sigmaB, sigmaR, sigmaZ = sigma
        self.flux_bFilter = gaussian_filter(self.flux_b, sigmaB)
        self.flux_rFilter = gaussian_filter(self.flux_r, sigmaR)
        self.flux_zFilter = gaussian_filter(self.flux_z, sigmaZ)

    
    def apply_savgolFilter(self, window_length, polyorder):
        if np.array(window_length).shape == (): windowB, windowR, windowZ = window_length, window_length, window_length
        else: windowB, windowR, windowZ = window_length
        if np.array(polyorder).shape == (): polyB, polyR, polyZ = polyorder, polyorder, polyorder
        else: polyB, polyR, polyZ = polyorder
        self.flux_bFilter = savgol_filter(self.flux_b, windowB, polyB)
        self.flux_rFilter = savgol_filter(self.flux_r, windowR, polyR)
        self.flux_zFilter = savgol_filter(self.flux_z, windowZ, polyZ)

