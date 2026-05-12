"""A package containing classes to manipulate spectra."""

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


class Spectrum():
    """A class to read and manipulate spectra."""
    
    def __init__(self, flux, wave):
        self.flux, self.wavelength = flux, wave
        

    def _select_useFlux(self, suffix=''):
        """Retun the attribut flux represented by suffix. Especially used to choose which attribut flux a method has to act on.
        Example of values for suffix:
        - "" : return self.flux (default)
        - "Filter" : return self. self.flux_Filter"""
        flux = self.__dict__["flux" + suffix].copy()
        return flux


    def apply_gaussianFilter(self, sigma):
        self.fluxFilter = gaussian_filter(self.flux, sigma)

    
    def apply_savgolFilter(self, window_length, polyorder):
        self.fluxFilter = savgol_filter(self.flux, window, poly)


    def plot(self, use_flux='', **kwargs):
        flux =  self._select_useFlux(use_flux)
        wave =  self.wavelength.copy()
        z_renorm = kwargs.get('z_renorm', None)
        if z_renorm is not None: wave /= (1+z_renorm)
        
        if "figax" in kwargs.keys(): fig, ax = kwargs["figax"] #figax have to be tuple (fig, ax).
        else: fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(wave, flux)
        
        title = kwargs.get('title', "Spectrum")
        xscale = kwargs.get('xscale', 'log')
        yscale = kwargs.get('yscale', 'log')
        ax.set_title(title)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_xlabel("Wave Length [Angström]")
        ax.set_ylabel("Flux $[erg/s/cm^2/Hz]$")
        return fig, ax

    
    @classmethod
    def read_sed(cls, filename):
        with open(filename, "r") as sed:
            wave, flux = np.loadtxt(sed, dtype='float', usecols=(0,1), unpack=True)
        return cls(flux=flux, wave=wave)
        



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

