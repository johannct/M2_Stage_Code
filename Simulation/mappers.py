"""A package containing classes to manipulate maps."""

import fitsio
import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt
from astropy.table import Table, vstack, MaskedColumn

try: from simulMap import raDec2map_Table
except: from Simulation.simulMap import raDec2map_Table


##### Parent class to all others: ##### 
class Mapper():
    """A class to read and manipulate maps. It is the parent to derivated classes."""
    _mapNameBase = "map" #base name for all map attributs (ex: self.map, self.mapMasked, self.mapCut...)
    _settingsPlot = {"graticule": True, #default settings to use in self.plot()
        "graticule_labels": True,
        "xlabel": "RA", "ylabel": "DEC"}
    
    def __init__(self, data, nest: bool = True, map=None, dataName="table"):
        self.nest = nest
        self.__dict__[dataName] = data
        if map is not None: self.__dict__[self._mapNameBase] = map
        self._instance_settingsPlot = {} #to create new default settings to use in self.plot(), specific to the instance.

    
    def _set_instance_settingsPlot(self, **kwargs):
        """Allow to set default settings specific to the instance self, in order to be used in self.plot()."""
        self._instance_settingsPlot = self._instance_settingsPlot | kwargs  #take values in kwargs if their exist, else take values in self._instance_settingsPlot
        

    def _select_useMap(self, suffix=''):
        """Retun the attribut map represented by suffix. Especially used to choose which attribut map a method has to act on.
        Example of values for suffix:
        - "" : return self.map (default)
        - "Masked" : return self.mapMasked
        - "Cut" : return self.mapCut"""
        return self.__dict__[self._mapNameBase + suffix].copy()

    
    def _create_newMap(self, map, suffix):
        """Add an attribut map to the instance self, named self._mapNameBase + suffix."""
        if suffix=="": raise ValueError(f"You gave suffix = '', which means you are trying to rewrite the attribut {self._mapNameBase}. This attribut cannot by modify by this function.")
        mapName = self._mapNameBase + suffix
        if hasattr(self, mapName): print(f"Attribut {mapName} already existed and has been replaced.")
        self.__dict__[mapName] = map
    
    
    @classmethod
    def read(cls, filename: str, nest: bool = True, **kwargs):
        """Load a map from an external file with Table.read."""
        data = Table.read(filename)
        return cls(data=data, nest=nest, **kwargs)

    @classmethod
    def read_hdf(cls, filename: str, nest: bool = True, **kwargs):
        """Load a map from an external file with pandas.read_hdf."""
        data = pd.read_hdf(filename)
        return cls(data=data, nest=nest, **kwargs)

    
    def plot(self, use_map: str = '', **kwargs):
        """Plot the HealPIX map by using healpy.projview().
        Parameters:
        - use_map: suffix of the attribut map that has to be plotted. Default: use_map = "".
        - kwargs: optionnal parametters to give to hp.projview(). Can also accept xlabel and ylabel to custom axes labels.
        Example :
        - use_map = "": plot self.map.
        - use_map = "Masked": plot self.mapMasked.
        - use_map = "Cut": plot self.mapCut."""
        settings = self._settingsPlot | self._instance_settingsPlot | kwargs
        xlabel, ylabel = settings.pop("xlabel"), settings.pop("ylabel")
        map = self._select_useMap(use_map) #choosing which attribut map to plot.
        hp.projview(map, nest=self.nest, **settings)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)



##### Density map: #####
class DensityMapper(Mapper):
    """A class to read and manipulate density maps."""
    _settingsPlot = Mapper._settingsPlot | {
        "unit": "Source density in $[\deg^{-2}]$"} #default settings to use in self.plot()
    
    def __init__(self, data, nside: int, nest: bool = True, map=None):
        self.nside = nside
        self.area_deg2 = hp.nside2pixarea(self.nside, degrees=True)
        super().__init__(data=data, nest=nest, map=map)
        if map is None: self.__dict__[self._mapNameBase] = raDec2map_Table(self.nside, self.table, nest=self.nest)/self.area_deg2
    
    
    def add(self, mapper):
        """Add another DensityMapper to the instance, and return a new DensityMapper with:
        - new.map = self.map + mapper.map
        - new.table = vstack([self.table, mapper.table])"""
        data = vstack([self.table, mapper.table])
        map = self.map + mapper.map
        return self.__class__(data=data, nside=self.nside, nest=self.nside, map=map)

    
    def set_mask(self, mask=None, badval=0, **kwargs):
        """Set a mask to self.map and stack it in the attribut self.mapMasked.
        Parameters:
        - mask: if given, self.mapMasked.mask = mask. Else, a default mask is set using badval.
        - badval: if mask is not given, value used to define the mask by healpy.ma(self.map, badval=badval).
        - kwargs: if mask is not given, other parametters of healpy.ma() can be given here."""
        self.mapMasked = hp.ma(self.map, badval=badval, **kwargs)
        if mask is not None: self.mapMasked.mask = mask

    
    def set_cutMask(self, cut_mask, invert=False, fromMap="Masked", toMap="Cut"):
        """Add a cut mask to a map without replacing the former mask.
        Parameters:
        - cut_mask: mask to add; only its True values will act, without changing the other values of a former mask.
        - invert: if True, add ~cut_mask instead of cut_mask. Default: invert = False.
        - fromMap: suffix of the map from which the cut_mask will be added. Default: fromMap = "Masked", which means the cut_mask will be added from self.mapMasked.mask.
        - toMap: suffix of the map to which the cut masked map will be stocked. Default: toMap = "Cut", which means the cut masked map will be stocked into self.mapCut."""
        sel = self._select_useMap(fromMap) #choosing which attribut map to strart from.
        if invert: cut_mask = ~cut_mask
        sel.mask[cut_mask] = True
        self._create_newMap(sel, toMap)

    
    def select_bounds(self, map_min=None, map_max=None, fromMap="", toMap="Cut"):
        """Mask all pixels the density of which is not in [map_min, map_max].
        Parameters:
        - map_min, map_max: minimum and maximum values to keep. If None, all values are kept. Deault: map_min=None, map_max=None.
        - fromMap: suffix of the map from which the min and max are computed. Default: fromMap = "", which means they will be computed from self.map.
          However, since self.map has no attribut mask, the cut_mask will be added from self.mapMasked.mask.
        - toMap: suffix of the map to which the cut masked map will be stocked. Default: toMap = "Cut", which means the cut masked map will be stocked into self.mapCut."""
        map = self._select_useMap(fromMap)
        if map_min is None: map_min = np.nanmin(map)
        if map_max is None: map_max = np.nanmax(map)
        is_in = (map_min <= map) & (map <= map_max) #True if counts in bounds
        if fromMap == "":
            print("Attribut map has no attribut mask. Using mask from mapMasked instead.")
            fromMap = "Masked"
        self.set_cutMask(is_in, invert=True, fromMap=fromMap, toMap=toMap)



##### FRACCOV map: #####
class FracCovMapper(Mapper):
    """A class to read and manipulate FRACCOV maps."""
    _mapNameBase = "FRACCOV" #base name for all map attributs (ex: self.FRACCOV, self.FRACCOVnside, self.FRACCOVcut...)
    _settingsPlot = Mapper._settingsPlot | {"unit": "FRACCOV",
        "cmap": "jet"} #default settings to use in self.plot()
    
    def __init__(self, data, nest: bool = True):
        super().__init__(data=data, nest=nest, map=None)
        self.__dict__[self._mapNameBase] = self.table['FRACCOV'].copy()
        self.nsideOriginal = hp.npix2nside(len(self.table))
        
    
    def get_nside(self, nside: int):
        """"From the original FRACCOV map, create a new one with a new nside value, by computing the mean of FRACCOV corresponding to each pixel.
        If the original FRACCOV map is masked, a new mask is also created by taking True if the mask of all original pixels sent to the new pixel are True.
        The new FRACCOV and mask are stocked in self.FRACCOVnside and self.mask_FRACCOVnside, so as the original FRACCOV map is not replaced.
        A pandas DataFrame is also add to the instance, in order to make other operations with the new nside settings."""
        #adding nside information:
        col_ipix = f'HealPIX_{nside}'
        self.nside = nside
        if not col_ipix in self.table.columns: self.table[col_ipix] = hp.ang2pix(nside, self.table['RA'], self.table['DEC'], nest=self.nest, lonlat=True)

        #adding DataFrames:
        self.df = self.table[[col_ipix, 'FRACCOV']].to_pandas() # Taking olnly useful columns
        self.df['is_masked'] = self.FRACCOV.mask
        self.df_grouped = self.df.groupby(col_ipix)

        #adding FRACCOV:
        self.mask_FRACCOVnside = self.df_grouped['is_masked'].all() #True if all rows for a pixel are masked
        self.FRACCOVnside = MaskedColumn(self.df_grouped['FRACCOV'].mean())

    
    def cut_FracCov(self, frac_min=0, frac_max=1, fromMap="nside", toMap="cut"):
        """Mask all pixels the density of which is not in [frac_min, frac_max].
        Parameters:
        - frac_min, frac_max: minimum and maximum values to keep. Default: frac_min=0, frac_max=1.
        - fromMap: suffix of the map from which the min and max are computed. Default: fromMap = "nside", which means they will be computed from self.FRACCOVnside.
        - toMap: suffix of the map to which the cut masked map will be stocked. Default: toMap = "cut", which means the cut masked map will be stocked into self.FRACCOVcut."""
        map = self._select_useMap(fromMap)
        is_in = (frac_min <= map) & (map <= frac_max) #True if FRACCOV in bounds
        map.mask[~is_in] = True
        self._create_newMap(map, toMap)
        self.frac_min, self.frac_max = frac_min, frac_max



##### depth map: #####
class DepthMapper(Mapper):
    """A class to read and manipulate m5 maps."""
    _mapNameBase = "m5" #base name for all map attributs (ex: self.map, self.mapMasked, self.mapCut...)
    _settingsPlot = Mapper._settingsPlot | {'norm': 'hist', } #default settings to use in self.plot()
    
    def __init__(self, data, nest: bool = True):
        super().__init__(data=data, nest=nest, map=None, dataName="df")

    def _get_band_map(self, band : str = None):
        if band is None: band = self.band
        band_map = self._select_useMap(band)
        return band, band_map
    
    def select_year(self, year : int = 1):
        """Select a year in the data, and and create the corresponding DataFrame with the indices of healPIX pixels, in order to obtain the full healpIX map.
        This DataFrame is stocked in the attribut m5year of the instance."""
        self.year = year
        idx = self.df['year'] == year
        sel = self.df[idx]
        dfMap = sel.set_index('healpixID')
        IDpix = pd.Index(np.arange(49152))
        self._create_newMap(dfMap.reindex(IDpix), 'year')

    def set_mask(self, band=None, mask=None, badval=-999, **kwargs):
        """Set a mask to self.map and stack it in the attribut self.mapMasked.
        Parameters:
        - mask: if given, self.mapMasked.mask = mask. Else, a default mask is set using badval.
        - badval: if mask is not given, value used to define the mask by healpy.ma(self.map, badval=badval).
        - kwargs: if mask is not given, other parametters of healpy.ma() can be given here."""
        # if band is None: band = self.band
        # map = self._select_useMap(band)
        band, map = self._get_band_map(band)
        masked = hp.ma(map, badval=badval, **kwargs)
        if mask is not None: masked.mask = mask
        self._create_newMap(masked, band+'Masked')
        
    def select_band(self, band : str):
        """Select the m5 map corresponding to a band and stock it into an attribut of the instance."""
        self.band = band
        band_map = self.m5year['m5_'+band].copy()
        band_map.fillna(-999, inplace=True)
        band_map = np.array(band_map)
        self._create_newMap(band_map, band)
    
    def plot(self, band=None, use_map: str = 'Masked', **kwargs):
        """Plot the m5 map corresponding to a given band. If no band is given, plot the last one that has been selected by self.select_band()."""
        # if band is None: band = self.band
        # band_map = self._select_useMap(band)
        band, band_map = self._get_band_map(band)
        settings = self._settingsPlot | self._instance_settingsPlot | {'unit': f'{self._mapNameBase}_{band}'} | kwargs
        use_map = band + use_map
        super().plot(use_map=use_map, **settings)
    
    def get_m5(self, band=None, **kwargs):
        """return the m5 value corresponding to a given pixel or a given ra, dec, depending on a given band.
        If no band is given, use the last one that has been selected by self.select_band()."""
        # if band is None: band = self.band
        # band_map = self._select_useMap(band)
        band, band_map = self._get_band_map(band)
        if 'healpix_id' in kwargs:
            print("use healpix id to obtain the m5 value on the map")
            healpix_id = kwargs['healpix_id']
        elif 'ra' in kwargs and 'dec' in kwargs:
            print("use ra and dec to find m5 value in the map")
            nside = hp.npix2nside(len(band_map))
            healpix_id = hp.ang2pix(nside, kwargs['ra'], kwargs['dec'], nest=self.nest, lonlat=True)
        return band_map[healpix_id]