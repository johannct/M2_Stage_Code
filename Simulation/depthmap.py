import pandas as pd
import numpy as np
import healpy as hp

class DepthMapper():
    def __init__(self, filename : str):
        self.df = pd.read_hdf(filename)
    def select_year(self, year : int):
        idx = self.df['year'] == 1
        sel = self.df[idx]
        dfMap = sel.set_index('healpixID')
        IDpix = pd.Index(np.arange(49152))
        self.m5map = dfMap.reindex(IDpix)
    def select_band(self, band : str):
        self.band = band
        self.band_map = self.m5map['m5_'+band]
        self.band_map.fillna(-999, inplace=True)
    def plot(self, nest: bool = True, norm: str = 'hist', cmap: str = 'viridis_r'):
        map_masked = hp.ma(self.band_map, badval=-999)
        hp.projview(map_masked, nest=nest, norm=norm, unit='m5_' + self.band, cmap=cmap)
    def m5(self, **kwargs):
        if 'healpix_id' in kwargs:
            print("use healpix id to obtain the m5 value on the map")
        elif 'ra' in kwargs and 'dec' in kwargs:
            print("use ra and dec to find m5 value in the map")
        
