import numpy as np
import healpy as hp
from ulid import ULID

try:
    from cluster import *
    from simulMap import *
    from mappers import CountMapper
except:
    from Simulation.cluster import *
    from Simulation.simulMap import *
    from Simulation.mappers import CountMapper


class Generator():

    def __init__(self, nside, nest, zmin=0.1, zmax=3.0, zsize=400, build_nz=build_nz_model, Omega_c=0.25, Omega_b=0.05, h=0.67, sigma8=0.8, n_s=0.96, cosmo=None, lognormal=False):
        self.nside, self.nest = nside, nest
        self.npix = hp.nside2npix(self.nside)
        
        self.zmin, self.zmax = zmin, zmax
        self.build_nz = build_nz_model
        self.z, self.nz = self.build_nz(self.zmin, self.zmax, zsize)

        self.Omega_c, self.Omega_b, self.h, self.sigma8, self.n_s = Omega_c, Omega_b, h, sigma8, n_s
        if cosmo is None: self.cosmo = ccl.Cosmology(Omega_c=self.Omega_c, Omega_b=self.Omega_b, h=self.h, sigma8=self.sigma8, n_s=self.n_s)
        else: self.cosmo

        self.cl, self.ell = get_Cl_ccl(self.nside, self.zmin, self.zmax, zsize, self.build_nz, self.Omega_c, self.Omega_b, self.h, self.sigma8, self.n_s, True, self.cosmo)
        self.lognormal = lognormal


    def generate_clusterContrast(self, dipoleMapper=None, unit=None, IDinunit=True):
        hpmap, mapID = get_clusterContrast(self.cl, self.nside, self.lognormal, self.nest), ULID()
        clMapper.from_map(hpmap, nest=self.nest)
        if dipoleMapper is not None:
            clMapper = dipoleMapper.add(clMapper)
            if unit is None: unit = "Clustering and dipole density contrast"
        elif unit is None: unit = "Clustering density contrast"
        clMapper._set_instance_settingsPlot(unit=unit)
        clMapper.set_mapID(mapID, inunit=IDinunit)
        return clMapper


    def generate_dipoleContrast(self, A, ra, dec, clMapper=None, unit=None, IDinunit=True):
        hpmap = apply_dipole_ARaDec(np.arange(self.npix), A, ra, dec, self.nest)
        dipoleMapper = CountMapper.from_map(hpmap, nest=self.nest)
        if clMapper is not None:
            dipoleMapper = clMapper.add(dipoleMapper)
            if unit is None: unit = "Clustering and dipole density contrast"
        elif unit is None: unit = f"Dipole density constrast\nfor A = {A}, RA = {ra}, DEC = {dec}"
        dipoleMapper._set_instance_settingsPlot(unit = unit)
        if hasattr(clMapper, 'ID'): dipoleMapper.set_mapID(clMapper.ID, inunit=IDinunit)
        return dipoleMapper


    def generate_monopole(self, M, contrastMapper, unit=None, IDinunit=True):
        hpmap = apply_monopole_Mcontrast(contrastMapper._select_useMap(""), M)
        monopMapper = CountMapper.from_map(hpmap, nest=self.nest)
        if unit is not None: monopMapper._set_instance_settingsPlot(unit=unit)
        if hasattr(contrastMapper, 'ID'): monopMapper.set_mapID(contrastMapper.ID, inunit=IDinunit)
        return monopMapper