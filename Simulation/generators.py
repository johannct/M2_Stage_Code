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


    def generate_m5Ratio(self, countMapper, Nth, unit=None):
        hpmap = countMapper._select_useMap("")/Nth
        ratioMapper = CountMapper.from_map(hpmap)
        if unit is not None: ratioMapper._set_instance_settingsPlot(unit=unit)
        ratioMapper.set_mask()
        return ratioMapper
    
    
    def generate_m5Detect(self, sourceMapper, ratioMapper, unit=None):
        hpmap = sourceMapper._select_useMap("") * ratioMapper._select_useMap("")
        detectMapper = CountMapper.from_map(hpmap, nest=self.nest)
        if unit is not None: detectMapper._set_instance_settingsPlot(unit=unit)
        detectMapper.set_mask()
        return detectMapper
    
    
    def generate_m5Observ(self, detectMapper, ratioMapper, contrastMapper=None, poissNoise=True, unit=None, IDinunit=True):
        if contrastMapper is not None: observMapper = self.generate_monopole(detectMapper._select_useMap(""), contrastMapper)
        else: observMapper = CountMapper.from_map(detectMapper._select_useMap(""))
        if poissNoise: observMapper = observMapper.get_poisson_noise(IDinunit=IDinunit)
        if unit is not None: observMapper._set_instance_settingsPlot(unit=unit)
        hpmapMasked = detectMapper._select_useMap('Masked')
        observMapper.set_mask(hpmapMasked.mask)

        model_1 = observMapper._instance_settingsFit_MD["model"]
        model_2 = observMapper._instance_settingsFit_D["model"]
        observMapper._instance_settingsFit_MD["model"] = lambda hpmap, M, A, ra, dec, contrast : model_1(hpmap, M, A, ra, dec, contrast) * ratioMapper._select_useMap("Masked")
        observMapper._instance_settingsFit_D["model"] = lambda hpmap, A, ra, dec, contrast : model_2(hpmap, A, ra, dec, contrast) * ratioMapper._select_useMap("Masked")
        
        return observMapper


    def generate_m5Correct(self, observMapper, ratioMapper, unit=None, IDinunit=True):
        hpmap = observMapper._select_useMap("") / ratioMapper._select_useMap("")
        correctMapper = CountMapper.from_map(hpmap, nest=self.nest)
        correctMapper.fillna()
        if unit is not None: correctMapper._set_instance_settingsPlot(unit=unit)
        if hasattr(observMapper, 'ID'): correctMapper.set_mapID(observMapper.ID, inunit=IDinunit)
        correctMapper.set_mask()
        errmodel = correctMapper._get_map_errY
        correctMapper._get_map_errY = lambda hpmap: errmodel(hpmap) / np.sqrt(ratioMapper._select_useMap("Masked"))
        return correctMapper


    def alm2map(self, alms, **kwargs):
        hpmap = hp.alm2map(alms, nside=self.nside, **kwargs)
        mapper = CountMapper.from_map(hpmap, nest = False)
        if self.nest: mapper = mapper.invert_nest()
        return mapper


    def generate_blindedMap(self, mapper, use_map: str = '', alm: bool = True, IDinunit=True, scaleFactor=2, **kwargs):
        cl, alms = mapper.anafast(alm=True, use_map=use_map, **kwargs)
        lmax = hp.Alm.getlmax(len(alms))
        idx1, idx0, idx_1 = hp.Alm.getidx(lmax, 1, 1), hp.Alm.getidx(lmax, 1, 0), hp.Alm.getidx(lmax, 1, -1)
        alms10_1 = np.array(alms[[idx1, idx0, idx_1]])
        almsOrigin = [alms10_1.real, alms10_1.imag]
        
        anew, ID = np.random.normal(almsOrigin, scale = scaleFactor*np.abs(almsOrigin)), ULID()
        anew = anew[0] + 1j*anew[1]
        almsNew = alms.copy()
        almsNew[[idx1, idx0, idx_1]] = anew
        
        newMapper = self.alm2map(almsNew, **kwargs)
        newMapper.set_mapID(ID, inunit=IDinunit)
        if alm: return newMapper, almsNew
        else: return newMapper

    