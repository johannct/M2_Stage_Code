import numpy as np
import healpy as hp
import pyccl as ccl

try: from simulMap import nz_model, build_nz
except: from Simulation.simulMap import nz_model, build_nz


def get_Cl_ccl(nside, zmin, Omega_c=0.25, Omega_b=0.05, h=0.67, sigma8=0.8, n_s=0.96, get_ell=False, cosmo=None):
    lmax = 3*nside - 1
    if cosmo is None: cosmo = ccl.Cosmology(
        Omega_c=Omega_c,
        Omega_b=Omega_b,
        h=h,
        sigma8=sigma8,
        n_s=n_s
    )

    ell = np.arange(0, lmax+1)
    z, nz = build_nz(zmin)
    bias = np.ones_like(z)  # b=1
    tracer = ccl.NumberCountsTracer(
        cosmo,
        has_rsd=False,
        dndz=(z, nz),
        bias=(z, bias)
    )

    cl = ccl.angular_cl(cosmo, tracer, tracer, ell, l_limber='auto')
    #print("Cl[ℓ=1] =", cl[0])
    if get_ell: return cl, ell
    else: return cl


def get_clusterContrast(cl, nside, lognormal=False):
    m = hp.synfast(cl, nside=nside)
    if lognormal: m = np.exp(m - 0.5 * np.var(m)) #Log-Normale transformation
    return m