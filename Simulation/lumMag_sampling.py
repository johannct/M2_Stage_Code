'''A module containing several useful functions to simulate luminosities and magnitudes.'''

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.table import Table
from scipy.integrate import quad
from scipy.stats import gamma
import pyccl as ccl

try:
    from simulMap import nz_model
    from spectra import Spectrum
except:
    from Simulation.simulMap import nz_model
    from Simulation.spectra import Spectrum


### Functions definition:

## Convertion functions:
def proba_schechter_lum(L, L_star):
        return (L/L_star)**alpha * np.exp(-L/L_star)


def lum2absMag(L, M_sun=4.83, L_sun=1):
    return M_sun - 2.5*np.log10(L/L_sun)


def absMag2lum(M, M_star):
    """Return the luminosity from the absolute magnitude M and a reference  absolute magnitude M_sar."""
    return 10**(0.4 * (M_star - M))


def lum2flux(L, dL):
    return L/(4*np.pi * (dL**2))


def lumBolom2lumBand(L, sed, response, renorm=True):
    if renorm: sed = sed.renormalize(L)
    return sed.integrate(response)


## Sampling functions:
def sample_truncated_power_law(alpha, xmin, xmax, size=None):
    """
    Sample from a truncated power-law distribution p(x) ∝ x^alpha
    for xmin <= x <= xmax, with alpha != -1.

    Parameters:
       - alpha (float): exponent (must be < -1)
       - xmin (float): lower bound (> 0)
       - xmax (float): upper bound (> xmin)
       - size (int): number of samples

    Returns:
        numpy array of samples
    """
    if alpha == -1:
        raise ValueError("alpha = -1 requires a different (logarithmic) treatment")

    # Generate uniform random numbers
    u = np.random.uniform(0, 1, size)

    # Inverse CDF
    exponent = alpha + 1
    xmin_exp = xmin ** exponent
    xmax_exp = xmax ** exponent

    samples = (u * (xmax_exp - xmin_exp) + xmin_exp) ** (1 / exponent)

    return samples


def proba_schechter_lumRatio(x, alpha, phi_star=1):
        return phi_star * np.power(x, alpha) * np.exp(-x)


def acceptReject(N, sampling_func, acceptance_func, args, pmax_func=None):
    """Create a sample of size N by using an accept-reject test from a sampling function and an acceptance function.
    args is a tuple containing the arguments to give to sampling_func.
    acceptance_func is the function computing the acceptance ratio by acceptance_ratio = acceptance_func(x_cand)."""
    samples = []
    reject = 0
    remaining = N
    total = 0
    
    while remaining > 0:
        sampling_func_size = lambda *args: sampling_func(*args, size=remaining * 3)
        x_cand = sampling_func_size(*args)
        acceptance_ratio = acceptance_func(x_cand)
        
        # Acceptance test:
        if pmax_func is None: pmax = 1
        else: pmax = pmax_func(x_cand)
        u = np.random.uniform(0, pmax, size=x_cand.shape)
        accepted = x_cand[u < acceptance_ratio]
        reject += remaining * 3 - len(accepted)
        total += remaining * 3
        samples.append(accepted)
        remaining -= len(accepted)
        
    print('Number of rejects =', reject)
    print(f"Accepted ratio = {int((total-reject)/total*100)}%")
    return np.concatenate(samples)[:N]


def generate_redshift(N, z_min=None, z_max=None, sigma=0.5,  beta=1.5):
    """Generate N randomized redshifts, by using a reject test and nz_model distribution."""
    # Analytically finding the maximum: d/dz[dist] = 0
    # => 2z - beta/sigma * (z/sigma)^(beta-1) * z^2 = 0
    # => z_mode = sigma * (2/beta)^(1/(beta-1))
    z_mode = sigma * (2 / beta) ** (1 / beta)
    peak = nz_model(z_mode, sigma, beta)

    # Global envelope: use a Gamma(3, scale) distribution which has the same
    # z^2 * exp(-z/scale) shape, matching both the power-law rise and exponential decay.
    # Choose scale to match the mode: mode of Gamma(a, scale) = (a-1)*scale
    # For a=3: mode = 2*scale => scale = z_mode / 2
    a = 3
    scale = z_mode / (a - 1)

    g_dist = gamma(a, scale=scale)
    g_peak_ratio = peak / g_dist.pdf(z_mode)  # constant M such that dist <= M * g(z)
    
    args = (z_min, z_max)
    def sampling_func(z_min, z_max, size):
        z_cand = g_dist.rvs(size=size)
        if z_min is not None: z_cand = z_cand[z_cand >= z_min]
        if z_max is not None: z_cand = z_cand[z_cand <= z_max]
        return z_cand
    
    acceptance_func = lambda z: nz_model(z, sigma, beta)
    pmax_func = lambda z: g_peak_ratio * g_dist.pdf(z)
    
    samples = acceptReject(N, sampling_func, acceptance_func, args, pmax_func=pmax_func)
    return np.array(samples)
    

def generate_schechter_lumRatio(N, alpha, x_min, x_max, phi_star=1):
    """Generate N randomized luminosities, by using a reject test and Schechter's law."""
    sampling_func = sample_truncated_power_law
    args = (alpha, x_min, x_max)
    # Acceptance ratio:
    # f(x) = x^alpha * exp(-x)
    # g(x) = x^alpha (on ignore la constante de normalisation qui s'annule)
    # f(x)/g(x) = exp(-x)
    # acceptance_ratio = np.exp(-L_cand)
    acceptance_func = lambda x: np.exp(-x)
    
    samples = acceptReject(N, sampling_func, acceptance_func, args)
    return np.array(samples)


def generate_schechter_lum(N, L_star, alpha, L_min, L_max, phi_star):
    x_min, x_max = L_min/L_star, L_max/L_star
    return L_star*generate_schechter_lumRatio(N, alpha, x_min, x_max, phi_star)


def get_dL(zi, H0=67.4, Om=0.315, Ol=0.685, c=3e8):
    '''Compute the luminosity distance depending on the redshift z, the Hubble constant H0, the cosmological parameters Om and Ol, and the speed of ligt c.'''
    inv_Ez = lambda zp: 1.0 / np.sqrt(Om * (1 + zp)**3 + Ol)
    integral, _ = quad(inv_Ez, 0, zi)
    return (c / H0) * (1 + zi) * integral


def get_dL_cosmo(z, Omega_c=0.25, Omega_b=0.05, h=0.67, sigma8=0.8, n_s=0.96, cosmo=None):
    if cosmo is None: cosmo = ccl.Cosmology(
        Omega_c=Omega_c,
        Omega_b=Omega_b,
        h=h,
        sigma8=sigma8,
        n_s=n_s
    )
    a = 1.0 / (1.0 + z)
    return ccl.luminosity_distance(cosmo, a)


def generate_lumMag(N, L_min=1e7, L_max=1e11,  L_star=1e10, alpha=-1.1, z_min=0.1, z_max=3.0, phi_star=1, L_bandRatio=1, sed=None, response=1, renorm=True, to_table=True, **kwargs):
    #Cosmological Parameters  (Planck 2018):
    H0 = kwargs.get('H0', 67.4)
    Om = kwargs.get('Om', 0.315)
    Ol = kwargs.get('Ol', 0.685)
    Oc = kwargs.get('Oc', 0.25)
    Ob = kwargs.get('Ob', 0.05)
    h = kwargs.get('h', 0.67)
    sigma8 = kwargs.get('sigma8', 0.8)
    n_s = kwargs.get('n_s', 0.8)
    cosmo = kwargs.get('cosmo', None)
    c = kwargs.get('c', 3e5) #speed of ligt in km/s (because H0 is in Km/s/Mpc)
    M_sun = kwargs.get('M_sun', 4.83) #Absolute magnitude of Sun
    L_sun = kwargs.get('L_sun', 1) #Luminosity of Sun

    print("\nGenerating redshifts")
    z = generate_redshift(N, z_min, z_max) #Redshift
    print("\nGenerating dL")
    #dL_mpc = np.array([get_dL(zi, H0, Om, Ol, c) for zi in z]) #luminosity distance in Mpc
    dL_mpc = get_dL_cosmo(z, Oc, Ob, h, sigma8, n_s, cosmo) #luminosity distance in Mpc
    print("\nGenerating luminosities") 
    L = generate_schechter_lum(N, L_star, alpha, L_min, L_max, phi_star) #luminosities
    if sed is not None:
        print("\nConverting to band luminosities")
        L = pd.DataFrame({"L": L})
        L = np.array(L['L'].apply(lumBolom2lumBand, args=(sed, response, renorm))) #from bolometric to band
    else: L = L*L_bandRatio               #from bolometric to band
    M = lum2absMag(L, M_sun, L_sun) #Absolute magnitudes
    m = M + 5*np.log10(dL_mpc) + 25 #Aparent magnitudes

    if to_table:
        table = Table({"z":z, "dL":dL_mpc, "L":L, "M":M, "m":m})
        table["dL"].unit = u.Mpc
        table["L"].unit = u.Lsun
        return table
    else:
        return z, m, M, L, dL_mpc

