'''A module containing several useful functions to simulate luminosities and magnitudes.'''

import numpy as np
import astropy.units as u
from astropy.table import Table
from scipy.integrate import quad


### Functions definition:

## Convertion functions:
def proba_schechter_lum(L, L_star):
        return (L/L_star)**alpha * np.exp(-L/L_star)


def lum2absMag(L, M_sun=4.83, L_sun=1):
    return M_sun - 2.5*np.log10(L/L_sun)


def lum2flux(L, dL):
    return L/(4*np.pi * (dL**2))


## Sampling functions:
def sample_truncated_power_law(alpha, xmin, xmax, size=None):
    """
    Sample from a truncated power-law distribution p(x) ∝ x^alpha
    for xmin <= x <= xmax, with alpha < -1.

    Parameters:
        alpha (float): exponent (must be < -1)
        xmin (float): lower bound (> 0)
        xmax (float): upper bound (> xmin)
        size (int): number of samples

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


def generate_schechter_lumRatio(N, alpha, x_min, x_max, phi_star=1, get_proba=False):
    """Generate N randomized luminosities, by using a reject test and Schechter's law."""
    samples = []
    reject = 0
    
    while len(samples) < N:
        L_cand = sample_truncated_power_law(alpha, x_min, x_max)
        
        # Acceptance ratio:
        # f(x) = x^alpha * exp(-x)
        # g(x) = x^alpha (on ignore la constante de normalisation qui s'annule)
        # f(x)/g(x) = exp(-x)
        acceptance_ratio = np.exp(-L_cand)
        
        # Acceptance test:
        u = np.random.uniform(0, 1)
        if u < acceptance_ratio:
            samples.append(L_cand)
        else:
            reject += 1
    print('Number of rejects =', reject)
    if get_proba: return np.array(samples), np.array(proba)
    else: return np.array(samples)


def generate_schechter_lum(N, L_star, alpha, L_min, L_max, phi_star):
    x_min, x_max = L_min/L_star, L_max/L_star
    return L_star*generate_schechter_lumRatio(N, alpha, x_min, x_max, phi_star)


def get_dL(zi, H0=67.4, Om=0.315, Ol=0.685, c=3e8):
    '''Compute the luminosity distance depending on the redshift z, the Hubble constant H0, the cosmological parameters Om and Ol, and the speed of ligt c.'''
    inv_Ez = lambda zp: 1.0 / np.sqrt(Om * (1 + zp)**3 + Ol)
    integral, _ = quad(inv_Ez, 0, zi)
    return (c / H0) * (1 + zi) * integral


def generate_lumMag(N, L_min=1e7, L_max=1e11,  L_star=1e10, alpha=-1.1, z_min=0.01, z_max=3.0, phi_star=1, to_table=True, **kwargs):
    #Cosmological Parameters  (Planck 2018):
    H0 = kwargs.get('H0', 67.4)
    Om = kwargs.get('Om', 0.315)
    Ol = kwargs.get('Ol', 0.685)
    c = kwargs.get('c', 3e5) #speed of ligt in km/s (because H0 is in Km/s/Mpc)
    M_sun = kwargs.get('M_sun', 4.83) #Absolute magnitude of Sun
    L_sun = kwargs.get('L_sun', 1) #Luminosity of Sun
    
    z = np.random.uniform(z_min, z_max, N) #Redshift
    dL_mpc = np.array([get_dL(zi, H0, Om, Ol, c) for zi in z]) #luminosity distance in Mpc
    L = generate_schechter_lum(N, L_star, alpha, L_min, L_max, phi_star) #luminosities
    M = lum2absMag(L, M_sun, L_sun) #Absolute magnitudes
    m = M + 5*np.log10(dL_mpc) + 25 #Aparent magnitudes

    if to_table:
        table = Table({"z":z, "dL":dL_mpc, "L":L, "M":M, "m":m})
        table["dL"].unit = u.Mpc
        table["L"].unit = u.Lsun
        return table
    else:
        return z, m, M, L, dL_mpc

