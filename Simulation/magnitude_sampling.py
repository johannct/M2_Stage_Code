import numpy as np

def proba_schechter_lum(L, L_star):
        return (L/L_star)**alpha * np.exp(-L/L_star)


def generate_schechter_lum(n_samples, L_star, alpha, L_min, L_max):
    """Generate N randomized luminosities, by using a reject test and Schechter's law."""
    samples = []
    f_max = proba_schechter_lum(L_min, L_star) # The maximum value of the function on [L_min, L_max] is often at L_min if alpha < -1
    while len(samples) < n_samples:
        L_cand = np.random.uniform(L_min, L_max)
        p_test = np.random.uniform(0, f_max)
        
        if p_test < proba_schechter_lum(L_cand, L_star):
            samples.append(L_cand)
            
    return np.array(samples)


def lum2absMag(L, M_sun=4.83, L_sun=3.828e33): #with L_sun in erg/s
    return M_sun - 2.5*np.log10(L/L_sun)


def get_dL(zi, H0=67.4, Om=0.315, Ol=0.685, c=3e8):
    '''Compute the luminosity distance depending on the redshift z, the Hubble constant H0, the cosmological parameters Om and Ol, and the speed of ligt c.'''
    inv_Ez = lambda zp: 1.0 / np.sqrt(Om * (1 + zp)**3 + Ol)
    integral, _ = quad(inv_Ez, 0, zi)
    return (c / H0) * (1 + zi) * integral


def generate_lumMag(N, L_star, alpha, L_min, L_max, z_min=0.01, z_max=3.0, **kwargs):
    #Cosmological Parameters  (Planck 2018):
    H0 = kwargs.get('H0', 67.4)
    Om = kwargs.get('Om', 0.315)
    Ol = kwargs.get('Ol', 0.685)
    c = kwargs.get('c', 3e5) #speed of ligt in km/s (because H0 is in Km/s/Mpc)
    M_sun = kwargs.get('M_sun', 4.83) #Absolute magnitude of Sun
    L_sun = kwargs.get('L_sun', 3.828e33) #Luminosity of Sun in  erg/s
    
    z = np.random.uniform(z_min, z_max, N) #Redshift
    dL_mpc = np.array([get_dL(zi, H0, Om, Ol, c) for zi in z]) #luminosity distance in Mpc
    L = generate_schechter_lum(n_samples, L_star, alpha, L_min, L_max)  #luminosities
    M = lum2absMag(L, M_sun, L_sun) #Absolute magnitudes
    m = M + 5*np.log10(dL_mpc) + 25 #Aparent magnitudes
    L = absMag2lum(M, M_sun)
    return z, m, M, L, dL_mpc

