'''A module containing several useful functions to simulate and fit sky maps.'''

### Package importation:
import os
import numpy as np
import dask.array as da
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord



### Functions definition:

## Simulation functions:
def raDec2map(NSIDE, RA, DEC, **kwargs):
    '''Return the number of sources by pixel, depending on the resolution, the right-ascention RA and the declination DEC.'''
    NPIX = hp.nside2npix(NSIDE)
    NSource_px = hp.ang2pix(NSIDE, RA, DEC, lonlat=True, **kwargs)
    NSource_px = np.bincount(NSource_px, minlength=NPIX)
    return NSource_px


def raDec2map_Table(NSIDE, table, col_RA="RA", col_DEC="DEC", **kwargs):
    '''Return the number of sources by pixel, depending on the resolution, and a table or a dataframe having right-ascention and declination in its colums.
    col_RA and col_DEC allow to precise the names of these columns; by default: col_RA="RA", col_DEC="DEC"'''
    return raDec2map(NSIDE, table[col_RA], table[col_DEC], **kwargs)
    

def get_raDec2map(NSIDE, NSource_px_th):
    '''Return the simulated number of sources by pixel, the right-ascention and the declination, by uniformally randomizing RA and DEC, depending on the resolution NSIDE and the theorical number of sources by pixel NSource_px_th.'''
    #RA, DEC simulation:
    NPIX = hp.nside2npix(NSIDE)
    RA = np.random.uniform(0, 360, int(NPIX*NSource_px_th))
    DEC_sin = np.random.uniform(-1, 1, int(NPIX*NSource_px_th)) #pour DEC, il faut passer par sin(DEC), compris entre -1 et 1
    DEC = np.degrees(np.arcsin(DEC_sin))

    #Number of sources by pixel conversion:
    NSource_px = raDec2map(NSIDE, RA, DEC)
    return NSource_px, RA, DEC


def cut_m52map(m, m5, chunk_size=1e4):
    m = da.from_array(np.array(m), chunks=chunk_size)
    m5 = da.from_array(np.array(m5), chunks=chunk_size)

    if (m.ndim == 1) and (m5.ndim == 1): #to broadcast m and m5
        m = m.reshape(1, -1)
        m5 = m5.reshape(-1, 1)

    mask = m <= m5
    result_dask = mask.sum(axis=1)
    return result_dask.compute()



## Plot functions:
def get_savefig(fig, output_path, sufix, **kwargs):
    """Save the figure fig; several format for the images can be chosen in the same time, by inputing a tuple or a list in the parmateer format.
    This function is especially created to be used in other functions."""
    os.makedirs(output_path, exist_ok=True)
    ext = kwargs.get('format', 'pdf')
    if type(ext)==tuple or type(ext)==list: #to save in several formats.
        for e in ext:
            output_file = output_path + '{}.{}'.format(sufix, e)
            print('Saving {} in {}'.format(sufix, output_file))
            fig.savefig(output_file, bbox_inches='tight') #bbox_inches ti$o not cut labels and title.
    else:
        output_file = output_path + '{}.{}'.format(sufix, ext)
        print('Saving {} in {}'.format(sufix, output_file))
        fig.savefig(output_file, bbox_inches='tight') #bbox_inches to not cut labels and title.


def get_hist(x, title='', xlabel='Nb. of sources by pixel', ylabel='', show=False, **kwargs):
    '''Plot the histogram of x, and return the corresponding var, bins obtained from this histogram by:
    var, bins = plt.hist(x, **kwargs)[:-1]
    bins = (bins[1:] + bins[:-1])/2

    Optionnal parameters:
    - title, xlabel, ylabel and **kwargs: allow to custom the histogram.
    - show: says if the histogram has to be showed by "if show: plt.show()".
    '''
    if "figax" in kwargs.keys(): fig, ax = kwargs.pop("figax") #figax have to be tuple (fig, ax); using pop because hist doesn't accept figax argument.
    else: fig, ax = plt.subplots() #not inserted as default value because it would create a new useless figure.
    get_fig = kwargs.pop("get_fig", False)
    var, bins = ax.hist(x, **kwargs)[:-1]
    if title: ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_xlabel(ylabel)
    bins = (bins[1:] + bins[:-1])/2
    if show: plt.show()

    #Saving figure:
    output_path = kwargs.get("output_path", False)
    if output_path:
        ext = kwargs.get('format', 'pdf')
        get_savefig(fig=fig, output_path=output_path, sufix="Hist", format=ext)
    if get_fig: return var, bins, fig, ax
    else: return var, bins







    
## Fit models:
def gauss(x,A,mu,sigma):
    '''Return the usual gauss function of x, depending on the amplitude A, the mean mu, and the std sigma.'''
    return (A / (sigma*np.sqrt(2*np.pi)))* np.exp(-np.square(x-mu)/(2*sigma**2))


def get_pixSide(map, cut_masked=False):
    """From a map, return npix, nside, ipix."""
    npix = len(map) #nb. of pixels
    nside= hp.npix2nside(npix)
    ipix = np.arange(npix)
    if hasattr(map, "mask") and cut_masked: ipix = ipix[~map.mask] #to filter masked pixels
    return npix, nside, ipix


def apply_dipole_Alb(map, A, l, b, nest, cut_masked=False):
    """Return the measured map, when it is modified by a a dipole with amplitude A and direction l, b, depending on the true map."""
    npix, nside, ipix = get_pixSide(map, cut_masked)
    if hasattr(map, "mask"): ipix = ipix[~map.mask] #to filter masked pixels
    dipole = SkyCoord(l=l*u.degree, b=b*u.degree, frame='galactic')
    u_dipole = dipole.cartesian.xyz.value
    ra, dec = hp.pix2ang(nside, ipix, nest=nest, lonlat=True)
    sources = SkyCoord(l=ra*u.degree, b=dec*u.degree, frame="galactic")
    u_source = sources.cartesian.xyz.value
    return A * np.dot(u_dipole, u_source)


def apply_dipole_ARaDec(map, A, ra, dec, nest, cut_masked=False):
    """Return the measured map, when it is modified by a a dipole with amplitude A and direction l, b, depending on the true map."""
    npix, nside, ipix = get_pixSide(map, cut_masked)
    if hasattr(map, "mask"): ipix = ipix[~map.mask] #to filter masked pixels
    dipole = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame="icrs")
    u_dipole = dipole.cartesian.xyz.value
    raS, decS = hp.pix2ang(nside, ipix, nest=nest, lonlat=True)
    sources = SkyCoord(ra=raS*u.degree, dec=decS*u.degree, frame="icrs")
    u_source = sources.cartesian.xyz.value
    return A * np.dot(u_dipole, u_source)


def apply_dipole_MD(map, M, D0, D1, D2, nest, frame='icrs', contrast=True, cut_masked=False):
    """Returned the measured map, when it is modified by a a monopole M and a kinematic dipole D, depending on the true map."""
    if frame == 'icrs': Acostheta = apply_dipole_ARaDec(map, D0, D1, D2, nest, cut_masked)
    elif frame == 'galactic': Acostheta = apply_dipole_Alb(map, D0, D1, D2, nest, cut_masked)
    elif frame == 'cartesian':
        npix, nside, ipix = get_pixSide(map, cut_masked)
        u_source = hp.pix2vec(nside, ipix, nest)
        D = np.array([D0, D1, D2])
        Acostheta = np.dot(D, u_source)
    if contrast: return M + Acostheta
    else: return M*(1 + Acostheta)
