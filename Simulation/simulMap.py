'''A module containing several useful functions to simulate and fit sky maps.'''

### Package importation:
import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord
from ulid import ULID

from iminuit import Minuit
from iminuit.cost import LeastSquares

from utile_fitsFile import *


### Functions definition:

## Simulation functions:
def RADEC2NSource(NSIDE, RA, DEC, **kwargs):
    '''Return the number of sources by pixel, depending on the resolution, the right-ascention RA and the declination DEC.'''
    NPIX = hp.nside2npix(NSIDE)
    NSource_px = hp.ang2pix(NSIDE, RA, DEC, lonlat=True, **kwargs)
    NSource_px = np.bincount(NSource_px, minlength=NPIX)
    return NSource_px


def RADEC2NSource_Table(NSIDE, table, col_RA="RA", col_DEC="DEC", **kwargs):
    '''Return the number of sources by pixel, depending on the resolution, and a table or a dataframe having right-ascention and declination in its colums.
    col_RA and col_DEC allow to precise the names of these columns; by default: col_RA="RA", col_DEC="DEC"'''
    return RADEC2NSource(NSIDE, table[col_RA], table[col_DEC], **kwargs)
    

def get_RADEC2NSource(NSIDE, NSource_px_th):
    '''Return the simulated number of sources by pixel, the right-ascention and the declination, by uniformally randomizing RA and DEC, depending on the resolution NSIDE and the theorical number of sources by pixel NSource_px_th.'''
    #RA, DEC simulation:
    NPIX = hp.nside2npix(NSIDE)
    RA = np.random.uniform(0, 360, int(NPIX*NSource_px_th))
    DEC_sin = np.random.uniform(-1, 1, int(NPIX*NSource_px_th)) #pour DEC, il faut passer par sin(DEC), compris entre -1 et 1
    DEC = np.degrees(np.arcsin(DEC_sin))

    #Number of sources by pixel conversion:
    NSource_px = RADEC2NSource(NSIDE, RA, DEC)
    return NSource_px, RA, DEC


def proba_schechter_mag(M, M_star, alpha):
    """Compute the relative probability of a magnitude M.
    M_star is the caracteristic magnitude of Schechter's luminosity function."""
    L_ratio = 10**(0.4 * (M_star - M)) #Conversion magnitude -> relative luminosity L/L*
    return (L_ratio**alpha) * np.exp(-L_ratio)


def generate_schechter_magnitudes(N, M_min=-23, M_max=-15, M_star=-20.44, alpha=-1.1):
    """Generate N randomized absolute magnitudes, by using a reject test and Schechter's law."""
    magnitudes = []
    while len(magnitudes) < N:
        M_test = np.random.uniform(M_min, M_max)
        p_test = np.random.uniform(0, 1) #Probability for rejet test
        if p_test < proba_schechter_mag(M_test, M_star, alpha): magnitudes.append(M_test)         
    return np.array(magnitudes)


def get_dL(zi, H0=67.4, Om=0.315, Ol=0.685, c=3e8):
    '''Compute the luminosity distance depending on the redshift z, the Hubble constant H0, the cosmological parameters Om and Ol, and the speed of ligt c.'''
    inv_Ez = lambda zp: 1.0 / np.sqrt(Om * (1 + zp)**3 + Ol)
    integral, _ = quad(inv_Ez, 0, zi)
    return (c / H0) * (1 + zi) * integral


def generate_galaxies(N, z_min=0.01, z_max=2.0, M_min=-23, M_max=-15, M_star=-20.44, alpha=-1.1, sigma_M_star=1.5, rejectTest_M=False, only_coord=False, **kwargs):
    #Cosmological Parameters  (Planck 2018):
    H0 = kwargs.get('H0', 67.4)
    Om = kwargs.get('Om', 0.315)
    Ol = kwargs.get('Ol', 0.685)
    c = kwargs.get('c', 3e5) #speed of ligt in km/s (because H0 is in Km/s/Mpc)
    M_sun = kwargs.get('M_sun', 4.83) #Absolute magnitude of Sun

    #Coord ra, dec, z:
    ra = np.random.uniform(0, 360, N)
    dec = np.random.uniform(-1, 1, N)
    dec = np.degrees(np.arcsin(dec))
    z = np.random.uniform(z_min, z_max, N) #Redshift
    if only_coord: return ra, dec, z

    #Other data:
    dL_mpc = np.array([get_dL(zi, H0, Om, Ol, c) for zi in z]) #luminosity distance in Mpc
    #Absolute magnitudes:
    if rejectTest_M: M = generate_schechter_magnitudes(N, M_min, M_max, M_star, alpha)
    else: M = np.random.normal(M_star, sigma_M_star, N)
    m = M + 5*np.log10(dL_mpc) + 25 #Aparent magnitudes
    L = 10**(0.4 * (M_sun - M)) #luminosities
    return ra, dec, z, m, M, L, dL_mpc


## Plot functions:
def get_savefig(fig, output_path, sufix, **kwargs):
    """Save the figure fig; several format for the images can be chosen in the same time, by inputing a tuple or a list in the parmateer format.
    This function is especially created to be used in other functions."""
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


def get_hist(x, title='', xlabel='Nb. of sources by pixel', ylabel='', show=True, **kwargs):
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


def plot_fit(x_fit, y_fit, values, model, **kwargs):
    '''Plot on a same figure data and its fit, and return the corresponding fig, ax variables.
    
    Parameters:
    - x_fit, y_fit: original data used to compute the fit.
    - values: tuple containing the parameter values obtainds by the fit.
    - model: fitted function. The fit plot is obtained by: ax.plot(bins, model(x_fit, *values))

    Spetial keys for **kwargs:
     - title, xlabel, ylabel: allow to custom the figure.
     - figax: allow to use existing fig, ax variables, rather creating new ones; have to be tuple (fig, ax).
    '''
    if "figax" in kwargs.keys(): fig, ax = kwargs["figax"] #figax have to be tuple (fig, ax).
    else: fig, ax = plt.subplots()
    ax.scatter(x_fit, y_fit, label="data")
    ax.plot(x_fit, model(x_fit, *values), c="r", label="fit")
    ax.legend()
    if "title" in kwargs.keys(): ax.set_title(kwargs["title"])
    if "xlabel" in kwargs.keys(): ax.set_xlabel(kwargs["xlabel"])
    if "ylabel" in kwargs.keys(): ax.set_xlabel(kwargs["ylabel"])

    #Saving figure:
    output_path = kwargs.get("output_path", False)
    if output_path:
        ext = kwargs.get('format', 'pdf')
        get_savefig(fig=fig, output_path=output_path, sufix="Fit", format=ext)
    return fig, ax


def get_LeastSquare_plot(least_square, param, index=-2, start=0, stop=1e7, step=10000, **kwargs):
    '''Plot the LeastSquare function of a fit vs. param[index].'''
    par = param.copy()
    parx = np.linspace(start, stop, step)
    ls = []
    for x in parx:
        par[index] = x
        ls.append(least_square(*par))
    
    fig, ax = plt.subplots()
    ax.plot(parx, ls)
    xlabel = kwargs.get("xlabel", "param[{}]".format(index))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("$\chi_2$ function")
    if "title" in kwargs.keys(): ax.set_title(kwargs["title"])
    
    output_path = kwargs.get("output_path", False)
    if output_path:
        ext = kwargs.get('format', 'pdf')
        get_savefig(fig=fig, output_path=output_path, sufix="chi2_vs_param{}".format(index), format=ext)
    
    return ls


## Fit functions:
def fit_minuit(x_fit, y_fit, y_err, model, init, par_name, bounds=None, fixed=[], get_fig=False, plot_fig=True, **kwargs):
    '''Fit data with iminuit and the least squares methode, and return the corresponding Minuit() instance.
    Also show the figure with both data and fit, by using plot_fit(x_fit, y_fit, values, model, **kwargs).

    Parameters:
    - x_fit, y_fit: original data used to compute the fit.
    - y_err: error on y_fit.
    - model: function to fit.
    - init: tuple containing the initial parameter values for the fit.
    - par_name: tuple containing the names of the model parameters.

    Optionnal parameters:
    - get_fig: if True, return also the fig, ax variables from the data&fit figure. By default, get_fig=False.
    '''
    verbose = kwargs.pop('verbose', True)
    cost_func = kwargs.pop("cost_func", LeastSquares(x_fit, y_fit, y_err, model))
    m = Minuit(cost_func, *init, name=par_name)
    if bounds: m.limits = bounds
    if fixed: m.fixed[fixed] = True
    m.migrad()       # finds minimum of least_squares function
    m.hesse()        # accurately computes uncertainties
    try: m.minos()   # computes non symetrics uncertainties
    except:
        if verbose: print('Unable to use minos()')

    if plot_fig: fig, ax = plot_fit(x_fit, y_fit, m.values, model, **kwargs)
    else: get_fig = False #fig, ax only exist if plot_fig = True, so they can't be returned if plot_fig = False.
    if get_fig: return m, fig, ax
    else: return m


def fit_dipole_err(model, map, init, names, bounds=([0, 0, -90], [1, 360, 90]), fixed=[], fit_mode="minuit", **kwargs):
    """Reteurn the result of a dipole fit, depending on a model. Can use either iminuit or scipy."""
    map_errY = np.sqrt(np.abs(map)) 
    
    npix = len(map) #nb. of pixels
    ipix = np.arange(npix) #indices of each pixel
    #print("ipix =", ipix)
    nside= hp.npix2nside(npix)

    if fit_mode == "minuit": return fit_minuit(ipix, map, map_errY, model, init, names, list(zip(bounds[0], bounds[1])), fixed, title="Fit dipole", xlabel="Pixels", **kwargs)
    else: return curve_fit(model, ipix, map, p0=init, bounds=bounds, sigma= map_errY)


def get_saveFit_minuit_csv(m, output_path, sufix, x_fit=None, y_fit=None, y_fit_err=None, dic_param={}, index_col=None, verbose=True):
    """Save the resulf of a fit frim iminuit into a CSV file from a pandas dataframe."""
    #For list items, drop_duplicates() returns an error.
    #Values will be saved as str, and adapted to be later reconstructed by eval().
    def adapt_to_str(ch):
        ch = repr(ch)
        if str(ch) == "None": ch = "NaN" #easier to treat in DataFrame.
        if "dtype" in ch: ch = ch[:ch.index("dtype")] +')'
        if "array" in ch: ch = "np."+ch
        return ch
        
    for k, v in m.values.to_dict().items():
        dic_param[k] = v
        dic_param[k + "_init"] = m.init_params[k].value
        dic_param[k + "_err"] = m.errors[k]
        dic_param[k + "_fixed"] = m.fixed[k]
        dic_param[k + "_limits"] = [adapt_to_str(m.limits[k])]
    dic_param["valid"] = m.valid
    dic_param["x_fit"] = [adapt_to_str(x_fit)]
    dic_param["y_fit"] = [adapt_to_str(y_fit)]
    dic_param["y_fit_err"] = [adapt_to_str(y_fit_err)]
    
    table_param = pd.DataFrame(dic_param)
    if index_col: table_param.set_index(index_col, inplace=True)

    output_file = output_path + sufix + '.csv'
    try:
        if verbose: print("\nTrying to open Dataframe from", output_file)
        table_param2 = pd.read_csv(output_file, index_col=index_col)
        #if 'Unnamed: 0' in table_param2..columns: table_param2.pop('Unnamed: 0') #because a column can be added by the indices.
        if verbose: print("Concatenating new and former dataframes.\n")
        table_param2 = pd.concat([table_param, table_param2])
        table_param2.drop_duplicates()
    except:
        if verbose:
            print("No existing file found for the path", output_file)
            print("Creating a new one.\n")
        table_param2 = table_param
        
    if verbose: print('Saving Dataframe results in {}'.format(output_file))
    has_name = table_param2.index.name is not None #to save the index column only if it has a name.
    table_param2.to_csv(output_file, index=has_name)
    return table_param


def get_dicParam_minuit(m, mapID, add_param={}, to_pandas=True):
    """Return the dictionnary of fit results useful to be saved in an external file."""
    dic_param = {"Map_ID": mapID}
    for k, v in m.values.to_dict().items():
        dic_param[k] = v
        dic_param[k + "_init"] = m.init_params[k].value
        dic_param[k + "_err"] = m.errors[k]
        dic_param[k + "_fixed"] = m.fixed[k]
        dic_param[k + "_limit_0"] = m.limits[k][0] #separated to make easier convertion Table to pd.DataFrame.
        dic_param[k + "_limit_1"] = m.limits[k][1]
    dic_param["valid"] = m.valid
    for k, v in add_param.items(): dic_param[k] = v
    if to_pandas: dic_param = pd.DataFrame([dic_param])
    return dic_param


def prep_df_to_fits(df):
    """Prepare a dataframe to be saved in a fits file by adaoting some columns."""
    data = df.copy()
    data['Coord'] = data['Coord'].astype('U9')
    for col in data.columns:
        if col.endswith('_fixed'):
            data[col].fillna(False, inplace=True)
    return data


def save_fit_minuit(dicMinuit, outputfile, HDU_target='FIT_MINUIT'):
    with fitsio.FITS(outputfile, 'rw') as fits: #Ouvrir le fichier en mode écriture ('rw' crée ou écrase)
        if HDU_target in fits: 
            fits[HDU_target].append(dicMinuit)
        else:
            print(f"Coulndn't find HDU {HDU_target} in the data ; creating one.")
            fits.write(dicMinuit, extname=HDU_target)
    print('Saving Dataframe results in {}'.format(output_file)
    print("Saving complete.")


def get_save_fit_dfMinuit(dfMinuit, outputfile, HDU_target='FIT_MINUIT'):
    df = dfMinuit.copy()
    df = prep_df_to_fits(df)
    data = fitsio.FITS(outputfile) #to avoid duplicates.
    if HDU_target in data: df = get_row_not_in(df, Table(data[HDU_target].read()))
    data.close()
    dic = df.as_array()
    if len(dic) == 0: print(f'All the rows already exists in {HDU_target}, or empty table given. No row was saved.')
    else: save_fit_minuit(dic, outputfile, HDU_target)
    return df

    
## Fit models:
def gauss(x,A,mu,sigma):
    '''Return the usual gauss function of x, depending on the amplitude A, the mean mu, and the std sigma.'''
    return (A / (sigma*np.sqrt(2*np.pi)))* np.exp(-np.square(x-mu)/(2*sigma**2))


def apply_dipole_Alb(map, A, l, b, nest):
    """Return the measured map, when it is modified by a a dipole with amplitude A and direction l, b, depending on the true map."""
    npix = len(map) #nb. of pixels
    nside= hp.npix2nside(npix)
    ipix = np.arange(npix)
    dipole = SkyCoord(l=l*u.degree, b=b*u.degree, frame='galactic')
    u_dipole = dipole.cartesian.xyz.value
    ra, dec = hp.pix2ang(nside, ipix, nest=nest, lonlat=True)
    sources = SkyCoord(l=ra*u.degree, b=dec*u.degree, frame="galactic")
    u_source = sources.cartesian.xyz.value
    return A * np.dot(u_dipole, u_source)


def apply_dipole_ARaDec(map, A, ra, dec, nest):
    """Return the measured map, when it is modified by a a dipole with amplitude A and direction l, b, depending on the true map."""
    npix = len(map) #nb. of pixels
    nside= hp.npix2nside(npix)
    ipix = np.arange(npix)
    dipole = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame="icrs")
    u_dipole = dipole.cartesian.xyz.value
    raS, decS = hp.pix2ang(nside, ipix, nest=nest, lonlat=True)
    sources = SkyCoord(ra=raS*u.degree, dec=decS*u.degree, frame="icrs")
    u_source = sources.cartesian.xyz.value
    return A * np.dot(u_dipole, u_source)


def apply_dipole_MD(map, M, D0, D1, D2, nest, frame='icrs', contrast=True):
    """Returned the measured map, when it is modified by a a monopole M and a kinematic dipole D, depending on the true map."""
    if frame == 'icrs': Acostheta = apply_dipole_ARaDec(map, D0, D1, D2, nest)
    elif frame == 'galactic': Acostheta = apply_dipole_Alb(map, D0, D1, D2, nest)
    elif frame == 'cartesian':
        npix = len(map) #nb. of pixels
        nside = hp.npix2nside(npix)
        ipix = np.arange(npix)
        u_source = hp.pix2vec(nside, ipix, nest)
        D = np.array([D0, D1, D2])
        Acostheta = np.dot(D, u_source)
    if contrast: return M + Acostheta
    else: return M*(1 + Acostheta)