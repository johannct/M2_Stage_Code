'''A module containing several useful functions to simulate and fit sky maps.'''

### Package importation:
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from iminuit import Minuit
from iminuit.cost import LeastSquares


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
    cost_func = kwargs.pop("cost_func", LeastSquares(x_fit, y_fit, y_err, model))
    m = Minuit(cost_func, *init, name=par_name)
    if bounds: m.limits = bounds
    if fixed: m.fixed[fixed] = True
    m.migrad()       # finds minimum of least_squares function
    m.hesse()        # accurately computes uncertainties
    try: m.minos()   # computes non symetrics uncertainties
    except: print('Unable to use minos()')

    if plot_fig: fig, ax = plot_fit(x_fit, y_fit, m.values, model, **kwargs)
    else: get_fig = False #fig, ax only exist if plot_fig = True, so they can't be returned if plot_fig = False.
    if get_fig: return m, fig, ax
    else: return m

    
## Fit models:
def gauss(x,A,mu,sigma):
    '''Return the usual gauss function of x, depending on the amplitude A, the mean mu, and the std sigma.'''
    return (A / (sigma*np.sqrt(2*np.pi)))* np.exp(-np.square(x-mu)/(2*sigma**2))