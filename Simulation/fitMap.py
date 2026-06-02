'''A module functions to fit and save sky maps.'''

### Package importation:
import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt
#from ulid import ULID

from iminuit import Minuit
from iminuit.cost import LeastSquares

try:
    from utile_fitsFile import *
    from simulMap import get_savefig
except:
    from Simulation.utile_fitsFile import *
    from Simulation.simulMap import get_savefig



### Functions definition:

## Plot functions:
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


def get_LeastSquare_plot(least_square, param, index=-2, start=0, stop=1e7, step=10000, get_fig=False, **kwargs):
    '''Plot the LeastSquare function of a fit vs. param[index].'''
    if type(param) == tuple: par = list(param)
    else: par = param.copy()
    parx = np.linspace(start, stop, step)
    ls = []
    for x in parx:
        par[index] = x
        ls.append(least_square(*par))

    output_path = kwargs.pop("output_path", False) #to save the figure.
    ext = kwargs.pop('format', 'pdf')
    if "figax" in kwargs.keys(): fig, ax = kwargs.pop("figax") #figax have to be tuple (fig, ax).
    else: fig, ax = plt.subplots()
    xlabel = kwargs.pop("xlabel", "param[{}]".format(index))
    if "title" in kwargs.keys(): ax.set_title(kwargs.pop("title"))
    ax.plot(parx, ls, **kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("$\chi_2$ function")
    
    if output_path: get_savefig(fig=fig, output_path=output_path, sufix="chi2_vs_param{}".format(index), format=ext)
    
    if get_fig: return ls, fig, ax
    else: return ls


def compareFit_val(df, x, y, xscale='log', yscale='linear', nsigma=1, surface=True, **kwargs):
    """Plot the comparison between the measured value y and the true one depending on a study parater x, when fit results are loaded in the dataframe df."""
    x = x + "_true"
    xlabel = kwargs.pop("xlabel", x)
    ylabel = kwargs.pop("ylabel", y)
    if "figax" in kwargs.keys(): fig, ax = kwargs["figax"] #figax have to be tuple (fig, ax).
    else: fig, ax = plt.subplots()

    if surface:
        y_inf = df[y] - nsigma*df[y + "_err"]
        y_sup = df[y] + nsigma*df[y + "_err"]
        ax.fill_between(df[x], y_inf, y_sup, alpha=0.5, label=f"Error surface to {nsigma}$\sigma$")
        ax.plot(df[x], df[y], label="Fit results")
    else: ax.errorbar(df[x], df[y], yerr=df[y + "_err"], label="Fit results")
    ax.plot(df[x], df[y + "_true"], label="True value")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.legend()


def compareFit_diff(df, x, y, xscale='log', yscale='linear', nsigma=1, surface=True, **kwargs):
    """Plot the difference between the measured value y and the true one depending on a study parater x, when fit results are loaded in the dataframe df."""
    x = x + "_true"
    ylegend = kwargs.pop("ylegend", y)
    xlabel = kwargs.pop("xlabel", x)
    ylabel = kwargs.pop("ylabel", f'abs({ylegend} - {ylegend}_true)')
    if "figax" in kwargs.keys(): fig, ax = kwargs["figax"] #figax have to be tuple (fig, ax).
    else: fig, ax = plt.subplots()
    
    y_diff = np.abs(df[y] - df[y + "_true"])
    if surface:
        y_inf = y_diff - nsigma*df[y + "_err"]
        y_sup = y_diff + nsigma*df[y + "_err"]
        ax.fill_between(df[x], y_inf, y_sup, alpha=0.5, label=f"Error surface to {nsigma}$\sigma$")
        ax.plot(df[x], y_diff, label=f"Difference {ylegend} - {ylegend}_true")
    else: ax.errorbar(df[x], y_diff, yerr=df[y + "_err"], label=f"Difference {ylegend} - {ylegend}_true")
    ax.hlines(0, df[x].min(), df[x].max(), linestyles="--", color="orange", label="y = 0")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel);
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.legend()


def compareFit_ratio(df, x, y, xscale='log', yscale='linear', nsigma=1, surface=True, **kwargs):
    """Plot the difference between the measured value y and the true one depending on a study parater x, when fit results are loaded in the dataframe df."""
    x = x + "_true"
    ylegend = kwargs.pop("ylegend", y)
    xlabel = kwargs.pop("xlabel", x)
    ylabel = kwargs.pop("ylabel", f'{ylegend} / {ylegend}_true')
    if "figax" in kwargs.keys(): fig, ax = kwargs["figax"] #figax have to be tuple (fig, ax).
    else: fig, ax = plt.subplots()
    
    y_ratio = df[y] / df[y + "_true"]
    if surface:
        y_inf = y_ratio - nsigma*df[y + "_err"]/df[y + "_true"]
        y_sup = y_ratio + nsigma*df[y + "_err"]/df[y + "_true"]
        ax.fill_between(df[x], y_inf, y_sup, alpha=0.5, label=f"Error surface to {nsigma}$\sigma$")
        ax.plot(df[x], y_ratio, label=f"Ratio {ylegend} / {ylegend}_true")
    else: ax.errorbar(df[x], y_ratio, yerr=df[y + "_err"]/np.abs(df[y + "_true"]), label=f"Ratio {ylegend} / {ylegend}_true")
    ax.hlines(1, df[x].min(), df[x].max(), linestyles="--", color="orange", label="y = 1")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel);
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.legend()



## Fit functions:
def fit_minuit(x_fit, y_fit, y_err, model, init, par_name, bounds=None, fixed=[], get_fig=False, get_cost=False, plot_fig=True, **kwargs):
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
    - get_cost: if True, return also the cost function used to the fit. By default, get_cost=False.
    '''
    verbose = kwargs.pop('verbose', True)
    weights = kwargs.pop('weights', None)
    if weights is not None: y_err = y_err/np.sqrt(weights)
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
    result = [m]
    if get_cost: result.append(cost_func)
    if get_fig:
        result.append(fig)
        result.append(ax)
    if len(result) > 1: return tuple(result)
    else: return m


def fit_dipole_err(model, map, init, names, bounds=([0, 0, -90], [1, 360, 90]), fixed=[], fit_mode="minuit", **kwargs):
    """Reteurn the result of a dipole fit, depending on a model. Can use either iminuit or scipy."""
    map_errY = np.sqrt(np.abs(map))
    if hasattr(map, "mask"): map_errY[map.mask] = np.inf #to filter masked pixels
    
    npix = len(map) #nb. of pixels
    ipix = np.arange(npix) #indices of each pixel
    nside= hp.npix2nside(npix)

    if fit_mode == "minuit": return fit_minuit(ipix, map, map_errY, model, init, names, list(zip(bounds[0], bounds[1])), fixed, title="Fit dipole", xlabel="Pixels", **kwargs)
    else: return curve_fit(model, ipix, map, p0=init, bounds=bounds, sigma= map_errY)



## Saving functions:
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
    """Prepare a dataframe to be saved in a fits file by adapting some columns."""
    data = df.copy()
    if 'Coord' in data.columns: data['Coord'] = data['Coord'].astype('U9')
    if data['Map_ID'].dtype == "O": data['Map_ID'] = data['Map_ID'].astype(str)
    for col in data.columns:
        if col.endswith('_fixed'):
            data[col].fillna(False, inplace=True)
    return data


def save_fit_minuit(dicMinuit, outputfile, HDU_target='FIT_MINUIT', header_global=None):
    if not os.path.isfile(outputfile):
        print(f"File {outputfile} doesn't exist ; creating one.")
        create_primaryHDU(outputfile, header_global)
    
    with fitsio.FITS(outputfile, 'rw') as fits: #Ouvrir le fichier en mode écriture ('rw' crée ou écrase)
        if HDU_target in fits: 
            fits[HDU_target].append(dicMinuit)
        else:
            print(f"Coulndn't find HDU {HDU_target} in the data ; creating one.")
            fits.write(dicMinuit, extname=HDU_target)
    print('Saving Dataframe results in {}'.format(outputfile))
    print("Saving complete.")


def get_save_fit_dfMinuit(dfMinuit, outputfile, HDU_target='FIT_MINUIT', header_global=None):
    df = dfMinuit.copy()
    df = prep_df_to_fits(df)
    if os.path.isfile(outputfile):
        data = fitsio.FITS(outputfile) #to avoid duplicates.
        if HDU_target in data: df = get_row_not_in(df, Table(data[HDU_target].read()))
        data.close()
    if type(df) == pd.DataFrame: df = Table.from_pandas(df) #to be able to use as_array()
    dic = df.as_array()
    if len(dic) == 0: print(f'All the rows already exists in {HDU_target}, or empty table given. No row was saved.')
    else: save_fit_minuit(dic, outputfile, HDU_target, header_global)
    return df


def get_dicMaps(map_IDs, maps, key_ID="Map_ID", key_map="Map"):
    return {key_ID: np.array(map_IDs, dtype='S26'), key_map: np.array(maps)}


def save_dicMaps(dicMaps, outputfile, HDU_target='MAPS', header_global=None):
    if not os.path.isfile(outputfile):
        print(f"File {outputfile} doesn't exist ; creating one.")
        create_primaryHDU(outputfile, header_global)
    
    with fitsio.FITS(outputfile, 'rw') as fits: #Ouvrir le fichier en mode écriture ('rw' crée ou écrase)
        if HDU_target in fits: 
            fits[HDU_target].append(dicMaps)
        else:
            print(f"Coulndn't find HDU {HDU_target} in the data ; creating one.")
            fits.write(dicMaps, extname=HDU_target)
    print('Saving maps in {}'.format(outputfile))
    print("Saving complete.")


def get_save_dicMaps(dicMaps, outputfile, HDU_target='MAPS', header_global=None, ID_col='Map_ID'):
    dic = dicMaps.copy()
    if os.path.isfile(outputfile):
        data = fitsio.FITS(outputfile) #to avoid duplicates.
        if HDU_target in data: dic = get_not_in_fits_ID(dic, data, HDU=HDU_target, ID_col=ID_col)
        data.close()
    
    if len(dic[ID_col]) == 0: print(f'All the maps already exists in {HDU_target}, or empty dict given. No map was saved.')
    else: save_dicMaps(dic, outputfile, HDU_target=HDU_target, header_global=header_global)
    return dic