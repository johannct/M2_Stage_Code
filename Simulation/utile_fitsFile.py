'''A module containing several useful functions to make operation of .fits files.'''

import fitsio
import numpy as np
import pandas as pd
from astropy.table import Table


### Functions definition:

## Queries functions:
def get_indexID(fits, ID, HDU='MAPS'):
    if type(ID) == np.ndarray or type(ID) == list or type(ID) == tuple:
        idx = [get_index(fits, id, HDU) for id in ID]
    else: idx = fits[HDU].where(f'Map_ID == "{ID}"')
    return idx
    

def hasID_fits(fits, ID, HDU='MAPS'):
    """Check if an ID or a list of IDs is in the HDU of the fits.
    Return True if it is, and False if not."""
    return np.isin(ID, fits[HDU]['Map_ID'][:])


def get_ID_not_in(fits, HDU_ref ='FIT_MINUIT', HDU_target='MAPS'):
    """Return the IDs in HDU_target that are not in HDU_ref."""
    ID_ref = fits[HDU_ref].read(columns='Map_ID')
    ID_target = fits[HDU_target].read(columns='Map_ID')
    not_in = np.isin(ID_target, ID_ref, invert=True)
    return ID_target[not_in]


## Avoiding duplicates:
def get_not_in_fits_ID(df, fits, HDU='MAPS', ID_col='Map_ID'):
    """From a dataframe, return only the rows corresponding to IDs that are not already in the HDU of a fits.
    Espectially usefull to avoid saving twice the same map."""
    in_fits = hasID_fits(fits, np.array(df[ID_col]), HDU)
    return df[~in_fits]


def get_row_not_in(df1, df2, as_table=True):
    """From a dataframe df1, return only the rows that are not in another dataframe df2.
    Espectially usefull to avoid saving twice the same fit result from iminuit."""
    if type(df1) == Table: df1 = df1.to_pandas()
    if type(df2) == Table: df2 = df2.to_pandas()
    df_merge = df1.merge(df2, how='left', indicator=True)
    rows = df_merge[df_merge['_merge'] == 'left_only'].drop('_merge', axis=1)
    if as_table: rows = Table.from_pandas(rows)
    return rows


def remove_HDU(original, clear, extNB):
    """Copy a fits file from original to clear by avoiding the HDY extension extNB."""
    with fitsio.FITS(original) as fin:
        with fitsio.FITS(clear, 'rw', clobber=True) as fout:
            for i in range(len(fin)):
                if i == extNB: # index to remove
                    continue
                
                data = fin[i].read()
                header = fin[i].read_header()
                
                # On récupère le nom de l'extension dans le header
                # .get('EXTNAME') renvoie None si le mot-clé n'existe pas (cas de la Primary HDU)
                name = header.get('EXTNAME')
                
                # On écrit en passant l'argument extname
                fout.write(data, header=header, extname=name)