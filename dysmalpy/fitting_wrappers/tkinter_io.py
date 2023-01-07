# Methods for loading data for fitting wrappers

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tkinter as tk
from tkinter import filedialog


####
def get_paramfile_tkinter():
    root = tk.Tk()
    root.withdraw()
    title = "Select Dysmalpy parameter file"
    filetypes = (("PARAM files","*.params *.param *.PARAMS *.PARAM"),
                 ("All files","*.*"))
    param_filename = filedialog.askopenfilename(parent=root,filetypes=filetypes, title=title)
    root.destroy()

    print("Parameter file: {}".format(param_filename))

    return param_filename


def get_datadir_tkinter(ndim=None):
    root = tk.Tk()
    root.withdraw()
    title = "Data directory not set. Select the (main) data file."
    if ndim is None:
        filetypes = (("Data files","*.txt *.dat *.fits *.fit *.FITS *.FIT *.TXT *.DAT *.gz *.GZ"),
                 ("All files","*.*"))
    else:
        if ndim == 1:
            filetypes = (("Data files","*.txt *.dat *.TXT *.DAT"),
                 ("All files","*.*"))
        elif ndim == 2:
            filetypes = (("Data files","*.fits *.fit *.FITS *.FIT *.gz *.GZ"),
                 ("All files","*.*"))
        elif ndim == 3:
            filetypes = (("Data files","*.fits *.fit *.FITS *.FIT *.gz *.GZ"),
                 ("All files","*.*"))
        else:
            raise ValueError("ndim={} not recognized!".format(ndim))
    fname = filedialog.askopenfilename(parent=root,filetypes=filetypes, title=title)
    root.destroy()

    # Strip dir from fname
    if fname != '':
        delim = '/'
        f_arr = fname.split(delim)
        datadir = delim.join(f_arr[:-1]) + delim
    else:
        datadir = ''

    return datadir
