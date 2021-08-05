#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A GUI for the galaxy kinematics modeling and fitting tool DysmalPy.

Last updates:
    - 2021-06-23 initialized by Daizhong Liu.
    - 2021-06-28 first version finalized by Daizhong Liu.
    - 2021-07-06 small adjustments, e.g., data points, panel widgets
    - 2021-07-25 adding lensing
    - 2021-08-03 xshift yshift can be free
    - 2021-08-04 can save lensing source plane and image plane cubes
"""

import os, sys, re, copy, json, ast, shutil, operator
import numpy as np
from enum import Enum
from pprint import pprint
from collections import OrderedDict

import logging
#logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
#logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from dysmalpy import galaxy, models, fitting, instrument, parameters, plotting, config, data_classes
from dysmalpy.fitting_wrappers import utils_io
from dysmalpy.fitting_wrappers.plotting import plot_bundle_1D
from dysmalpy.fitting_wrappers.dysmalpy_fit_single import dysmalpy_fit_single
from dysmalpy.fitting_wrappers.setup_gal_models import setup_gal_model_base, setup_single_object_1D, setup_single_object_2D, setup_single_object_3D, setup_fit_dict
from dysmalpy.fitting_wrappers import data_io
from dysmalpy.utils import apply_smoothing_3D, rebin, gaus_fit_sp_opt_leastsq

# <DZLIU><20210726> ++++++++++
from dysmalpy.lensing import LensingTransformer
# <DZLIU><20210726> ----------

import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import SqrtStretch, AsinhStretch, MinMaxInterval, ImageNormalize

from regions import (DS9Parser, read_ds9, write_ds9, RectangleSkyRegion, PixelRegion, PolygonPixelRegion, CirclePixelRegion, RegionMask, PixCoord, ds9_objects_to_string)

from spectral_cube import SpectralCube

import matplotlib as mpl
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['legend.borderpad'] = 0.1
mpl.rcParams['legend.labelspacing'] = 0.1
mpl.rcParams['legend.handletextpad'] = 0.1
mpl.rcParams['legend.borderaxespad'] = 0.1
mpl.rcParams['legend.handlelength'] = 1.05
mpl.rcParams['legend.frameon'] = True
mpl.rcParams['legend.framealpha'] = 0.1
mpl.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patheffects as path_effects
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, Ellipse, Polygon
from matplotlib.backend_bases import MouseButton
from mpl_toolkits.axes_grid1 import make_axes_locatable

import PyQt5
from PyQt5 import QtWidgets, QtCore, uic
from PyQt5.QtWidgets import (QMainWindow, QMenuBar, QMenu, QAction, qApp, QTabWidget, QFrame, QSpacerItem, QTabBar, QCheckBox, 
                             QWidget, QToolTip, QLabel, QLineEdit, QTextEdit, QPushButton, QRadioButton, QButtonGroup, QGroupBox, 
                             QSplitter, QHBoxLayout, QVBoxLayout, QGridLayout, QFormLayout, QSizePolicy, QApplication, 
                             QFileDialog, QDialog, QDialogButtonBox, QMessageBox, QScrollArea, QComboBox, QLayout, 
                             QShortcut)
from PyQt5.QtGui import (QIcon, QFont, QImage, QPixmap, QPolygon, QPolygonF, QPainter, QPen, QTransform, 
                         QRegExpValidator, QKeySequence)
from PyQt5.QtCore import (Qt, QObject, QPoint, QPointF, QRect, QRectF, QSize, QSizeF, QThread, QRegExp, QUrl, pyqtSignal, pyqtSlot)

#galaxy.logger.setLevel(logging.DEBUG)
#models.logger.setLevel(logging.DEBUG)
#fitting.logger.setLevel(logging.DEBUG)
logging.getLogger('DysmalPy').setLevel(logging.DEBUG)


logger.debug('models._dir_noordermeer: '+str(models._dir_noordermeer))


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)



class QDysmalPyGUI(QMainWindow):
    """A PyQt5 based DysmalPy GUI."""
    
    def __init__(self, ScreenSize=None):
        super(QDysmalPyGUI, self).__init__()
        self.DysmalPyParamFile = ''
        self.DysmalPyParams = {}
        self.DefaultDirectory = os.getcwd()
        self.DysmalPyFittingWorker = QDysmalPyFittingWorker()
        self.DysmalPyFittingThread = None
        self.logger = logging.getLogger('QDysmalPyGUI')
        self.logger.setLevel(logging.getLevelName(logging.getLogger(__name__).level)) # self.logger.setLevel(logging.DEBUG)
        self.initUI(ScreenSize=ScreenSize) # initUI
        # 
        # debug
        #self.selectParamFile('outdir/fitting_1D_mpfit.params')
    
    #def initDysmalPyParams(self):
    #    filepath = 'TODO' #<TODO># a template params file?
    #    self.DysmalPyParams = data_io.read_fitting_params(filepath)
    #    pass
    
    def initUI(self, ScreenSize=None):
        # 
        self.CentralWidget = QWidget()
        self.PanelLeft = QWidget()
        self.PanelMiddle = QWidget()
        self.PanelRight = QWidget()
        self.PanelLeft.setMinimumWidth(400)
        self.PanelMiddle.setMinimumWidth(180)
        self.PanelRight.setMinimumWidth(400)
        # 
        # Panel Left widgets
        self.LabelForTabWidgetA = QLabel(self.tr('Data'))
        self.LabelForTabWidgetB = QLabel(self.tr('Model'))
        self.TabWidgetHolderA = QWidget()
        self.TabWidgetHolderB = QWidget()
        self.TabWidgetA = QTabWidget()
        self.TabWidgetB = QTabWidget()
        self.TabPageA1 = QWidget()
        self.TabPageA2 = QWidget()
        self.TabPageA3 = QWidget()
        self.TabPageA4 = QWidget()
        self.TabPageA5 = QWidget()
        self.TabPageB1 = QWidget()
        self.TabPageB2 = QWidget()
        self.TabPageB3 = QWidget()
        self.TabPageB4 = QWidget()
        self.TabPageB5 = QWidget()
        self.TabPageB6 = QWidget()
        self.TabPageB7 = QWidget()
        self.TabPageA1.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred) # this is needed otherwise labels will be partially hidden
        self.TabPageA2.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred) # this is needed otherwise labels will be partially hidden
        self.TabPageA3.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred) # this is needed otherwise labels will be partially hidden
        self.TabPageA4.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred) # this is needed otherwise labels will be partially hidden
        self.TabPageA5.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred) # this is needed otherwise labels will be partially hidden
        self.TabPageB1.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred) # this is needed otherwise labels will be partially hidden
        self.TabPageB2.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred) # this is needed otherwise labels will be partially hidden
        self.TabPageB3.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred) # this is needed otherwise labels will be partially hidden
        self.TabPageB4.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred) # this is needed otherwise labels will be partially hidden
        self.TabPageB5.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred) # this is needed otherwise labels will be partially hidden
        self.TabPageB6.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred) # this is needed otherwise labels will be partially hidden
        self.TabPageB7.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred) # this is needed otherwise labels will be partially hidden
        self.ScrollAreaForTabPageA1 = QScrollArea(self)
        self.ScrollAreaForTabPageA2 = QScrollArea(self)
        self.ScrollAreaForTabPageA3 = QScrollArea(self)
        self.ScrollAreaForTabPageA4 = QScrollArea(self)
        self.ScrollAreaForTabPageA5 = QScrollArea(self)
        self.ScrollAreaForTabPageB1 = QScrollArea(self)
        self.ScrollAreaForTabPageB2 = QScrollArea(self)
        self.ScrollAreaForTabPageB3 = QScrollArea(self)
        self.ScrollAreaForTabPageB4 = QScrollArea(self)
        self.ScrollAreaForTabPageB5 = QScrollArea(self)
        self.ScrollAreaForTabPageB6 = QScrollArea(self)
        self.ScrollAreaForTabPageB7 = QScrollArea(self)
        self.ScrollAreaForTabPageA1.setWidgetResizable(True)
        self.ScrollAreaForTabPageA2.setWidgetResizable(True)
        self.ScrollAreaForTabPageA3.setWidgetResizable(True)
        self.ScrollAreaForTabPageA4.setWidgetResizable(True)
        self.ScrollAreaForTabPageA5.setWidgetResizable(True)
        self.ScrollAreaForTabPageB1.setWidgetResizable(True)
        self.ScrollAreaForTabPageB2.setWidgetResizable(True)
        self.ScrollAreaForTabPageB3.setWidgetResizable(True)
        self.ScrollAreaForTabPageB4.setWidgetResizable(True)
        self.ScrollAreaForTabPageB5.setWidgetResizable(True)
        self.ScrollAreaForTabPageB6.setWidgetResizable(True)
        self.ScrollAreaForTabPageB7.setWidgetResizable(True)
        self.ScrollAreaForTabPageA1.setWidget(self.TabPageA1)
        self.ScrollAreaForTabPageA2.setWidget(self.TabPageA2)
        self.ScrollAreaForTabPageA3.setWidget(self.TabPageA3)
        self.ScrollAreaForTabPageA4.setWidget(self.TabPageA4)
        self.ScrollAreaForTabPageA5.setWidget(self.TabPageA5)
        self.ScrollAreaForTabPageB1.setWidget(self.TabPageB1)
        self.ScrollAreaForTabPageB2.setWidget(self.TabPageB2)
        self.ScrollAreaForTabPageB3.setWidget(self.TabPageB3)
        self.ScrollAreaForTabPageB4.setWidget(self.TabPageB4)
        self.ScrollAreaForTabPageB5.setWidget(self.TabPageB5)
        self.ScrollAreaForTabPageB6.setWidget(self.TabPageB6)
        self.ScrollAreaForTabPageB7.setWidget(self.TabPageB7)
        self.LabelForLineEditParamFile = QLabel(self.tr('Param File:'))
        self.ButtonForLineEditParamFile = QPushButton(self.tr('...'))
        self.ButtonForLineEditParamFile.setToolTip('Open a param file')
        self.ButtonForLineEditParamFile.clicked.connect(self.onOpenParamFileCall)
        self.LineEditParamFile = QLineEdit()
        self.LineEditParamFile.setToolTip(self.tr('Input a Dysmal params file.'))
        # 
        self.LineEditDataParamsDict = OrderedDict()
        self.LineEditDataParamsDict['datadir'] = QWidgetForParamInput(\
                    keyname=self.tr('datadir'), 
                    keycomment=self.tr('Input data directory.'), 
                    datatype=str, 
                    fullwidth=True,
                    isdatadir=True, defaultdir=self.DefaultDirectory, enabled=False)
        self.LineEditDataParamsDict['fdata'] = QWidgetForParamInput(\
                    keyname=self.tr('fdata'), 
                    keycomment=self.tr('1D data file.'), 
                    datatype=str, 
                    fullwidth=True,
                    isdatafile=True, namefilter=self.tr('1D data file (*.txt *.dat *.*)'), defaultdir=self.DefaultDirectory, enabled=False)
        self.LineEditDataParamsDict['fdata_flux'] = QWidgetForParamInput(\
                    keyname=self.tr('fdata_flux'), 
                    keycomment=self.tr('2D flux map data file.'), 
                    datatype=str, 
                    fullwidth=True,
                    isdatafile=True, namefilter=self.tr('FITS file (*.fits *.fits.gz)'), defaultdir=self.DefaultDirectory, enabled=False)
        self.LineEditDataParamsDict['fdata_ferr'] = QWidgetForParamInput(\
                    keyname=self.tr('fdata_ferr'), 
                    keycomment=self.tr('2D flux error map data file.'), 
                    datatype=str, 
                    fullwidth=True,
                    isdatafile=True, namefilter=self.tr('FITS file (*.fits *.fits.gz)'), defaultdir=self.DefaultDirectory, enabled=False)
        self.LineEditDataParamsDict['fdata_vel'] = QWidgetForParamInput(\
                    keyname=self.tr('fdata_vel'), 
                    keycomment=self.tr('2D velocity map data file.'), 
                    datatype=str, 
                    fullwidth=True,
                    isdatafile=True, namefilter=self.tr('FITS file (*.fits *.fits.gz)'), defaultdir=self.DefaultDirectory, enabled=False)
        self.LineEditDataParamsDict['fdata_verr'] = QWidgetForParamInput(\
                    keyname=self.tr('fdata_verr'), 
                    keycomment=self.tr('2D velocity map data file.'), 
                    datatype=str, 
                    fullwidth=True,
                    isdatafile=True, namefilter=self.tr('FITS file (*.fits *.fits.gz)'), defaultdir=self.DefaultDirectory, enabled=False)
        self.LineEditDataParamsDict['fdata_disp'] = QWidgetForParamInput(\
                    keyname=self.tr('fdata_disp'), 
                    keycomment=self.tr('2D dispersion map data file.'), 
                    datatype=str, 
                    fullwidth=True,
                    isdatafile=True, namefilter=self.tr('FITS file (*.fits *.fits.gz)'), defaultdir=self.DefaultDirectory, enabled=False)
        self.LineEditDataParamsDict['fdata_derr'] = QWidgetForParamInput(\
                    keyname=self.tr('fdata_derr'), 
                    keycomment=self.tr('2D dispersion error map data file.'), 
                    datatype=str, 
                    fullwidth=True,
                    isdatafile=True, namefilter=self.tr('FITS file (*.fits *.fits.gz)'), defaultdir=self.DefaultDirectory, enabled=False)
        self.LineEditDataParamsDict['fdata_mask'] = QWidgetForParamInput(\
                    keyname=self.tr('fdata_mask'), 
                    keycomment=self.tr('2D mask map data file.'), 
                    datatype=str, 
                    fullwidth=True,
                    isdatafile=True, namefilter=self.tr('FITS file (*.fits *.fits.gz)'), defaultdir=self.DefaultDirectory, enabled=False)
        self.LineEditDataParamsDict['fdata_cube'] = QWidgetForParamInput(\
                    keyname=self.tr('fdata_cube'), 
                    keycomment=self.tr('3D data cube.'), 
                    datatype=str, 
                    fullwidth=True,
                    isdatafile=True, namefilter=self.tr('FITS file (*.fits *.fits.gz)'), defaultdir=self.DefaultDirectory, enabled=False)
        self.LineEditDataParamsDict['outdir'] = QWidgetForParamInput(\
                    keyname=self.tr('outdir'), 
                    keycomment=self.tr('Output directory.'), 
                    datatype=str, 
                    fullwidth=True,
                    isoutdir=True, defaultdir=self.DefaultDirectory, enabled=False)
        self.LineEditDataParamsDict['galID'] = QWidgetForParamInput(\
                    keyname=self.tr('galID'), 
                    keycomment=self.tr('Name of your object'), 
                    datatype=str)
        self.LineEditDataParamsDict['z'] = QWidgetForParamInput(\
                    keyname=self.tr('z'), 
                    keycomment=self.tr('Redshift'), 
                    datatype=float)
        self.LineEditDataParamsDict['data_inst_corr'] = QWidgetForParamInput(\
                    keyname=self.tr('data_inst_corr'), 
                    keycomment=self.tr('Is the dispersion corrected for instrumental broadening?'), 
                    datatype=bool)
        self.LineEditDataParamsDict['symmetrize_data'] = QWidgetForParamInput(\
                    keyname=self.tr('symmetrize_data'), 
                    keycomment=self.tr('Symmetrize data before fitting?'), 
                    datatype=bool)
        self.LineEditDataParamsDict['slit_width'] = QWidgetForParamInput(\
                    keyname=self.tr('slit_width'), 
                    keycomment=self.tr('arcsecs'), 
                    datatype=float)
        self.LineEditDataParamsDict['slit_pa'] = QWidgetForParamInput(\
                    keyname=self.tr('slit_pa'), 
                    keycomment=self.tr('Degrees from N towards blue'), 
                    datatype=float)
        self.LineEditDataParamsDict['profile1d_type'] = QWidgetForParamInput(\
                    keyname=self.tr('profile1d_type'), 
                    keycomment=self.tr('Default 1D aperture extraction shape'), 
                    datatype=str,
                    options=['circ_ap_cube', 'rect_ap_cube', 'square_ap_cube', 'circ_ap_pv', 'single_pix_pv'])
        self.LineEditDataParamsDict['aperture_radius'] = QWidgetForParamInput(\
                    keyname=self.tr('aperture_radius'),
                    keycomment=self.tr('Circular aperture radius, in ARCSEC. Have used half slit width in past'), 
                    datatype=float,
                    default=0.2)
        self.LineEditDataParamsDict['smoothing_type'] = QWidgetForParamInput(\
                    keyname=self.tr('smoothing_type'), 
                    keycomment=self.tr('Is the data median smoothed before extracting maps?'), 
                    datatype=str,
                    options=['median'])
        self.LineEditDataParamsDict['smoothing_npix'] = QWidgetForParamInput(\
                    keyname=self.tr('smoothing_npix'), 
                    keycomment=self.tr('Number of pixels for smoothing aperture'), 
                    datatype=float,
                    default=3.0)
        self.LineEditDataParamsDict['xcenter'] = QWidgetForParamInput(\
                    keyname=self.tr('xcenter'), 
                    keycomment=self.tr('Galaxy center in pixel coordinate, starting from 0 to NX-1. Need +1 for QFitsView. None means using image center.'), 
                    datatype=float,
                    default='None')
        self.LineEditDataParamsDict['ycenter'] = QWidgetForParamInput(\
                    keyname=self.tr('ycenter'), 
                    keycomment=self.tr('Galaxy center in pixel coordinate, starting from 0 to NY-1. Need +1 for QFitsView. None means using image center.'), 
                    datatype=float,
                    default='None')
        self.LineEditDataParamsDict['linked_posteriors'] = QWidgetForParamInput(\
                    keyname=self.tr('linked_posteriors'), 
                    keycomment=self.tr(''), 
                    datatype=str, 
                    listtype=list,
                    default="['total_mass', 'r_eff_disk', 'bt', 'fdm', 'sigma0']", 
                    fullwidth=True)
        self.LineEditDataParamsDict['pixscale'] = QWidgetForParamInput(\
                    keyname=self.tr('pixscale'), 
                    keycomment=self.tr('Pixel scale in arcsec/pixel'), 
                    datatype=float)
        self.LineEditDataParamsDict['fov_npix'] = QWidgetForParamInput(\
                    keyname=self.tr('fov_npix'), 
                    keycomment=self.tr('Number of pixels on a side of model cube'), 
                    datatype=int)
        self.LineEditDataParamsDict['spec_type'] = QWidgetForParamInput(\
                    keyname=self.tr('spec_type'), 
                    keycomment=self.tr('Spectral type, must be velocity.'), 
                    datatype=str, 
                    options=['velocity'])
        self.LineEditDataParamsDict['spec_start'] = QWidgetForParamInput(\
                    keyname=self.tr('spec_start'), 
                    keycomment=self.tr('Starting value for spectral axis'), 
                    datatype=float)
        self.LineEditDataParamsDict['spec_step'] = QWidgetForParamInput(\
                    keyname=self.tr('spec_step'), 
                    keycomment=self.tr('Step size for spectral axis in km/s'), 
                    datatype=float)
        self.LineEditDataParamsDict['nspec'] = QWidgetForParamInput(\
                    keyname=self.tr('nspec'), 
                    keycomment=self.tr('Number of spectral steps'), 
                    datatype=int)
        self.LineEditDataParamsDict['use_lsf'] = QWidgetForParamInput(\
                    keyname=self.tr('use_lsf'), 
                    keycomment=self.tr('True/False if using an LSF'), 
                    datatype=bool)
        self.LineEditDataParamsDict['sig_inst_res'] = QWidgetForParamInput(\
                    keyname=self.tr('sig_inst_res'), 
                    keycomment=self.tr('Instrumental dispersion in km/s'), 
                    datatype=float)
        self.LineEditDataParamsDict['psf_type'] = QWidgetForParamInput(\
                    keyname=self.tr('psf_type'), 
                    keycomment=self.tr('PSF type, Gaussian or Moffat or DoubleGaussian.'), 
                    datatype=str, 
                    options=['Gaussian','Moffat', 'DoubleGaussian'])
        self.LineEditDataParamsDict['psf_fwhm'] = QWidgetForParamInput(\
                    keyname=self.tr('psf_fwhm'), 
                    keycomment=self.tr('PSF FWHM in arcsecs'), 
                    datatype=float)
        self.LineEditDataParamsDict['psf_beta'] = QWidgetForParamInput(\
                    keyname=self.tr('psf_beta'), 
                    keycomment=self.tr('Beta parameter for a Moffat PSF'), 
                    datatype=float)
        # 
        #self.LineEditInstrumentParamsDict = OrderedDict()
        #self.LineEditInstrumentParamsDict['datadir'] = QWidgetForParamInput(\
        #            TODO
        # 
        self.LineEditLensingParamsDict = OrderedDict()
        self.LineEditLensingParamsDict['lensing_mesh'] = QWidgetForParamInput(\
                    keyname=self.tr('lensing_mesh'), 
                    keycomment=self.tr('Glafic lensing model mesh.dat file.'), 
                    datatype=str, 
                    fullwidth=True,
                    isdatafile=True, namefilter=self.tr('Glafic lensing model mesh.dat file (*.dat)'), 
                    defaultdir=self.DefaultDirectory, enabled=False)
        self.LineEditLensingParamsDict['lensing_ra'] = QWidgetForParamInput(\
                    keyname=self.tr('lensing_ra'), 
                    keycomment=self.tr('Lensing model reference WCS RA.'), 
                    datatype=float)
        self.LineEditLensingParamsDict['lensing_dec'] = QWidgetForParamInput(\
                    keyname=self.tr('lensing_dec'), 
                    keycomment=self.tr('Lensing model reference WCS Dec.'), 
                    datatype=float)
        self.LineEditLensingParamsDict['lensing_sra'] = QWidgetForParamInput(\
                    keyname=self.tr('lensing_sra'), 
                    keycomment=self.tr('Source plane map center RA.'), 
                    datatype=float)
        self.LineEditLensingParamsDict['lensing_sdec'] = QWidgetForParamInput(\
                    keyname=self.tr('lensing_sdec'), 
                    keycomment=self.tr('Source plane map center Dec.'), 
                    datatype=float)
        self.LineEditLensingParamsDict['lensing_ssizex'] = QWidgetForParamInput(\
                    keyname=self.tr('lensing_ssizex'), 
                    keycomment=self.tr('Source plane map size in pixels.'), 
                    datatype=float)
        self.LineEditLensingParamsDict['lensing_ssizey'] = QWidgetForParamInput(\
                    keyname=self.tr('lensing_ssizey'), 
                    keycomment=self.tr('Source plane map size in pixels.'), 
                    datatype=float)
        self.LineEditLensingParamsDict['lensing_spixsc'] = QWidgetForParamInput(\
                    keyname=self.tr('lensing_spixsc'), 
                    keycomment=self.tr('Source plane map pixel size in units of arcsec.'), 
                    datatype=float)
        self.LineEditLensingParamsDict['lensing_imra'] = QWidgetForParamInput(\
                    keyname=self.tr('lensing_imra'), 
                    keycomment=self.tr('Image plane map center RA.'), 
                    datatype=float)
        self.LineEditLensingParamsDict['lensing_imdec'] = QWidgetForParamInput(\
                    keyname=self.tr('lensing_imdec'), 
                    keycomment=self.tr('Image plane map center Dec.'), 
                    datatype=float)
        # 
        self.LineEditModelParamsDictForBulgeDisk = OrderedDict()
        self.LineEditModelParamsDictForBulgeDisk['total_mass'] = QWidgetForParamInput(\
                    keyname=self.tr('total_mass'), 
                    keycomment=self.tr("Total mass of disk and bulge log(Msun)"), 
                    datatype=float)
        self.LineEditModelParamsDictForBulgeDisk['total_mass_fixed'] = QWidgetForParamInput(\
                    keyname=self.tr('total_mass_fixed'), 
                    keycomment=self.tr(""), 
                    datatype=bool)
        self.LineEditModelParamsDictForBulgeDisk['bt'] = QWidgetForParamInput(\
                    keyname=self.tr('bt'), 
                    keycomment=self.tr("Bulge-to-Total Ratio"), 
                    datatype=float)
        self.LineEditModelParamsDictForBulgeDisk['bt_fixed'] = QWidgetForParamInput(\
                    keyname=self.tr("bt_fixed"), 
                    keycomment=self.tr("Fix bulge-to-total ratio?"), 
                    datatype=bool)
        self.LineEditModelParamsDictForBulgeDisk['r_eff_disk'] = QWidgetForParamInput(\
                    keyname=self.tr('r_eff_disk'), 
                    keycomment=self.tr("Effective radius of disk in kpc"), 
                    datatype=float)
        self.LineEditModelParamsDictForBulgeDisk['r_eff_disk_fixed'] = QWidgetForParamInput(\
                    keyname=self.tr("r_eff_disk_fixed"), 
                    keycomment=self.tr("Fix R_eff?"), 
                    datatype=bool)
        self.LineEditModelParamsDictForBulgeDisk['n_disk'] = QWidgetForParamInput(\
                    keyname=self.tr('n_disk'), 
                    keycomment=self.tr("Sersic index for disk"), 
                    datatype=float)
        self.LineEditModelParamsDictForBulgeDisk['n_disk_fixed'] = QWidgetForParamInput(\
                    keyname=self.tr("n_disk_fixed"), 
                    keycomment=self.tr("Fix n_disk?"), 
                    datatype=bool)
        self.LineEditModelParamsDictForBulgeDisk['invq_disk'] = QWidgetForParamInput(\
                    keyname=self.tr('invq_disk'), 
                    keycomment=self.tr("disk scale length to zheight ratio for disk"), 
                    datatype=float, 
                    default=5.0)
        self.LineEditModelParamsDictForBulgeDisk['invq_disk_NULL'] = QWidgetForParamInput(\
                    keyname=self.tr('NULL'), 
                    keycomment=self.tr(""))
        self.LineEditModelParamsDictForBulgeDisk['r_eff_bulge'] = QWidgetForParamInput(\
                    keyname=self.tr('r_eff_bulge'), 
                    keycomment=self.tr("Effective radius of bulge in kpc"), 
                    datatype=float)
        self.LineEditModelParamsDictForBulgeDisk['r_eff_bulge_fixed'] = QWidgetForParamInput(\
                    keyname=self.tr("r_eff_bulge_fixed"), 
                    keycomment=self.tr("Fix R_eff and bulge-to-total?"), 
                    datatype=bool)
        self.LineEditModelParamsDictForBulgeDisk['n_bulge'] = QWidgetForParamInput(\
                    keyname=self.tr('n_bulge'), 
                    keycomment=self.tr("Sersic index for bulge"), 
                    datatype=float)
        self.LineEditModelParamsDictForBulgeDisk['n_bulge_fixed'] = QWidgetForParamInput(\
                    keyname=self.tr("n_bulge_fixed"), 
                    keycomment=self.tr("Fix n_disk and bulge-to-total?"), 
                    datatype=bool)
        self.LineEditModelParamsDictForBulgeDisk['invq_bulge'] = QWidgetForParamInput(\
                    keyname=self.tr('invq_bulge'), 
                    keycomment=self.tr("disk scale length to zheight ratio for bulge"), 
                    datatype=float, 
                    default=2.0)
        self.LineEditModelParamsDictForBulgeDisk['invq_bulge_NULL'] = QWidgetForParamInput(\
                    keyname=self.tr('NULL'), 
                    keycomment=self.tr(""))
        # 
        self.LineEditModelParamsDictForDispersion = OrderedDict()
        self.LineEditModelParamsDictForDispersion['sigma0'] = QWidgetForParamInput(\
                    keyname=self.tr('sigma0'), 
                    keycomment=self.tr("Constant intrinsic dispersion value"), 
                    datatype=float)
        self.LineEditModelParamsDictForDispersion['sigma0_fixed'] = QWidgetForParamInput(\
                    keyname=self.tr('sigma0_fixed'), 
                    keycomment=self.tr(""), 
                    datatype=bool)
        # 
        self.LineEditModelParamsDictForZHeight = OrderedDict()
        self.LineEditModelParamsDictForZHeight['sigmaz'] = QWidgetForParamInput(\
                    keyname=self.tr('sigmaz'), 
                    keycomment=self.tr("Gaussian width of the galaxy in z"), 
                    datatype=float)
        self.LineEditModelParamsDictForZHeight['sigmaz_fixed'] = QWidgetForParamInput(\
                    keyname=self.tr('sigmaz_fixed'), 
                    keycomment=self.tr(""), 
                    datatype=bool)
        # 
        self.LineEditModelParamsDictForGeometry = OrderedDict()
        self.LineEditModelParamsDictForGeometry['inc'] = QWidgetForParamInput(\
                    keyname=self.tr('inc'), 
                    keycomment=self.tr("Inclination of galaxy, 0=face-on, 90=edge-on"), 
                    datatype=float)
        self.LineEditModelParamsDictForGeometry['inc_fixed'] = QWidgetForParamInput(\
                    keyname=self.tr('inc_fixed'), 
                    keycomment=self.tr(""), 
                    datatype=bool)
        self.LineEditModelParamsDictForGeometry['pa'] = QWidgetForParamInput(\
                    keyname=self.tr('pa'), 
                    keycomment=self.tr("Position angle of galaxy."), 
                    datatype=float)
        self.LineEditModelParamsDictForGeometry['pa_fixed'] = QWidgetForParamInput(\
                    keyname=self.tr('pa_fixed'), 
                    keycomment=self.tr(""), 
                    datatype=bool)
        self.LineEditModelParamsDictForGeometry['xshift'] = QWidgetForParamInput(\
                    keyname=self.tr('xshift'), 
                    keycomment=self.tr("xshift in pixels"), 
                    datatype=float, 
                    default=0.0)
        self.LineEditModelParamsDictForGeometry['xshift_fixed'] = QWidgetForParamInput(\
                    keyname=self.tr('xshift_fixed'), 
                    keycomment=self.tr(""), 
                    datatype=bool)
        self.LineEditModelParamsDictForGeometry['yshift'] = QWidgetForParamInput(\
                    keyname=self.tr('yshift'), 
                    keycomment=self.tr("yshift in pixels"), 
                    datatype=float, 
                    default=0.0)
        self.LineEditModelParamsDictForGeometry['yshift_fixed'] = QWidgetForParamInput(\
                    keyname=self.tr('yshift_fixed'), 
                    keycomment=self.tr(""), 
                    datatype=bool)
        self.LineEditModelParamsDictForGeometry['vel_shift'] = QWidgetForParamInput(\
                    keyname=self.tr('vel_shift'), 
                    keycomment=self.tr("vel_shift in km/s"), 
                    datatype=float, 
                    default=0.0)
        self.LineEditModelParamsDictForGeometry['vel_shift_fixed'] = QWidgetForParamInput(\
                    keyname=self.tr('vel_shift_fixed'), 
                    keycomment=self.tr(""), 
                    datatype=bool)
        # 
        self.LineEditModelParamsDictForDarkMatterHalo = OrderedDict()
        self.LineEditModelParamsDictForDarkMatterHalo['halo_profile_type'] = QWidgetForParamInput(\
                    keyname=self.tr('halo_profile_type'), 
                    keycomment=self.tr("Halo type"),
                    datatype=str, 
                    options=['NFW', 'twopowerhalo', 'burkert', 'einasto', 'dekelzhao'])
        self.LineEditModelParamsDictForDarkMatterHalo['halo_profile_type_NULL'] = QWidgetForParamInput(\
                    keyname=self.tr('NULL'), 
                    keycomment=self.tr(""))
        self.LineEditModelParamsDictForDarkMatterHalo['mvirial'] = QWidgetForParamInput(\
                    keyname=self.tr('mvirial'), 
                    keycomment=self.tr("Halo virial mass in log(Msun)"), 
                    datatype=float)
        self.LineEditModelParamsDictForDarkMatterHalo['mvirial_fixed'] = QWidgetForParamInput(\
                    keyname=self.tr('mvirial_fixed'), 
                    keycomment=self.tr(""), 
                    datatype=bool)
        self.LineEditModelParamsDictForDarkMatterHalo['halo_conc'] = QWidgetForParamInput(\
                    keyname=self.tr('halo_conc'), 
                    keycomment=self.tr("Halo concentration parameter"), 
                    datatype=float)
        self.LineEditModelParamsDictForDarkMatterHalo['halo_conc_fixed'] = QWidgetForParamInput(\
                    keyname=self.tr('halo_conc_fixed'), 
                    keycomment=self.tr(""), 
                    datatype=bool)
        self.LineEditModelParamsDictForDarkMatterHalo['fdm'] = QWidgetForParamInput(\
                    keyname=self.tr('fdm'), 
                    keycomment=self.tr("Dark matter fraction at Reff"), 
                    datatype=float)
        self.LineEditModelParamsDictForDarkMatterHalo['fdm_fixed'] = QWidgetForParamInput(\
                    keyname=self.tr('fdm_fixed'), 
                    keycomment=self.tr(""), 
                    datatype=bool)
        # 
        initial_param_bounds = {}
        initial_param_bounds['total_mass']   = [10.0, 13.0]
        initial_param_bounds['bt']           = [0.0, 1.0]
        initial_param_bounds['r_eff_disk']   = [0.1, 30.0]
        initial_param_bounds['n_disk']       = [1.0, 8.0]
        initial_param_bounds['r_eff_bulge']  = [1.0, 5.0]
        initial_param_bounds['n_bulge']      = [1.0, 8.0]
        initial_param_bounds['sigma0']       = [5.0, 300.0]
        initial_param_bounds['sigmaz']       = [0.1, 1.0]
        initial_param_bounds['inc']          = [0.0, 90.0]
        initial_param_bounds['pa']           = [-180.0, 180.0]
        initial_param_bounds['mvirial']      = [10.0, 14.0]
        initial_param_bounds['halo_conc']    = [1.0, 20.0]
        initial_param_bounds['fdm']          = [0.0, 0.99]
        initial_param_stddev = {}
        initial_param_stddev['total_mass']   = 1.0
        initial_param_stddev['bt']           = 0.1
        initial_param_stddev['r_eff_disk']   = 1.0
        initial_param_stddev['n_disk']       = 0.1
        initial_param_stddev['r_eff_bulge']  = 1.0
        initial_param_stddev['n_bulge']      = 0.1
        initial_param_stddev['sigma0']       = 25.0
        initial_param_stddev['sigmaz']       = 0.1
        initial_param_stddev['inc']          = 0.1
        initial_param_stddev['pa']           = 0.1
        initial_param_stddev['mvirial']      = 0.5
        initial_param_stddev['halo_conc']    = 0.5
        initial_param_stddev['fdm']          = 1.0
        # 
        self.LineEditModelParamsDictForLimits = OrderedDict()
        for key in ['total_mass', 'bt', 'r_eff_disk', 'n_disk', 'r_eff_bulge', 'n_bulge', 
                    'sigma0', 'sigmaz', 'inc', 'pa', 
                    'mvirial', 'halo_conc', 'fdm']:
            self.LineEditModelParamsDictForLimits[key+'_bounds'] = QWidgetForParamInput(\
                    keyname=self.tr(key+'_bounds'), 
                    keycomment=self.tr(""), 
                    datatype=float, 
                    listtype=list, 
                    default=str(initial_param_bounds[key]),
                    fullwidth=True)
            self.LineEditModelParamsDictForLimits[key+'_prior'] = QWidgetForParamInput(\
                    keyname=self.tr(key+'_prior'), 
                    keycomment=self.tr(""), 
                    datatype=str, 
                    options=['flat','gaussian'])
            self.LineEditModelParamsDictForLimits[key+'_stddev'] = QWidgetForParamInput(\
                    keyname=self.tr(key+'_stddev'), 
                    keycomment=self.tr(""), 
                    datatype=float, 
                    default=initial_param_stddev[key])
        # 
        self.LineEditModelParamsDictForFitting = OrderedDict()
        self.LineEditModelParamsDictForFitting['fit_method'] = QWidgetForParamInput(\
                    keyname=self.tr('fit_method'), 
                    keycomment=self.tr("mcmc or mpfit"), 
                    datatype=str, 
                    options=['mpfit', 'mcmc'])
        self.LineEditModelParamsDictForFitting['moment_calc'] = QWidgetForParamInput(\
                    keyname=self.tr('moment_calc'), 
                    keycomment=self.tr('If False, observed maps fit with GAUSSIANS'), 
                    datatype=bool, 
                    default=False)
        self.LineEditModelParamsDictForFitting['fitdispersion'] = QWidgetForParamInput(\
                    keyname=self.tr('fitdispersion'), 
                    keycomment=self.tr("Simultaneously fit the velocity and dispersion?"), 
                    datatype=bool)
        self.LineEditModelParamsDictForFitting['fitflux'] = QWidgetForParamInput(\
                    keyname=self.tr('fitflux'), 
                    keycomment=self.tr("Simultaneously fit the flux?"), 
                    datatype=bool, 
                    default=False)
        self.LineEditModelParamsDictForFitting['do_plotting'] = QWidgetForParamInput(\
                    keyname=self.tr('do_plotting'), 
                    keycomment=self.tr("Produce all output plots?"), 
                    datatype=bool)
        self.LineEditModelParamsDictForFitting['do_plotting_NULL'] = QWidgetForParamInput(\
                    keyname=self.tr('do_plotting_NULL'), 
                    keycomment=self.tr("NULL"), 
                    datatype=bool, 
                    default=False)
        self.LineEditModelParamsDictForFitting['oversample'] = QWidgetForParamInput(\
                    keyname=self.tr('oversample'), 
                    keycomment=self.tr("Oversampling the model cube"), 
                    datatype=int, 
                    default=3)
        self.LineEditModelParamsDictForFitting['oversize'] = QWidgetForParamInput(\
                    keyname=self.tr('oversize'), 
                    keycomment=self.tr("Oversize of the model cube"), 
                    datatype=int, 
                    default=1)
        self.LineEditModelParamsDictForFitting['nWalkers'] = QWidgetForParamInput(\
                    keyname=self.tr('nWalkers'), 
                    keycomment=self.tr("Number of walkers. Must be even and >= 2x the number of free parameters."), 
                    datatype=int, 
                    default=1)
        self.LineEditModelParamsDictForFitting['nCPUs'] = QWidgetForParamInput(\
                    keyname=self.tr('nCPUs'), 
                    keycomment=self.tr("Number of CPUs to use for parallelization"), 
                    datatype=int, 
                    default=1)
        self.LineEditModelParamsDictForFitting['nBurn'] = QWidgetForParamInput(\
                    keyname=self.tr('nBurn'), 
                    keycomment=self.tr("Number of steps during burn-in"), 
                    datatype=int, 
                    default=1)
        self.LineEditModelParamsDictForFitting['nSteps'] = QWidgetForParamInput(\
                    keyname=self.tr('nSteps'), 
                    keycomment=self.tr("Number of steps for sampling"), 
                    datatype=int, 
                    default=1)
        self.LineEditModelParamsDictForFitting['scale_param_a'] = QWidgetForParamInput(\
                    keyname=self.tr('scale_param_a'), 
                    keycomment=self.tr(""), 
                    datatype=float, 
                    default=3.0)
        self.LineEditModelParamsDictForFitting['minAF'] = QWidgetForParamInput(\
                    keyname=self.tr('minAF'), 
                    keycomment=self.tr("Minimum acceptance fraction"), 
                    datatype=float, 
                    default='None')
        self.LineEditModelParamsDictForFitting['maxAF'] = QWidgetForParamInput(\
                    keyname=self.tr('maxAF'), 
                    keycomment=self.tr("Maximum acceptance fraction"), 
                    datatype=float, 
                    default='None')
        self.LineEditModelParamsDictForFitting['nEff'] = QWidgetForParamInput(\
                    keyname=self.tr('nEff'), 
                    keycomment=self.tr("Number of auto-correlation times before convergence"), 
                    datatype=int, 
                    default=10)
        self.LineEditModelParamsDictForFitting['maxiter'] = QWidgetForParamInput(\
                    keyname=self.tr('maxiter'), 
                    keycomment=self.tr("Maximum number of iterations before mpfit quits"), 
                    datatype=int, 
                    default=200)
        self.LineEditModelParamsDictForFitting['overwrite'] = QWidgetForParamInput(\
                    keyname=self.tr('overwrite'), 
                    keycomment=self.tr("Overwrite outputs"), 
                    datatype=bool, 
                    default='False')
        # 
        self.LineEditModelParamsDicts = []
        self.LineEditModelParamsDicts.append(self.LineEditModelParamsDictForBulgeDisk)
        self.LineEditModelParamsDicts.append(self.LineEditModelParamsDictForDarkMatterHalo)
        self.LineEditModelParamsDicts.append(self.LineEditModelParamsDictForDispersion)
        self.LineEditModelParamsDicts.append(self.LineEditModelParamsDictForGeometry)
        self.LineEditModelParamsDicts.append(self.LineEditModelParamsDictForZHeight)
        self.LineEditModelParamsDicts.append(self.LineEditModelParamsDictForLimits)
        self.LineEditModelParamsDicts.append(self.LineEditModelParamsDictForFitting)
        # 
        self.TabWidgetA.addTab(self.ScrollAreaForTabPageA1, self.tr('Data'))
        self.TabWidgetA.addTab(self.ScrollAreaForTabPageA2, self.tr('Data 2D'))
        self.TabWidgetA.addTab(self.ScrollAreaForTabPageA3, self.tr('Data 3D'))
        self.TabWidgetA.addTab(self.ScrollAreaForTabPageA4, self.tr('Instrument'))
        self.TabWidgetA.addTab(self.ScrollAreaForTabPageA5, self.tr('Lensing'))
        self.TabWidgetA.setTabToolTip(0, self.tr('Data setup'))
        self.TabWidgetA.setTabToolTip(1, self.tr('Instrument setup'))
        self.TabWidgetA.setTabToolTip(2, self.tr('Lensing transformation'))
        # 
        self.TabWidgetB.addTab(self.ScrollAreaForTabPageB1, self.tr('Disk+Bulge'))
        self.TabWidgetB.addTab(self.ScrollAreaForTabPageB2, self.tr('Halo'))
        self.TabWidgetB.addTab(self.ScrollAreaForTabPageB3, self.tr('Disp'))
        self.TabWidgetB.addTab(self.ScrollAreaForTabPageB4, self.tr('Geometry'))
        self.TabWidgetB.addTab(self.ScrollAreaForTabPageB5, self.tr('ZHeight'))
        self.TabWidgetB.addTab(self.ScrollAreaForTabPageB6, self.tr('Limits'))
        self.TabWidgetB.addTab(self.ScrollAreaForTabPageB7, self.tr('Fitting'))
        self.TabWidgetB.setTabToolTip(0, self.tr('Disk+Bulge model parameters'))
        self.TabWidgetB.setTabToolTip(1, self.tr('Halo model parameters'))
        self.TabWidgetB.setTabToolTip(2, self.tr('Dispersion profile parameters'))
        self.TabWidgetB.setTabToolTip(3, self.tr('Geometric parameters'))
        self.TabWidgetB.setTabToolTip(4, self.tr('Z-direction parameters'))
        self.TabWidgetB.setTabToolTip(5, self.tr('All parameter limits'))
        self.TabWidgetB.setTabToolTip(6, self.tr('Parameters related to the fitting'))
        # 
        self.TabWidgetA.setStyleSheet("""
            QTabWidget::tab-bar {
                left: 0; 
            }""")
        self.TabWidgetB.setStyleSheet("""
            QTabWidget::tab-bar {
                left: 0; 
            }""")
        #self.PanelSpecA = QWidget()
        #self.PanelSpecB = QWidget()
        # 
        for keyname in ['fdata', 'fdata_flux', 'fdata_ferr', 'fdata_vel', 'fdata_verr', 'fdata_disp', 'fdata_derr', 'fdata_mask', 'fdata_cube']:
            self.LineEditDataParamsDict['datadir'].ParamUpdateSignal.connect(self.LineEditDataParamsDict[keyname].onDataDirParamUpdateCall)
        # 
        # Panel Left layout
        self.LayoutForLineEditParamFile = QHBoxLayout()
        self.LayoutForLineEditParamFile.setContentsMargins(0, 0, 0, 0)
        self.LayoutForLineEditParamFile.setSpacing(0)
        self.LayoutForLineEditParamFile.addWidget(self.LabelForLineEditParamFile)
        self.LayoutForLineEditParamFile.addWidget(self.LineEditParamFile)
        self.LayoutForLineEditParamFile.addWidget(self.ButtonForLineEditParamFile)
        self.LayoutForLineEditDataParams = QGridLayout()
        self.LayoutForLineEditDataParams.setHorizontalSpacing(30)
        self.LayoutForLineEditDataParams.setVerticalSpacing(5)
        ncolumn = 2
        icount = 0 # self.LayoutForLineEditDataParams.count() * ncolumn # the widgets above are full width
        for key in self.LineEditDataParamsDict:
            if not (key.find('NULL')>=0):
                self.LineEditDataParamsDict[key].ParamUpdateSignal.connect(self.onParamUpdateCall)
                if self.LineEditDataParamsDict[key].isEnabled():
                    self.DysmalPyParams[key] = self.LineEditDataParamsDict[key].keyvalue()
                icount, irow, icol, rowSpan, colSpan = self.LineEditDataParamsDict[key].getPositionInQGridLayout(icount, ncolumn)
                self.LayoutForLineEditDataParams.addWidget(self.LineEditDataParamsDict[key], irow, icol, rowSpan, colSpan)
            else:
                icount += 1
        self.LayoutForTabPageA1 = QVBoxLayout()
        self.LayoutForTabPageA1.setContentsMargins(0, 0, 0, 0)
        self.LayoutForTabPageA1.setSpacing(0)
        self.LayoutForTabPageA1.addItem(self.LayoutForLineEditParamFile)
        self.LayoutForTabPageA1.addItem(self.LayoutForLineEditDataParams)
        self.LayoutForTabPageA1.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        self.TabPageA1.setLayout(self.LayoutForTabPageA1)
        # 
        self.LayoutForLineEditLensingParams = QGridLayout()
        self.LayoutForLineEditLensingParams.setHorizontalSpacing(30)
        self.LayoutForLineEditLensingParams.setVerticalSpacing(5)
        ncolumn = 2
        icount = 0 # self.LayoutForLineEditLensingParams.count() * ncolumn # the widgets above are full width
        for key in self.LineEditLensingParamsDict:
            if not (key.find('NULL')>=0):
                self.LineEditLensingParamsDict[key].ParamUpdateSignal.connect(self.onParamUpdateCall)
                if self.LineEditLensingParamsDict[key].isEnabled():
                    self.DysmalPyParams[key] = self.LineEditLensingParamsDict[key].keyvalue()
                icount, irow, icol, rowSpan, colSpan = self.LineEditLensingParamsDict[key].getPositionInQGridLayout(icount, ncolumn)
                self.LayoutForLineEditLensingParams.addWidget(self.LineEditLensingParamsDict[key], irow, icol, rowSpan, colSpan)
            else:
                icount += 1
        self.LayoutForTabPageA5 = QVBoxLayout()
        self.LayoutForTabPageA5.setContentsMargins(0, 0, 0, 0)
        self.LayoutForTabPageA5.setSpacing(0)
        self.LayoutForTabPageA5.addItem(self.LayoutForLineEditLensingParams)
        self.LayoutForTabPageA5.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        self.TabPageA5.setLayout(self.LayoutForTabPageA5)
        # 
        for idict in range(len(self.LineEditModelParamsDicts)):
            this_layout = QGridLayout()
            this_layout.setHorizontalSpacing(30)
            this_layout.setVerticalSpacing(5)
            ncolumn = 2
            icount = 0
            for key in self.LineEditModelParamsDicts[idict]:
                if not (key.find('NULL')>=0):
                    self.LineEditModelParamsDicts[idict][key].ParamUpdateSignal.connect(self.onParamUpdateCall)
                    if self.LineEditModelParamsDicts[idict][key].isEnabled():
                        self.DysmalPyParams[key] = self.LineEditModelParamsDicts[idict][key].keyvalue()
                    icount, irow, icol, rowSpan, colSpan = self.LineEditModelParamsDicts[idict][key].getPositionInQGridLayout(icount, ncolumn)
                    this_layout.addWidget(self.LineEditModelParamsDicts[idict][key], irow, icol, rowSpan, colSpan)
                else:
                    icount += 1
            this_tabpage_layout = QVBoxLayout()
            this_tabpage_layout.setContentsMargins(0, 0, 0, 0)
            this_tabpage_layout.setSpacing(0)
            this_tabpage_layout.addItem(this_layout)
            this_tabpage_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
            getattr(self, 'TabPageB%d'%(idict+1)).setLayout(this_tabpage_layout)
        # 
        #self.LayoutForLineEditModelParams2 = QGridLayout()
        #self.LayoutForLineEditModelParams2.setHorizontalSpacing(30)
        #self.LayoutForLineEditModelParams2.setVerticalSpacing(5)
        #ncolumn = 2
        #icount = 0
        #for key in self.LineEditModelParamsDict2:
        #    if not (key.find('NULL')>=0):
        #        self.LineEditModelParamsDict2[key].ParamUpdateSignal.connect(self.onParamUpdateCall)
        #        if self.LineEditModelParamsDict2[key].isEnabled():
        #            self.DysmalPyParams[key] = self.LineEditModelParamsDict2[key].keyvalue()
        #        icount, irow, icol, rowSpan, colSpan = self.LineEditModelParamsDict2[key].getPositionInQGridLayout(icount, ncolumn)
        #        self.LayoutForLineEditModelParams2.addWidget(self.LineEditModelParamsDict2[key], irow, icol, rowSpan, colSpan)
        #    else:
        #        icount += 1
        #self.LayoutForTabPageB2 = QVBoxLayout()
        #self.LayoutForTabPageB2.setContentsMargins(0, 0, 0, 0)
        #self.LayoutForTabPageB2.setSpacing(0)
        #self.LayoutForTabPageB2.addItem(self.LayoutForLineEditModelParams2)
        #self.LayoutForTabPageB2.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        #self.TabPageB2.setLayout(self.LayoutForTabPageB2)
        # 
        #self.LayoutForLineEditModelParams3 = QGridLayout()
        #self.LayoutForLineEditModelParams3.setHorizontalSpacing(30)
        #self.LayoutForLineEditModelParams3.setVerticalSpacing(5)
        #ncolumn = 2
        #icount = 0
        #for key in self.LineEditModelParamsDict3:
        #    if not (key.find('NULL')>=0):
        #        self.LineEditModelParamsDict3[key].ParamUpdateSignal.connect(self.onParamUpdateCall)
        #        if self.LineEditModelParamsDict3[key].isEnabled():
        #            self.DysmalPyParams[key] = self.LineEditModelParamsDict3[key].keyvalue()
        #        icount, irow, icol, rowSpan, colSpan = self.LineEditModelParamsDict3[key].getPositionInQGridLayout(icount, ncolumn)
        #        self.LayoutForLineEditModelParams3.addWidget(self.LineEditModelParamsDict3[key], irow, icol, rowSpan, colSpan)
        #    else:
        #        icount += 1
        #self.LayoutForTabPageB3 = QVBoxLayout()
        #self.LayoutForTabPageB3.setContentsMargins(0, 0, 0, 0)
        #self.LayoutForTabPageB3.setSpacing(0)
        #self.LayoutForTabPageB3.addItem(self.LayoutForLineEditModelParams3)
        #self.LayoutForTabPageB3.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        #self.TabPageB3.setLayout(self.LayoutForTabPageB3)
        # 
        #self.LayoutForLineEditModelParams4 = QGridLayout()
        #self.LayoutForLineEditModelParams4.setHorizontalSpacing(30)
        #self.LayoutForLineEditModelParams4.setVerticalSpacing(5)
        #ncolumn = 2
        #icount = 0
        #for key in self.LineEditModelParamsDict4:
        #    if not (key.find('NULL')>=0):
        #        self.LineEditModelParamsDict4[key].ParamUpdateSignal.connect(self.onParamUpdateCall)
        #        if self.LineEditModelParamsDict4[key].isEnabled():
        #            self.DysmalPyParams[key] = self.LineEditModelParamsDict4[key].keyvalue()
        #        icount, irow, icol, rowSpan, colSpan = self.LineEditModelParamsDict4[key].getPositionInQGridLayout(icount, ncolumn)
        #        self.LayoutForLineEditModelParams4.addWidget(self.LineEditModelParamsDict4[key], irow, icol, rowSpan, colSpan)
        #    else:
        #        icount += 1
        #self.LayoutForTabPageB4 = QVBoxLayout()
        #self.LayoutForTabPageB4.setContentsMargins(0, 0, 0, 0)
        #self.LayoutForTabPageB4.setSpacing(0)
        #self.LayoutForTabPageB4.addItem(self.LayoutForLineEditModelParams4)
        #self.LayoutForTabPageB4.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        #self.TabPageB4.setLayout(self.LayoutForTabPageB4)
        # 
        self.LayoutForTabWidgetHolderA = QVBoxLayout()
        self.LayoutForTabWidgetHolderA.addWidget(self.LabelForTabWidgetA)
        self.LayoutForTabWidgetHolderA.addWidget(self.TabWidgetA)
        self.TabWidgetHolderA.setLayout(self.LayoutForTabWidgetHolderA)
        self.LayoutForTabWidgetHolderB = QVBoxLayout()
        self.LayoutForTabWidgetHolderB.addWidget(self.LabelForTabWidgetB)
        self.LayoutForTabWidgetHolderB.addWidget(self.TabWidgetB)
        self.TabWidgetHolderB.setLayout(self.LayoutForTabWidgetHolderB)
        self.SplitterForTabWidgets = QSplitter(Qt.Vertical)
        self.SplitterForTabWidgets.addWidget(self.TabWidgetHolderA)
        self.SplitterForTabWidgets.addWidget(self.TabWidgetHolderB)
        self.SplitterForTabWidgets.setStyleSheet("""
            QSplitter::handle:vertical {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #eee, stop:1 #ccc);
                border: 1px solid #aaa;
                width: 20px;
                margin-top: 2px;
                margin-bottom: 2px;
                border-radius: 4px; 
            }""")
        self.LayoutForPanelLeft = QVBoxLayout()
        self.LayoutForPanelLeft.addWidget(self.SplitterForTabWidgets)
        self.LayoutForPanelLeft.setContentsMargins(0, 0, 0, 0)
        self.PanelLeft.setLayout(self.LayoutForPanelLeft)
        # 
        # Panel Middle widgets
        self.LabelForControlButtons = QLabel(self.tr("Control"))
        self.LabelComponents = QLabel(self.tr("Components"))
        # 
        self.CheckBoxModelParamsDict = OrderedDict()
        self.CheckBoxModelParamsDict['include_halo'] = QWidgetForParamInput(\
                     keyname=self.tr("include_halo"), 
                     keycomment=self.tr("Include the halo as a component in fitting?"), 
                     datatype=bool, 
                     checkbox=True, 
                     default='True')
        self.CheckBoxModelParamsDict['adiabatic_contract'] = QWidgetForParamInput(\
                     keyname=self.tr("adiabatic_contract"), 
                     keycomment=self.tr("Apply adiabatic contraction?"), 
                     datatype=bool, 
                     checkbox=True)
        self.CheckBoxModelParamsDict['pressure_support'] = QWidgetForParamInput(\
                     keyname=self.tr("pressure_support"), 
                     keycomment=self.tr("Apply assymmetric drift correction?"), 
                     datatype=bool, 
                     checkbox=True, 
                     default='True')
        self.CheckBoxModelParamsDict['noord_flat'] = QWidgetForParamInput(\
                     keyname=self.tr("noord_flat"), 
                     keycomment=self.tr("Apply Noordermeer 2008 flattenning for the velocity field of Sersic profile? (Need \"$SERSIC_PROFILE_MASS_VC_DATADIR/mass_VC_profile_sersic_n*.fits\")"), 
                     datatype=bool, 
                     checkbox=True, 
                     default='True')
        self.CheckBoxModelParamsDict['zheight_tied'] = QWidgetForParamInput(\
                     keyname=self.tr('zheight_tied'), 
                     keycomment=self.tr("Tie the zheight to the effective radius of the disk?"), 
                     datatype=bool, 
                     checkbox=True, 
                     default='True')
        self.CheckBoxModelParamsDict['fdm_tied'] = QWidgetForParamInput(\
                     keyname=self.tr('fdm_tied'), 
                     keycomment=self.tr("For NFW, fdm_tied=True determines fDM from Mvirial (+baryons)."), 
                     datatype=bool, 
                     checkbox=True, 
                     default='False')
        # 
        self.CheckBoxModelParamsDict['__gauss_extract__'] = QWidgetForParamInput(\
                     keyname=self.tr('__gauss_extract__'), 
                     keycomment=self.tr("Using Gaussian line profile fitting instead of moment to extract the velocity field. Inverse of moment_calc."), 
                     datatype=bool, 
                     checkbox=True, 
                     default='True')
        self.CheckBoxModelParamsDict['__gauss_extract__'].CheckBoxWidget.stateChanged.disconnect()
        self.CheckBoxModelParamsDict['__gauss_extract__'].CheckBoxWidget.stateChanged.connect(self.onGaussExtractCheckStateChangedCall)
        # 
        self.CheckBoxModelParamsDict['__oversampling__'] = QWidgetForParamInput(\
                     keyname=self.tr('__oversampling__'), 
                     keycomment=self.tr("Oversampling the model cube. See also the oversample input box."), 
                     datatype=bool, 
                     checkbox=True, 
                     default=str(self.LineEditModelParamsDictForFitting['oversample'].keyvalue()>1))
        self.CheckBoxModelParamsDict['__oversampling__'].CheckBoxWidget.stateChanged.disconnect()
        self.CheckBoxModelParamsDict['__oversampling__'].CheckBoxWidget.stateChanged.connect(self.onOverSamplingCheckStateChangedCall)
        # 
        self.CheckBoxModelParamsDict['__overwriting__'] = QWidgetForParamInput(\
                     keyname=self.tr('__overwriting__'), 
                     keycomment=self.tr("Overwriting the fitting. See also the overwrite check box."), 
                     datatype=bool, 
                     checkbox=True, 
                     default=self.LineEditModelParamsDictForFitting['overwrite'].keyvalue())
        self.CheckBoxModelParamsDict['__overwriting__'].CheckBoxWidget.stateChanged.disconnect()
        self.CheckBoxModelParamsDict['__overwriting__'].CheckBoxWidget.stateChanged.connect(self.onOverWritingCheckStateChangedCall)
        # 
        # 
        self.ButtonInitRandomParams = QPushButton(self.tr("Init Random Params"))
        self.ButtonInitRandomParams.setMinimumWidth(160)
        self.ButtonInitRandomParams.clicked.connect(self.onInitRandomParamsCall)
        self.ButtonOpenParamsFile = QPushButton(self.tr("Open Params File"))
        self.ButtonOpenParamsFile.setMinimumWidth(160)
        self.ButtonOpenParamsFile.clicked.connect(self.onOpenParamFileCall)
        self.ButtonFitData = QPushButton(self.tr("Fit Data"))
        self.ButtonFitData.setMinimumWidth(160)
        self.ButtonFitData.clicked.connect(self.onFitDataCall)
        self.ButtonLoadFittingResult = QPushButton(self.tr("Load Fitting Result"))
        self.ButtonLoadFittingResult.setMinimumWidth(160)
        self.ButtonLoadFittingResult.clicked.connect(self.onLoadFittingResultCall)
        self.ButtonGenerateModelCube = QPushButton(self.tr("Generate Model Cube"))
        self.ButtonGenerateModelCube.setMinimumWidth(160)
        self.ButtonGenerateModelCube.clicked.connect(self.onGenerateModelCubeCall)
        self.ButtonGenerateMomentMaps = QPushButton(self.tr("Generate Mom. Maps"))
        self.ButtonGenerateMomentMaps.setMinimumWidth(160)
        self.ButtonGenerateMomentMaps.clicked.connect(self.onGenerateMomentMapsCall)
        self.ButtonGenerateRotationCurves = QPushButton(self.tr("Generate Rot. Curves"))
        self.ButtonGenerateRotationCurves.setMinimumWidth(160)
        self.ButtonGenerateRotationCurves.clicked.connect(self.onGenerateRotationCurvesCall)
        self.ButtonHideSlit = QPushButton(self.tr("Hide Slit"))
        self.ButtonHideSlit.setMinimumWidth(160)
        self.ButtonHideSlit.clicked.connect(self.onHideSlitCall)
        self.ButtonSaveParams = QPushButton(self.tr("Save Params"))
        self.ButtonSaveParams.setMinimumWidth(160)
        self.ButtonSaveParams.clicked.connect(self.onSaveParamFileCall)
        self.ButtonSaveModelFiles = QPushButton(self.tr("Save Model Files"))
        self.ButtonSaveModelFiles.setToolTip(self.tr("Save model cube, flux map, velocity map and dispersion map as FITS files."))
        self.ButtonSaveModelFiles.setMinimumWidth(160)
        self.ButtonSaveModelFiles.clicked.connect(self.onSaveModelFilesCall)
        self.ButtonExit = QPushButton(self.tr("Exit"))
        self.ButtonExit.setMinimumWidth(160)
        self.ButtonExit.clicked.connect(self.onExitCall)
        # 
        # Panel Middle layout
        self.LayoutForCheckBoxModelParams = QGridLayout()
        self.LayoutForCheckBoxModelParams.setContentsMargins(0, 0, 0, 0)
        self.LayoutForCheckBoxModelParams.setHorizontalSpacing(30)
        self.LayoutForCheckBoxModelParams.setVerticalSpacing(5)
        self.LayoutForCheckBoxModelParams.setSizeConstraint(QLayout.SetMinimumSize)
        ncolumn = 1
        icount = 0
        for key in self.CheckBoxModelParamsDict:
            if not (key.find('NULL')>=0):
                self.CheckBoxModelParamsDict[key].setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
                self.CheckBoxModelParamsDict[key].ParamUpdateSignal.connect(self.onParamUpdateCall)
                if self.CheckBoxModelParamsDict[key].isEnabled():
                    self.DysmalPyParams[key] = self.CheckBoxModelParamsDict[key].keyvalue()
                icount, irow, icol, rowSpan, colSpan = self.CheckBoxModelParamsDict[key].getPositionInQGridLayout(icount, ncolumn)
                self.LayoutForCheckBoxModelParams.addWidget(self.CheckBoxModelParamsDict[key], irow, icol, rowSpan, colSpan)
            else:
                icount += 1
        #self.LayoutForCheckBoxModelParamsHolder = QHBoxLayout()
        #self.LayoutForCheckBoxModelParamsHolder.addItem(self.LayoutForCheckBoxModelParams)
        #self.LayoutForCheckBoxModelParamsHolder.addItem(QSpacerItem(2, 40, QSizePolicy.Expanding, QSizePolicy.Minimum))
        self.LayoutForPanelMiddle = QVBoxLayout()
        self.LayoutForPanelMiddle.setAlignment(Qt.AlignCenter)
        self.LayoutForPanelMiddle.setContentsMargins(0, 0, 0, 0)
        self.LayoutForPanelMiddle.addWidget(self.LabelForControlButtons)
        self.LayoutForPanelMiddle.addItem(self.LayoutForCheckBoxModelParams) # self.LayoutForCheckBoxModelParamsHolder
        self.LayoutForPanelMiddle.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        self.LayoutForPanelMiddle.addWidget(self.ButtonInitRandomParams)
        self.LayoutForPanelMiddle.addWidget(self.ButtonOpenParamsFile)
        self.LayoutForPanelMiddle.addWidget(self.ButtonFitData)
        self.LayoutForPanelMiddle.addWidget(self.ButtonLoadFittingResult)
        self.LayoutForPanelMiddle.addWidget(self.ButtonGenerateModelCube)
        self.LayoutForPanelMiddle.addWidget(self.ButtonGenerateMomentMaps)
        self.LayoutForPanelMiddle.addWidget(self.ButtonGenerateRotationCurves)
        self.LayoutForPanelMiddle.addWidget(self.ButtonHideSlit)
        self.LayoutForPanelMiddle.addWidget(self.ButtonSaveParams)
        self.LayoutForPanelMiddle.addWidget(self.ButtonSaveModelFiles)
        self.LayoutForPanelMiddle.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        self.LayoutForPanelMiddle.addWidget(self.ButtonExit)
        self.PanelMiddle.setLayout(self.LayoutForPanelMiddle)
        self.PanelMiddle.setStyleSheet("""
            QPushButton {
                max-width: 100px;
            }
            QCheckBox {
                max-width: 100px;
            }
            """)
        # 
        # Panel Right widgets
        self.LabelForSpecViewerA = QLabel(self.tr("Rotation Curves"))
        self.LabelForImageViewerA = QLabel(self.tr("Data Images"))
        self.LabelForImageViewerB = QLabel(self.tr("Model Images"))
        self.LabelForImageViewerC = QLabel(self.tr("Residual Images"))
        self.ImageViewerA1 = QFitsImageWidget(title='Flux', name='Data Flux Map', with_colorbar=True)
        self.ImageViewerA2 = QFitsImageWidget(title='Vel', name='Data Vel Map', with_colorbar=True)
        self.ImageViewerA3 = QFitsImageWidget(title='Disp', name='Data Disp Map', with_colorbar=True)
        self.ImageViewerB1 = QFitsImageWidget(title='Flux', name='Model Flux Map', with_colorbar=True)
        self.ImageViewerB2 = QFitsImageWidget(title='Vel', name='Model Vel Map', with_colorbar=True)
        self.ImageViewerB3 = QFitsImageWidget(title='Disp', name='Model Disp Map', with_colorbar=True)
        self.ImageViewerC1 = QFitsImageWidget(title='Flux', name='Residual Flux Map', with_colorbar=True)
        self.ImageViewerC2 = QFitsImageWidget(title='Vel', name='Residual Vel Map', with_colorbar=True)
        self.ImageViewerC3 = QFitsImageWidget(title='Disp', name='Residual Disp Map', with_colorbar=True)
        self.ImageViewerA1.PixelSelectedSignal.connect(self.selectPixelInImageViewers)
        self.ImageViewerA2.PixelSelectedSignal.connect(self.selectPixelInImageViewers)
        self.ImageViewerA3.PixelSelectedSignal.connect(self.selectPixelInImageViewers)
        self.ImageViewerB1.PixelSelectedSignal.connect(self.selectPixelInImageViewers)
        self.ImageViewerB2.PixelSelectedSignal.connect(self.selectPixelInImageViewers)
        self.ImageViewerB3.PixelSelectedSignal.connect(self.selectPixelInImageViewers)
        self.ImageViewerC1.PixelSelectedSignal.connect(self.selectPixelInImageViewers)
        self.ImageViewerC2.PixelSelectedSignal.connect(self.selectPixelInImageViewers)
        self.ImageViewerC3.PixelSelectedSignal.connect(self.selectPixelInImageViewers)
        self.SpecViewerA1 = QSpectrumWidget(title='Flux', name='1D Flux')
        self.SpecViewerA2 = QSpectrumWidget(title='Vel', name='1D Vel')
        self.SpecViewerA3 = QSpectrumWidget(title='Disp', name='1D Disp')
        self.SpecViewerA1.ChannelSelectedSignal.connect(self.selectChannelInSpecViewers)
        self.SpecViewerA2.ChannelSelectedSignal.connect(self.selectChannelInSpecViewers)
        self.SpecViewerA3.ChannelSelectedSignal.connect(self.selectChannelInSpecViewers)
        # 
        # Panel Right layout
        self.LayoutForSpecViewerA = QHBoxLayout()
        self.LayoutForSpecViewerA.setContentsMargins(0, 0, 0, 0)
        self.LayoutForSpecViewerA.addWidget(self.SpecViewerA1)
        self.LayoutForSpecViewerA.addWidget(self.SpecViewerA2)
        self.LayoutForSpecViewerA.addWidget(self.SpecViewerA3)
        self.LayoutForImageViewerA = QHBoxLayout()
        self.LayoutForImageViewerA.setContentsMargins(0, 0, 0, 0)
        self.LayoutForImageViewerA.addWidget(self.ImageViewerA1)
        self.LayoutForImageViewerA.addWidget(self.ImageViewerA2)
        self.LayoutForImageViewerA.addWidget(self.ImageViewerA3)
        self.LayoutForImageViewerB = QHBoxLayout()
        self.LayoutForImageViewerB.setContentsMargins(0, 0, 0, 0)
        self.LayoutForImageViewerB.addWidget(self.ImageViewerB1)
        self.LayoutForImageViewerB.addWidget(self.ImageViewerB2)
        self.LayoutForImageViewerB.addWidget(self.ImageViewerB3)
        self.LayoutForImageViewerC = QHBoxLayout()
        self.LayoutForImageViewerC.setContentsMargins(0, 0, 0, 0)
        self.LayoutForImageViewerC.addWidget(self.ImageViewerC1)
        self.LayoutForImageViewerC.addWidget(self.ImageViewerC2)
        self.LayoutForImageViewerC.addWidget(self.ImageViewerC3)
        self.LayoutForSpecViewerWithLabelA = QVBoxLayout()
        self.LayoutForImageViewerWithLabelA = QVBoxLayout()
        self.LayoutForImageViewerWithLabelB = QVBoxLayout()
        self.LayoutForImageViewerWithLabelC = QVBoxLayout()
        self.LayoutForSpecViewerWithLabelA.setContentsMargins(0, 0, 0, 0)
        self.LayoutForImageViewerWithLabelA.setContentsMargins(0, 0, 0, 0)
        self.LayoutForImageViewerWithLabelB.setContentsMargins(0, 0, 0, 0)
        self.LayoutForImageViewerWithLabelC.setContentsMargins(0, 0, 0, 0)
        self.LayoutForSpecViewerWithLabelA.addWidget(self.LabelForSpecViewerA)
        self.LayoutForSpecViewerWithLabelA.addItem(self.LayoutForSpecViewerA)
        self.LayoutForImageViewerWithLabelA.addWidget(self.LabelForImageViewerA)
        self.LayoutForImageViewerWithLabelA.addItem(self.LayoutForImageViewerA)
        self.LayoutForImageViewerWithLabelB.addWidget(self.LabelForImageViewerB)
        self.LayoutForImageViewerWithLabelB.addItem(self.LayoutForImageViewerB)
        self.LayoutForImageViewerWithLabelC.addWidget(self.LabelForImageViewerC)
        self.LayoutForImageViewerWithLabelC.addItem(self.LayoutForImageViewerC)
        self.LayoutForPanelRight = QVBoxLayout()
        self.LayoutForPanelRight.setContentsMargins(0, 0, 0, 0)
        self.LayoutForPanelRight.addItem(self.LayoutForSpecViewerWithLabelA)
        self.LayoutForPanelRight.addItem(self.LayoutForImageViewerWithLabelA)
        self.LayoutForPanelRight.addItem(self.LayoutForImageViewerWithLabelB)
        self.LayoutForPanelRight.addItem(self.LayoutForImageViewerWithLabelC)
        self.LayoutForPanelRight.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        self.PanelRight.setLayout(self.LayoutForPanelRight)
        # 
        #self.SplitterA = QSplitter(Qt.Horizontal)
        #self.SplitterB = QSplitter(Qt.Horizontal)
        #self.SplitterA.addWidget(self.PanelImageA)
        #self.SplitterB.addWidget(self.PanelImageB)
        #self.SplitterA.addWidget(self.PanelSpecA)
        #self.SplitterB.addWidget(self.PanelSpecB)
        #self.SplitterA.setStretchFactor(0, 1)
        #self.SplitterB.setStretchFactor(0, 1)
        #self.SplitterA.setStretchFactor(1, 3)
        #self.SplitterB.setStretchFactor(1, 3)
        #self.SplitterA.splitterMoved.connect(self.moveSplitterAB)
        #self.SplitterB.splitterMoved.connect(self.moveSplitterAB)
        # 
        self.SplitterForCentralWidget = QSplitter(Qt.Horizontal)
        self.SplitterForCentralWidget.addWidget(self.PanelLeft)
        self.SplitterForCentralWidget.addWidget(self.PanelMiddle)
        self.SplitterForCentralWidget.addWidget(self.PanelRight)
        self.SplitterForCentralWidget.setStretchFactor(0, 6)
        self.SplitterForCentralWidget.setStretchFactor(1, 0)
        self.SplitterForCentralWidget.setStretchFactor(2, 6)
        self.SplitterForCentralWidget.setSizes([9, 1, 9])
        self.SplitterForCentralWidget.setStyleSheet("""
            QSplitter::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #eee, stop:1 #ccc);
                border: 1px solid #aaa;
                width: 20px;
                margin-top: 2px;
                margin-bottom: 2px;
                border-radius: 4px; 
            }""")
        self.PanelLeft.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.PanelMiddle.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        self.PanelRight.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        # 
        # set main layout
        self.LayoutForCentralWidget = QHBoxLayout()
        self.LayoutForCentralWidget.addWidget(self.SplitterForCentralWidget)
        self.LayoutForCentralWidget.setContentsMargins(0, 0, 0, 0)
        self.LayoutForCentralWidget.setSpacing(0)
        self.CentralWidget.setLayout(self.LayoutForCentralWidget)
        self.setCentralWidget(self.CentralWidget)
        # 
        #self.adjustSize()
        # 
        # set status bar
        self.logMessage(self.tr('Initializing...'))
        # 
        # set menu actions
        exitAct = QAction(self.tr('&Exit'), self)
        exitAct.setShortcut('Ctrl+Q')
        exitAct.setStatusTip(self.tr('Exit application'))
        exitAct.setMenuRole(QAction.NoRole) # see https://stackoverflow.com/questions/39574105/missing-menubar-in-pyqt5
        exitAct.triggered.connect(self.onExitCall)
        openParamFileAct = QAction(self.tr('&Open Param File'), self)
        openParamFileAct.setShortcut('Ctrl+O')
        openParamFileAct.setStatusTip(self.tr('Open a *.params file'))
        openParamFileAct.triggered.connect(self.onOpenParamFileCall)
        # 
        # set menu
        menubar = QMenuBar(self)
        menubar.setNativeMenuBar(False)
        fileMenu = QMenu(self.tr('&File'), menubar)
        fileMenu.addAction(openParamFileAct)
        fileMenu.addAction(exitAct)
        menubar.addMenu(fileMenu)
        self.setMenuBar(menubar)
        # 
        # set fitting worker qobject
        self.DysmalPyFittingWorker.LogMessageUpdateSignal.connect(self.onLogMessageUpdateCall)
        self.LineEditDataParamsDict['outdir'].ParamUpdateSignal.connect(self.onOutDirParamUpdateCall)
        # 
        # set status bar
        self.logMessage(self.tr('Ready'))
        # 
        # set main window geometry size to 95% x 85% of the screen size
        if ScreenSize is not None:
            self.setGeometry(ScreenSize.width() * (1.0-0.95)/2.0, ScreenSize.height() * (1.0-0.85)/2.0, 
                             ScreenSize.width() * 0.95, ScreenSize.height() * 0.85) # make it centered on screen, 75% screen width and 85% screen height.
        #self.setWindowTitle('Icon')
        #self.setWindowIcon(QIcon('web.png'))
        self.SplitterForCentralWidget.setSizes([9, 1, 9])
        # 
        self.setButtonsEnabledDisabled()
        # 
        #self.ShortCutOpenParamFile = QShortcut(QKeySequence('Alt+O'), self)
        #self.ShortCutOpenParamFile.activated.connect(self.onOpenParamFileCall)
        # 
        self.setAcceptDrops(True)
        # 
        self.show()

    def moveSplitterAB(self, index, pos):
        self.SplitterA.blockSignals(True)
        self.SplitterB.blockSignals(True)
        self.SplitterA.moveSplitter(index, pos)
        self.SplitterB.moveSplitter(index, pos)
        self.SplitterA.blockSignals(False)
        self.SplitterB.blockSignals(False)

    def closeEvent(self, event):
        self.logger.debug('Closing... Bye.')
        super(QDysmalPyGUI, self).closeEvent(event)

    @pyqtSlot()
    def onExitCall(self):
        self.logger.debug('Closing... Bye.')
        qApp.quit()

    def logMessage(self, message):
        if self.statusBar():
            self.logger.info(message)
            self.statusBar().showMessage(message)
        
    @pyqtSlot(str)
    def onLogMessageUpdateCall(self, message):
        self.logMessage(message)
        
    @pyqtSlot(str, str, type, type)
    def onParamUpdateCall(self, keyname, keyvalue, datatype, listtype):
        if keyname.find('NULL')>=0 or keyname.startswith('__'):
            return
        if self.DysmalPyParams is None:
            self.DysmalPyParams = {} # create an empty DysmalPyParams
        
        if keyvalue == '':
            keyvalue = None
        elif keyvalue == 'None':
            keyvalue = None
        elif keyvalue == 'True':
            keyvalue = True
        elif keyvalue == 'False':
            keyvalue = False
        elif keyvalue == 'inf':
            keyvalue = np.inf
        else:
            if listtype is list:
                try:
                    keyvalue = eval(keyvalue)
                    keyvalue = np.array(keyvalue).astype(datatype)
                except:
                    self.logMessage(self.tr('Could not update DysmalPyParams key ')+str(keyname)+self.tr(' value ')+'"'+str(keyvalue)+'"'+self.tr(' as type ')+str(datatype)+self.tr('. Current value '+str(self.DysmalPyParams[keyname])))
                    return
            else:
                try:
                    keyvalue = np.array([keyvalue]).astype(datatype)[0]
                except:
                    self.logMessage(self.tr('Could not update DysmalPyParams key ')+str(keyname)+self.tr(' value ')+'"'+str(keyvalue)+'"'+self.tr(' as type ')+str(datatype)+self.tr('. Current value '+str(self.DysmalPyParams[keyname])))
                    return
        
        if keyname in ['fdata', 'fdata_flux', 'fdata_ferr', 'fdata_vel', 'fdata_verr', 'fdata_disp', 'fdata_derr', 'fdata_mask', 'fdata_cube'] and \
           keyname in self.DysmalPyParams and \
           keyvalue is None:
            del self.DysmalPyParams[keyname]
            self.logMessage(self.tr('Deleted DysmalPyParams key ')+str(keyname))
        elif keyname.startswith('lensing_') and \
             self.DysmalPyFittingWorker.LensingTransformer is not None:
            del self.DysmalPyFittingWorker.LensingTransformer
            self.DysmalPyFittingWorker.LensingTransformer = None
            self.logMessage(self.tr('Cleared DysmalPyFittingWorker.LensingTransformer due to changing key ')+str(keyname))
        else:
            self.DysmalPyParams[keyname] = keyvalue
            self.logMessage(self.tr('Updated DysmalPyParams key ')+str(keyname)+self.tr(' value ')+str(keyvalue)+self.tr(' type ')+str(datatype))
    
    @pyqtSlot(int)
    def onGaussExtractCheckStateChangedCall(self, state):
        self.logger.debug('QWidgetForParamInput::onGaussExtractCheckStateChangedCall()')
        for this_dict in self.LineEditModelParamsDicts:
            for this_key in this_dict:
                if this_key == 'moment_calc':
                    this_dict[this_key].setText(str(np.invert(state>0)))
    
    @pyqtSlot(int)
    def onOverSamplingCheckStateChangedCall(self, state):
        self.logger.debug('QWidgetForParamInput::onOverSamplingCheckStateChangedCall()')
        for this_dict in self.LineEditModelParamsDicts:
            for this_key in this_dict:
                if this_key == 'oversample':
                    if state>0:
                        this_dict[this_key].setEnabled(True)
                        this_dict[this_key].setText(str(this_dict[this_key].ParamValue))
                    else:
                        this_dict[this_key].ParamValue = this_dict[this_key].keyvalue()
                        this_dict[this_key].setText('1')
                        this_dict[this_key].setEnabled(False)
    
    @pyqtSlot(int)
    def onOverWritingCheckStateChangedCall(self, state):
        self.logger.debug('QWidgetForParamInput::onOverWritingCheckStateChangedCall()')
        for this_dict in self.LineEditModelParamsDicts:
            for this_key in this_dict:
                if this_key == 'overwrite':
                    if state>0:
                        this_dict[this_key].setText('True')
                    else:
                        this_dict[this_key].setText('False')
    
    @pyqtSlot(str, str, type, type)
    def onOutDirParamUpdateCall(self, keyname, keyvalue, datatype, listtype):
        self.logger.debug('QWidgetForParamInput::onOutDirParamUpdateCall()')
        if keyname == 'outdir':
            if keyvalue is not None and keyvalue != '':
                self.logger.debug('QWidgetForParamInput::onOutDirParamUpdateCall() self.DysmalPyFittingWorker.setDirectory("{}")'.format(keyvalue))
                self.DysmalPyFittingWorker.setDirectory(keyvalue)
    
    @pyqtSlot()
    def onOpenParamFileCall(self):
        self.logMessage(self.tr('Opening Dysmal param file...'))
        filepath = None
        filepath, selectedfilter = QFileDialog.getOpenFileName(self, self.tr('Open file'), self.DefaultDirectory, self.tr("Dysmal params file (*.params *.*)"))
        if filepath is None or filepath == '':
            self.logMessage(self.tr('No file selected.'))
            self.deselectParamFile()
        else:
            self.logMessage(self.tr('Selected Dysmal param file: '+str(filepath)))
            self.selectParamFile(filepath)

    def selectParamFile(self, filepath):
        if not os.path.isfile(filepath):
            self.logMessage(self.tr('Error! Dysmal param file does not exist: ')+filepath)
            return
        self.deselectParamFile()
        self.LineEditParamFile.setText(filepath)
        self.LineEditParamFile.setFocus()
        self.readParamFile(filepath)

    def deselectParamFile(self):
        if self.DysmalPyParams is not None:
            # deselect the current param file, pop up a window to ask if the user wants to save it 
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Question)
            msgBox.setText("Parameters changed! Do you want to save current parameters?")
            msgBox.setWindowTitle("Save param file?")
            clickedButton = msgBox.setStandardButtons(QMessageBox.Save | QMessageBox.Cancel)
            if clickedButton == QMessageBox.Save:
                self.saveParamFile()
        self.LineEditParamFile.setText('')
    
    def readParamFile(self, filepath):
        # 
        #if self.DysmalPyParamFile is not None:
        #    self.logMessage(self.tr('Selecting Dysmal param file: ')+str(filepath))
        #    if filepath == self.DysmalPyParamFile:
        #        self.logger.debug('The selected file is the same as current file. Do nothing.')
        #        return
        #
        self.logMessage(self.tr('Reading Dysmal param file: ')+str(filepath))
        self.setButtonsEnabledDisabled()
        errormessages = []
        params = utils_io.read_fitting_params(fname=filepath) #<TODO><DEBUG>#
        try:
            params = utils_io.read_fitting_params(fname=filepath)
        except Exception as err:
            params = None
            errormessages.append(str(err))
        if params is None:
            self.logMessage(self.tr('Error! Failed to load the Dysmal param file: ')+str(filepath))
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Critical)
            msgBox.setText(self.tr('Error! Failed to load the Dysmal param file: ')+'\n'+str(filepath)+'\n'+self.tr('Please check your Dysmal params file and open again.'))
            msgBox.setInformativeText(self.tr('Error messages:')+'\n'+'\n'.join(errormessages))
            msgBox.setWindowTitle("Failed to load the Dysmal param file")
            msgBox.setStandardButtons(QMessageBox.Ok)
            #msgBox.buttonClicked.connect(msgButtonClick)
            msgBox.exec()
            return
        # 
        self.DysmalPyParamFile = filepath
        self.DysmalPyParams = params
        for keyname in ['fdata', 'fdata_flux', 'fdata_ferr', 'fdata_vel', 'fdata_verr', 'fdata_disp', 'fdata_derr', 'fdata_mask', 'fdata_cube']:
            if keyname not in params:
                self.LineEditDataParamsDict[keyname].setText('', blocksignal=True)
                self.LineEditDataParamsDict[keyname].setEnabled(False)
        # 
        if 'datadir' not in self.DysmalPyParams:
            self.DysmalPyParams['datadir'] = None
        if self.DysmalPyParams['datadir'] is not None and self.DysmalPyParams['datadir'] != '':
            self.DysmalPyParams['datadir'] = utils_io.ensure_path_trailing_slash(self.DysmalPyParams['datadir'])
        if 'outdir' not in self.DysmalPyParams:
            self.DysmalPyParams['outdir'] = None
        if self.DysmalPyParams['outdir'] is not None and self.DysmalPyParams['outdir'] != '':
            self.DysmalPyParams['outdir'] = utils_io.ensure_path_trailing_slash(self.DysmalPyParams['outdir'])
        # 
        #self.logger.debug("self.DysmalPyParams['datadir']: "+str(self.DysmalPyParams['datadir']))
        #self.logger.debug("self.DysmalPyParams['outdir']: "+str(self.DysmalPyParams['outdir']))
        # 
        # fill in GUI LineEdit
        for key in self.LineEditDataParamsDict:
            if key in self.DysmalPyParams:
                self.LineEditDataParamsDict[key].setText(self.DysmalPyParams[key], blocksignal=True)
        for key in self.LineEditLensingParamsDict:
            if key in self.DysmalPyParams:
                self.LineEditLensingParamsDict[key].setText(self.DysmalPyParams[key], blocksignal=True)
        for i in range(len(self.LineEditModelParamsDicts)):
            for key in self.LineEditModelParamsDicts[i]:
                if key in self.DysmalPyParams:
                    self.LineEditModelParamsDicts[i][key].setText(self.DysmalPyParams[key], blocksignal=True)
        for key in self.CheckBoxModelParamsDict:
            if key in self.DysmalPyParams:
                self.CheckBoxModelParamsDict[key].setChecked(self.DysmalPyParams[key], blocksignal=True)
        # 
        # check data file data directory, see if we can proceed to load existing best fit result
        errormessages = []
        if self.DysmalPyParams['datadir'] is None or self.DysmalPyParams['datadir'] == '':
            errormessages.append('datadir is None')
        elif not os.path.isdir(self.DysmalPyParams['datadir']):
            errormessages.append('datadir is not found on disk: '+self.DysmalPyParams['datadir'])
        if self.DysmalPyParams['outdir'] is None or self.DysmalPyParams['outdir'] == '':
            errormessages.append('outdir is None')
        hasdata = False
        for keyname in ['fdata', 'fdata_flux', 'fdata_ferr', 'fdata_vel', 'fdata_verr', 'fdata_disp', 'fdata_derr', 'fdata_mask', 'fdata_cube']:
            if keyname in self.DysmalPyParams:
                hasdata = True
                if self.DysmalPyParams['datadir'] is None or self.DysmalPyParams['datadir'] == '': 
                    filepath_check = self.DysmalPyParams[keyname]
                else:
                    filepath_check = os.path.join(self.DysmalPyParams['datadir'], self.DysmalPyParams[keyname])
                if not os.path.isfile(filepath_check):
                    errormessages.append(keyname+' is not found on disk: '+self.DysmalPyParams[keyname])
        if not hasdata:
            errormessages.append('No fdata* key set!')
        # 
        if len(errormessages) > 0:
            self.logMessage(self.tr('Found errors in the Dysmal param file: ')+str(filepath))
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Critical)
            msgBox.setText(self.tr('Found errors in the Dysmal param file: ')+'\n'+str(filepath)+'\n'+self.tr('Please check your Dysmal params file and correct the errors.'))
            msgBox.setInformativeText(self.tr('Error messages:')+'\n'+'\n'.join(errormessages))
            msgBox.setWindowTitle("Found errors in the Dysmal param file")
            msgBox.setStandardButtons(QMessageBox.Ok)
            #msgBox.buttonClicked.connect(msgButtonClick)
            msgBox.exec()
            self.setButtonsEnabledDisabled()
            return
        # 
        self.logMessage(self.tr('Successfully loaded Dysmal param file: ')+str(self.DysmalPyParamFile))
        # 
        # check if there are already best fit file
        output_pickle_file = os.path.join(self.DysmalPyParams['outdir'], self.DysmalPyParams['galID']+'_'+self.DysmalPyParams['fit_method']+'_results.pickle')
        if os.path.exists(output_pickle_file):
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Question)
            msgBox.setText("Would you like to load previous bestfits?")
            msgBox.setWindowTitle("Load previous bestfits?")
            msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            clickedButton = msgBox.exec()
            if clickedButton == QMessageBox.Yes:
                self.fitDataAsync(dofit=False, clearFitResults=False)
        self.setButtonsEnabledDisabled()
    
    def writeOneLineToParamFile(self, fp, keyname, keyvalue, keycomment, datatype, listtype, ensure_path_trailing_slash=False):
        if keyname.find('NULL')>=0 or keyname.startswith('__'):
            return
        if keycomment is None:
            keycomment = ''
        if ensure_path_trailing_slash:
            if not keyvalue.endswith(os.sep):
                keyvalue += os.sep
        if keyvalue == '' and datatype in [int, float] and listtype is not list:
            keyvalue = 'None'
        if listtype is list:
            #keyvalue = re.sub(r'[, ]+', r' ', re.sub(r'^\[(.*)\]$', r'\1', str(keyvalue).strip()))
            keyvalue = ' '.join([str(t) for t in eval(keyvalue)])
        if keycomment.strip() != '':
            fp.write('{:<26} {:<62} {} {} \n'.format(keyname+',', keyvalue, '#', keycomment))
        else:
            fp.write('{:<26} {:<62} \n'.format(keyname+',', keyvalue))
    
    def saveParamFile(self, filepath=None):
        if self.DysmalPyParams is None:
            self.logMessage(self.tr('Error! DysmalPyParams not loaded!'))
            return
        if filepath is None or filepath == '':
            filepath, _ = QFileDialog.getSaveFileName(self, self.tr('Saving Dysmal params file'), self.DefaultDirectory, self.tr('Dysmal Params File (*.params *.*)')) # QFileDialog::DontConfirmOverwrite
            if filepath is None or filepath == '':
                self.logMessage(self.tr('No file selected.'))
                return
        if not filepath.endswith('.params'):
            filepath += '.params'
        if os.path.isfile(filepath):
            self.logger.debug('Found existing param file %r. Backing up as %r.'%(filepath, filepath+'.backup'))
            shutil.move(filepath, filepath+'.backup')
        with open(filepath, 'w') as fp:
            #self.writeOneLineToParamFile(fp, 'datadir', self.LineEditDataDirDisplayed.text(), self.LineEditDataDirDisplayed.toolTip())
            #keyname = 'fdata'
            #if 'fdata' in self.DysmalPyParams:
            #    keyname = 'fdata'
            #elif 'fdata_vel' in self.DysmalPyParams:
            #    keyname = 'fdata_vel'
            #elif 'fdata_cube' in self.DysmalPyParams:
            #    keyname = 'fdata_cube'
            #self.writeOneLineToParamFile(fp, keyname, self.LineEditDataFileDisplayed.text(), self.LineEditDataFileDisplayed.toolTip())
            #self.writeOneLineToParamFile(fp, 'outdir', self.LineEditOutDirDisplayed.text(), self.LineEditOutDirDisplayed.toolTip())
            for key in self.LineEditDataParamsDict:
                if self.LineEditDataParamsDict[key].isEnabled():
                    self.writeOneLineToParamFile(fp, key, \
                                                 self.LineEditDataParamsDict[key].text(),\
                                                 self.LineEditDataParamsDict[key].toolTip(),\
                                                 self.LineEditDataParamsDict[key].dataType(),\
                                                 self.LineEditDataParamsDict[key].listType()\
                                                 )
            for key in self.LineEditLensingParamsDict:
                if self.LineEditLensingParamsDict[key].isEnabled():
                    self.writeOneLineToParamFile(fp, key, \
                                                 self.LineEditLensingParamsDict[key].text(),\
                                                 self.LineEditLensingParamsDict[key].toolTip(),\
                                                 self.LineEditLensingParamsDict[key].dataType(),\
                                                 self.LineEditLensingParamsDict[key].listType()\
                                                 )
            for i in range(len(self.LineEditModelParamsDicts)): 
                for key in self.LineEditModelParamsDicts[i]:
                    if self.LineEditModelParamsDicts[i][key].isEnabled():
                        self.writeOneLineToParamFile(fp, key, \
                                                     self.LineEditModelParamsDicts[i][key].text(),\
                                                     self.LineEditModelParamsDicts[i][key].toolTip(),\
                                                     self.LineEditModelParamsDicts[i][key].dataType(),\
                                                     self.LineEditModelParamsDicts[i][key].listType()\
                                                     )
            for key in self.CheckBoxModelParamsDict:
                if self.CheckBoxModelParamsDict[key].isEnabled():
                    self.writeOneLineToParamFile(fp, key, \
                                                 self.CheckBoxModelParamsDict[key].text(),\
                                                 self.CheckBoxModelParamsDict[key].toolTip(),\
                                                 self.CheckBoxModelParamsDict[key].dataType(),\
                                                 self.CheckBoxModelParamsDict[key].listType()\
                                                 )
        self.logMessage(self.tr('Saved parameters to param file: ')+str(filepath))
    
    @pyqtSlot()
    def onSaveParamFileCall(self, filepath=None):
        self.saveParamFile(filepath=None)
    
    @pyqtSlot()
    def onSaveModelCubeCall(self, filepath=None):
        if self.DysmalPyFittingWorker.model_cube is None:
            self.logMessage(self.tr('Error! No valid model cube! Could not save to FITS file.'))
            return
        self.saveFitsFile(data=self.DysmalPyFittingWorker.model_cube)
    
    @pyqtSlot()
    def onSaveModelFluxCall(self):
        if self.DysmalPyFittingWorker.model_flux_map is None:
            self.logMessage(self.tr('Error! No valid model flux map! Could not save to FITS file.'))
            return
        self.saveFitsFile(data=self.DysmalPyFittingWorker.model_flux_map)
    
    @pyqtSlot()
    def onSaveModelVelCall(self):
        if self.DysmalPyFittingWorker.model_vel_map is None:
            self.logMessage(self.tr('Error! No valid model vel map! Could not save to FITS file.'))
            return
        self.saveFitsFile(data=self.DysmalPyFittingWorker.model_vel_map)
    
    @pyqtSlot()
    def onSaveModelVdispCall(self):
        if self.DysmalPyFittingWorker.model_disp_map is None:
            self.logMessage(self.tr('Error! No valid model disp map! Could not save to FITS file.'))
            return
        self.saveFitsFile(data=self.DysmalPyFittingWorker.model_disp_map)
    
    @pyqtSlot()
    def onSaveModelFilesCall(self, filepath=None):
        errormessages = []
        if self.DysmalPyFittingWorker.model_cube is None:
            errormessages.append(self.tr('Error! No valid model cube! Could not save the FITS file.'))
        if self.DysmalPyFittingWorker.model_flux_map is None:
            errormessages.append(self.tr('Error! No valid model flux map! Could not save the FITS file.'))
        if self.DysmalPyFittingWorker.model_vel_map is None:
            errormessages.append(self.tr('Error! No valid model vel map! Could not save the FITS file.'))
        if self.DysmalPyFittingWorker.model_disp_map is None:
            errormessages.append(self.tr('Error! No valid model disp map! Could not save the FITS file.'))
        if len(errormessages) != 0:
            return
        # 
        defaultpath = self.DefaultDirectory
        if self.DysmalPyParams['outdir'] is not None:
            defaultpath = self.DysmalPyParams['outdir']
        filepath, _ = QFileDialog.getSaveFileName(self, self.tr('Saving model files'), defaultpath, self.tr('Base name for output (*.*)'))
        if filepath is None or filepath == '':
            self.logMessage(self.tr('No file selected. No file saved.'))
            return
        filepath = re.sub(r'^(.*)\.fits$', r'\1', filepath)
        self.saveFitsFile(data=self.DysmalPyFittingWorker.model_cube, filepath=filepath+'_model_cube.fits')
        self.saveFitsFile(data=self.DysmalPyFittingWorker.model_flux_map, filepath=filepath+'_model_flux_map.fits')
        self.saveFitsFile(data=self.DysmalPyFittingWorker.model_vel_map, filepath=filepath+'_model_vel_map.fits')
        self.saveFitsFile(data=self.DysmalPyFittingWorker.model_disp_map, filepath=filepath+'_model_disp_map.fits')
        self.saveFitsFile(data=self.DysmalPyFittingWorker.residual_flux_map, filepath=filepath+'_residual_flux_map.fits')
        self.saveFitsFile(data=self.DysmalPyFittingWorker.residual_vel_map, filepath=filepath+'_residual_vel_map.fits')
        self.saveFitsFile(data=self.DysmalPyFittingWorker.residual_disp_map, filepath=filepath+'_residual_disp_map.fits')
        self.saveFitsFile(data=self.DysmalPyFittingWorker.data_flux_map, filepath=filepath+'_data_flux_map.fits')
        self.saveFitsFile(data=self.DysmalPyFittingWorker.data_vel_map, filepath=filepath+'_data_vel_map.fits')
        self.saveFitsFile(data=self.DysmalPyFittingWorker.data_disp_map, filepath=filepath+'_data_disp_map.fits')
        self.saveFitsFile(data=self.DysmalPyFittingWorker.data_mask_map.astype(int), filepath=filepath+'_data_mask_map.fits')
        self.saveJsonFile(data=self.DysmalPyFittingWorker.model_flux_curve, filepath=filepath+'_model_flux_curve.json')
        self.saveJsonFile(data=self.DysmalPyFittingWorker.model_vel_curve, filepath=filepath+'_model_vel_curve.json')
        self.saveJsonFile(data=self.DysmalPyFittingWorker.model_disp_curve, filepath=filepath+'_model_disp_curve.json')
        self.SpecViewerA1.fig.savefig(filepath+'_model_flux_curve.pdf', dpi=300, transparent=True)
        self.SpecViewerA2.fig.savefig(filepath+'_model_vel_curve.pdf', dpi=300, transparent=True)
        self.SpecViewerA3.fig.savefig(filepath+'_model_disp_curve.pdf', dpi=300, transparent=True)
        self.ImageViewerA1.fig.savefig(filepath+'_data_flux_map.pdf', dpi=300, transparent=True)
        self.ImageViewerA2.fig.savefig(filepath+'_data_vel_map.pdf', dpi=300, transparent=True)
        self.ImageViewerA3.fig.savefig(filepath+'_data_disp_map.pdf', dpi=300, transparent=True)
        self.ImageViewerB1.fig.savefig(filepath+'_model_flux_map.pdf', dpi=300, transparent=True)
        self.ImageViewerB2.fig.savefig(filepath+'_model_vel_map.pdf', dpi=300, transparent=True)
        self.ImageViewerB3.fig.savefig(filepath+'_model_disp_map.pdf', dpi=300, transparent=True)
        self.ImageViewerC1.fig.savefig(filepath+'_residual_flux_map.pdf', dpi=300, transparent=True)
        self.ImageViewerC2.fig.savefig(filepath+'_residual_vel_map.pdf', dpi=300, transparent=True)
        self.ImageViewerC3.fig.savefig(filepath+'_residual_disp_map.pdf', dpi=300, transparent=True)
        if self.DysmalPyFittingWorker.LensingTransformer is not None:
            if self.DysmalPyFittingWorker.LensingTransformer.image_plane_data_cube is not None:
                self.saveFitsFile(data=self.DysmalPyFittingWorker.LensingTransformer.image_plane_data_cube, 
                                  header=self.DysmalPyFittingWorker.LensingTransformer.image_plane_data_info, 
                                  filepath=filepath+'_lensing_image_plane_data_cube.fits')
            if self.DysmalPyFittingWorker.LensingTransformer.source_plane_data_cube is not None:
                self.saveFitsFile(data=self.DysmalPyFittingWorker.LensingTransformer.source_plane_data_cube, 
                                  header=self.DysmalPyFittingWorker.LensingTransformer.source_plane_data_info, 
                                  filepath=filepath+'_lensing_source_plane_data_cube.fits')
        self.saveParamFile(filepath=filepath+'.params')
    
    def saveFitsFile(self, data, header=None, filepath=None):
        if data is None:
            return
        if filepath is None or filepath == '':
            filepath, _ = QFileDialog.getSaveFileName(self, self.tr('Saving as FITS file'), self.DefaultDirectory, self.tr('FITS file (*.fits)'))
            if filepath is None or filepath == '':
                self.logMessage(self.tr('No file selected. No file saved.'))
                return
        if os.path.isfile(filepath):
            self.logger.debug('Found existing FITS file %r. Backing up as %r.'%(filepath, filepath+'.backup'))
            shutil.move(filepath, filepath+'.backup')
        if header is not None:
            if isinstance(header, (dict, OrderedDict)):
                header2 = copy.deepcopy(header)
                header = fits.Header()
                for key in header2:
                    header[key] = header2[key]
            hdu = fits.PrimaryHDU(data=data, header=header)
        else:
            hdu = fits.PrimaryHDU(data=data)
        hdu.writeto(filepath)
        self.logMessage(self.tr('Saved to FITS file: ')+str(filepath))
    
    def saveJsonFile(self, data, filepath=None, sort_keys=True, indent=4):
        if data is None:
            return
        if filepath is None or filepath == '':
            filepath, _ = QFileDialog.getSaveFileName(self, self.tr('Saving as JSON file'), self.DefaultDirectory, self.tr('JSON file (*.json)'))
            if filepath is None or filepath == '':
                self.logMessage(self.tr('No file selected. No file saved.'))
                return
        if os.path.isfile(filepath):
            self.logger.debug('Found existing JSON file %r. Backing up as %r.'%(filepath, filepath+'.backup'))
            shutil.move(filepath, filepath+'.backup')
        with open(filepath, 'w') as fp:
            json.dump(data, fp, sort_keys=sort_keys, indent=indent, cls=NumpyEncoder) # write spwmap to json file
        self.logMessage(self.tr('Saved to JSON file: ')+str(filepath))
    
    def readJsonFile(self, filepath=None):
        if filepath is None or filepath == '':
            filepath, _ = QFileDialog.getOpenFileName(self, self.tr('Opening a JSON file'), self.DefaultDirectory, self.tr('JSON file (*.json)'))
            if filepath is None or filepath == '':
                self.logMessage(self.tr('No file selected.'))
                return None
        self.logMessage(self.tr('Reading JSON file: ')+str(filepath))
        with open(filepath, 'r') as fp:
            data = json.load(fp)
            data = ast.literal_eval(json.dumps(data)) # Removing uni-code chars
        return data
    
    @pyqtSlot()
    def onGenerateModelCubeCall(self):
        self.logger.debug('onGenerateModelCubeCall()')
        if self.checkDysmalPyParams():
            self.generateModelCubeAsync(clearFitResults=True)
    
    @pyqtSlot()
    def onGenerateMomentMapsCall(self):
        self.logger.debug('onGenerateMomentMapsCall()')
        if self.checkDysmalPyParams():
            self.generateMomentMapsAsync()
    
    @pyqtSlot()
    def onGenerateRotationCurvesCall(self):
        self.logger.debug('onGenerateRotationCurvesCall()')
        if self.checkDysmalPyParams():
            self.generateRotationCurvesAsync()
    
    @pyqtSlot()
    def onHideSlitCall(self):
        self.logger.debug('onHideSlitCall()')
        if self.checkDysmalPyParams():
            # hide/show the slit in the image viewer
            if self.DysmalPyFittingWorker.data_flux_map is not None:
                if self.ImageViewerA1.slit.get_visible() == True:
                    self.ImageViewerA1.hideSlit()
                    self.ButtonHideSlit.setText(self.tr('Show Slit'))
                else:
                    self.ImageViewerA1.showSlit()
                    self.ButtonHideSlit.setText(self.tr('Hide Slit'))
            if self.DysmalPyFittingWorker.data_vel_map is not None:
                if self.ImageViewerA2.slit.get_visible() == True:
                    self.ImageViewerA2.hideSlit()
                    self.ButtonHideSlit.setText(self.tr('Show Slit'))
                else:
                    self.ImageViewerA2.showSlit()
                    self.ButtonHideSlit.setText(self.tr('Hide Slit'))
            if self.DysmalPyFittingWorker.data_disp_map is not None:
                if self.ImageViewerA3.slit.get_visible() == True:
                    self.ImageViewerA3.hideSlit()
                    self.ButtonHideSlit.setText(self.tr('Show Slit'))
                else:
                    self.ImageViewerA3.showSlit()
                    self.ButtonHideSlit.setText(self.tr('Hide Slit'))
            # 
            if self.DysmalPyFittingWorker.model_flux_map is not None:
                if self.ImageViewerB1.slit.get_visible() == True:
                    self.ImageViewerB1.hideSlit()
                    self.ButtonHideSlit.setText(self.tr('Show Slit'))
                else:
                    self.ImageViewerB1.showSlit()
                    self.ButtonHideSlit.setText(self.tr('Hide Slit'))
            if self.DysmalPyFittingWorker.model_vel_map is not None:
                if self.ImageViewerB2.slit.get_visible() == True:
                    self.ImageViewerB2.hideSlit()
                    self.ButtonHideSlit.setText(self.tr('Show Slit'))
                else:
                    self.ImageViewerB2.showSlit()
                    self.ButtonHideSlit.setText(self.tr('Hide Slit'))
            if self.DysmalPyFittingWorker.model_disp_map is not None:
                if self.ImageViewerB3.slit.get_visible() == True:
                    self.ImageViewerB3.hideSlit()
                    self.ButtonHideSlit.setText(self.tr('Show Slit'))
                else:
                    self.ImageViewerB3.showSlit()
                    self.ButtonHideSlit.setText(self.tr('Hide Slit'))
            # 
            #if self.ButtonHideSlit.text() == self.tr('Show Slit'):
            #    self.ButtonHideSlit.setText(self.tr('Hide Slit'))
            #else:
            #    self.ButtonHideSlit.setText(self.tr('Show Slit'))
    
    def plotDataMomentMaps(self):
        self.logMessage(self.tr('Computing and displaying data cube moment maps'))
        if self.DysmalPyFittingWorker.data_flux_map is not None:
            self.ImageViewerA1.showImage(self.DysmalPyFittingWorker.data_flux_map, with_colorbar=True)
        if self.DysmalPyFittingWorker.data_vel_map is not None:
            self.ImageViewerA2.showImage(self.DysmalPyFittingWorker.data_vel_map, with_colorbar=True, cmap='RdYlBu_r')
        if self.DysmalPyFittingWorker.data_disp_map is not None:
            self.ImageViewerA3.showImage(self.DysmalPyFittingWorker.data_disp_map, with_colorbar=True, cmap='plasma')
        self.logMessage(self.tr('Successfully computed and displayed data cube moment maps'))

    def plotModelMomentMaps(self):
        self.logMessage(self.tr('Computing and displaying model cube moment maps'))
        if self.DysmalPyFittingWorker.model_flux_map is not None:
            self.ImageViewerB1.showImage(self.DysmalPyFittingWorker.model_flux_map, with_colorbar=True)
        if self.DysmalPyFittingWorker.model_vel_map is not None:
            self.ImageViewerB2.showImage(self.DysmalPyFittingWorker.model_vel_map, with_colorbar=True, cmap='RdYlBu_r')
        if self.DysmalPyFittingWorker.model_disp_map is not None:
            self.ImageViewerB3.showImage(self.DysmalPyFittingWorker.model_disp_map, with_colorbar=True, cmap='plasma')
        if self.DysmalPyFittingWorker.residual_flux_map is not None: 
            self.ImageViewerC1.showImage(self.DysmalPyFittingWorker.residual_flux_map, with_colorbar=True)
        if self.DysmalPyFittingWorker.residual_vel_map is not None: 
            self.ImageViewerC2.showImage(self.DysmalPyFittingWorker.residual_vel_map, with_colorbar=True, cmap='RdYlBu_r')
        if self.DysmalPyFittingWorker.residual_disp_map is not None: 
            self.ImageViewerC3.showImage(self.DysmalPyFittingWorker.residual_disp_map, with_colorbar=True, cmap='RdYlBu_r')
        self.logMessage(self.tr('Successfully computed and displayed model cube moment maps'))
    
    def plotModelRotationCurves(self):
        self.logMessage(self.tr('Computing and displaying model rotation curve'))
        # 
        clear_plot = True
        has_plot = False
        if self.DysmalPyFittingWorker.data_flux_curve is not None:
            self.logger.debug('plotModelRotationCurves() self.SpecViewerA1.plotSpectrum self.DysmalPyFittingWorker.data_flux_curve')
            self.SpecViewerA1.plotSpectrum(**(self.DysmalPyFittingWorker.data_flux_curve), clear_plot=clear_plot, label='data', color='k', zorder=90)
            clear_plot = False
            has_plot = True
        if self.DysmalPyFittingWorker.model_flux_curve is not None:
            self.logger.debug('plotModelRotationCurves() self.SpecViewerA1.plotSpectrum self.DysmalPyFittingWorker.model_flux_curve')
            self.SpecViewerA1.plotSpectrum(**(self.DysmalPyFittingWorker.model_flux_curve), clear_plot=clear_plot, label='model', color='red', zorder=99)
            clear_plot = False
            has_plot = True
        if has_plot:
            #self.SpecViewerA1.axes.set_title('Flux')
            self.SpecViewerA1.axes.legend(loc='upper left')
            self.SpecViewerA1.draw()
            # also show a slit as a line
            if self.DysmalPyFittingWorker.data_flux_map is not None:
                self.ImageViewerA1.showSlit(self.getSlitShapeInPixel())
                self.ButtonHideSlit.setText(self.tr('Hide Slit'))
            if self.DysmalPyFittingWorker.model_flux_map is not None:
                self.ImageViewerB1.showSlit(self.getSlitShapeInPixel())
                self.ButtonHideSlit.setText(self.tr('Hide Slit'))
        # 
        clear_plot = True
        has_plot = False
        if self.DysmalPyFittingWorker.data_vel_curve is not None:
            self.logger.debug('plotModelRotationCurves() self.SpecViewerA2.plotSpectrum self.DysmalPyFittingWorker.data_vel_curve')
            self.SpecViewerA2.plotSpectrum(**(self.DysmalPyFittingWorker.data_vel_curve), clear_plot=clear_plot, label='data', color='k', zorder=90)
            clear_plot = False
            has_plot = True
        if self.DysmalPyFittingWorker.model_vel_curve is not None:
            self.logger.debug('plotModelRotationCurves() self.SpecViewerA2.plotSpectrum self.DysmalPyFittingWorker.model_vel_curve')
            self.SpecViewerA2.plotSpectrum(**(self.DysmalPyFittingWorker.model_vel_curve), clear_plot=clear_plot, label='model', color='red', zorder=99)
            clear_plot = False
            has_plot = True
        if has_plot:
            #self.SpecViewerA2.axes.set_title('Vel')
            self.SpecViewerA2.axes.legend(loc='upper left')
            self.SpecViewerA2.draw()
            # also show a slit as a line
            if self.DysmalPyFittingWorker.data_vel_map is not None:
                self.ImageViewerA2.showSlit(self.getSlitShapeInPixel())
                self.ButtonHideSlit.setText(self.tr('Hide Slit'))
            if self.DysmalPyFittingWorker.model_vel_map is not None:
                self.ImageViewerB2.showSlit(self.getSlitShapeInPixel())
                self.ButtonHideSlit.setText(self.tr('Hide Slit'))
        # 
        clear_plot = True
        has_plot = False
        if self.DysmalPyFittingWorker.data_disp_curve is not None:
            self.logger.debug('plotModelRotationCurves() self.SpecViewerA3.plotSpectrum self.DysmalPyFittingWorker.data_disp_curve')
            self.SpecViewerA3.plotSpectrum(**(self.DysmalPyFittingWorker.data_disp_curve), clear_plot=clear_plot, label='data', color='k', zorder=90)
            clear_plot = False
            has_plot = True
        if self.DysmalPyFittingWorker.model_disp_curve is not None:
            self.logger.debug('plotModelRotationCurves() self.SpecViewerA3.plotSpectrum self.DysmalPyFittingWorker.model_disp_curve')
            self.SpecViewerA3.plotSpectrum(**(self.DysmalPyFittingWorker.model_disp_curve), clear_plot=clear_plot, label='model', color='red', zorder=99)
            clear_plot = False
            has_plot = True
        if has_plot:
            #self.SpecViewerA3.axes.set_title('Vdisp')
            self.SpecViewerA3.axes.legend(loc='upper left')
            self.SpecViewerA3.draw()
            # also show a slit as a line
            if self.DysmalPyFittingWorker.data_disp_map is not None:
                self.ImageViewerA3.showSlit(self.getSlitShapeInPixel())
                self.ButtonHideSlit.setText(self.tr('Hide Slit'))
            if self.DysmalPyFittingWorker.model_disp_map is not None:
                self.ImageViewerB3.showSlit(self.getSlitShapeInPixel())
                self.ButtonHideSlit.setText(self.tr('Hide Slit'))
        # 
        self.logMessage(self.tr('Successfully computed and displayed model rotation curve'))
    
    @pyqtSlot(PixCoord)
    def selectPixelInImageViewers(self, pixel:PixCoord = None):
        if pixel is not None:
            # 
            if self.DysmalPyFittingWorker.data_flux_map is not None:
                self.ImageViewerA1.blockSignals(True)
            if self.DysmalPyFittingWorker.data_vel_map is not None:
                self.ImageViewerA2.blockSignals(True)
            if self.DysmalPyFittingWorker.data_disp_map is not None:
                self.ImageViewerA3.blockSignals(True)
            if self.DysmalPyFittingWorker.model_flux_map is not None:
                self.ImageViewerB1.blockSignals(True)
            if self.DysmalPyFittingWorker.model_vel_map is not None:
                self.ImageViewerB2.blockSignals(True)
            if self.DysmalPyFittingWorker.model_disp_map is not None:
                self.ImageViewerB3.blockSignals(True)
            if self.DysmalPyFittingWorker.residual_flux_map is not None:
                self.ImageViewerC1.blockSignals(True)
            if self.DysmalPyFittingWorker.residual_vel_map is not None:
                self.ImageViewerC2.blockSignals(True)
            if self.DysmalPyFittingWorker.residual_disp_map is not None:
                self.ImageViewerC3.blockSignals(True)
            # 
            if self.DysmalPyFittingWorker.data_flux_map is not None:
                self.ImageViewerA1.setSelectedPixel(pixel.x, pixel.y)
            if self.DysmalPyFittingWorker.data_vel_map is not None:
                self.ImageViewerA2.setSelectedPixel(pixel.x, pixel.y)
            if self.DysmalPyFittingWorker.data_disp_map is not None:
                self.ImageViewerA3.setSelectedPixel(pixel.x, pixel.y)
            if self.DysmalPyFittingWorker.model_flux_map is not None:
                self.ImageViewerB1.setSelectedPixel(pixel.x, pixel.y)
            if self.DysmalPyFittingWorker.model_vel_map is not None:
                self.ImageViewerB2.setSelectedPixel(pixel.x, pixel.y)
            if self.DysmalPyFittingWorker.model_disp_map is not None:
                self.ImageViewerB3.setSelectedPixel(pixel.x, pixel.y)
            if self.DysmalPyFittingWorker.residual_flux_map is not None:
                self.ImageViewerC1.setSelectedPixel(pixel.x, pixel.y)
            if self.DysmalPyFittingWorker.residual_vel_map is not None:
                self.ImageViewerC2.setSelectedPixel(pixel.x, pixel.y)
            if self.DysmalPyFittingWorker.residual_disp_map is not None:
                self.ImageViewerC3.setSelectedPixel(pixel.x, pixel.y)
            # 
            if self.DysmalPyFittingWorker.data_flux_map is not None:
                self.ImageViewerA1.blockSignals(False)
            if self.DysmalPyFittingWorker.data_vel_map is not None:
                self.ImageViewerA2.blockSignals(False)
            if self.DysmalPyFittingWorker.data_disp_map is not None:
                self.ImageViewerA3.blockSignals(False)
            if self.DysmalPyFittingWorker.model_flux_map is not None:
                self.ImageViewerB1.blockSignals(False)
            if self.DysmalPyFittingWorker.model_vel_map is not None:
                self.ImageViewerB2.blockSignals(False)
            if self.DysmalPyFittingWorker.model_disp_map is not None:
                self.ImageViewerB3.blockSignals(False)
            if self.DysmalPyFittingWorker.residual_flux_map is not None:
                self.ImageViewerC1.blockSignals(False)
            if self.DysmalPyFittingWorker.residual_vel_map is not None:
                self.ImageViewerC2.blockSignals(False)
            if self.DysmalPyFittingWorker.residual_disp_map is not None:
                self.ImageViewerC3.blockSignals(False)
    
    @pyqtSlot(int)
    def selectChannelInSpecViewers(self, ichan:int = None):
        if ichan is not None:
            # 
            if (self.DysmalPyFittingWorker.data_flux_curve is not None or self.DysmalPyFittingWorker.model_flux_curve is not None):
                self.SpecViewerA1.blockSignals(True)
            if (self.DysmalPyFittingWorker.data_vel_curve is not None or self.DysmalPyFittingWorker.model_vel_curve is not None):
                self.SpecViewerA2.blockSignals(True)
            if (self.DysmalPyFittingWorker.data_disp_curve is not None or self.DysmalPyFittingWorker.model_disp_curve is not None):
                self.SpecViewerA3.blockSignals(True)
            # 
            if (self.DysmalPyFittingWorker.data_flux_curve is not None or self.DysmalPyFittingWorker.model_flux_curve is not None):
                self.SpecViewerA1.setSelectedChannel(ichan)
            if (self.DysmalPyFittingWorker.data_vel_curve is not None or self.DysmalPyFittingWorker.model_vel_curve is not None):
                self.SpecViewerA2.setSelectedChannel(ichan)
            if (self.DysmalPyFittingWorker.data_disp_curve is not None or self.DysmalPyFittingWorker.model_disp_curve is not None):
                self.SpecViewerA3.setSelectedChannel(ichan)
            # 
            if (self.DysmalPyFittingWorker.data_flux_curve is not None or self.DysmalPyFittingWorker.model_flux_curve is not None):
                self.SpecViewerA1.blockSignals(False)
            if (self.DysmalPyFittingWorker.data_vel_curve is not None or self.DysmalPyFittingWorker.model_vel_curve is not None):
                self.SpecViewerA2.blockSignals(False)
            if (self.DysmalPyFittingWorker.data_disp_curve is not None or self.DysmalPyFittingWorker.model_disp_curve is not None):
                self.SpecViewerA3.blockSignals(False)
    
    @pyqtSlot()
    def onInitRandomParamsCall(self):
        if self.DysmalPyParamFile is not None and self.DysmalPyParamFile != '':
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Question)
            msgBox.setText("Parameters exist! Do you want to re-initialize and override current parameters?")
            msgBox.setWindowTitle("Re-initializing and overriding parameters?")
            msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
            clickedButton = msgBox.exec()
            if clickedButton != QMessageBox.Yes:
                return
        self.initRandomParams()
    
    def initRandomParams(self):
        self.logMessage(self.tr('Initializing DysmalPyParams with random values...'))
        if self.LineEditDataParamsDict['galID'].text() == '':
            self.LineEditDataParamsDict['galID'].setText('galID')
        if self.LineEditDataParamsDict['z'].text() == '':
            self.LineEditDataParamsDict['z'].setText(str(np.round(np.random.uniform(low=0.5, high=3.0, size=(1))[0], 3)))
        if self.LineEditDataParamsDict['aperture_radius'].text() == '':
            self.LineEditDataParamsDict['aperture_radius'].setText('0.2')
        if self.LineEditDataParamsDict['linked_posteriors'].text() == '':
            self.LineEditDataParamsDict['linked_posteriors'].setText("['total_mass', 'r_eff_disk', 'bt', 'fdm', 'sigma0']")
        if self.LineEditDataParamsDict['pixscale'].text() == '':
            self.LineEditDataParamsDict['pixscale'].setText('0.05')
        if self.LineEditDataParamsDict['fov_npix'].text() == '':
            self.LineEditDataParamsDict['fov_npix'].setText('50')
        if self.LineEditDataParamsDict['spec_type'].text() == '':
            self.LineEditDataParamsDict['spec_type'].setText('velocity')
        if self.LineEditDataParamsDict['spec_start'].text() == '':
            self.LineEditDataParamsDict['spec_start'].setText('-1000.')
        if self.LineEditDataParamsDict['spec_step'].text() == '':
            self.LineEditDataParamsDict['spec_step'].setText('25.')
        if self.LineEditDataParamsDict['nspec'].text() == '':
            self.LineEditDataParamsDict['nspec'].setText('80')
        if self.LineEditDataParamsDict['sig_inst_res'].text() == '':
            self.LineEditDataParamsDict['sig_inst_res'].setText('5.')
        if self.LineEditDataParamsDict['psf_fwhm'].text() == '':
            self.LineEditDataParamsDict['psf_fwhm'].setText('0.3') # arcsec
        if self.LineEditDataParamsDict['slit_width'].text() == '':
            self.LineEditDataParamsDict['slit_width'].setText('0.5') # arcsec
        if self.LineEditDataParamsDict['slit_pa'].text() == '':
            self.LineEditDataParamsDict['slit_pa'].setText(str(np.round(np.random.uniform(low=0.0, high=180.0, size=(1))[0], 3)))
        # 
        self.LineEditModelParamsDictForBulgeDisk['total_mass'].setText(str(np.round(np.random.uniform(low=10.5, high=13.0, size=(1))[0], 3)))
        self.LineEditModelParamsDictForBulgeDisk['bt'].setText(str(np.round(np.random.uniform(low=0.0, high=1.0, size=(1))[0], 3)))
        self.LineEditModelParamsDictForBulgeDisk['r_eff_disk'].setText(str(np.round(np.random.uniform(low=5.0, high=27.0, size=(1))[0], 3)))
        self.LineEditModelParamsDictForBulgeDisk['n_disk'].setText(str(np.round(np.random.uniform(low=0.5, high=4.0, size=(1))[0], 3)))
        self.LineEditModelParamsDictForBulgeDisk['r_eff_bulge'].setText(str(np.round(np.random.uniform(low=0.3, high=4.5, size=(1))[0], 3))) # must be smaller than r_eff_disk and 5 kpc.
        self.LineEditModelParamsDictForBulgeDisk['n_bulge'].setText(str(np.round(np.random.uniform(low=2.0, high=4.0, size=(1))[0], 3)))
        self.LineEditModelParamsDictForDispersion['sigma0'].setText(str(np.round(np.random.uniform(low=10.0, high=100.0, size=(1))[0], 3)))
        self.LineEditModelParamsDictForZHeight['sigmaz'].setText(str(np.round(np.random.uniform(low=0.0, high=0.5, size=(1))[0], 3)))
        self.LineEditModelParamsDictForGeometry['inc'].setText(str(np.round(np.random.uniform(low=15.0, high=65.0, size=(1))[0], 3)))
        self.LineEditModelParamsDictForGeometry['pa'].setText(str(np.round(np.random.uniform(low=-180.0, high=180.0, size=(1))[0], 3)))
        # 
        self.LineEditModelParamsDictForDarkMatterHalo['mvirial'].setText(str(np.round(self.LineEditModelParamsDictForBulgeDisk['total_mass'].keyvalue()+np.random.uniform(low=0.5, high=1.0, size=(1))[0], 3)))
        self.LineEditModelParamsDictForDarkMatterHalo['halo_conc'].setText(str(np.round(np.random.uniform(low=3.0, high=6.0, size=(1))[0], 3)))
        self.LineEditModelParamsDictForDarkMatterHalo['fdm'].setText(str(np.round(np.random.uniform(low=0.1, high=0.6, size=(1))[0], 3)))
        # 
        self.setButtonsEnabledDisabled()
        # 
        self.logMessage(self.tr('Initialized DysmalPyParams with some values'))
    
    def setButtonsEnabledDisabled(self, allDisabled=None):
        has_gui = True
        has_par = (len(self.DysmalPyParams) != 0)
        has_validpar = False
        has_data = False
        has_bestfit = False
        has_model = False
        if allDisabled:
            has_gui = False
            has_par = False
        elif has_par:
            if np.any([t in self.DysmalPyParams for t in ['fdata', 'fdata_vel', 'fdata_cube']]):
                has_data = True
            if np.all([self.DysmalPyParams[t] is not None for t in ['total_mass', 'bt', 'r_eff_disk', 
                        'n_disk', 'r_eff_bulge', 'n_bulge', 'sigma0', 'sigmaz', 'inc', 'pa', 
                        'mvirial', 'halo_conc', 'fdm']]):
                has_validpar = True
            if self.DysmalPyFittingWorker is not None:
                if self.DysmalPyFittingWorker.DysmalPyFitResultFile is not None:
                    if os.path.exists(self.DysmalPyFittingWorker.DysmalPyFitResultFile):
                        has_bestfit = True
                if self.DysmalPyFittingWorker.model_cube is not None:
                    has_model = True
        self.ButtonInitRandomParams.setEnabled(has_gui)
        self.ButtonOpenParamsFile.setEnabled(has_gui)
        self.ButtonFitData.setEnabled(has_data)
        self.ButtonLoadFittingResult.setEnabled(has_bestfit)
        self.ButtonGenerateModelCube.setEnabled(has_validpar)
        self.ButtonGenerateMomentMaps.setEnabled(has_model)
        self.ButtonGenerateRotationCurves.setEnabled(has_model)
        self.ButtonHideSlit.setEnabled(has_model)
        self.ButtonSaveParams.setEnabled(has_validpar)
        self.ButtonSaveModelFiles.setEnabled(has_model)
    
    @pyqtSlot()
    def onFitDataCall(self):
        self.logger.debug('onFitDataCall()')
        if self.checkDysmalPyParams(require_data=True):
            self.fitDataAsync()
    
    @pyqtSlot()
    def onLoadFittingResultCall(self):
        self.logger.debug('onLoadFittingResultCall()')
        if self.checkDysmalPyParams(require_data=True):
            self.fitDataAsync(dofit=False, clearFitResults=False)
    
    def checkDysmalPyParams(self, require_data=False, check_limits=True):
        self.logger.debug('checkDysmalPyParams()')
        checkOK = True
        errormessages = []
        if self.DysmalPyParams is None:
            errormessages.append('DysmalPyParams is None!')
            checkOK = False
        else:
            for keyname in ['z']:
                if not (keyname in self.DysmalPyParams):
                    errormessages.append('Key '+keyname+' not set!')
                    checkOK = False
                elif self.DysmalPyParams[keyname] is None or self.DysmalPyParams[keyname] == '':
                    errormessages.append('Key '+keyname+' is empty!')
                    checkOK = False
            if require_data:
                check_data = False
                for keyname in ['fdata', 'fdata_vel', 'fdata_cube']:
                    if (keyname in self.DysmalPyParams):
                        check_data = True
                if not check_data:
                    errormessages.append('No data key defined! Please check \'fdata\', \'fdata_vel\' or \'fdata_cube\' key.')
                    checkOK = False
            if check_limits:
                for keyname in self.DysmalPyParams:
                    if keyname+'_bounds' in self.DysmalPyParams:
                        this_limits = self.DysmalPyParams[keyname+'_bounds']
                        if not np.isscalar(this_limits) and len(this_limits) == 2:
                            if self.DysmalPyParams[keyname] < this_limits[0] or self.DysmalPyParams[keyname] > this_limits[1]:
                                errormessages.append('Key '+keyname+' value '+str(self.DysmalPyParams[keyname])+\
                                    ' is not in the limit '+str(this_limits)+' as defined by key '+keyname+'_bounds!')
                                checkOK = False
        if not checkOK:
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Critical)
            msgBox.setText("Error! DysmalPyParams is invalid!")
            msgBox.setInformativeText(self.tr('Error messages:')+'\n'+'\n'.join(errormessages))
            msgBox.setWindowTitle("Invalid DysmalPyParams")
            msgBox.setStandardButtons(QMessageBox.Ok)
            msgBox.exec()
        return checkOK
    
    @pyqtSlot()
    def fitDataAsync(self, dofit=True, clearFitResults=True):
        self.logger.debug('fitDataAsync(dofit={},clearFitResults={})'.format(dofit, clearFitResults))
        # 
        if self.DysmalPyFittingThread is not None:
            if self.DysmalPyFittingThread.isRunning():
                self.logMessage(self.tr('Current thread is still running.'))
                return
            #else:
            #    del self.DysmalPyFittingThread
            #    self.DysmalPyFittingThread = None
        self.setButtonsEnabledDisabled(allDisabled=True)
        self.DysmalPyFittingWorker.DysmalPyParams = self.DysmalPyParams
        self.DysmalPyFittingWorker.dofit = dofit
        self.DysmalPyFittingWorker.overwrite = self.DysmalPyParams['overwrite']
        if clearFitResults:
            self.DysmalPyFittingWorker.DysmalPyFitResults = None
            #self.DysmalPyFittingWorker.DysmalPyFitResultFile = None
        if self.DysmalPyFittingThread is None:
            self.DysmalPyFittingThread = QThread()
            self.DysmalPyFittingWorker.moveToThread(self.DysmalPyFittingThread)
            self.DysmalPyFittingThread.started.connect(self.DysmalPyFittingWorker.fitData)
            self.DysmalPyFittingWorker.finished.connect(self.DysmalPyFittingThread.quit)
            self.DysmalPyFittingWorker.finished.connect(self.onFittingWorkerFinished)
            self.DysmalPyFittingWorker.finishedWithError.connect(self.DysmalPyFittingThread.quit)
            self.DysmalPyFittingWorker.finishedWithError.connect(self.onFittingWorkerFinishedWithError)
            self.DysmalPyFittingWorker.finishedWithWarning.connect(self.DysmalPyFittingThread.quit)
            self.DysmalPyFittingWorker.finishedWithWarning.connect(self.onFittingWorkerFinishedWithWarning)
            #self.DysmalPyFittingWorker.finished.connect(self.DysmalPyFittingWorker.deleteLater)
            #self.DysmalPyFittingThread.finished.connect(self.DysmalPyFittingThread.deleteLater)
            #self.DysmalPyFittingWorker.progress.connect(self.reportProgress)
        else:
            self.DysmalPyFittingThread.started.disconnect()
            self.DysmalPyFittingThread.started.connect(self.DysmalPyFittingWorker.fitData)
        self.DysmalPyFittingThread.start()
    
    @pyqtSlot()
    def generateModelCubeAsync(self, clearFitResults=False):
        self.logger.debug('generateModelCubeAsync()')
        # 
        if self.DysmalPyFittingThread is not None:
            if self.DysmalPyFittingThread.isRunning():
                self.logMessage(self.tr('Current thread is still running.'))
                return
        self.setButtonsEnabledDisabled(allDisabled=True)
        self.DysmalPyFittingWorker.DysmalPyParams = self.DysmalPyParams
        if clearFitResults:
            self.DysmalPyFittingWorker.DysmalPyFitResults = None
            #self.DysmalPyFittingWorker.DysmalPyFitResultFile = None
        if self.DysmalPyFittingThread is None:
            self.DysmalPyFittingThread = QThread()
            self.DysmalPyFittingWorker.moveToThread(self.DysmalPyFittingThread)
            self.DysmalPyFittingThread.started.connect(self.DysmalPyFittingWorker.generateModelCube)
            self.DysmalPyFittingWorker.finished.connect(self.DysmalPyFittingThread.quit)
            self.DysmalPyFittingWorker.finished.connect(self.onFittingWorkerFinished)
            self.DysmalPyFittingWorker.finishedWithError.connect(self.DysmalPyFittingThread.quit)
            self.DysmalPyFittingWorker.finishedWithError.connect(self.onFittingWorkerFinishedWithError)
            self.DysmalPyFittingWorker.finishedWithWarning.connect(self.DysmalPyFittingThread.quit)
            self.DysmalPyFittingWorker.finishedWithWarning.connect(self.onFittingWorkerFinishedWithWarning)
        else:
            self.DysmalPyFittingThread.started.disconnect()
            self.DysmalPyFittingThread.started.connect(self.DysmalPyFittingWorker.generateModelCube)
        self.DysmalPyFittingThread.start()
    
    @pyqtSlot()
    def generateMomentMapsAsync(self, clearFitResults=False):
        self.logger.debug('generateMomentMapsAsync()')
        # 
        if self.DysmalPyFittingThread is not None:
            if self.DysmalPyFittingThread.isRunning():
                self.logMessage(self.tr('Current thread is still running.'))
                return
        self.DysmalPyFittingWorker.DysmalPyParams = self.DysmalPyParams
        if clearFitResults:
            self.DysmalPyFittingWorker.DysmalPyFitResults = None
            #self.DysmalPyFittingWorker.DysmalPyFitResultFile = None
        if self.DysmalPyFittingThread is None:
            self.DysmalPyFittingThread = QThread()
            self.DysmalPyFittingWorker.moveToThread(self.DysmalPyFittingThread)
            self.DysmalPyFittingThread.started.connect(self.DysmalPyFittingWorker.generateMomentMaps)
            self.DysmalPyFittingWorker.finished.connect(self.DysmalPyFittingThread.quit)
            self.DysmalPyFittingWorker.finished.connect(self.onFittingWorkerFinished)
            self.DysmalPyFittingWorker.finishedWithError.connect(self.DysmalPyFittingThread.quit)
            self.DysmalPyFittingWorker.finishedWithError.connect(self.onFittingWorkerFinishedWithError)
            self.DysmalPyFittingWorker.finishedWithWarning.connect(self.DysmalPyFittingThread.quit)
            self.DysmalPyFittingWorker.finishedWithWarning.connect(self.onFittingWorkerFinishedWithWarning)
        else:
            self.DysmalPyFittingThread.started.disconnect()
            self.DysmalPyFittingThread.started.connect(self.DysmalPyFittingWorker.generateMomentMaps)
        self.DysmalPyFittingThread.start()
    
    @pyqtSlot()
    def generateRotationCurvesAsync(self, clearFitResults=False):
        self.logger.debug('generateRotationCurvesAsync()')
        # 
        if self.DysmalPyFittingThread is not None:
            if self.DysmalPyFittingThread.isRunning():
                self.logMessage(self.tr('Current thread is still running.'))
                return
        self.DysmalPyFittingWorker.DysmalPyParams = self.DysmalPyParams
        if clearFitResults:
            self.DysmalPyFittingWorker.DysmalPyFitResults = None
            #self.DysmalPyFittingWorker.DysmalPyFitResultFile = None
        if self.DysmalPyFittingThread is None:
            self.DysmalPyFittingThread = QThread()
            self.DysmalPyFittingWorker.moveToThread(self.DysmalPyFittingThread)
            self.DysmalPyFittingThread.started.connect(self.DysmalPyFittingWorker.generateRotationCurves)
            self.DysmalPyFittingWorker.finished.connect(self.DysmalPyFittingThread.quit)
            self.DysmalPyFittingWorker.finished.connect(self.onFittingWorkerFinished)
            self.DysmalPyFittingWorker.finishedWithError.connect(self.DysmalPyFittingThread.quit)
            self.DysmalPyFittingWorker.finishedWithError.connect(self.onFittingWorkerFinishedWithError)
            self.DysmalPyFittingWorker.finishedWithWarning.connect(self.DysmalPyFittingThread.quit)
            self.DysmalPyFittingWorker.finishedWithWarning.connect(self.onFittingWorkerFinishedWithWarning)
        else:
            self.DysmalPyFittingThread.started.disconnect()
            self.DysmalPyFittingThread.started.connect(self.DysmalPyFittingWorker.generateRotationCurves)
        self.DysmalPyFittingThread.start()
    
    @pyqtSlot()
    def onFittingWorkerFinished(self):
        self.logger.debug('onFittingWorkerFinished()')
        #<TODO># we can optimize here, no need to update all these things if generateMomentMapsAsync or generateRotationCurvesAsync is run.
        self.plotDataMomentMaps()
        self.plotModelMomentMaps()
        self.plotModelRotationCurves()
        self.updateFittingParams()
        self.setButtonsEnabledDisabled()
    
    @pyqtSlot(str)
    def onFittingWorkerFinishedWithError(self, errormessage):
        self.logger.debug('onFittingWorkerFinishedWithError()')
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Critical)
        msgBox.setText("Error! Failed to run DysmalPyFittingWorker!")
        msgBox.setInformativeText(self.tr('Error messages:')+'\n'+errormessage)
        msgBox.setWindowTitle("Failed to run DysmalPyFittingWorker")
        msgBox.setStandardButtons(QMessageBox.Ok)
        msgBox.exec()
        self.setButtonsEnabledDisabled()
    
    @pyqtSlot(str)
    def onFittingWorkerFinishedWithWarning(self, errormessage):
        self.logger.debug('onFittingWorkerFinishedWithWarning()')
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Warning)
        msgBox.setText("Warning from DysmalPyFittingWorker:")
        msgBox.setInformativeText(self.tr('Warning messages:')+'\n'+errormessage)
        msgBox.setWindowTitle("Warning from DysmalPyFittingWorker")
        msgBox.setStandardButtons(QMessageBox.Ok)
        msgBox.exec()
        self.setButtonsEnabledDisabled()
    
    def reportProgress(self):
        pass
    
    def updateFittingParams(self):
        if self.DysmalPyFittingWorker.DysmalPyFitResults is not None:
            free_param_names = self.DysmalPyFittingWorker.DysmalPyFitResults.free_param_names
            bestfit_parameters = self.DysmalPyFittingWorker.DysmalPyFitResults.bestfit_parameters
            #bestfit_parameters_l68_err = self.DysmalPyFittingWorker.DysmalPyFitResults.bestfit_parameters_l68_err
            #bestfit_parameters_u68_err = self.DysmalPyFittingWorker.DysmalPyFitResults.bestfit_parameters_u68_err
            self.logger.debug("updateFittingParams() DysmalPyFittingWorker.DysmalPyFitResults -> UI")
            self.logger.debug("free_param_names = %s"%(free_param_names))
            self.logger.debug("bestfit_parameters = [%s]"%(', '.join(np.array(bestfit_parameters).astype(str))))
            ipar = 0
            for component_name in free_param_names:
                for param_name in free_param_names[component_name]:
                    #self.logger.debug("bestfit %r %r = %r"%(component_name, param_name, bestfit_parameters[ipar]))
                    for i in range(len(self.LineEditModelParamsDicts)):
                        if param_name in self.LineEditModelParamsDicts[i]:
                            self.LineEditModelParamsDicts[i][param_name].setText(bestfit_parameters[ipar])
                    ipar += 1
            #for key in bestfit_parameters:
            #    self.logger.debug("bestfit_parameters[%r] = %r"%(key, bestfit_parameters[key]))
        return
    
    def getSlitShapeInPixel(self):
        xcenter = None
        ycenter = None
        fov_npix = None
        slit_width = None
        slit_pa = None
        pixscale = None
        slit_shape_in_pixel = None
        if self.DysmalPyParams is not None:
            if 'xcenter' in self.DysmalPyParams:
                if self.DysmalPyParams['xcenter'] is not None:
                    xcenter = self.DysmalPyParams['xcenter']
            if xcenter is None:
                if 'fov_npix' in self.DysmalPyParams:
                    if self.DysmalPyParams['fov_npix'] is not None:
                        xcenter = (self.DysmalPyParams['fov_npix']-1.0)/2.0
            # 
            if 'ycenter' in self.DysmalPyParams:
                if self.DysmalPyParams['ycenter'] is not None:
                    ycenter = self.DysmalPyParams['ycenter']
            if ycenter is None:
                if 'fov_npix' in self.DysmalPyParams:
                    if self.DysmalPyParams['fov_npix'] is not None:
                        ycenter = (self.DysmalPyParams['fov_npix']-1.0)/2.0
            # 
            if 'fov_npix' in self.DysmalPyParams:
                if self.DysmalPyParams['fov_npix'] is not None:
                    fov_npix = self.DysmalPyParams['fov_npix']
            # 
            if 'slit_width' in self.DysmalPyParams:
                if self.DysmalPyParams['slit_width'] is not None:
                    slit_width = self.DysmalPyParams['slit_width']
            # 
            if 'slit_pa' in self.DysmalPyParams:
                if self.DysmalPyParams['slit_pa'] is not None:
                    slit_pa = self.DysmalPyParams['slit_pa']
            # 
            if 'pixscale' in self.DysmalPyParams:
                if self.DysmalPyParams['pixscale'] is not None:
                    pixscale = self.DysmalPyParams['pixscale']
            # 
            if np.all([t is not None for t in [xcenter, ycenter, fov_npix, slit_width, slit_pa, pixscale]]):
                nx = fov_npix
                ny = fov_npix
                #if np.isclose(slit_pa, 0.0) or np.isclose(slit_pa, 180.0):
                #    x = np.array([xcenter-slit_width/pixscale/2.0, xcenter+slit_width/pixscale/2.0])
                #    y1 = np.array([0, 0]).astype(float)
                #    y2 = np.array([ny-1, ny-1]).astype(float)
                #    mask = np.array([True, True])
                #else:
                #    x = np.arange(nx).astype(float)
                #    y = (x-xcenter) * np.tan(np.deg2rad(slit_pa+90.0)) + ycenter
                #    y1 = y - np.abs(slit_width/pixscale/2.0*np.cos(np.deg2rad(slit_pa+90.0)))
                #    y2 = y + np.abs(slit_width/pixscale/2.0*np.cos(np.deg2rad(slit_pa+90.0)))
                #    mask = np.logical_and.reduce((x>=0, y1>=0, y2>=0, x<=nx-1, y1<=ny-1, y2<=ny-1))
                #slit_shape_in_pixel = {'x':x, 'y1':y1, 'y2':y2, 'where':mask, 'step':'mid', 'alpha':0.445, 'color':'cyan'} 
                # slit transparency 0.445
                #self.logger.debug('slit_shape_in_pixel: '+str(slit_shape_in_pixel))
                # 
                # use rectangle slit
                dx_pix = fov_npix
                dy_pix = slit_width/pixscale
                x0 = xcenter - (dx_pix/2.0*np.cos(np.deg2rad(slit_pa+90.0))) \
                             + (dy_pix/2.0*np.sin(np.deg2rad(slit_pa+90.0)))
                y0 = ycenter - (dx_pix/2.0*np.sin(np.deg2rad(slit_pa+90.0))) \
                             - (dy_pix/2.0*np.cos(np.deg2rad(slit_pa+90.0)))
                self.logger.debug('Slit x0 %s y0 %s w %s h %s angle %s'%(x0, y0, dx_pix, dy_pix, slit_pa+90.0))
                slit_shape_in_pixel = Rectangle((x0, y0), dx_pix, dy_pix, angle=slit_pa+90.0, 
                                                alpha=0.225, facecolor='cyan', edgecolor='cyan')
        # 
        return slit_shape_in_pixel
    
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        #super(QDysmalPyGUI, self).dragEnterEvent(event)
    
    def dropEvent(self, event):
        if self.ButtonOpenParamsFile.isEnabled():
            for url in event.mimeData().urls():
                filepath = url.toLocalFile()
                if filepath.endswith('.params'):
                    self.selectParamFile(filepath)
                break
            event.accept()
        #super(QDysmalPyGUI, self).dropEvent(event)



class QDysmalPyFittingWorker(QObject):
    
    LogMessageUpdateSignal = pyqtSignal(str)
    finished = pyqtSignal()
    finishedWithWarning = pyqtSignal(str)
    finishedWithError = pyqtSignal(str)
    
    def __init__(self):
        super(QDysmalPyFittingWorker, self).__init__()
        self.logger = logging.getLogger('QDysmalPyFittingWorker')
        self.logger.setLevel(logging.getLevelName(logging.getLogger(__name__).level)) # self.logger.setLevel(logging.DEBUG)
        self.DysmalPyParams = None
        self.DysmalPyGal = None
        self.DysmalPyFitDict = None
        self.DysmalPyFitResults = None
        self.DysmalPyFitResultFile = None
        self.DefaultDirectory = os.getcwd()
        self.LastLogMessage = ''
        self.LensingTransformer = None
        self.dofit = True
        self.overwrite = False
        self.data_cube = None
        self.data_flux_map = None
        self.data_vel_map = None
        self.data_disp_map = None
        self.data_mask_map = None
        self.data_flux_curve = None
        self.data_vel_curve = None
        self.data_disp_curve = None
        self.data_rotation_curve = None
        self.model_cube = None
        self.model_flux_map = None
        self.model_vel_map = None
        self.model_disp_map = None
        self.model_flux_curve = None
        self.model_vel_curve = None
        self.model_disp_curve = None
        self.model_rotation_curve = None
        self.residual_flux_map = None
        self.residual_vel_map = None
        self.residual_disp_map = None
        self.residual_flux_curve = None
        self.residual_vel_curve = None
        self.residual_disp_curve = None
        self.residual_rotation_curve = None
    
    def logMessage(self, message):
        self.LastLogMessage = message
        self.LogMessageUpdateSignal.emit(message)
    
    def setDirectory(self, directory):
        self.DefaultDirectory = directory
    
    def getDysmalPyGal(self):
        self.logMessage(self.tr('Getting DysmalPyGal and DysmalPyFitDict from DysmalPyParams.'))
        if self.DysmalPyParams is None:
            self.logMessage(self.tr('Error! DysmalPyParams is invalid! Could not get a DysmalPyGal.'))
            return
        # 
        # debug print
        #logger.debug("self.DysmalPyParams['total_mass_bounds']: "+str(self.DysmalPyParams['total_mass_bounds'])+" "+str(type(self.DysmalPyParams['total_mass_bounds'])))
        # 
        self.DysmalPyGal = None
        self.DysmalPyFitDict = None
        if 'fdata' in self.DysmalPyParams.keys():
            ndim_fit = 1
            self.DysmalPyGal, self.DysmalPyFitDict = setup_single_object_1D(params=self.DysmalPyParams, data=None)
        elif 'fdata_vel' in self.DysmalPyParams.keys():
            ndim_fit = 2
            self.DysmalPyGal, self.DysmalPyFitDict = setup_single_object_2D(params=self.DysmalPyParams, data=None)
        elif 'fdata_cube' in self.DysmalPyParams.keys():
            ndim_fit = 3
            self.DysmalPyGal, self.DysmalPyFitDict = setup_single_object_3D(params=self.DysmalPyParams, data=None)
        else:
            self.logMessage(self.tr('Warning! No data defined in DysmalPyParams. Please check the \'fdata\', \'fdata_vel\' or \'fdata_cube\' keys.'))
            self.logMessage(self.tr('Setting up galaxy model base...'))
            self.DysmalPyGal = setup_gal_model_base(params=self.DysmalPyParams)
            #self.DysmalPyGal.data = data_io.load_single_object_1D_data(fdata=params['fdata'], fdata_mask=fdata_mask, params=params, datadir=datadir)
            #self.DysmalPyGal.data = data_classes.Data1D(...)
            #self.DysmalPyGal.data.filename_velocity = datadir+params['fdata']
            #if (params['profile1d_type'] != 'circ_ap_pv') & (params['profile1d_type'] != 'single_pix_pv'):
            #self.DysmalPyGal.data.apertures = setup_basic_aperture_types(gal=self.DysmalPyGal, params=self.DysmalPyParams)
            #self.DysmalPyGal.data.profile1d_type = self.DysmalPyParams['profile1d_type']
            #self.DysmalPyFitDict = setup_fit_dict(params=self.DysmalPyParams, ndim_data=1)
        # 
        # <DZLIU><20210726> ++++++++++
        if self.LensingTransformer is None:
            lensing_keys = [\
                    'lensing_mesh', 
                    'lensing_ra', 
                    'lensing_dec', 
                    'lensing_sra', 
                    'lensing_sdec', 
                    'lensing_ssizex', 
                    'lensing_ssizey', 
                    'lensing_spixsc', 
                    'lensing_imra', 
                    'lensing_imdec', 
                ]
            for lensing_key in lensing_keys:
                self.logger.debug('DysmalPyParams.has_key(%r) %s'%(lensing_key, lensing_key in self.DysmalPyParams))
            has_lensing_model = np.all([lensing_key in self.DysmalPyParams for lensing_key in lensing_keys])
            if has_lensing_model:
                for lensing_key in lensing_keys:
                    self.logger.debug('DysmalPyParams[%r] = %r'%(lensing_key, self.DysmalPyParams[lensing_key]))
                has_lensing_model = np.all([self.DysmalPyParams[lensing_key] is not None for lensing_key in lensing_keys])
            if has_lensing_model:
                self.logger.debug('Setting LensingTransformer')
                self.LensingTransformer = LensingTransformer(\
                        mesh_file = self.DysmalPyParams['datadir'] + os.sep + self.DysmalPyParams['lensing_mesh'], 
                        mesh_ra = self.DysmalPyParams['lensing_ra'], 
                        mesh_dec = self.DysmalPyParams['lensing_dec'], 
                        source_plane_nx = self.DysmalPyParams['lensing_ssizex'], 
                        source_plane_ny = self.DysmalPyParams['lensing_ssizey'], 
                        source_plane_nchan = self.DysmalPyParams['nspec'], 
                        source_plane_cenra = self.DysmalPyParams['lensing_sra'], 
                        source_plane_cendec = self.DysmalPyParams['lensing_sdec'], 
                        source_plane_pixsc = self.DysmalPyParams['lensing_spixsc'], 
                        image_plane_cenra = self.DysmalPyParams['lensing_imra'], 
                        image_plane_cendec = self.DysmalPyParams['lensing_imdec'], 
                        image_plane_pixsc = self.DysmalPyParams['pixscale'], 
                        image_plane_sizex = self.DysmalPyParams['fov_npix'], 
                        image_plane_sizey = self.DysmalPyParams['fov_npix'], 
                    )
        # <DZLIU><20210726> ----------
        # 
        return
    
    @pyqtSlot()
    def fitData(self):
        if self.dofit:
            self.logMessage(self.tr('Preparing to fit the data...'))
        else:
            self.logMessage(self.tr('Preparing to load fitting result...'))
        if self.DysmalPyParams is None:
            self.logMessage(self.tr('Error! DysmalPyParams is invalid! Could not proceed to fit the data.'))
            self.finishedWithError.emit(self.LastLogMessage)
            return
        # 
        self.getDysmalPyGal()
        if self.DysmalPyGal is None or self.DysmalPyFitDict is None:
            self.logMessage(self.tr('Error! Could not set DysmalPyGal and DysmalPyFitDict from DysmalPyParams?!'))
            self.finishedWithError.emit(self.LastLogMessage)
            return
        # 
        if self.DysmalPyFitDict['fit_method'] == 'mcmc':
            output_pickle_file = os.path.join(self.DysmalPyParams['outdir'], self.DysmalPyParams['galID']+'_mcmc_results.pickle')
        elif self.DysmalPyFitDict['fit_method'] == 'mpfit':
            output_pickle_file = os.path.join(self.DysmalPyParams['outdir'], self.DysmalPyParams['galID']+'_mpfit_results.pickle')
        else:
            self.logMessage(self.tr('Error! fit_method must be \'mcmc\' or \'mpfit\'! Could not do the fitting!'))
            self.finishedWithError.emit(self.LastLogMessage)
            return
        self.logMessage(self.tr('Fitting with fit_method '+self.DysmalPyFitDict['fit_method']+'.'))
        # 
        config_c_m_data = config.Config_create_model_data(**self.DysmalPyFitDict)
        config_sim_cube = config.Config_simulate_cube(**self.DysmalPyFitDict)
        kwargs_galmodel = {**config_c_m_data.dict, **config_sim_cube.dict}
        # <DZLIU><20210726> ++++++++++
        # setup LensingTransformer if needed
        if self.LensingTransformer is not None:
            self.logger.debug('Setting lensing_transformer in kwargs')
            kwargs_galmodel['lensing_transformer'] = self.LensingTransformer
        # <DZLIU><20210726> ----------
        kwargs_all = {**kwargs_galmodel, **self.DysmalPyFitDict}
        # 
        if ((not os.path.isfile(output_pickle_file)) or self.overwrite) and self.dofit:
            # check DysmalPy param limits
            #self.checkDysmalPyParams()
            # run the DysmalPy fitting
            if self.DysmalPyFitDict['fit_method'] == 'mcmc':
                self.DysmalPyFitResults = fitting.fit_mcmc(self.DysmalPyGal, **kwargs_all)
            elif self.DysmalPyFitDict['fit_method'] == 'mpfit':
                self.DysmalPyFitResults = fitting.fit_mpfit(self.DysmalPyGal, **kwargs_all)
            else:
                self.logMessage(self.tr('Error! fit_method must be \'mcmc\' or \'mpfit\'! Could not do the fitting!'))
                self.finishedWithError.emit(self.LastLogMessage)
                return
            # check output file
            if not os.path.isfile(output_pickle_file):
                self.logMessage(self.tr('Error! Output file not produced?! ')+str(output_pickle_file))
                self.finishedWithError.emit(self.LastLogMessage)
                return
            else:
                self.logMessage(self.tr('Successfully fitted the data.'))
        elif os.path.isfile(output_pickle_file):
            # load existing fitting result
            self.logMessage(self.tr('Loading previous fitting...'))
            self.DysmalPyGal, self.DysmalPyFitResults = fitting.reload_all_fitting(\
                                        filename_galmodel=self.DysmalPyFitDict['f_model'],
                                        filename_results=self.DysmalPyFitDict['f_results'],
                                        fit_method=self.DysmalPyParams['fit_method'])
            self.logMessage(self.tr('Successfully loaded previous fitting.'))
        else:
            #self.logMessage(self.tr('Warning! Previous fitting result not found and not doing a fitting. Please click the "Fit Data" button and try again.'))
            self.logMessage(self.tr('Warning! Previous fitting result not found. Please click the "Fit Data" button to run a fit.'))
            self.finishedWithWarning.emit(self.LastLogMessage)
            return
        self.model_cube = copy.copy(self.DysmalPyGal.model_cube.data)
        self.DysmalPyFitResultFile = output_pickle_file
        # 
        self.generateMomentMaps(blocksignal=True)
        if self.LastLogMessage.find(self.tr('Error!')) >= 0:
            self.finishedWithError.emit(self.LastLogMessage)
        # 
        self.generateRotationCurves(blocksignal=True)
        if self.LastLogMessage.find(self.tr('Error!')) >= 0:
            self.finishedWithError.emit(self.LastLogMessage)
        # 
        if self.dofit:
            self.logMessage(self.tr('Successfully generated the model cube, moment maps and rotation curves.'))
        # 
        self.finished.emit()
    
    @pyqtSlot()
    def generateModelCube(self):
        self.logMessage(self.tr('Generating model cube...'))
        if self.DysmalPyParams is None:
            self.logMessage(self.tr('Error! DysmalPyParams is invalid! Could not proceed to generate the model cube.'))
            self.finishedWithError.emit(self.LastLogMessage)
            return
        # 
        self.getDysmalPyGal()
        if self.DysmalPyGal is None:
            self.logMessage(self.tr('Error! Could not set DysmalPyGal from DysmalPyParams?!'))
            self.finishedWithError.emit(self.LastLogMessage)
            return
        if self.DysmalPyGal.instrument is None:
            self.logMessage(self.tr('Error! DysmalPyGal.instrument is invalid! Could not proceed to generate the model cube.'))
            self.finishedWithError.emit(self.LastLogMessage)
            return
        # 
        #config_c_m_data = config.Config_create_model_data(**self.DysmalPyFitDict)
        #config_sim_cube = config.Config_simulate_cube(**self.DysmalPyFitDict)
        #kwargs_galmodel = {**config_c_m_data.dict, **config_sim_cube.dict}
        #self.DysmalPyGal.create_model_data(**kwargs_galmodel)
        # 
        # from_instrument: nx_sky, ny_sky, spec_type, spec_start, spec_step, spec_unit, nspec, rstep will be read from gal.instrument
        # xcenter ycenter are needed only if the center is not at the cube center.
        # line_center is needed only if spec_type is 'wavelength'
        ndim_final = 3
        profile1d_type = self.DysmalPyParams['profile1d_type']
        oversample = self.DysmalPyParams['oversample']
        oversize = self.DysmalPyParams['oversize']
        aperture_radius = self.DysmalPyParams['aperture_radius'] # if profile1d_type == 'circ_ap_cube'
        pixscale = self.DysmalPyParams['pixscale']
        fov_npix = self.DysmalPyParams['fov_npix']
        xcenter = self.DysmalPyParams['xcenter']
        ycenter = self.DysmalPyParams['ycenter']
        #aper_centers = np.linspace(-np.abs(fov_npix/pixscale/2.0), +np.abs(fov_npix/pixscale/2.0), num=30, endpoint=True) #<TODO># 
        aper_centers = np.linspace(-np.abs(fov_npix*pixscale/2.0), +np.abs(fov_npix*pixscale/2.0), num=30, endpoint=True) # in arcsec
        kwargs_galmodel = {}
        #if profile1d_type == 'circ_ap_cube':
        #    kwargs_galmodel['aperture_radius'] = 0.1
        #elif profile1d_type == 'rect_ap_cube':
        #    kwargs_galmodel['pix_perp'] = 5
        #    kwargs_galmodel['pix_parallel'] = 20
        #elif profile1d_type == 'square_ap_cube':
        #    kwargs_galmodel['pix_length'] = 20
        # <DZLIU><20210726> ++++++++++
        # setup LensingTransformer if needed
        if self.LensingTransformer is not None:
            kwargs_galmodel['lensing_transformer'] = self.LensingTransformer
        # <DZLIU><20210726> ----------
        self.DysmalPyGal.create_model_data(\
                            ndim_final = ndim_final, 
                            profile1d_type = profile1d_type, 
                            aperture_radius = aperture_radius, 
                            aper_centers = aper_centers,
                            from_instrument = True, 
                            from_data = False,
                            oversample = oversample, 
                            oversize = oversize,
                            xcenter = xcenter, 
                            ycenter = ycenter,
                            skip_downsample = False, 
                            **kwargs_galmodel, 
                            )
        #if self.model_cube is not None:
        #    self.logger.debug('DEBUG DEBUG old model cube %s vs new model cube %s'%(self.model_cube[:,25,25], self.DysmalPyGal.model_cube.data[:,25,25]))
        self.model_cube = copy.copy(self.DysmalPyGal.model_cube.data)
        # 
        self.generateMomentMaps(blocksignal=True)
        if self.LastLogMessage.find(self.tr('Error!')) >= 0:
            self.finishedWithError.emit(self.LastLogMessage)
        # 
        self.generateRotationCurves(blocksignal=True)
        if self.LastLogMessage.find(self.tr('Error!')) >= 0:
            self.finishedWithError.emit(self.LastLogMessage)
        # 
        self.logMessage(self.tr('Successfully generated the model cube, moment maps and rotation curves.'))
        # 
        self.finished.emit()
        return
    
    def computeMomentMapsFromCube(self, data_cube):
        # see "galaxy.py"
        if not isinstance(data_cube, SpectralCube):
            self.logMessage(self.tr('Error! The input data cube to generateMomentMapsFromCube is not a SpectralCube!'))
            return None, None, None
        mom0 = data_cube.moment0().to(u.km/u.s).value
        mom1 = data_cube.moment1().to(u.km/u.s).value
        mom2 = data_cube.linewidth_sigma().to(u.km/u.s).value
        flux = np.zeros(mom0.shape)
        vel = np.zeros(mom0.shape)
        disp = np.zeros(mom0.shape)
        for i in range(mom0.shape[0]):
            for j in range(mom0.shape[1]):
                best_fit = gaus_fit_sp_opt_leastsq(\
                                    data_cube.spectral_axis.to(u.km/u.s).value,
                                    data_cube.unmasked_data[:,i,j].value,
                                    mom0[i,j], mom1[i,j], mom2[i,j])
                flux[i,j] = best_fit[0] * np.sqrt(2 * np.pi) * best_fit[2]
                vel[i,j] = best_fit[1]
                disp[i,j] = best_fit[2]
        return flux, vel, disp
    
    @pyqtSlot()
    def generateMomentMaps(self, blocksignal=False):
        self.logMessage(self.tr('Generating moment maps...'))
        if self.DysmalPyParams is None:
            self.logMessage(self.tr('Error! DysmalPyParams is invalid! Could not proceed to generate the moment maps. Please set DysmalPyFittingWorker.DysmalPyParams first!'))
            if not blocksignal:
                self.finishedWithError.emit(self.LastLogMessage)
            return
        if self.DysmalPyGal is None:
            self.logMessage(self.tr('Error! DysmalPyGal is invalid! Could not proceed to generate the moment maps. Please run DysmalPyFittingWorker.generateModelCube first!'))
            if not blocksignal:
                self.finishedWithError.emit(self.LastLogMessage)
            return
        if self.DysmalPyGal.model_cube is None:
            self.logMessage(self.tr('Error! DysmalPyGal.model_cube is invalid! Could not proceed to generate the moment maps. Please run DysmalPyFittingWorker.generateModelCube first!'))
            if not blocksignal:
                self.finishedWithError.emit(self.LastLogMessage)
            return
        # 
        model_flux_map = None
        model_vel_map = None
        model_disp_map = None
        # 
        model_flux_map, model_vel_map, model_disp_map = \
            self.computeMomentMapsFromCube(self.DysmalPyGal.model_cube.data)
        # 
        #if self.model_flux_map is not None:
        #    self.logger.debug('DEBUG DEBUG old model flux map %s \nvs new model flux map %s'%(\
        #        self.model_flux_map[20:30,20:30], \
        #        model_flux_map[20:30,20:30]))
        # 
        self.model_flux_map = model_flux_map
        self.model_vel_map = model_vel_map
        self.model_disp_map = model_disp_map
        # 
        if hasattr(self.DysmalPyGal, 'data'):
            if self.DysmalPyGal.data is not None:
                if self.DysmalPyGal.data.ndim == 2:
                    self.data_flux_map = self.DysmalPyGal.data.data['flux']
                    self.data_vel_map = self.DysmalPyGal.data.data['velocity']
                    self.data_disp_map = self.DysmalPyGal.data.data['dispersion']
                    self.data_mask_map = self.DysmalPyGal.data.mask
                    if self.data_flux_map is not None:
                        nan_mask = np.invert(self.data_mask_map) # (self.data_flux_map < -9.99999e5)
                        if np.count_nonzero(nan_mask)>0:
                            self.data_flux_map[nan_mask] = np.nan
                    if self.data_vel_map is not None:
                        nan_mask = np.invert(self.data_mask_map) # (self.data_vel_map < -9.99999e5)
                        if np.count_nonzero(nan_mask)>0:
                            self.data_vel_map[nan_mask] = np.nan
                    if self.data_disp_map is not None:
                        nan_mask = np.invert(self.data_mask_map) # (self.data_disp_map < -9.99999e5)
                        if np.count_nonzero(nan_mask)>0:
                            self.data_disp_map[nan_mask] = np.nan
                    if self.data_flux_map is not None:
                        if self.data_flux_map.shape == self.model_flux_map.shape:
                            self.residual_flux_map = self.data_flux_map - self.model_flux_map
                    if self.data_vel_map is not None:
                        if self.data_vel_map.shape == self.model_vel_map.shape:
                            self.residual_vel_map = self.data_vel_map - self.model_vel_map
                    if self.data_disp_map is not None:
                        if self.data_disp_map.shape == self.model_disp_map.shape:
                            self.residual_disp_map = self.data_disp_map - self.model_disp_map
        # 
        self.logMessage(self.tr('Successfully generated moment maps.'))
        # 
        # emit finish signal
        if not blocksignal:
            self.finished.emit()
        return
    
    @pyqtSlot()
    def generateRotationCurves(self, blocksignal=False):
        self.logMessage(self.tr('Generating rotation curves...'))
        if self.DysmalPyParams is None:
            self.logMessage(self.tr('Error! DysmalPyParams is invalid! Could not proceed to generate the rotation curve. Please set DysmalPyFittingWorker.DysmalPyParams first!'))
            if not blocksignal:
                self.finishedWithError.emit(self.LastLogMessage)
            return
        if self.DysmalPyGal is None:
            self.logMessage(self.tr('Error! DysmalPyGal is invalid! Could not proceed to generate the rotation curve. Please run DysmalPyFittingWorker.generateModelCube first!'))
            if not blocksignal:
                self.finishedWithError.emit(self.LastLogMessage)
            return
        if self.DysmalPyGal.model_cube is None:
            self.logMessage(self.tr('Error! DysmalPyGal.model_cube is invalid! Could not proceed to generate the rotation curve. Please run DysmalPyFittingWorker.generateModelCube first!'))
            if not blocksignal:
                self.finishedWithError.emit(self.LastLogMessage)
            return
        # 
        self.data_flux_curve = None
        self.data_vel_curve = None
        self.data_disp_curve = None
        self.model_flux_curve = None
        self.model_vel_curve = None
        self.model_disp_curve = None
        inst_corr = True
        # 
        if 'fdata' in self.DysmalPyParams:
            self.logMessage(self.tr('Computing rotation velocity profile along the data slit at PA ')+str(self.DysmalPyGal.data.slit_pa))
            self.DysmalPyGal.create_model_data(ndim_final=1, from_data=True)
            self.data_flux_curve  = {'x':self.DysmalPyGal.data.rarr, 
                                     'y':self.DysmalPyGal.data.data['flux'], 
                                     'yerr':self.DysmalPyGal.data.error['flux'], 
                                     'marker':'o', 'markersize':5, 'markeredgecolor':'none', 'linestyle':'none', 'alpha':0.7, 'capsize':2}
            self.data_vel_curve   = {'x':self.DysmalPyGal.data.rarr, 
                                     'y':self.DysmalPyGal.data.data['velocity'], 
                                     'yerr':self.DysmalPyGal.data.error['velocity'], 
                                     'marker':'o', 'markersize':5, 'markeredgecolor':'none', 'linestyle':'none', 'alpha':0.7, 'capsize':2}
            self.data_disp_curve  = {'x':self.DysmalPyGal.data.rarr, 
                                     'y':self.DysmalPyGal.data.data['dispersion'], 
                                    'yerr':self.DysmalPyGal.data.error['dispersion'], 
                                    'marker':'o', 'markersize':5, 'markeredgecolor':'none', 'linestyle':'none', 'alpha':0.7, 'capsize':2}
            self.model_flux_curve = {'x':self.DysmalPyGal.model_data.rarr, 
                                     'y':self.DysmalPyGal.model_data.data['flux'], 
                                     'marker':'s', 'markersize':5, 'markeredgecolor':'none', 'linestyle':'none', 'alpha':0.7}
            self.model_vel_curve  = {'x':self.DysmalPyGal.model_data.rarr, 
                                     'y':self.DysmalPyGal.model_data.data['velocity'], 
                                     'marker':'s', 'markersize':5, 'markeredgecolor':'none', 'linestyle':'none', 'alpha':0.7}
            self.model_disp_curve = {'x':self.DysmalPyGal.model_data.rarr, 
                                     'y':self.DysmalPyGal.model_data.data['dispersion'], 
                                     'marker':'s', 'markersize':5, 'markeredgecolor':'none', 'linestyle':'none', 'alpha':0.7}
            if 'inst_corr' in self.DysmalPyGal.data.data.keys():
                inst_corr = self.DysmalPyGal.data.data['inst_corr']
        else:
            slit_width = self.DysmalPyParams['slit_width']
            slit_pa = self.DysmalPyParams['slit_pa']
            oversample = self.DysmalPyParams['oversample']
            oversize = self.DysmalPyParams['oversize']
            if slit_pa is None or slit_width is None:
                self.logMessage(self.tr('Error! Invalid slit_width or slit_pa. Could not proceed to generate the rotation curves.'))
                if not blocksignal:
                    self.finishedWithError.emit(self.LastLogMessage)
                return
            profile1d_type = self.DysmalPyParams['profile1d_type']
            aperture_radius = self.DysmalPyParams['aperture_radius'] # if profile1d_type == 'circ_ap_cube'
            pixscale = self.DysmalPyParams['pixscale']
            fov_npix = self.DysmalPyParams['fov_npix']
            xcenter = self.DysmalPyParams['xcenter']
            ycenter = self.DysmalPyParams['ycenter']
            aper_centers = np.linspace(-np.abs(fov_npix*pixscale/2.0), +np.abs(fov_npix*pixscale/2.0), num=35, endpoint=True) # in arcsec
            this_DysmalPyGal = copy.deepcopy(self.DysmalPyGal)
            if this_DysmalPyGal.data is not None:
                this_DysmalPyGal.data.aper_center_pix_shift = None
            self.logMessage(self.tr('Computing rotation velocity profile along the assumed slit at PA ')+str(slit_pa))
            self.logger.debug('generateRotationCurves: this_DysmalPyGal.model.line_center = '+str(this_DysmalPyGal.model.line_center))
            self.logger.debug('generateRotationCurves: this_DysmalPyGal.instrument.spec_type = '+str(this_DysmalPyGal.instrument.spec_type))
            self.logger.debug('generateRotationCurves: this_DysmalPyGal.instrument.spec_start.value = '+str(this_DysmalPyGal.instrument.spec_start.value))
            self.logger.debug('generateRotationCurves: this_DysmalPyGal.instrument.spec_step.value = '+str(this_DysmalPyGal.instrument.spec_step.value))
            self.logger.debug('generateRotationCurves: this_DysmalPyGal.instrument.spec_start.unit = '+str(this_DysmalPyGal.instrument.spec_start.unit))
            self.logger.debug('generateRotationCurves: this_DysmalPyGal.instrument.pixscale.value (rstep) = '+str(this_DysmalPyGal.instrument.pixscale.value))
            self.logger.debug('generateRotationCurves: this_DysmalPyGal.instrument.nspec = '+str(this_DysmalPyGal.instrument.nspec))
            self.logger.debug('generateRotationCurves: this_DysmalPyGal.instrument.fov[0] = '+str(this_DysmalPyGal.instrument.fov[0]))
            self.logger.debug('generateRotationCurves: this_DysmalPyGal.instrument.fov[1] = '+str(this_DysmalPyGal.instrument.fov[1]))
            #self.logger.debug('generateRotationCurves: this_DysmalPyGal.instrument.slit_width = '+str(this_DysmalPyGal.instrument.slit_width))
            self.logger.debug('generateRotationCurves: this_DysmalPyGal.dscale = '+str(this_DysmalPyGal.dscale))
            self.logger.debug('generateRotationCurves: this_DysmalPyGal.slit_width = '+str(slit_width))
            self.logger.debug('generateRotationCurves: this_DysmalPyGal.slit_pa = '+str(slit_pa))
            self.logger.debug('generateRotationCurves: this_DysmalPyGal.profile1d_type = '+str(profile1d_type))
            self.logger.debug('generateRotationCurves: this_DysmalPyGal.aperture_radius = '+str(aperture_radius))
            self.logger.debug('generateRotationCurves: this_DysmalPyGal.aper_centers = '+str(aper_centers))
            self.logger.debug('generateRotationCurves: this_DysmalPyGal.oversample = '+str(oversample))
            self.logger.debug('generateRotationCurves: this_DysmalPyGal.oversize = '+str(oversize))
            self.logger.debug('generateRotationCurves: this_DysmalPyGal.xcenter = '+str(xcenter))
            self.logger.debug('generateRotationCurves: this_DysmalPyGal.ycenter = '+str(ycenter))
            this_DysmalPyGal.create_model_data(ndim_final=1, from_data=False, from_instrument=True, 
                                               slit_width = slit_width, 
                                               slit_pa = slit_pa, 
                                               profile1d_type = profile1d_type, 
                                               aperture_radius = aperture_radius, 
                                               aper_centers = aper_centers, 
                                               oversample = oversample, 
                                               oversize = oversize, 
                                               skip_downsample = False, 
                                               xcenter = xcenter, 
                                               ycenter = ycenter, 
                                               )
            #logger.debug('this_DysmalPyGal.model_data.rarr: '+str(this_DysmalPyGal.model_data.rarr))
            #logger.debug('this_DysmalPyGal.model_data.data: '+str(this_DysmalPyGal.model_data.data))
            #aper_model = aperture_classes.setup_aperture_types(gal=this_DysmalPyGal,
            #            profile1d_type=profile1d_type,
            #            slit_width = slit_width,
            #            aper_centers=aper_centers,
            #            slit_pa=slit_pa,
            #            aperture_radius=aperture_radius,
            #            pix_perp=None,
            #            pix_parallel=None,
            #            pix_length=None,
            #            partial_weight=False,
            #            from_data=False)
            #aper_centers, flux1d, vel1d, disp1d = aper_model.extract_1d_kinematics(spec_arr=vel_arr,
            #        cube=cube_data, center_pixel = center_pixel,
            #        pixscale=rstep)
            #self.model_data = Data1D(r=aper_centers, velocity=vel1d,
            #                         vel_disp=disp1d, flux=flux1d, mask=None,
            #                         slit_width=slit_width, slit_pa=slit_pa)
            self.model_flux_curve = {'x':this_DysmalPyGal.model_data.rarr, 
                                     'y':this_DysmalPyGal.model_data.data['flux'], 
                                     'marker':'s', 'markersize':5, 'markeredgecolor':'none', 'linestyle':'none', 'alpha':0.7}
            self.model_vel_curve  = {'x':this_DysmalPyGal.model_data.rarr, 
                                     'y':this_DysmalPyGal.model_data.data['velocity'], 
                                     'marker':'s', 'markersize':5, 'markeredgecolor':'none', 'linestyle':'none', 'alpha':0.7}
            self.model_disp_curve = {'x':this_DysmalPyGal.model_data.rarr, 
                                     'y':this_DysmalPyGal.model_data.data['dispersion'], 
                                     'marker':'s', 'markersize':5, 'markeredgecolor':'none', 'linestyle':'none', 'alpha':0.7}
        if inst_corr:
            try: 
                lsf_dispersion = this_DysmalPyGal.instrument.lsf.dispersion
            except:
                self.logMessage(self.tr('LSF dispersion not defined. Will not do inst_corr.'))
                inst_corr = False
        if inst_corr:
            self.model_disp_curve['y'] = np.sqrt( self.model_disp_curve['y']**2 - lsf_dispersion.to(u.km/u.s).value**2 ) # see "dysmalpy/plotting.py" def plot_data_model_comparison_1D
        # 
        self.logMessage(self.tr('Successfully generated rotation curves.'))
        # 
        # emit finish signal
        if not blocksignal:
            self.finished.emit()
        return
    
    def saveModelCube(self, filepath=None):
        if self.model_cube is None:
            self.logMessage(self.tr('Error! No valid model cube! Could not save the model cube.'))
            return
        if filepath is None or filepath == '':
            filepath, _ = QFileDialog.getSaveFileName(self, self.tr('Saving mode flux as a FITS file'), self.DefaultDirectory, self.tr('FITS cube (*.fits)')) # QFileDialog::DontConfirmOverwrite
            if filepath is None or filepath == '':
                self.logMessage(self.tr('No file selected.'))
                return
        if os.path.isfile(filepath):
            self.logger.debug('Found existing param file %r. Backing up as %r.'%(filepath, filepath+'.backup'))
            shutil.move(filepath, filepath+'.backup')
        hdu = fits.PrimaryHDU(data=self.model_cube) #<TODO># header
        hdu.writeto(filepath)
        self.logMessage(self.tr('Saved model flux to FITS file: ')+str(filepath))

    def saveModelFlux(self, filepath=None):
        if self.model_flux_map is None:
            self.logMessage(self.tr('Error! No valid model cube! Could not save the model flux.'))
            return
        if filepath is None or filepath == '':
            filepath, _ = QFileDialog.getSaveFileName(self, self.tr('Saving mode flux as a FITS file'), self.DefaultDirectory, self.tr('FITS image (*.fits)')) # QFileDialog::DontConfirmOverwrite
            if filepath is None or filepath == '':
                self.logMessage(self.tr('No file selected.'))
                return
        if os.path.isfile(filepath):
            self.logger.debug('Found existing param file %r. Backing up as %r.'%(filepath, filepath+'.backup'))
            shutil.move(filepath, filepath+'.backup')
        hdu = fits.PrimaryHDU(data=self.model_flux_map) #<TODO># header
        hdu.writeto(filepath)
        self.logMessage(self.tr('Saved model flux to FITS file: ')+str(filepath))
    
    def saveModelVel(self, filepath=None):
        if self.model_vel_map is None:
            self.logMessage(self.tr('Error! No valid model cube! Could not save the model vel. map.'))
            return
        if filepath is None or filepath == '':
            filepath, _ = QFileDialog.getSaveFileName(self, self.tr('Saving mode vel. map as a FITS file'), self.DefaultDirectory, self.tr('FITS image (*.fits)')) # QFileDialog::DontConfirmOverwrite
            if filepath is None or filepath == '':
                self.logMessage(self.tr('No file selected.'))
                return
        if os.path.isfile(filepath):
            self.logger.debug('Found existing param file %r. Backing up as %r.'%(filepath, filepath+'.backup'))
            shutil.move(filepath, filepath+'.backup')
        hdu = fits.PrimaryHDU(data=self.model_vel_map) #<TODO># header
        hdu.writeto(filepath)
        self.logMessage(self.tr('Saved model vel. map to FITS file: ')+str(filepath))
        
    def saveModelVdisp(self, filepath=None):
        if self.model_disp_map is None:
            self.logMessage(self.tr('Error! No valid model vel. disp.! Could not save the model vel. disp.'))
            return
        if filepath is None or filepath == '':
            filepath, _ = QFileDialog.getSaveFileName(self, self.tr('Saving mode vel. disp. as a FITS file'), self.DefaultDirectory, self.tr('FITS image (*.fits)')) # QFileDialog::DontConfirmOverwrite
            if filepath is None or filepath == '':
                self.logMessage(self.tr('No file selected.'))
                return
        if os.path.isfile(filepath):
            self.logger.debug('Found existing param file %r. Backing up as %r.'%(filepath, filepath+'.backup'))
            shutil.move(filepath, filepath+'.backup')
        hdu = fits.PrimaryHDU(data=self.model_disp_map) #<TODO># header
        hdu.writeto(filepath)
        self.logMessage(self.tr('Saved model vel. disp. to FITS file: ')+str(filepath))
        return
    
    


# 

class QWidgetForParamInput(QWidget):
    """QWidget for a param input.
    
    The widget can be a QCheckBox, if the parameter has a boolean data type, 
    or QComboBox, if the parameter has multiple options or is a boolean data type 
    but with checkbox = False, or more commonly, the widget is a QLineEdit, 
    and parameter data type can be a list or a str or a int or a float. 
    """
    
    ParamUpdateSignal = pyqtSignal(str, str, type, type)
    
    def __init__(self, 
            keyname, keyvalue=None, keycomment='', datatype=str, listtype=str, 
            default=None, options=None, checkbox=False, readonly=False, 
            fullwidth=False, isdatadir=False, defaultdir=None, isdatafile=False, namefilter=None, isoutdir=False, 
            enabled=True, 
            parent=None
        ):
        super(QWidgetForParamInput, self).__init__()
        self.ParamName = keyname
        self.ParamValue = keyvalue # note that this will change if the user edited anything via the GUI
        self.ParamComment = keycomment
        self.ParamDefaultValue = default
        self.ParamDataType = datatype
        self.ParamListType = listtype
        self.ParamRegExpValidator = ''
        self.LabelText = keyname
        self.IsFullWidth = fullwidth
        self.IsDataDir = isdatadir
        self.IsDataFile = isdatafile
        self.IsOutDir = isoutdir
        self.NameFilter = namefilter
        self.DefaultDirectory = defaultdir if defaultdir is not None else os.getcwd()
        self.RestrictedToDirectory = None
        self.IndexInWidgetGrid = -1
        self.LabelWidget = None
        self.LineEditWidget = None
        self.CheckBoxWidget = None
        self.ComboBoxWidget = None
        self.ButtonToOpen = None
        self.ButtonToEdit = None
        self.ButtonToClose = None
        self.Layout = QHBoxLayout()
        self.Layout.setContentsMargins(0, 0, 0, 0)
        self.Layout.setSpacing(0)
        if self.LabelText.startswith('__'):
            if re.match(r'__(.*)__', self.LabelText):
                self.LabelText = re.sub(r'__(.*)__', r'\1', self.LabelText)
            else:
                self.LabelText = re.sub(r'__(.*)', r'\1', self.LabelText)
        if self.ParamDataType is bool and self.ParamListType is not list and checkbox == True:
            # checkbox for bool
            self.CheckBoxWidget = QCheckBox(self.LabelText)
            self.CheckBoxWidget.setMinimumWidth(160)
            self.CheckBoxWidget.setToolTip(self.ParamComment)
            self.Layout.addWidget(self.CheckBoxWidget)
            self.CheckBoxWidget.stateChanged.connect(self.onCheckBoxStateChangedCall)
        elif self.ParamDataType is bool and self.ParamListType is not list:
            # combobox for bool
            self.LabelWidget = QLabel(self.LabelText+':')
            self.LabelWidget.setMinimumWidth(80)
            self.ComboBoxWidget = QComboBox()
            self.ComboBoxWidget.setToolTip(self.ParamComment)
            self.ComboBoxWidget.addItem(self.tr('True'))
            self.ComboBoxWidget.addItem(self.tr('False'))
            self.Layout.addWidget(self.LabelWidget)
            self.Layout.addWidget(self.ComboBoxWidget)
            self.ComboBoxWidget.currentIndexChanged.connect(self.onComboBoxIndexChangedCall)
        elif options is not None:
            # combobox for options
            self.LabelWidget = QLabel(self.LabelText+':')
            self.LabelWidget.setMinimumWidth(80)
            self.ComboBoxWidget = QComboBox()
            self.ComboBoxWidget.setToolTip(self.ParamComment)
            for this_option in options:
                self.ComboBoxWidget.addItem(this_option)
            self.Layout.addWidget(self.LabelWidget)
            self.Layout.addWidget(self.ComboBoxWidget)
            self.ComboBoxWidget.currentIndexChanged.connect(self.onComboBoxIndexChangedCall)
        else:
            self.LabelWidget = QLabel(self.LabelText+':')
            self.LabelWidget.setMinimumWidth(80)
            self.LineEditWidget = QLineEdit()
            self.LineEditWidget.setToolTip(self.ParamComment)
            self.LineEditWidget.setReadOnly(readonly)
            self.Layout.addWidget(self.LabelWidget)
            self.Layout.addWidget(self.LineEditWidget)
            if self.ParamDataType is not None:
                if self.ParamDataType is bool:
                    self.ParamRegExpValidator = r'(True|False)'
                elif self.ParamDataType is int:
                    self.ParamRegExpValidator = r'([+-]|)[0-9.]+'
                elif self.ParamDataType is float:
                    self.ParamRegExpValidator = r'([+-]|)[0-9.]+([eE][0-9]+|[eE][+-][0-9]+)'
                else:
                    pass
            if self.ParamListType is list:
                if self.ParamRegExpValidator == '':
                    self.ParamRegExpValidator = r'\[.*\]'
                else:
                    self.ParamRegExpValidator = r'\['+self.ParamRegExpValidator+r', *'+self.ParamRegExpValidator+r'.*'+r'\]'
            if self.ParamRegExpValidator != '':
                self.LineEditWidget.setValidator(QRegExpValidator(QRegExp(r'^'+self.ParamRegExpValidator+r'$')))
            self.LineEditWidget.setStyleSheet("""
                QLineEdit {
                    background: white;
                    border: 1px solid #cccccc;
                    border-radius: 3px;
                }
                QLineEdit[readOnly="true"] {
                    background: #dedede;
                    border: 1px solid #cccccc;
                }
                QLineEdit[enabled="false"] {
                    background: #dedede;
                    border: 1px solid #cccccc;
                }
                """)
            self.LineEditWidget.textChanged.connect(self.onLineEditTextChangedCall)
            # 
            if self.IsDataDir or self.IsDataFile or self.IsOutDir:
                self.ButtonToOpen = QPushButton(self.tr('...'))
                self.ButtonToEdit = QPushButton(self.tr('✓'))
                self.ButtonToClose = QPushButton(self.tr('✘'))
                if self.IsDataDir:
                    self.ButtonToOpen.setToolTip('Select a data directory.')
                elif self.IsDataFile:
                    self.ButtonToOpen.setToolTip('Select a data file.')
                elif self.IsOutDir:
                    self.ButtonToOpen.setToolTip('Select an output directory.')
                self.ButtonToEdit.setToolTip('Edit')
                self.ButtonToClose.setToolTip('Close')
                self.ButtonToOpen.clicked.connect(self.onButtonToOpenCall)
                self.ButtonToEdit.clicked.connect(self.onButtonToEditCall)
                self.ButtonToClose.clicked.connect(self.onButtonToCloseCall)
                self.Layout.addWidget(self.ButtonToOpen)
                self.Layout.addWidget(self.ButtonToEdit)
                self.Layout.addWidget(self.ButtonToClose)
                self.setAcceptDrops(True)
            # 
        self.setLayout(self.Layout)
        # 
        if self.ParamDefaultValue is not None:
            self.setText(self.ParamDefaultValue)
        # 
        if not enabled:
            self.setEnabled(enabled)

    def getPositionInQGridLayout(self, icount, ncolumn):
        """Return a position to add this widget to a QGridLayout. 
        
        The input is the searlized index `icount` in the QGridLayout and the grid column width `ncolumn`.
        
        The output is the incremented searlized index for the next widget in the QGridLayout, 
        the row index, column index, row span and column span of this widget, which are to be 
        used when calling the QGridLayout.addWidget() function.
        """
        irow = int(icount / ncolumn)
        icol = (icount % ncolumn)
        rowSpan = 1
        colSpan = 1
        # if the widget is set to be full width, then we need to make sure its icol is 0 and colSpan is ncolumn
        if self.IsFullWidth:
            if icol != 0:
                irow += 1
                icount += ncolumn-1-icol+1
            colSpan = ncolumn
        icount += colSpan
        return icount, irow, icol, rowSpan, colSpan

    def isFullWidth(self):
        return self.IsFullWidth

    def keyname(self):
        return self.ParamName

    def keyvalue(self):
        #return self.ParamValue
        return self.textToValue(self.text())

    def keycomment(self):
        return self.ParamComment
    
    def textToValue(self, text):
        keyvalue = None
        if text == '':
            keyvalue = None
        elif text == 'None':
            keyvalue = None
        elif text == 'True':
            keyvalue = True
        elif text == 'False':
            keyvalue = False
        elif text == 'inf':
            keyvalue = np.inf
        elif text == 'nan':
            keyvalue = np.nan
        else:
            if self.ParamListType is list:
                keyvalue = eval(text)
                keyvalue = np.array(keyvalue).astype(self.ParamDataType)
                # note that here may cause exception, better to use try except when calling this function
            else:
                keyvalue = np.array([text]).astype(self.ParamDataType)[0]
                # note that here may cause exception, better to use try except when calling this function
        return keyvalue

    def text(self):
        if self.LineEditWidget is not None:
            return self.LineEditWidget.text()
        elif self.ComboBoxWidget is not None:
            return self.ComboBoxWidget.currentText()
        elif self.CheckBoxWidget is not None:
            if self.CheckBoxWidget.isChecked():
                return 'True'
            else:
                return 'False'
        else:
            return ''

    def toolTip(self):
        if self.LineEditWidget is not None:
            return self.LineEditWidget.toolTip()
        elif self.ComboBoxWidget is not None:
            return self.ComboBoxWidget.toolTip()
        elif self.CheckBoxWidget is not None:
            return self.CheckBoxWidget.toolTip()
        else:
            return ''
    
    def dataType(self):
        return self.ParamDataType
    
    def listType(self):
        return self.ParamListType
    
    def setText(self, text, blocksignal=False):
        if text is None:
            text = ''
        if self.LineEditWidget is not None:
            if blocksignal:
                self.LineEditWidget.blockSignals(True)
            if text != '' and (not self.isEnabled()):
                self.setEnabled(True)
            self.LineEditWidget.setText(str(text))
            if blocksignal:
                self.LineEditWidget.blockSignals(False)
        elif self.ComboBoxWidget is not None:
            found_index = self.ComboBoxWidget.findText(str(text))
            if found_index >= 0:
                if blocksignal:
                    self.ComboBoxWidget.blockSignals(True)
                self.ComboBoxWidget.setCurrentIndex(found_index)
                if blocksignal:
                    self.ComboBoxWidget.blockSignals(False)
        elif self.CheckBoxWidget is not None:
            if blocksignal:
                self.CheckBoxWidget.blockSignals(True)
            if self.CheckBoxWidget.isChecked() and text == 'False':
                self.CheckBoxWidget.setChecked(False)
            elif not self.CheckBoxWidget.isChecked() and text == 'True':
                self.CheckBoxWidget.setChecked(True)
            if blocksignal:
                self.CheckBoxWidget.blockSignals(False)

    def setChecked(self, checked, blocksignal=False):
        if self.CheckBoxWidget is not None:
            if blocksignal:
                self.CheckBoxWidget.blockSignals(True)
            self.CheckBoxWidget.setChecked(checked)
            if blocksignal:
                self.CheckBoxWidget.blockSignals(False)

    def setEnabled(self, enabled):
        #logger.debug('QWidgetForParamInput::setEnabled({})'.format(enabled))
        if self.isEnabled() == enabled:
            return
        if self.LineEditWidget is not None:
            #self.LineEditWidget.setEnabled(enabled)
            self.LineEditWidget.setReadOnly(operator.not_(enabled))
            #logger.debug('QWidgetForParamInput::setEnabled() self.LineEditWidget.setReadOnly({})'.format(operator.not_(enabled)))
            self.LineEditWidget.style().unpolish(self.LineEditWidget)
            self.LineEditWidget.style().polish(self.LineEditWidget)

    def isEnabled(self):
        if self.LineEditWidget is not None:
            #return self.LineEditWidget.isEnabled()
            return operator.not_(self.LineEditWidget.isReadOnly())
        return True

    def setEditable(self, editable):
        if self.LineEditWidget is not None:
            self.LineEditWidget.setReadOnly(operator.not_(editable))
            self.LineEditWidget.style().unpolish(self.LineEditWidget)
            self.LineEditWidget.style().polish(self.LineEditWidget)

    def setReadOnly(self, readonly):
        if self.LineEditWidget is not None:
            self.LineEditWidget.setReadOnly(readonly)
            self.LineEditWidget.style().unpolish(self.LineEditWidget)
            self.LineEditWidget.style().polish(self.LineEditWidget)
        elif self.CheckBoxWidget is not None:
            self.CheckBoxWidget.setReadOnly(readonly)

    @pyqtSlot(str)
    def onLineEditTextChangedCall(self, updatedtext):
        #logger.debug('QWidgetForParamInput::onLineEditTextChangedCall')
        if self.LineEditWidget is not None:
            self.ParamUpdateSignal.emit(self.ParamName, updatedtext, self.ParamDataType, self.ParamListType)

    @pyqtSlot(int)
    def onComboBoxIndexChangedCall(self, state):
        if self.ComboBoxWidget is not None:
            self.ParamUpdateSignal.emit(self.ParamName, self.ComboBoxWidget.currentText(), self.ParamDataType, self.ParamListType)

    @pyqtSlot(int)
    def onCheckBoxStateChangedCall(self, state):
        if self.CheckBoxWidget is not None:
            if self.CheckBoxWidget.isChecked():
                self.ParamUpdateSignal.emit(self.ParamName, 'True', self.ParamDataType, self.ParamListType)
            else:
                self.ParamUpdateSignal.emit(self.ParamName, 'False', self.ParamDataType, self.ParamListType)
    
    def setRestrictedToDirectory(self, dirpath):
        if dirpath is None or dirpath == '':
            self.RestrictedToDirectory = None
        else:
            self.RestrictedToDirectory = dirpath
    
    @pyqtSlot()
    def onButtonToOpenCall(self):
        logger.debug('QWidgetForParamInput::onButtonToOpenCall()')
        #self.logMessage(self.tr('Selecting a filepath...'))
        if self.ButtonToOpen is None:
            return
        if self.LineEditWidget is None:
            return
        if self.IsDataDir or self.IsOutDir:
            dialog = QFileDialog(self, self.ButtonToOpen.toolTip(), self.DefaultDirectory, self.NameFilter)
            dialog.setFileMode(QFileDialog.DirectoryOnly) # select only directories
            if self.IsDataDir:
                dialog.setOption(QFileDialog.ReadOnly, True)
        elif self.IsDataFile:
            if self.RestrictedToDirectory is not None:
                # here we restrict to select only data files in self.DefaultDirectory
                dialog = QFileDialogRestrictedToDirectory(self, self.ButtonToOpen.toolTip(), self.RestrictedToDirectory, self.NameFilter)
            else:
                dialog = QFileDialog(self, self.ButtonToOpen.toolTip(), self.DefaultDirectory, self.NameFilter)
            dialog.setFileMode(QFileDialog.ExistingFiles)
            dialog.setOption(QFileDialog.ReadOnly, True)
        else:
            dialog = QFileDialog(self, self.ButtonToOpen.toolTip(), self.DefaultDirectory, self.NameFilter)
        #dialog.setSidebarUrls([QUrl.fromLocalFile(self.DefaultDirectory)])
        dialog.RestoreDirectory = True
        filepath = None
        if dialog.exec_() == QDialog.Accepted:
            filepath = dialog.selectedFiles()[0]
        del dialog
        if filepath is None or filepath == '':
            #self.logMessage(self.tr('No directory selected.'))
            #if not self.isEnabled():
            #    filepath = ''
            #    self.setText('', blocksignal=True)
            #    self.setEnabled(False)
            pass
        else:
            if self.IsDataDir or self.IsOutDir:
                if not filepath.endswith(os.sep):
                    filepath += os.sep
            elif self.IsDataFile:
                filepath = os.path.basename(filepath)
                #<TODO># how to make sure data files are under 'datadir'?
            #self.logMessage(self.tr('Selected data file: '+str(filepath)))
            #if not self.isEnabled():
            #    self.setEnabled(True)
            self.setText(filepath, blocksignal=True)
        self.ParamUpdateSignal.emit(self.ParamName, filepath, self.ParamDataType, self.ParamListType)
    
    @pyqtSlot()
    def onButtonToEditCall(self):
        logger.debug('QWidgetForParamInput::onButtonToEditCall()')
        self.setEnabled(True)
    
    @pyqtSlot()
    def onButtonToCloseCall(self):
        logger.debug('QWidgetForParamInput::onButtonToCloseCall()')
        self.setText('', blocksignal=True)
        self.setEnabled(False)
        self.ParamUpdateSignal.emit(self.ParamName, '', self.ParamDataType, self.ParamListType)

    @pyqtSlot(str, str, type, type)
    def onDataDirParamUpdateCall(self, keyname, keyvalue, datatype, listtype):
        logger.debug('QWidgetForParamInput::onDataDirParamUpdateCall()')
        if keyname == 'datadir':
            if self.IsDataFile:
                logger.debug('QWidgetForParamInput::onDataDirParamUpdateCall() self.setRestrictedToDirectory("{}")'.format(keyvalue))
                self.setRestrictedToDirectory(keyvalue)
    
    def dragEnterEvent(self, event):
        if self.isEnabled() and (self.IsDataDir or self.IsDataFile or self.IsOutDir):
            event.acceptProposedAction()
    
    def dropEvent(self, event):
        if self.isEnabled() and (self.IsDataDir or self.IsDataFile or self.IsOutDir):
            #input_text = event.mimeData().text()
            #self.LineEditWidget.setText(input_text)
            for url in event.mimeData().urls():
                filepath = url.toLocalFile()
                if self.IsDataFile:
                    filepath = os.path.basename(filepath) # only take file basename 
                elif not filepath.endswith(os.sep):
                    filepath += os.sep # append trailing os.sep to datadir or outdir
                self.LineEditWidget.setText(filepath)
                break
            event.accept()





class QFileDialogRestrictedToDirectory(QFileDialog):
    """A QFileDialog that selects files restricted in a given directory."""
    def __init__(self, parent=None, caption="", directory="", filter=""):
        super(QFileDialogRestrictedToDirectory, self).__init__()
        self.setViewMode(QFileDialog.List)
        self.setAcceptMode(QFileDialog.AcceptOpen)
        self.RestrictedToDirectory = directory
        self.directoryEntered.connect(self.onDirectoryEntered) # here we restrict to select only data files in self.DefaultDirectory
    
    def onDirectoryEntered(self, dirpath):
        logger.debug('QFileDialogRestrictedToDirectory::onDirectoryEntered("{}")'.format(dirpath))
        self.setDirectory(self.RestrictedToDirectory)
        #<TODO># not working!
    
    def setRestrictedToDirectory(self, dirpath):
        self.RestrictedToDirectory = dirpath
        
    




class QLineEditForParamInput(QWidget):
    
    ParamUpdateSignal = pyqtSignal(str, str, type, type)
    
    def __init__(self, labeltext, tooltiptext=None, datatype=str, listtype=str, parent=None):
        # datatype should be str, list, int, float
        # listtype should be list or str (stands for None)
        super(QLineEditForParamInput, self).__init__()
        self.ParamName = labeltext
        self.ParamDataType = datatype
        self.ParamListType = listtype
        self.ParamRegExpValidator = ''
        self.LineEdit = QLineEdit()
        #self.LineEdit.setMaximumWidth(120)
        self.LabelForLineEdit = QLabel(labeltext)
        self.LabelForLineEdit.setMinimumWidth(80)
        self.LayoutForLineEdit = QHBoxLayout()
        self.LayoutForLineEdit.setContentsMargins(0, 0, 0, 0)
        self.LayoutForLineEdit.setSpacing(0)
        self.LayoutForLineEdit.addWidget(self.LabelForLineEdit)
        self.LayoutForLineEdit.addWidget(self.LineEdit)
        self.setLayout(self.LayoutForLineEdit)
        if tooltiptext is not None:
            self.LineEdit.setToolTip(tooltiptext)
        if datatype is not None:
            if datatype is bool:
                self.ParamRegExpValidator = r'(True|False)'
            elif datatype is int:
                self.ParamRegExpValidator = r'([+-]|)[0-9.]+'
            elif datatype is float:
                self.ParamRegExpValidator = r'([+-]|)[0-9.]+([eE][0-9]+|[eE][+-][0-9]+)'
            else:
                pass
        if listtype is list:
            if self.ParamRegExpValidator == '':
                self.ParamRegExpValidator = r'\[.*\]'
            else:
                self.ParamRegExpValidator = r'\['+self.ParamRegExpValidator+r', *'+self.ParamRegExpValidator+r'.*'+r'\]'
        if self.ParamRegExpValidator != '':
            self.LineEdit.setValidator(QRegExpValidator(QRegExp(r'^'+self.ParamRegExpValidator+r'$')))
        self.LineEdit.setStyleSheet("""
            QLineEdit {
                background: white;
                border: 1px solid #cccccc;
                border-radius: 3px;
            }
            QLineEdit[readOnly="true"] {
                background: #dedede;
                border: 1px solid #cccccc;
            }
            """)
        self.LineEdit.textChanged.connect(self.onLineEditTextChangedCall)
    
    def text(self):
        return self.LineEdit.text()
    
    def toolTip(self):
        return self.LineEdit.toolTip()
    
    def setText(self, text):
        self.LineEdit.blockSignals(True)
        self.LineEdit.setText(str(text))
        self.LineEdit.blockSignals(False)

    def setEditable(self, editable):
        self.LineEdit.setReadOnly(operator.not_(editable))
        self.LineEdit.style().unpolish(self.LineEdit)
        self.LineEdit.style().polish(self.LineEdit)

    def setReadOnly(self, readonly):
        self.LineEdit.setReadOnly(readonly)
        self.LineEdit.style().unpolish(self.LineEdit)
        self.LineEdit.style().polish(self.LineEdit)

    @pyqtSlot(str)
    def onLineEditTextChangedCall(self, updatedtext):
        self.ParamUpdateSignal.emit(self.ParamName, updatedtext, self.ParamDataType, self.ParamListType)
        



class QCheckBoxForParamInput(QWidget):
    
    ParamUpdateSignal = pyqtSignal(str, str, type, type)
    
    def __init__(self, labeltext, tooltiptext=None, datatype=bool, listtype=str, parent=None):
        super(QCheckBoxForParamInput, self).__init__()
        self.ParamName = labeltext
        self.ParamDataType = datatype
        self.ParamListType = listtype
        self.Layout = QHBoxLayout()
        self.Layout.setContentsMargins(0, 0, 0, 0)
        self.Layout.setSpacing(0)
        self.CheckBox = QCheckBox(labeltext)
        self.CheckBox.setMinimumWidth(160)
        #self.CheckBox.setMaximumWidth(160)
        self.Layout.addWidget(self.CheckBox)
        self.setLayout(self.Layout)
        if tooltiptext is not None:
            self.CheckBox.setToolTip(tooltiptext)
        #self.CheckBox.setStyleSheet("""
        #    QCheckBox {
        #        background: transparent;
        #        border: 1px solid #cccccc;
        #        border-radius: 3px;
        #    }
        #    """)
        self.CheckBox.stateChanged.connect(self.onCheckBoxStateChangedCall)
    
    def text(self):
        if self.CheckBox.isChecked():
            return 'True'
        else:
            return 'False'
    
    def toolTip(self):
        return self.CheckBox.toolTip()

    def setReadOnly(self, readonly):
        self.CheckBox.blockSignals(True)
        self.CheckBox.setReadOnly(readonly)
        self.CheckBox.blockSignals(False)

    def setChecked(self, checked):
        self.CheckBox.blockSignals(True)
        self.CheckBox.setChecked(checked)
        self.CheckBox.blockSignals(False)

    @pyqtSlot(int)
    def onCheckBoxStateChangedCall(self, state):
        if self.CheckBox.isChecked():
            self.ParamUpdateSignal.emit(self.ParamName, '1', self.ParamDataType, self.ParamListType)
        else:
            self.ParamUpdateSignal.emit(self.ParamName, '0', self.ParamDataType, self.ParamListType)



# 
# 
# 
class QInputDialogSetColorBarLimits(QDialog):
    
    def __init__(self, vmin=None, vmax=None, parent = None):
        logger.debug('QInputDialogSetColorBarLimits::__init__()')
        super(QInputDialogSetColorBarLimits, self).__init__(parent)
        layout = QFormLayout()
        self.Label1 = QLabel(self.tr('vmin'))
        self.Label2 = QLabel(self.tr('vmax'))
        self.LineEdit1 = QLineEdit('')
        self.LineEdit2 = QLineEdit('')
        self.LineEdit1.setMinimumWidth(200)
        self.LineEdit2.setMinimumWidth(200)
        self.LineEdit1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.LineEdit2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        if vmin is not None:
            self.LineEdit1.setText(str(vmin))
        if vmax is not None:
            self.LineEdit2.setText(str(vmax))
        self.ButtonBox = QDialogButtonBox()
        self.ButtonBox.addButton("OK", QDialogButtonBox.AcceptRole)
        self.ButtonBox.addButton("Cancel", QDialogButtonBox.RejectRole)
        #self.ButtonBox.setStandardButtons(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.ButtonBox.accepted.connect(self.accept)
        self.ButtonBox.rejected.connect(self.reject)
        layout.addRow(self.Label1, self.LineEdit1)
        layout.addRow(self.Label2, self.LineEdit2)
        layout.addRow(self.ButtonBox)
        self.LineEdit1.setValidator(QRegExpValidator(QRegExp(r'^([0-9eE .+-]+)$')))
        self.LineEdit2.setValidator(QRegExpValidator(QRegExp(r'^([0-9eE .+-]+)$')))
        self.setLayout(layout)
        self.setWindowTitle(self.tr('Set Color Bar Limits'))
        self.setMinimumWidth(400)
        self.setMinimumHeight(100)
    
    def getResults(self):
        logger.debug('QInputDialogSetColorBarLimits::getResults()')
        if self.exec_() == QDialog.Accepted:
            # get all values
            val1 = float(self.LineEdit1.text())
            val2 = float(self.LineEdit2.text())
            return val1, val2
        else:
            return None, None
    
    #def onAccepted(self):
    #    self.done(1)
    
    #def onRejected(self):
    #    self.done(0)




# 
# class QFitsImageWidget
# for ImageA, ImageB, ImageC, ImageD, ImageE
# 
class QFitsImageWidget(FigureCanvasQTAgg):
    
    PixelSelectedSignal = pyqtSignal(PixCoord)
    RegionSelectedSignal = pyqtSignal(PixelRegion)
    PositionVelocityRidgeSelectedSignal = pyqtSignal(list)
    
    class SelectingMode(Enum):
        Nothing = 0
        Pixel = 1
        Polygon = 2
        PositionVelocity = 3
    NoneMode = SelectingMode.Nothing
    PixelMode = SelectingMode.Pixel
    PolygonMode = SelectingMode.Polygon
    PositionVelocityMode = SelectingMode.PositionVelocity
    
    #def __colorbar_number_formatter__(self, x, pos):
    #    if x < 0:
    #        return '\N{MINUS SIGN}%g'%(-x)
    #    else:
    #        return '%g'%(x)
    
    def __init__(self, parent=None, width=5, height=5, title=None, name=None, with_colorbar=False, tight_layout=False):
        self.dataimage = None
        self.dataimagewcs = None
        self.dataimagewidth = None
        self.dataimageheight = None
        self.title = title
        self.name = name
        self.slit = None
        #self.pixmap = None
        #self.origin = QPoint(0, 0)
        #self.transform = None
        #self.scalefactor = None
        #self.datapixel = None # currently selected pixel in data coordinate (int)
        #self.datapolygon = None # polygons in data coordinate (int)
        #self.DataPixelF = None # currently selected pixel in data coordinate (float)
        #self.DataPolygonF = None # polygons in data coordinate (float)
        self.SelectedPixel = None # selected pixel in data coordinate (float)
        self.SelectedRegion = None # must be a PixelRegion
        self.SelectedPositionVelocityRidge = None # a list of positions for Position-Velocity (PV) analysis
        self.StoredRegions = []
        self.StoredPositionVelocityRidges = []
        self.SelectingMode = self.PixelMode
        self.PreviouslySelectedPixelCoord = None # 
        self.PreviouslySelectedPixels = None # for adding pixels in polygon selection mode
        self.PlottedImshow = None
        self.PlottedContour = None
        self.PlottedColorbar = None
        self.PlottedPixel = [] # must be a list
        self.PlottedPolygon = [] # must be a list
        # setup matplotlib figure
        self.fig = Figure(figsize=(width, height), tight_layout=tight_layout)
        if not tight_layout:
            self.fig.subplots_adjust(left=0.01, right=0.80, bottom=0.01, top=0.87) #<TODO># adjust this
        self.axes = self.fig.add_subplot(111)
        self.axes.get_xaxis().set_visible(False)
        self.axes.get_yaxis().set_visible(False)
        self.caxdiv = None
        self.cax = None
        self.caxorientation = "vertical"
        if with_colorbar:
            self.caxdiv = make_axes_locatable(self.axes)
            self.cax = self.caxdiv.append_axes("right", size="7%", pad="2%")
            self.cax.yaxis.set_ticks_position("right")
            self.cax.tick_params(labelsize='small')
            #self.cax.yaxis.set_major_formatter(ticker.FuncFormatter(self.__colorbar_number_formatter__))
            self.caxorientation = "vertical"
        # 
        super(QFitsImageWidget, self).__init__(self.fig)
        self.mpl_connect('button_press_event', self.mplMousePressEvent)
        self.mpl_connect('motion_notify_event', self.mplMouseMoveEvent)
        self.mpl_connect('key_press_event', self.mplKeyPressEvent)
        # set qt cursor style
        self.setCursor(Qt.CrossCursor)
        #self.setStyleSheet('.QWidget{border: 2px solid black; border-radius: 2px; background-color: rgb(0, 0, 0);}')
        self.setMinimumSize(50, 50)
        # 
        self.ContextMenu = QMenu()
        self.ActionSetColorBarLimits = QAction('Set Color Bar limits')
        self.ActionSetColorBarLimits.triggered.connect(self.onActionSetColorBarLimits)
        self.ContextMenu.addAction(self.ActionSetColorBarLimits)
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.onImageViewerContextMenuCall)
        # 
        if title is not None:
            self.axes.set_title(title)
    
    def setSelectingMode(self, selecting_mode):
        self.SelectingMode = selecting_mode
    
    def setImageWCS(self, data_image_wcs):
        self.dataimagewcs = data_image_wcs
    
    def showImage(self, 
            image=None, cmap='viridis', extent=None, vmin=None, vmax=None, 
            selected_pixel=None, selected_pixel_color='k', selected_pixel_alpha=0.9, selected_pixel_linewidth=1.5, 
            selected_polygon=None, selected_polygon_color='k', selected_polygon_alpha=0.9, selected_polygon_linewidth=1.5, 
            selected_polygon_closed=True, 
            clear_plot=True, clear_selected_pixel=True, clear_selected_polygon=True, 
            with_contour=False, with_colorbar=False, clear_colorbar=False, 
            with_title=None, with_title_color='k', 
            ):
        """
        selected_chunk are filled with color from zero baseline to y value.
        selected_channel is filled from ylim[0] to ylim[1].
        """
        # check axes
        if self.axes is None:
            return False
        # clear previous plot
        if clear_selected_pixel:
            for pobj in self.PlottedPixel:
                pobj.remove()
            self.PlottedPixel = []
        if clear_selected_polygon:
            for pobj in self.PlottedPolygon:
                pobj.remove()
            self.PlottedPolygon = []
        if clear_plot:
            for axitem in self.axes.lines + self.axes.collections + self.axes.texts:
                axitem.remove()
            if self.slit is not None:
                self.slit.remove()
                self.slit = None
        # show image
        if image is not None:
            self.dataimage = image
            if extent is None:
                extent = [-0.5, image.shape[1]-0.5, -0.5, image.shape[0]-0.5] # left, right, bottom, top
            if self.PlottedImshow is None:
                imshow_params = {'origin':'lower', 'interpolation':'nearest', 'cmap':cmap, 'extent':extent}
                self.PlottedImshow = self.axes.imshow(self.dataimage, **imshow_params)
                # 
                # fix axes range
                self.axes.set_xlim([extent[0], extent[1]])
                self.axes.set_ylim([extent[2], extent[3]])
                # 
                if with_contour:
                    contour_params = {'origin':'lower', 'colors':'white', 'linewidths':0.605, 'alpha':0.85}
                    contour_params['extent'] = extent
                    self.PlottedContour = self.axes.contour(self.dataimage, **contour_params)
            else:
                self.PlottedImshow.set_data(self.dataimage)
                if vmin is None:
                    vmin = np.nanmin(self.dataimage)
                    #vmin = np.nanpercentile(self.dataimage.ravel(), 0.1)
                if vmax is None:
                    vmax = np.nanmax(self.dataimage)
                    #vmax = np.nanpercentile(self.dataimage.ravel(), 100.0)
                self.PlottedImshow.set_clim(vmin=vmin, vmax=vmax)
                self.PlottedImshow.set_extent(extent)
                # 
                # fix axes range
                self.axes.set_xlim([extent[0], extent[1]])
                self.axes.set_ylim([extent[2], extent[3]])
                # 
                if with_contour:
                    #if self.PlottedContour:
                    #    print('self.PlottedContour.collections', self.PlottedContour.collections)
                    #    for axitem in self.PlottedContour.collections:
                    #        print('axitem', axitem)
                    #        #axitem.remove()
                    #-- already removed by clear_plot
                    contour_params = {'origin':'lower', 'colors':'white', 'linewidths':0.605, 'alpha':0.85}
                    if extent is not None:
                        contour_params['extent'] = extent
                    self.PlottedContour = self.axes.contour(self.dataimage, **contour_params)
            # 
            if with_colorbar:
                if clear_colorbar and self.PlottedColorbar is not None:
                    self.PlottedColorbar.remove()
                if self.PlottedColorbar is None:
                    if self.cax is not None:
                        self.PlottedColorbar = self.fig.colorbar(self.PlottedImshow, cax=self.cax, orientation=self.caxorientation)
                    else:
                        self.PlottedColorbar = self.fig.colorbar(self.PlottedImshow)
                #else:
                #    self.PlottedColorbar.set_clim(vmin=np.nanmin(self.dataimage), vmax=np.nanmax(self.dataimage))
                #    self.PlottedColorbar.draw_all()
            # 
            if with_title is not None:
                self.axes.set_title(with_title, color=with_title_color)
                #self.axes.text(0.5, 1.02, with_title, ha='center', va='bottom', transform=self.axes.transAxes, color=with_title_color)
            # 
            self.update()
        # 
        # check image array
        if self.dataimage is None:
            return False
        # 
        # highlight selected pixel
        if selected_pixel is not None:
            print('QFitsImageWidget::showImage()', 'selected_pixel', selected_pixel, 'value', self.dataimage[selected_pixel[1], selected_pixel[0]], 'widget name', self.name)
            pobj = Rectangle(
                [selected_pixel[0]-0.5, selected_pixel[1]-0.5], 
                1, 1, 
                linewidth=selected_pixel_linewidth, 
                edgecolor=selected_pixel_color, 
                facecolor='none', 
                alpha=selected_pixel_alpha, 
                zorder=99)
            self.axes.add_artist(pobj)
            self.PlottedPixel.append(pobj)
        # 
        # highlight selected polygon
        if selected_polygon is not None:
            #print('QFitsImageWidget::showImage()', 'selected_polygon', selected_polygon)
            pobj = Polygon(
                selected_polygon, 
                closed=selected_polygon_closed,
                linewidth=selected_polygon_linewidth, 
                edgecolor=selected_polygon_color, 
                facecolor='none', 
                alpha=selected_polygon_alpha, 
                zorder=99)
            self.axes.add_artist(pobj)
            self.PlottedPolygon.append(pobj)
        # 
        # update canvas
        self.draw()
        return True
    
    def showSlit(self, slit_shape_in_pixel=None):
        if self.axes is not None and self.dataimage is not None:
            # if slit exists and not visible and slit_shape_in_pixel is None, set visible
            if self.slit is not None and self.slit.get_visible() == False and slit_shape_in_pixel is None:
                self.slit.set_visible(True)
            # otherwise if slit_shape_in_pixel is not None, plot the slit
            elif slit_shape_in_pixel is not None:
                #logger.debug('self.axes.get_xlim() = '+str(self.axes.get_xlim()))
                #logger.debug('self.axes.get_ylim() = '+str(self.axes.get_ylim()))
                #self.slit = self.axes.fill_between(**slit_shape_in_pixel, transform=self.axes.transData)
                # 
                # slit in rectangle
                if self.slit is not None:
                    self.slit.remove()
                self.slit = slit_shape_in_pixel
                self.axes.add_patch(self.slit)
            else:
                return False
            self.update()
            self.draw()
            return True
        return False
    
    def hideSlit(self):
        if self.axes is not None and self.dataimage is not None and self.slit is not None:
            self.slit.set_visible(False)
            self.update()
            self.draw()
            return True
        return False
    
    def setSelectedPixel(self, x, y, do_plot=True, emit_signal=True):
        print('QFitsImageWidget::setSelectedPixel()', 'x: %s, y: %s, widget name: %s'%(x, y, self.name))
        if self.axes is not None and self.dataimage is not None:
            self.SelectedPixel = PixCoord(x=x, y=y)
            if do_plot:
                self.showImage(
                    clear_plot=False, 
                    clear_selected_pixel=True, 
                    clear_selected_polygon=False, 
                    selected_pixel=(int(round(x)), int(round(y))))
            if emit_signal:
                self.PixelSelectedSignal.emit(self.SelectedPixel)
            return True
        return False
    
    def setSelectedRegion(self, region, closed=True, replaceId=-1):
        print('QFitsImageWidget::setSelectedRegion()', 'region: %s, closed: %s'%(str(region), closed))
        if self.axes is not None and self.dataimage is not None:
            if isinstance(region, PolygonPixelRegion):
                polygon = [(t.x,t.y) for t in region.vertices]
            elif isinstance(region, CirclePixelRegion):
                polygon = [(region.center.x+region.radius*np.cos(np.deg2rad(t)), 
                            region.center.y+region.radius*np.sin(np.deg2rad(t))) for t in np.arange(0, 360+15, 15)]
                #<TODO># here we convert anything to polygon. will do better later.
            else:
                polygon = region # here we assume the input is like [(1,2), (3,4), (5,6), ...]
            self.showImage(
                clear_plot=False, 
                clear_selected_pixel=False, 
                clear_selected_polygon=True, 
                selected_polygon=polygon, 
                selected_polygon_closed=closed,
                with_contour=True,
                )
            if closed:
                if isinstance(region, PixelRegion):
                    self.SelectedRegion = region
                else:
                    x = [t[0] for t in polygon]
                    y = [t[1] for t in polygon]
                    self.SelectedRegion = PolygonPixelRegion(vertices=PixCoord(x=x, y=y))
                #patch = self.SelectedRegion.as_artist(facecolor='none', edgecolor='red', lw=2)
                #ax.add_patch(patch)
                if replaceId >= 0 and replaceId <= len(self.StoredRegions)-1:
                    self.StoredRegions[replaceId] = self.SelectedRegion
                elif self.SelectedRegion not in self.StoredRegions:
                    #self.StoredRegions.append(self.SelectedRegion) #<TODO># Store all drawn regions
                    self.StoredRegions = [self.SelectedRegion] #<TODO># Just store one region
                self.RegionSelectedSignal.emit(self.SelectedRegion)
            return True
        return False
    
    def unsetSelectedRegion(self):
        print('QFitsImageWidget::unsetSelectedRegion()')
        self.SelectedRegion = None
        return True
    
    def setSelectedPositionVelocityRidge(self, positions, closed=True, replaceId=-1):
        """
        'positions' should be a list of (x,y) pair.
        """
        print('QFitsImageWidget::setSelectedPositionVelocityRidge()', 'positions: %s, closed: %s'%(str(positions), closed))
        if self.axes is not None and self.dataimage is not None:
            self.showImage(
                clear_plot=False, 
                clear_selected_pixel=False, 
                clear_selected_polygon=True, 
                selected_polygon=positions, 
                selected_polygon_closed=False,
                with_contour=True,
                ) # when drawing pv ridge, it is always a non-closed polygon
            if closed:
                print('QFitsImageWidget::setSelectedPositionVelocityRidge()', 'self.SelectedPositionVelocityRidge = positions')
                self.SelectedPositionVelocityRidge = positions
                if replaceId >= 0 and replaceId <= len(self.StoredPositionVelocityRidges)-1:
                    self.StoredPositionVelocityRidges[replaceId] = self.SelectedPositionVelocityRidge
                elif self.SelectedPositionVelocityRidge not in self.StoredPositionVelocityRidges:
                    #self.StoredPositionVelocityRidges.append(self.SelectedPositionVelocityRidge) #<TODO># Store all drawn pv ridges
                    self.StoredPositionVelocityRidges = [self.SelectedPositionVelocityRidge] #<TODO># Just store one pv ridge
                self.PositionVelocityRidgeSelectedSignal.emit(self.SelectedPositionVelocityRidge)
            return True
        return False
    
    def unsetSelectedPositionVelocityRidge(self):
        print('QFitsImageWidget::unsetSelectedPositionVelocityRidge()')
        self.SelectedPositionVelocityRidge = None
        return True
    
    def mplMousePressEvent(self, event):
        if self.axes is not None and self.dataimage is not None:
            print('QFitsImageWidget::mplMousePressEvent()', 'event.button: %s, event.x: %s, event.y: %s, event.xdata:%s, event.ydata:%s'%(
                event.button, event.x, event.y, event.xdata, event.ydata))
            #print('QFitsImageWidget::mplMousePressEvent()', 'self.PreviouslySelectedPixels: %s'%(
            #    self.PreviouslySelectedPixels))
            # 
            # if click in valid data area
            if event.xdata is not None and event.ydata is not None:
                # If in Polygon mode, right click once to enter the polygon editing mode, left click to add pixels, 
                # then right click again to end the polygon editing mode. 
                # Currently we can only add pixels in the editing mode. 
                # If left click while not in the editing mode, then select a region from the store. 
                if self.SelectingMode == self.PolygonMode:
                    if event.button == MouseButton.RIGHT:
                        if self.PreviouslySelectedPixels is None:
                            self.unsetSelectedRegion()
                            self.PreviouslySelectedPixels = [(event.xdata, event.ydata)]
                        else:
                            self.PreviouslySelectedPixels.extend([(event.xdata, event.ydata)]) # click right button again to close drawing the polygon
                            ok = self.setSelectedRegion(self.PreviouslySelectedPixels, closed=True)
                            self.PreviouslySelectedPixels = None
                    elif event.button == MouseButton.LEFT:
                        if self.PreviouslySelectedPixels is not None:
                            # if we are in polygon selection mode, then just add a pixel to the polygon and update the plot with unclosed polygon.
                            self.PreviouslySelectedPixels.extend([(event.xdata, event.ydata)])
                            ok = self.setSelectedRegion(self.PreviouslySelectedPixels, closed=False)
                        else:
                            # if we are not in polygon selection mode, i.e., self.PreviouslySelectedPixels is None, 
                            # then left click still selects a pixel.
                            #ok = self.setSelectedPixel(event.xdata, event.ydata)
                            pix = PixCoord(x=event.xdata, y=event.ydata)
                            # then check if current position is in any region, if yes, then select that region.
                            self.unsetSelectedRegion()
                            if self.StoredRegions is not None:
                                for ireg in range(len(self.StoredRegions)-1, -1, -1):
                                    reg = self.StoredRegions[ireg] # from the last stored region to the first one
                                    print('self.StoredRegions[ireg=%d]: %s'%(ireg, str(reg)))
                                    if reg.contains(pix):
                                        ok = self.setSelectedRegion(reg, closed=True)
                                        break
                            self.PreviouslySelectedPixelCoord = pix
                # 
                # If in PositionVelocity mode, right click once to enter the position editing mode, left click to add pixels, 
                # then right click again to end the position editing mode. 
                elif self.SelectingMode == self.PositionVelocityMode: 
                    if event.button == MouseButton.RIGHT:
                        if self.PreviouslySelectedPixels is None:
                            self.unsetSelectedPositionVelocityRidge()
                            self.PreviouslySelectedPixels = [(event.xdata, event.ydata)] # TODO: how to discard the first right click position in the record?
                        else:
                            #self.PreviouslySelectedPixels.extend([(event.xdata, event.ydata)]) # click right button again to close drawing the position, but do not record the last right click position
                            ok = self.setSelectedPositionVelocityRidge(self.PreviouslySelectedPixels, closed=True)
                            self.PreviouslySelectedPixels = None
                    elif event.button == MouseButton.LEFT:
                        if self.PreviouslySelectedPixels is not None:
                            # if we are in position selection mode, then just add a pixel to the position and update the plot with unclosed position.
                            self.PreviouslySelectedPixels.extend([(event.xdata, event.ydata)])
                            ok = self.setSelectedPositionVelocityRidge(self.PreviouslySelectedPixels, closed=False)
                        else:
                            pass
                            ## if we are not in position selection mode, i.e., self.PreviouslySelectedPixels is None, 
                            ## then left click still selects a pixel.
                            ##ok = self.setSelectedPixel(event.xdata, event.ydata)
                            #pix = PixCoord(x=event.xdata, y=event.ydata)
                            ## then check if current position is in any region, if yes, then select that region.
                            #self.unsetSelectedPositionVelocityRidge()
                            #if self.StoredPositionVelocityRidges is not None:
                            #    for ireg in range(len(self.StoredPositionVelocityRidges)-1, -1, -1):
                            #        reg = self.StoredPositionVelocityRidges[ireg] # from the last stored region to the first one
                            #        print('self.StoredPositionVelocityRidges[ireg=%d]: %s'%(ireg, str(reg)))
                            #        if reg.contains(pix):
                            #            ok = self.setSelectedPositionVelocityRidge(reg, closed=True)
                            #            break
                            #self.PreviouslySelectedPixelCoord = pix
                # 
                # If in Pixel mode, then left click to select a pixel.
                elif self.SelectingMode == self.PixelMode: 
                    if event.button == MouseButton.LEFT:
                        # if left click and in Pixel mode, select a pixel
                        ok = self.setSelectedPixel(event.xdata, event.ydata)
    
    def mplMouseMoveEvent(self, event):
        if self.axes is not None and self.dataimage is not None:
            #print('QFitsImageWidget::mplMouseMoveEvent()', 'event.button: %s, event.x: %s, event.y: %s, event.xdata:%s, event.ydata:%s'%(
            #    event.button, event.x, event.y, event.xdata, event.ydata))
            # 
            # if click in valid data area
            if event.xdata is not None and event.ydata is not None:
                # if in Polygon mode
                if self.SelectingMode == self.PolygonMode:
                    # if currently mouse left button is pressed and moved, then move the polygon and update spectrum
                    if event.button == MouseButton.LEFT:
                        print('QFitsImageWidget::mplMouseMoveEvent()', 'event.button: %s, event.x: %s, event.y: %s, event.xdata:%s, event.ydata:%s'%(
                            event.button, event.x, event.y, event.xdata, event.ydata))
                        if self.SelectedRegion is not None and self.PreviouslySelectedPixelCoord is not None:
                            pix = PixCoord(x=event.xdata, y=event.ydata)
                            dx = pix.x-self.PreviouslySelectedPixelCoord.x
                            dy = pix.y-self.PreviouslySelectedPixelCoord.y
                            if isinstance(self.SelectedRegion, PolygonPixelRegion):
                                reg = PolygonPixelRegion(vertices=PixCoord(x=self.SelectedRegion.vertices.x+dx, y=self.SelectedRegion.vertices.y+dy))
                            elif isinstance(self.SelectedRegion, CirclePixelRegion):
                                reg = CirclePixelRegion(center=PixCoord(x=self.SelectedRegion.center.x+dx, y=self.SelectedRegion.center.y+dy), 
                                                        radius=self.SelectedRegion.radius)
                            else:
                                raise NotImplementedError('Region type %s is not implemented!'%(type(self.SelectedRegion)))
                            ok = self.setSelectedRegion(reg, closed=True, replaceId=self.StoredRegions.index(self.SelectedRegion))
                            self.PreviouslySelectedPixelCoord = pix
    
    def mplKeyPressEvent(self, event):
        if self.axes is not None and self.dataimage is not None:
            print('QFitsImageWidget::mplKeyPressEvent()', 'event.key: %s, event.x: %s, event.y: %s, event.xdata:%s, event.ydata:%s'%(
                event.key, event.x, event.y, event.xdata, event.ydata))
            # 
            # if click in valid data area
            if event.xdata is not None and event.ydata is not None:
                # 
                # If in Pixel mode or Polygon mode, then press V key to print the coordinate of a pixel.
                if self.SelectingMode == self.PixelMode or self.SelectingMode == self.PolygonMode: 
                    if event.key == 'v' or event.key == 'V':
                        # if press key 'v' or 'V' in Pixel mode, select a pixel and print its coordinate
                        ok = self.setSelectedPixel(event.xdata, event.ydata)
                        if self.dataimagewcs is not None:
                            try:
                                this_skycoord_RA, this_skycoord_Dec = self.dataimagewcs.wcs_pix2world([event.xdata], [event.ydata], 0) # pixel coord origin is 0,0
                                if not np.isscalar(this_skycoord_RA): this_skycoord_RA = this_skycoord_RA[0]
                                if not np.isscalar(this_skycoord_Dec): this_skycoord_Dec = this_skycoord_Dec[0]
                                print('QFitsImageWidget::mplKeyPressEvent()', 'event.key: %s, event.x: %s, event.y: %s, event.xdata:%s, event.ydata:%s, RA:%.10f, Dec:%.10f'%(
                                    event.key, event.x, event.y, event.xdata, event.ydata, this_skycoord_RA, this_skycoord_Dec))
                            except:
                                print('QFitsImageWidget::mplKeyPressEvent()', 'Error! No WCS is defined! Please call class function setImageWCS() first!')
    
    
    @pyqtSlot(QPoint)
    def onImageViewerContextMenuCall(self, point):
        self.ContextMenu.exec_(self.mapToGlobal(point))
    
    
    @pyqtSlot()
    def onActionSetColorBarLimits(self):
        logger.debug('QFitsImageWidget::onActionSetColorBarLimits()')
        if self.PlottedImshow is None:
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Information)
            msgBox.setText("No data image plotted.")
            msgBox.setWindowTitle("No data image plotted.")
            msgBox.setStandardButtons(QMessageBox.Ok)
            msgBox.exec()
            return
        else:
            vmin, vmax = self.PlottedImshow.get_clim()
            inputDialog = QInputDialogSetColorBarLimits(vmin, vmax)
            inputDialog.show()
            vmin, vmax = inputDialog.getResults()
            if vmin is not None and vmax is not None:
                self.PlottedImshow.set_clim(vmin=vmin, vmax=vmax)
                self.draw()
                self.update()
    
    
    #def setDataImage(self, DataImage):
    #    previous_data_image_width = self.dataimagewidth
    #    previous_data_image_height = self.dataimageheight
    #    self.dataimage = DataImage
    #    self.dataimagewidth = DataImage.shape[1]
    #    self.dataimageheight = DataImage.shape[0]
    #    self.datanorm = ImageNormalize(self.dataimage, interval=MinMaxInterval()) # , stretch=AsinhStretch()
    #    #self.rgb888image = np.zeros((self.dataimagewidth, self.dataimageheight, 3), dtype=np.uint8)
    #    #self.rgb888image[:,:,0] = (self.datanorm(self.dataimage).data*255).astype(np.uint8)
    #    self.uint8image = (self.datanorm(self.dataimage).data*255).astype(np.uint8)
    #    self.uint8image
    #    bytesPerLine = 1 * self.dataimagewidth
    #    self.qimage = QImage(self.uint8image, self.dataimagewidth, self.dataimageheight, bytesPerLine, QImage.Format_Grayscale8) # Format_Grayscale8 Format_RGB888 Format_RGBA8888 Format_RGB32
    #    self.qpixmap = QPixmap.fromImage(self.qimage)
    #    #self.transform = QTransform().scale(1, -1) # flip/mirror y so that image origin is at the lower left corner
    #    ok = self.setPixmap(self.qpixmap)
    #    if ok:
    #        if previous_data_image_width != self.dataimagewidth or previous_data_image_height != self.dataimageheight:
    #            # if image size changed, reset transform matrix
    #            ok = self.setDataTransform()
    #            if ok:
    #                self.update()
    #                return True
    #        else:
    #            self.update()
    #    return False 
    
    #def setDataTransform(self):
    #    debug = 0
    #    if debug >= 1:
    #        print('QFitsImageWidget::setDataTransform()', 'self.size()', self.size())
    #    self.scalefactor = min(float(self.size().width())/self.dataimagewidth, float(self.size().height())/self.dataimageheight)
    #    if debug >= 1:
    #        print('QFitsImageWidget::setDataTransform()', 'self.scalefactor', self.scalefactor)
    #    self.transform = QTransform().scale(self.scalefactor, -self.scalefactor) # flip/mirror y so that image origin is at the lower left corner
    #    if debug >= 1:
    #        print('QFitsImageWidget::setDataTransform()', 'self.transform matrix: [[%+.3e %+.3e %+.3e]'%(self.transform.m11(), self.transform.m12(), self.transform.m13()))
    #        print('QFitsImageWidget::setDataTransform()', '                        [%+.3e %+.3e %+.3e]'%(self.transform.m21(), self.transform.m22(), self.transform.m23()))
    #        print('QFitsImageWidget::setDataTransform()', '                        [%+.3e %+.3e %+.3e]'%(self.transform.m31(), self.transform.m32(), self.transform.m33()))
    #    self.transform = QTransform().scale(self.scalefactor, -self.scalefactor).translate(0, -self.dataimageheight+1.) # flip/mirror y so that image origin is at the lower left corner
    #    if debug >= 1:
    #        print('QFitsImageWidget::setDataTransform()', 'self.transform matrix: [[%+.3e %+.3e %+.3e]'%(self.transform.m11(), self.transform.m12(), self.transform.m13()))
    #        print('QFitsImageWidget::setDataTransform()', '                        [%+.3e %+.3e %+.3e]'%(self.transform.m21(), self.transform.m22(), self.transform.m23()))
    #        print('QFitsImageWidget::setDataTransform()', '                        [%+.3e %+.3e %+.3e]'%(self.transform.m31(), self.transform.m32(), self.transform.m33()))
    #    return True
    
    #def setPixmap(self, pixmap):
    #    self.pixmap = pixmap
    #    return True
    
    #def setDataPixel(self, x, y):
    #    if self.scalefactor is not None and self.dataimagewidth is not None and self.dataimageheight is not None:
    #        ix = int(round(x))
    #        iy = int(round(y))
    #        if ix>=0 and ix<=self.dataimagewidth-1 and iy>=0 and iy<=self.dataimageheight-1:
    #            datapixel = QPoint(ix, iy)
    #            if self.datapixel != datapixel:
    #                self.datapixel = QPoint(ix, iy)
    #                self.DataPixelF = QPointF(float(x), float(y))
    #                self.update()
    #                self.PixelSelectedSignal.emit(ix, iy)
    #                return True
    #    return False
    
    #def setDataPixelFromEventPos(self, pos, printpixelvalue=False):
    #    if self.scalefactor is not None and self.dataimagewidth is not None and self.dataimageheight is not None:
    #        # get clicked pixel coordinate in data image pixel coordinate system
    #        x = (pos.x()-self.origin.x())
    #        y = (pos.y()-self.origin.y())
    #        x = x/self.scalefactor - 0.5 # pixel coordinates starting from (0,0), i.e., the first pixel's center has (0,0) and bottom-left corner (-0.5,-0.5)
    #        y = self.dataimageheight - (y/self.scalefactor) - 0.5 # mirrow y; pixel coordinates starting from (0,0), i.e., the first pixel's center has (0,0) and bottom-left corner (-0.5,-0.5)
    #        ok = self.setDataPixel(x, y)
    #        if ok:
    #            return True
    #    return False
    
    #def resizeEvent(self, event):
    #    if self.pixmap is not None:
    #        self.setDataTransform()
    #    super(QLabel, self).resizeEvent(event)
    
    #def paintEvent(self, event):
    #    #print('QFitsImageWidget::paintEvent()')
    #    if self.pixmap is not None and self.transform is not None:
    #        #print('QFitsImageWidget::paintEvent()', 'self.size()', self.size())
    #        size = self.size()
    #        painter = QPainter(self)
    #        #scaledPixmap = self.pixmap.scaled(size, Qt.KeepAspectRatio, transformMode=Qt.FastTransformation)
    #        scaledPixmap = self.pixmap.transformed(self.transform, Qt.FastTransformation)
    #        self.origin = QPoint(0, 0)
    #        self.origin.setX((size.width() - scaledPixmap.width()) / 2)
    #        self.origin.setY((size.height() - scaledPixmap.height()) / 2)
    #        painter.drawPixmap(self.origin, scaledPixmap)
    #        # 
    #        if self.datapixel is not None:
    #            rect = QRectF(self.transform.map(self.datapixel)+self.origin, QSizeF(self.scalefactor,self.scalefactor))
    #            #print('QFitsImageWidget::paintEvent()', 'self.datapixel', self.datapixel, 'rect', rect)
    #            pen = QPen(Qt.green, 2, Qt.SolidLine)
    #            painter.setPen(pen)
    #            painter.drawRect(rect)
    #        # 
    #    super(QLabel, self).paintEvent(event)
    
    #def mousePressEvent(self, event):
    #    #print('QFitsImageWidget::mousePressEvent()')
    #    #print('QFitsImageWidget::mousePressEvent()', 'event.pos()', event.pos(), 'event.buttons()', event.buttons())
    #    #print('QFitsImageWidget::mousePressEvent()', 'self.origin', self.origin)
    #    # 
    #    if self.scalefactor is not None:
    #        # select pixel to plot spectrum if left click
    #        if event.buttons() == Qt.LeftButton :
    #            #print('QFitsImageWidget::mousePressEvent()', 'Left clicked')
    #            ok = self.setDataPixelFromEventPos(event.pos())
    #            # 
    #            # also print pixel value
    #            #pixval = np.nan
    #            #if ok: pixval = self.dataimage[self.datapixel.y(), self.datapixel.x()]
    #            #print('QFitsImageWidget::mousePressEvent()', 'scalefactor: %s, event.pos(): %s, data pixel: %s (%s), pixel value: %s'%(\
    #            #    self.scalefactor, event.pos(), self.DataPixelF, self.datapixel, pixval))
    #        # 
    #        if event.buttons() == Qt.RightButton :
    #            #print('QFitsImageWidget::mousePressEvent()', 'Right clicked')
    #            #if QApplication.keyboardModifiers() == Qt.ControlModifier:
    #            #    print('Ctrl')
    #            pass
    
    #def mouseMoveEvent(self, event):
    #    # 
    #    if self.scalefactor is not None:
    #        # select pixel to plot spectrum
    #        if event.buttons() == Qt.LeftButton :
    #            #print('QFitsImageWidget::mouseMoveEvent()', 'Left pressed')
    #            ok = self.setDataPixelFromEventPos(event.pos())
    #        if event.buttons() == Qt.RightButton :
    #            #print('QFitsImageWidget::mouseMoveEvent()', 'Right pressed')
    #            pass




class QSpectrumWidget(FigureCanvasQTAgg):
    
    ChannelSelectedSignal = pyqtSignal(int)
    ChunkSelectedSignal = pyqtSignal(list)
    
    class SelectingMode(Enum):
        Channel = 1
        Chunk = 2
    ChannelMode = SelectingMode.Channel
    ChunkMode = SelectingMode.Chunk
    
    def __init__(self, parent=None, width=12, height=5, title=None, name=None, tight_layout=False):
        self.title = title
        self.name = name
        self.xarray = None
        self.yarray = None
        self.channelwidth = None
        self.SelectingMode = self.ChannelMode
        self.PreviouslySelectedChannel = None
        self.PlottedChannel = [] # must be a list
        self.PlottedChunk = [] # must be a list
        # setup matplotlib figure
        self.fig = Figure(figsize=(width, height), tight_layout=tight_layout)
        self.axes = self.fig.add_subplot(111)
        super(QSpectrumWidget, self).__init__(self.fig)
        #self.axes.plot(np.arange(100), np.random.random(100)-0.5, drawstyle='steps-mid') # random initial spectrum
        #self.axes.set_xlabel('Channel')
        #self.axes.set_xticks([])
        #self.axes.set_yticks([])
        self.axes.tick_params(labelsize='small')
        self.mpl_connect('button_press_event', self.mplMousePressEvent)
        # set qt cursor style
        self.setCursor(Qt.CrossCursor)
        self.setStyleSheet('.QWidget{border: 2px solid black; border-radius: 2px; background-color: rgb(255, 255, 255);}')
        self.setMinimumSize(100, 100)
        # 
        if not tight_layout:
            #self.fig.subplots_adjust(left=0.18, right=0.98, bottom=0.13, top=0.87) #<TODO># adjust this
            self.fig.subplots_adjust(left=0.01, right=0.80, bottom=0.13, top=0.87) #<TODO># adjust this
            self.axes.yaxis.tick_right()
        # 
        if title is not None:
            self.axes.set_title(title)
    
    def setSelectingMode(self, selecting_mode):
        self.SelectingMode = selecting_mode
    
    def determinePlotLimits(self, xarray, yarray, xmargin=0.1, ymargin=0.1):
        xmin = np.nanmin(xarray)
        xmax = np.nanmax(xarray)
        xdiv = xmax-xmin
        ymin = np.nanmin(yarray)
        ymax = np.nanmax(yarray)
        ydiv = ymax-ymin
        #print('QSpectrumWidget::determinePlotLimits()', 'xlim', [xmin-xmargin*xdiv, xmax+xmargin*xdiv], 'ylim', [ymin-ymargin*ydiv, ymax+ymargin*ydiv])
        return [xmin-xmargin*xdiv, xmax+xmargin*xdiv], [ymin-ymargin*ydiv, ymax+ymargin*ydiv]
    
    def plotSpectrum(self, 
            x=None, y=None, 
            xerr=None, yerr=None, 
            xtype=None, ytype=None, 
            xunit=None, yunit=None, 
            color='C0', drawstyle='steps-mid', linestyle='solid', linewidth=1.0, label='__none__', 
            plot_zero_line=True, update_xylimits=True, 
            selected_chunk=None, selected_chunk_baseline=0.0, selected_chunk_color='yellow', selected_chunk_alpha=0.8, 
            selected_channel=None, selected_channel_color='cyan', selected_channel_alpha=0.8, 
            clear_plot=True, clear_text=True, clear_selected_chunk=True, clear_selected_channel=True, 
            **kwargs, 
            ):
        """
        selected_chunk are filled with color from zero baseline to y value.
        selected_channel is filled from ylim[0] to ylim[1].
        """
        # check axes
        if self.axes is None:
            return False
        # clear previous plot
        if clear_selected_channel:
            for pobj in self.PlottedChannel:
                pobj.remove()
            self.PlottedChannel = []
        if clear_selected_chunk:
            for pobj in self.PlottedChunk:
                pobj.remove()
            self.PlottedChunk = []
        if clear_text:
            for axitem in self.axes.texts:
                axitem.remove()
        if clear_plot:
            for axitem in self.axes.lines + self.axes.collections + self.axes.containers + self.axes.texts:
                if not clear_selected_channel and axitem in self.PlottedChannel:
                    continue
                elif not clear_selected_chunk and axitem in self.PlottedChunk:
                    continue
                elif not clear_text and axitem in self.axes.texts:
                    continue
                axitem.remove()
        # plot spectrum
        if x is not None and y is not None:
            self.xarray = np.array(x)
            self.yarray = np.array(y)
            if np.all(np.isnan(self.xarray)):
                return False
            if np.all(np.isnan(self.yarray)):
                return False
            self.channelwidth = np.abs(self.xarray[-1]-self.xarray[0]) / float(len(self.xarray)-1)
            # determine plotting limits
            if update_xylimits:
                xlim, ylim = self.determinePlotLimits(x, y)
                self.axes.set_xlim(xlim)
                self.axes.set_ylim(ylim)
            # plot x y
            if xerr is not None or yerr is not None:
                self.axes.errorbar(x, y, xerr=xerr, yerr=yerr, color=color, linestyle=linestyle, linewidth=linewidth, label=label, **kwargs)
            elif linestyle == 'none':
                self.axes.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth, label=label, **kwargs)
            else:
                self.axes.plot(x, y, color=color, drawstyle=drawstyle, linewidth=linewidth, label=label, **kwargs)
            # plot zero line
            if plot_zero_line:
                self.axes.plot(xlim, [0.0, 0.0], ls='dotted', lw=0.5, color='k')
        # 
        # check x y arrays
        if self.xarray is None or self.yarray is None:
            return False
        # 
        # highlight channel chunks
        # ('selected_chunk' contains channel index starting from 0)
        # ('selected_chunk' should contain pairs of channels)
        if selected_chunk is not None:
            self.PlottedChunk = []
            if np.isscalar(selected_chunk):
                ichan = selected_chunk
                selected_chunk = [(ichan,ichan)]
            for ichanpair in selected_chunk:
                if np.isscalar(ichanpair):
                    ichan = ichanpair
                    ichanpair = (ichan, ichan)
                else:
                    ichanpair = tuple(ichanpair)
                if len(ichanpair) >= 2:
                    ichan1 = min(ichanpair)
                    ichan2 = max(ichanpair)
                    if ichan1 < 0: ichan1 = 0
                    if ichan2 < 0: ichan2 = 0
                    if ichan1 > len(self.xarray)-1: ichan1 = len(self.xarray)-1
                    if ichan2 > len(self.xarray)-1: ichan2 = len(self.xarray)-1
                    highlighting_xarray = self.xarray[ichan1:ichan2+1]
                    highlighting_yarray = self.yarray[ichan1:ichan2+1]
                    hbar = self.axes.bar(
                        highlighting_xarray, 
                        highlighting_yarray, 
                        width=self.channelwidth, 
                        align='center', 
                        edgecolor='none', 
                        facecolor=selected_chunk_color, 
                        alpha=selected_chunk_alpha)
                    self.PlottedChunk.append(hbar)
        # 
        # highlight current channel, fill between ylim[0] to ylim[1]
        if selected_channel is not None:
            ichan = int(round(selected_channel))
            if ichan < 0: ichan = 0
            if ichan > len(self.xarray)-1: ichan = len(self.xarray)-1
            pobj = self.axes.fill_between(
                [self.xarray[ichan]-self.channelwidth/2., self.xarray[ichan]+self.channelwidth/2.], 
                [self.axes.get_ylim()[0], self.axes.get_ylim()[0]], 
                [self.axes.get_ylim()[1], self.axes.get_ylim()[1]], 
                color=selected_channel_color, 
                alpha=selected_channel_alpha)
            self.PlottedChannel.append(pobj)
        # 
        # plot axis label
        xlabel = ''
        if xtype is not None:
            xlabel = xtype
        if xunit is not None:
            xlabel = (xlabel+' '+'['+xunit+']').strip()
        if xlabel != '':
            self.axes.set_xlabel(xlabel)
        ylabel = ''
        if ytype is not None:
            ylabel = ytype
        if yunit is not None:
            ylabel = (ylabel+' '+'['+yunit+']').strip()
        if ylabel != '':
            self.axes.set_ylabel(ylabel)
        # 
        # update canvas
        self.draw()
        return True
    
    def setSelectedChannel(self, ichan):
        if self.axes is not None and self.xarray is not None and self.yarray is not None:
            # highlight selected channel
            self.plotSpectrum(selected_channel=ichan, clear_plot=False, clear_selected_chunk=False, clear_selected_channel=True)
            #self.update()
            self.ChannelSelectedSignal.emit(ichan)
            return True
        return False
    
    def setSelectedChunk(self, chunk):
        if self.axes is not None and self.xarray is not None and self.yarray is not None:
            # highlight selected channel chunk
            self.plotSpectrum(selected_chunk=chunk, clear_plot=False, clear_selected_chunk=True, clear_selected_channel=False)
            #self.update()
            self.ChunkSelectedSignal.emit(chunk)
            # also emit channel selected signal
            ichanpair = chunk[0]
            ichanleft = min(ichanpair)
            ichanright = max(ichanpair)
            ichan = int(round((ichanleft+ichanright)/2.0))
            self.ChannelSelectedSignal.emit(ichan)
            return True
        return False
    
    def mplMousePressEvent(self, event):
        if self.axes is not None and self.xarray is not None and self.yarray is not None:
            print('QSpectrumWidget::mplMousePressEvent()', 'event.button: %s, event.x: %s, event.y: %s, event.xdata:%s, event.ydata:%s'%(
                event.button, event.x, event.y, event.xdata, event.ydata))
            # if click in valid data area
            if event.xdata is not None and event.ydata is not None:
                # if left click
                if event.button == MouseButton.LEFT:
                    ichan = np.argmin(np.abs(self.xarray-event.xdata)).ravel()[0]
                    if self.SelectingMode == self.ChannelMode:
                        ok = self.setSelectedChannel(ichan)
                        return ok
                    elif self.SelectingMode == self.ChunkMode:
                        if self.PreviouslySelectedChannel is None:
                            self.PreviouslySelectedChannel = ichan
                        else:
                            ok = self.setSelectedChunk([(self.PreviouslySelectedChannel, ichan)])
                            if ok:
                                self.PreviouslySelectedChannel = None
    
    #def setSelectedChannelFromEventPos(self, pos):
    #    if self.axes is not None and self.xarray is not None and self.yarray is not None:
    #        # translate screen coordinate to data coordinate
    #        x, y = self.axes.transData.inverted().transform((pos.x(), pos.y()))
    #        print('QSpectrumWidget::setSelectedChannelFromEventPos()', 'pos: %s, x: %s, y:%s'%(pos, x, y))
    #        #ok = self.setSelectedChannel(ichan)
    #    return False
    
    #def mousePressEvent(self, event):
    #    #print('QSpectrumWidget::mousePressEvent()')
    #    #print('QSpectrumWidget::mousePressEvent()', 'event.pos()', event.pos(), 'event.buttons()', event.buttons())
    #    # 
    #    if self.axes is not None:
    #        # hightlight channel if left click
    #        if event.buttons() == Qt.LeftButton :
    #            #print('QSpectrumWidget::mousePressEvent()', 'Left clicked')
    #            #self.setSelectedChannelFromEventPos(event.pos())
    #        # 
    #        if event.buttons() == Qt.RightButton :
    #            #print('QSpectrumWidget::mousePressEvent()', 'Right clicked')
    #            pass









# 
# MAIN
# 
if __name__ == '__main__':
    
    # read user input
    #DataFile = sys.argv[1]
    
    # 
    app = QApplication(sys.argv)
    
    ScreenSize = app.primaryScreen().size()
    
    logger.debug('ScreenSize = ' + str(ScreenSize.width()) + ', ' + str(ScreenSize.height()))

    w = QDysmalPyGUI(ScreenSize=ScreenSize)
    #w.resize(250, 150)
    #w.move(300, 300)
    #w.setWindowTitle('Simple')
    w.show()

    sys.exit(app.exec_())








