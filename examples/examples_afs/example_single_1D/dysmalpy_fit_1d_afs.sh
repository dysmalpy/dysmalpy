#!/bin/bash

# Usage: dysmalpy_fit_1d_afs.sh fitting_1D_mpfit.params

# Setup paths on AFS:
source /afs/mpe.mpg.de/astrosoft/dysmalpy/dysmalpy_setup.sh

# Run fitting
export DPY_PATH='/afs/mpe.mpg.de/astrosoft/dysmalpy/dysmalpy'

python $DPY_PATH/dysmalpy/fitting_wrappers/dysmalpy_fit_single_1D.py $1
