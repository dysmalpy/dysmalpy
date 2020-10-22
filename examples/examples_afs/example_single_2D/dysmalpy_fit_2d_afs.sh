#!/bin/bash

# Usage: dysmalpy_fit_2d_afs.sh fitting_2D_mpfit.params

# Setup paths on AFS:
source /afs/mpe.mpg.de/astrosoft/dysmalpy/dysmalpy_setup.sh

# Run fitting
export DPY_PATH='/afs/mpe.mpg.de/astrosoft/dysmalpy/dysmalpy'

python $DPY_PATH/fitting_wrappers/dysmalpy_fit_single_2D.py $1



