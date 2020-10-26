#!/bin/bash

# Usage: dysmalpy_fit_3d_afs.sh fitting_3D_mpfit.params

# Setup paths on AFS:
source /afs/mpe.mpg.de/astrosoft/dysmalpy/dysmalpy_setup.sh

# Add fitting_wrappers to path:
export PYTHONPATH="/afs/mpe.mpg.de/astrosoft/dysmalpy/fitting_wrappers/:$PYTHONPATH"

# Run fitting
export DPY_PATH='/afs/mpe.mpg.de/astrosoft/dysmalpy/dysmalpy'

python $DPY_PATH/dysmalpy/fitting_wrappers/dysmalpy_fit_single_3D.py $1
