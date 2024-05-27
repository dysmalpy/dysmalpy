"""
Provides functions for altering colormaps to make them more suitable for
particular use cases.

Originally written by Chris J. White (https://github.com/c-white);
used with permission.
Slightly modified for compatibility with python/matplotlib updates.

From Drummond Fielding and Chris White GSPS talk, 24 March 2017

"""

import matplotlib.cm as cm
import matplotlib as mpl
import matplotlib.colors as colors
import numpy as np

def new_sequential_cmap(name_original, gamma, bad=None, over=None, under=None,
    name_new=None):
  """
  Creates a stretched version of the given colormap.

  Inputs:

    name_original: String naming existing colormap

    gamma: Stretch parameter. Must be positive. Values greater than 1 compress
    colors near bottom of range, providing more color resolution there. Values
    less than 1 do the same near top of range.

    bad, over, under: Colors to be used for invalid values, values above the
    upper limit, and values below the lower limit. Default to values from
    original map.

    name_new: String under which new colormap will be registered. Defaults to
    prepending 'New' to original name.

  Returns new colormap.
  """
  
  # Get original colormap
  try:
      cmap_original_data = cm.datad[name_original]
  except:
      cmap_original_data = cm.datad[name_original.split('_r')[0]][::-1]
  cmap_original = mpl.colormaps[name_original]

  # Define new colormap data
  cmap_new_data = {}
  for color in ('red', 'green', 'blue'):
    try:
        cmap_new_data[color] = np.copy(cmap_original_data[color])
        cmap_new_data[color][:,0] **= gamma
    except:
        cmap_new_data[color] = np.array(cmap_original._segmentdata[color])
        cmap_new_data[color][:,0] **= gamma

  # Create new colormap
  if name_new is None:
    name_new = 'New' + name_original
  cmap_new = colors.LinearSegmentedColormap(name_new, cmap_new_data)
  bad = cmap_original(np.nan) if bad is None else bad
  over = cmap_original(np.inf) if over is None else over
  under = cmap_original(-np.inf) if under is None else under
  cmap_new.set_bad(bad)
  cmap_new.set_over(over)
  cmap_new.set_under(under)

  # Register and return new colormap
  mpl.colormaps.register(name=name_new, cmap=cmap_new)
  return cmap_new

def new_diverging_cmap(name_original, diverge=0.5, gamma_lower=1.0,
    gamma_upper=1.0, excise_middle=False, bad=None, over=None, under=None,
    name_new=None):
  """
  Creates a recentered and/or stretched and/or sharper version of the given
  colormap.

  Inputs:

    name_original: String naming existing colormap, which should be diverging
    and must have an anchor point at 0.5.

    diverge: Location of new center from which colors diverge. Defaults to 0.5.

    gamma_lower, gamma_upper: Stretch parameters for values below and above the
    diverging point. Must be positive. Values greater than 1 compress colors
    near the diverging point, providing more color resolution there. Values less
    than 1 do the same at the extremes of the range. Default to no stretching.

    excise_middle: Flag indicating the middle point should be removed, with the
    two color ranges joined sharply instead. Defaults to False.

    bad, over, under: Colors to be used for invalid values, values above the
    upper limit, and values below the lower limit. Default to values from
    original map.

    name_new: String under which new colormap will be registered. Defaults to
    prepending 'New' to original name.

  Returns new colormap.
  """

  # Get original colormap
  #cmap_original_data = cm.datad[name_original]
  try:
      cmap_original_data = cm.datad[name_original]
  except:
      cmap_original_data = cm.datad[name_original.split('_r')[0]][::-1]
  cmap_original = mpl.colormaps[name_original]

  # Define new colormap data
  cmap_new_data = {}
  for color in ('red', 'green', 'blue'):

    # Get original definition
    new_data = np.array(cmap_original._segmentdata[color])
    midpoint = np.where(new_data[:,0] == 0.5)[0][0]

    # Excise middle value if desired
    if excise_middle:
      anchor_lower = new_data[midpoint-1,0]
      anchors_lower = new_data[:midpoint,0]
      anchors_lower = 1.0/(2.0*anchor_lower) * anchors_lower
      anchor_upper = new_data[midpoint+1,0]
      anchors_upper = new_data[midpoint+1:,0]
      anchors_upper = 1.0/(2.0-2.0*anchor_upper) * (anchors_upper-1.0) + 1.0
      anchors = np.concatenate((anchors_lower[:-1], [0.5], anchors_upper[1:]))
      vals_below = \
          np.concatenate((new_data[:midpoint,1], new_data[midpoint+2:,1]))
      vals_above = \
          np.concatenate((new_data[:midpoint-1,2], new_data[midpoint+1:,2]))
      new_data = np.vstack((anchors, vals_below, vals_above)).T
      midpoint -= 1

    # Apply shift and stretch if desired
    anchors_lower = new_data[:midpoint,0]
    if diverge != 0.5 or gamma_lower != 1.0:
      anchors_lower = diverge * (1.0 - (1.0-2.0*anchors_lower) ** gamma_lower)
    anchors_upper = new_data[midpoint+1:,0]
    if diverge != 0.5 or gamma_upper != 1.0:
      anchors_upper = \
          diverge + (1.0-diverge) * (2.0*anchors_upper-1.0) ** gamma_upper
    anchors = np.concatenate((anchors_lower, [diverge], anchors_upper))
    anchors[0] = 0.0
    anchors[-1] = 1.0
    new_data[:,0] = anchors

    # Record changes
    cmap_new_data[color] = new_data

  # Create new colormap
  if name_new is None:
    name_new = 'New' + name_original
  cmap_new = colors.LinearSegmentedColormap(name_new, cmap_new_data)
  bad = cmap_original(np.nan) if bad is None else bad
  over = cmap_original(np.inf) if over is None else over
  under = cmap_original(-np.inf) if under is None else under
  cmap_new.set_bad(bad)
  cmap_new.set_over(over)
  cmap_new.set_under(under)

  # Register and return new colormap
  mpl.colormaps.register(name=name_new, cmap=cmap_new)
  return cmap_new
