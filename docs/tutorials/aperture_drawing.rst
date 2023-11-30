1D Aperture Setup
=================

Illustrations of the available 1D aperture setups from ``aperture_classes``.

Circular Aperture
-----------------
Keywords controlling the aperture shape: ``slit_pa``, ``slit_width``, ``aperture_radius``.

Set of circular apertures with a radius of ``aperture_radius`` along the slit described by ``slit_pa`` and ``slit_width``.
``slit_pa`` is the position angle in degrees from North pointing towards the blue side.
Normally, ``slit_width`` would be the FWHM of the beam size, while ``aperture_radius`` would be half of that.
If ``aperture_radius`` is not set, then the user must set a ``slit_width``.
The default center pixel is the center of the data cube.

.. image:: ../_static/dpy_apertures/circ_cp.png
  :width: 200
  :height: 200

Rectangular Aperture
---------------------
Keywords controlling the aperture shape: ``slit_pa``, ``slit_width``, or specify ``pix_perp`` and ``pix_parallel`` instead.

``pix_perp`` and ``pix_parallel`` are illustrated below.

``pix_perp`` and ``pix_parallel`` can be array objects defined by the user. In such a case, an array of the same length containing the aperture centers ``rarr`` has to be provided.
Otherwise, ``rarr`` will be taken from obs.data.rarr, which will then be treated as ``aper_centers``.

.. image:: ../_static/dpy_apertures/rect_cp.png
  :width: 200
  :height: 200

Single Pixel PV
----------------
Equivalent to Rectangular Aperture if ``pix_parallel`` is set to 1.
