1D aperture setup
============
Illustrations of the available 1D aperture setup from `aperture_classes`.

The available options that users can specify under `profile1d_type` are: `circ_ap_cube`, `rect_ap_cube` and `single_pix_pv`.

Circular aperture:
Keywords controlling the aperture shape: `slit_pa`, `slit_width`, `aperture_radius`.
Set of circular apertures with radius of `aperture_radius` along the slit described by `slit_pa` and `slit_width`.
`slit_pa` is the position angle in unit of degrees from North pointing towards the blue side.
Normally `slit_width` would be the FWHM of the beam size, while `aperture_radius` would be half of that.
If `aperture_radius` is not set, then the user must set a `slit_width`.
The default center pixel is the centre of the data cube.


.. image:: ../_static/dpy_apertures/circ_cp.pdf

Retangular Aperture
Keywords controlling the aperture shape: `slit_pa`, `slit_width` or specify `pix_perp` and `pix_parallel` instead.
`pix_perp` and `pix_parallel` are illustrated below.

`pix_perp` and pix_parallel` can be array object define by user, in such case an array of the same length containing the apertures centers `rarr` has to be provided.
Otherwise, `rarr` will be taken from obs.data.rarr, which then be treated as `aper_centers`.

.. image:: ../_static/dpy_apertures/rect_cp.pdf

Single pixel PV
Equivalent to Rectangular Aperture if `pix_parallel` is set to 1.
