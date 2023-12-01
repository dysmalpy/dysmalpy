1D Aperture Setup
=================

Illustrations of the available 1D aperture setups using the :code:`aperture_classes` module.

Circular Aperture
-----------------
Keywords controlling the aperture shape: :code:`slit_pa`, :code:`slit_width`, :code:`aperture_radius`.

This setup consists of circular apertures with a radius specified by :code:`aperture_radius`. The apertures are positioned along the slit, which is defined by the parameters :code:`slit_pa` and :code:`slit_width`. The angle :code:`slit_pa` represents the position angle in degrees from North pointing towards the blue side. Typically, :code:`slit_width` corresponds to the FWHM of the beam size. If :code:`aperture_radius` is not set, the user must provide a value for :code:`slit_width`. The default center pixel is the center of the data cube.

.. image:: ../_static/dpy_apertures/circ_cp.png
  :width: 200
  :height: 200

Rectangular Aperture
---------------------
Keywords controlling the aperture shape: :code:`slit_pa`, :code:`slit_width`, or alternatively, specify :code:`pix_perp` and :code:`pix_parallel`.

In this configuration, rectangular apertures are utilized. The dimensions of the rectangles are controlled by either the parameters :code:`slit_pa` and :code:`slit_width` or by directly specifying :code:`pix_perp` (semi-major axis) and :code:`pix_parallel` (semi-minor axis). These dimensions are illustrated below.

The user has the flexibility to define arrays for :code:`pix_perp` and :code:`pix_parallel`. If array objects are provided, ensure they match the length of the aperture centers specified in :code:`rarr`. Otherwise, the aperture centers will default to the values from :code:`obs.data.rarr`.

.. image:: ../_static/dpy_apertures/rect_cp.png

:width: 200
:height: 200

The user has the flexibility to define arrays for :code:`pix_perp` and :code:`pix_parallel`. If array objects are provided, ensure they match the length of the aperture centers specified in :code:`rarr`. Otherwise, the aperture centers will default to the values from :code:`obs.data.rarr`.

Single Pixel PV
----------------
This configuration is equivalent to a Rectangular Aperture when :code:`pix_parallel` is set to 1.
