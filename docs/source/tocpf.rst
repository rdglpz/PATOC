Total-Operating-Characteristic Curve's Probability Functions
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This project consist of classes and methods to compute and plot a TOC curve, as well as the computation of probability functions suitable for analyzing the relationship between feature values and hits (true positives).
The operations include, normalization, approximation by vectors (means of subsets of data).


TOCPF class
-------------
.. autoclass:: tocpf::TOCPF


Area
====
.. autoattribute:: tocpf::TOCPF.area

Area Ratio
==========
.. autoattribute:: tocpf::TOCPF.areaRatio

Kind of TOC curve
=================
.. autoattribute:: tocpf::TOCPF.kind

Indices of the sorted rank
==========================
.. autoattribute:: tocpf::TOCPF.isorted

Number of data
===============
.. autoattribute:: tocpf::TOCPF.ndata

Number of positive instances (presence)
=======================================
.. autoattribute:: tocpf::TOCPF.np

Number of coordinates in a discretized TOC
===========================================
.. autoattribute:: tocpf::TOCPF.ndiscretization

Array of Hits plus False Alarms
====================================
.. autoattribute:: tocpf::TOCPF.HpFA

Array of Hits
=======================
.. autoattribute:: tocpf::TOCPF.Hits

Thresholds used for the TOC computation
=======================================
.. autoattribute:: tocpf::TOCPF.Thresholds

Array of Hits plus False Alarms
======================================
.. autoattribute:: tocpf::TOCPF.dHpFA

Array of Hits
=======================
.. autoattribute:: tocpf::TOCPF.dHits


Array of Hits plus False Alarms in the smoothed TOC
=====================================================
.. autoattribute:: tocpf::TOCPF.smoothHpFA

Array of Hits in the smoothed TOC
==========================================
.. autoattribute:: tocpf::TOCPF.smoothHits


Presence-class data proportion
==============================
.. autoattribute:: tocpf::TOCPF.PDataProp

Constructor
=======================
.. automethod:: tocpf::TOCPF.__init__








.. Plotting the TOC curve
.. ======================
.. .. automethod:: tocpf::TOCPF.plot
..
.. Compute a probability from a rank value
.. =======================================
.. .. automethod:: tocpf::TOCPF.rank2prob
..
.. Increasing the TOC data by interpolation
.. =========================================
.. .. automethod:: tocpf::TOCPF.interpolation



.. TOCData class
.. --------------
..
.. This class is useful to translate longitude and latitud coordinates and values to a raster object (a rasterio file).
.. The intende use is to translate probabilities from the TOC density to a georeferenced raster object.
..
..
.. .. autoclass:: tocdata::TOCData
.. ..    :members:  DeltaLat,DeltaLon, lenLat, lenLon, maxLat, maxLon, minLat, minLon, nrow, ncol, raster, raster_file
..
..
.. Pixel size in latitude and longitude units
.. ==========================================
.. .. autoattribute:: tocdata::TOCData.DeltaLat
.. .. autoattribute:: tocdata::TOCData.DeltaLon
..
.. Dimensions of the raster in latitude and longitude units
.. ========================================================
.. .. autoattribute:: tocdata::TOCData.lenLat
.. .. autoattribute:: tocdata::TOCData.lenLon
..
.. Bounding box of the raster in latitude and longitude units
.. ==========================================================
.. .. autoattribute:: tocdata::TOCData.maxLat
.. .. autoattribute:: tocdata::TOCData.maxLon
.. .. autoattribute:: tocdata::TOCData.minLat
.. .. autoattribute:: tocdata::TOCData.minLon
..
.. Number of rows and cols of the raster
.. ======================================
.. .. autoattribute:: tocdata::TOCData.nrow
.. .. autoattribute:: tocdata::TOCData.ncol
..
.. Raster matrix
.. ==============
.. .. autoattribute:: tocdata::TOCData.raster
..
.. Raster file
.. ============
.. .. autoattribute:: tocdata::TOCData.raster_file
..
..
..
..
..
..
