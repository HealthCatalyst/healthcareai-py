.. HCPyTools documentation master file, created by
   sphinx-quickstart on Wed Oct 19 16:34:01 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the healthcareai documentation!
-------------------------------------------------

This package will get you started with healthcare machine learning

What can you do with it?
========================
 - Fill in missing data via imputation
 - Create and compare models based on your data
 - Save a model to produce daily predictions
 - Write predictions back to a database
 - Learn what factor drives each prediction

How to install 
==============

Windows

 - Download and install 64-bit python 3.5 from `here`_
 - ``pip install --upgrade https://github.com/HealthCatalystSLC/healthcareai-py/zipball/master``

.. _here: https://www.continuum.io/downloads

Non-Windows

 - Download and install python 3.5 64-bit version from `here`_
 - Download `docker`_
 - To install using docker, run ``docker build -t healthcareai .``
 - Run the docker instance with ``docker run -p 8888:8888 healthcareai``
 - You should see a jupyter notebook available at ``http://localhost:8888``

.. _here: https://www.continuum.io/downloads
.. _docker: https://docs.docker.com/engine/installation/


.. toctree::
   :maxdepth: 1

   begin
   FAQ
   develop
   deploy
