Getting started with healthcare.ai
----------------------------------

What can you do with this package?
==================================
 - Fill in missing data via imputation
 - Create and compare models based on your data
 - Save a model to produce daily predictions
 - Write predictions back to a database
 - Learn what factor drives each prediction

How to install 
==============

Windows

 - If you haven't, install 64-bit Python 3.5 via `the Anaconda distribution`_
 - Open Spyder (which installed with Anaconda)
 - Run ``conda install pyodbc``
 - Run ``pip install https://github.com/HealthCatalystSLC/healthcareai-py/zipball/master``

 .. _the Anaconda distribution: https://www.continuum.io/downloads

Non-Windows

 - Download and install python 3.5 64-bit version from `here`_
 - Download `docker`_
 - To install using docker, run ``docker build -t healthcareai .``
 - Run the docker instance with ``docker run -p 8888:8888 healthcareai``
 - You should see a jupyter notebook available at ``http://localhost:8888``

.. _here: https://www.continuum.io/downloads
.. _docker: https://docs.docker.com/engine/installation/

How to help
===========

Check out our github `repo`_

.. _repo: https://github.com/HealthCatalystSLC/healthcareai-py/blob/master/README.rst