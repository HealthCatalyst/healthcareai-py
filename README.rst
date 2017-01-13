healthcareai
---------

.. image::
   https://ci.appveyor.com/api/projects/status/17ap55llddwe16wy/branch/master?svg=true
   :width: 300
   :target: https://ci.appveyor.com/project/CatalystAdmin/healthcareai-py
   :alt: Appveyor build status
   
|

The aim of ``healthcareai`` is to streamline machine learning in healthcare. The package has two main goals:

-  Allow one to easily create models based on tabular data, and deploy a best model that pushes predictions to SQL Server.

-  Provide tools related to data cleaning, manipulation, and imputation.

Installation
=============

 - Using Windows
     - If you haven't, install 64-bit Python 3.5 via `the Anaconda distribution`_
     .. _the Anaconda distribution: https://www.continuum.io/downloads
     - Open the terminal (i.e., CMD or PowerShell, if using Windows)
     - run ``conda install pyodbc``
     - Upgrade to latest scipy (note that upgrade command took forever)
     - Run ``conda remove scipy``
     - Run ``conda install scipy``
     - To install the latest release, run 
     
       - ``pip install https://github.com/HealthCatalystSLC/healthcareai-py/zipball/v0.1.7-beta``
     - If you know what you're doing, and instead want the bleeding-edge version, run
       
       - ``pip install https://github.com/HealthCatalystSLC/healthcareai-py/zipball/master``
 - Using Linux / OSX (via docker)
     - Install `docker`_
     .. _docker: https://docs.docker.com/engine/installation/
     - Clone this repo (look for the green button on the repo main page)
     - cd into the cloned directory
     - run ``docker build -t healthcareai .``
     - run the docker instance with ``docker run -p 8888:8888 healthcareai`` 
     - You should then have a jupyter notebook available on ``http://localhost:8888``.

Getting started
=============
- Visit `healthcare.ai`_ to read the docs and find examples. Also, see this `notebook`_.

- Open Sphinx (which installed with Anaconda) and copy the examples into a new file

- Modify the queries and parameters to match your data

- If you plan on deploying a model (ie, pushing predictions to SQL Server), run this in SSMS beforehand:

.. _healthcare.ai: http://healthcare.ai/py/
.. _notebook: notebooks/HCPyToolsExample1.ipynb
.. code-block:: sql

   CREATE TABLE [SAM].[dbo].[HCPyDeployClassificationBASE] (
       [BindingID] [int] ,
       [BindingNM] [varchar] (255),
       [LastLoadDTS] [datetime2] (7),
       [PatientEncounterID] [decimal] (38, 0), --< change to your grain col
       [PredictedProbNBR] [decimal] (38, 2),
       [Factor1TXT] [varchar] (255),
       [Factor2TXT] [varchar] (255),
       [Factor3TXT] [varchar] (255))

   CREATE TABLE [SAM].[dbo].[HCPyDeployRegressionBASE] (
       [BindingID] [int],
       [BindingNM] [varchar] (255),
       [LastLoadDTS] [datetime2] (7),
       [PatientEncounterID] [decimal] (38, 0), --< change to your grain col
       [PredictedValueNBR] [decimal] (38, 2),
       [Factor1TXT] [varchar] (255),
       [Factor2TXT] [varchar] (255),
       [Factor3TXT] [varchar] (255))

Note that we're currently working on easy connections to other types of databases.

Contributing
=============

We welcome community contributions. See `here`_ to get started!

.. _here: https://github.com/HealthCatalystSLC/HCPyTools/blob/master/CONTRIBUTING.rst
