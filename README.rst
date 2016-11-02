healthcareai
---------

.. image::
   https://ci.appveyor.com/api/projects/status/17ap55llddwe16wy/branch/master?svg=true
   :width: 300
   :target: https://ci.appveyor.com/project/CatalystAdmin/hcpytools
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
     - In CMD or PowerShell, run ``conda install pyodbc``
     - In CMD or PowerShell, run
        - ``pip install https://github.com/HealthCatalystSLC/healthcareai-py/zipball/master``
 - Using POSIX
     - Clone this repo (look for the green button on the repo main page)
     - cd into the cloned directory
     - ``pip install -r dev-requirements.txt``
     - `cd` to the the cloned directory and run ``python setup.py install``

Getting started
=============
- Run the examples in the HCPyTools directory and check out this `notebook`_.

.. _notebook: notebooks/HCPyToolsExample1.ipynb

- Modify the queries and parameters to match your data

- If you plan on deploying a model (ie, pushing predictions to SQL Server), run this in SSMS beforehand:

   CREATE TABLE [SAM].[dbo].[HCPyDeployClassificationBASE] (
       [BindingID] [int] ,
       [BindingNM] [varchar] (255),
       [LastLoadDTS] [datetime2] (7),
       [PatientEncounterID] [decimal] (38, 0),
       [PredictedProbNBR] [decimal] (38, 2),
       [Factor1TXT] [varchar] (255),
       [Factor2TXT] [varchar] (255),
       [Factor3TXT] [varchar] (255))

   CREATE TABLE [SAM].[dbo].[HCPyDeployRegressionBASE] (
       [BindingID] [int],
       [BindingNM] [varchar] (255),
       [LastLoadDTS] [datetime2] (7),
       [PatientEncounterID] [decimal] (38, 0),
       [PredictedValueNBR] [decimal] (38, 2),
       [Factor1TXT] [varchar] (255),
       [Factor2TXT] [varchar] (255),
       [Factor3TXT] [varchar] (255))

Contributing
=============

We welcome community contributions. See `here`_ to get started!

.. _here: https://github.com/HealthCatalystSLC/HCPyTools/blob/master/CONTRIBUTING.rst

Documentation
=============

To render docs, create a virtualenvironment for ``hcpytools``
  - ``cd`` to directory where folder was downloaded
  - Type ``python -m virtualenv healthcare``

Install required python modules
  - Type ``pip install -r dev-requirements.txt``.

For Windows
 - Run ``sphinx-autobuild docs docs/_build/html`` in the root of the repo
 - Open a browser to http://127.0.0.1:8000

For non-Windows:
 - Simply run ``inv docs`` and a new browser window should open to http://127.0.0.1:8001

Installation
============

Docker
++++++

To install using docker, run ``docker build -t healthcareai .``
Then you can run the docker instance with ``docker run -p 8888:8888 healthcareai`` 
You should then have a jupyter notebook available on ``http://localhost:8888``.

Docker Compose
++++++++++++++

With ``docker-compose`` you can spin up a jupyter application and a database instance
for local development. This is useful for one-off development questions requiring a
database.
