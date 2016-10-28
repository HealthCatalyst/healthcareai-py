HCPyTools
---------

The aim of ``HCPyTools`` is to make it easy to do data science with healthcare
data. The package has two main goals:

-  Allow one to easily create models based on tabular data, and deploy a best model that pushes predictions to SQL Server.

-  Provide tools related to data cleaning, manipulation, and imputation.

To get started, check out this `notebook`_.

.. _notebook: notebooks/HCPyToolsExample1.ipynb

Installation
=============

- Download and run the Python 3.5 Windows x86-64 executable installer, from https://www.python.org/downloads/windows
    - On the first screen, check 'Add to PATH'

- Clone this repo (look for the green button above)

- Install the prerequisites
    - Using Windows
        - Install the following cp3.5 and amd64 type packages (in this order)
        - Use cmd or PowerShell: ``python -m pip install path\somepackage.whl``
        - numpy from http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy
        - scipy from http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy
        - scikit-learn from http://www.lfd.uci.edu/~gohlke/pythonlibs/#scikit-learn
        - pandas http://www.lfd.uci.edu/~gohlke/pythonlibs/#pandas
        - ceODBC http://www.lfd.uci.edu/~gohlke/pythonlibs/#ceodbc
        - matplotlib http://www.lfd.uci.edu/~gohlke/pythonlibs/#matplotlib
    - Using POSIX
        - cd into the cloned directory
        - ``pip install -r dev-requirements.txt``

    - `cd` to the the cloned directory and run ``python setup.py install``
    
Getting started
=============
- Run the examples in the HCPyTools directory

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



Documentation
=============

To render docs, create a virtualenvironment for ``hcpytools``
  - ``cd`` to directory where folder was downloaded
  - Type ``python -m virtualenv healthcare``

Install required python modules
  - Type ``pip install -r dev-requirements.txt``.

Then simply run ``inv docs`` and a new browser window should open to http://127.0.0.1:8001
