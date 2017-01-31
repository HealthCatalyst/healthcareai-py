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
     - Run ``conda install pyodbc``
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

For Issues
==========
- Double check that the code follows the examples at `healthcare.ai/py`_

- If you're still seeing an error, `create a post in our Google Group`_ that contains
  
  - Details on your environment (OS, database type, R vs Py)
  - Goals (ie, what are you trying to accomplish)
  - Crystal clear steps for reproducing the error
  
.. _healthcare.ai/py: http://healthcare.ai/py/
.. _create a post in our Google Group: https://groups.google.com/forum/#!forum/healthcareai-users

Contributing
=============
You want to help? Wohoo! We welcome that and are willing to help newbies get started.

First, See `here`_ for instructions on setting up your development environment

.. _here: https://github.com/HealthCatalystSLC/HCPyTools/blob/master/CONTRIBUTING.rst

After that's done, *here's the contribution workflow:*

1) `Identify an issue that`_ suits your skill level

   - Only look for issues in the Backlog category
   - If you're new to open source, please look for issues with the ``bug low``, ``help wanted``, or ``docs`` tags
   - Please reach out with questions on details and where to start
   
.. _Identify an issue that: https://github.com/HealthCatalystSLC/healthcareai-r/issues

2) Create a topic branch to work in; here are `instructions`_.

.. _instructions: CONTRIBUTING.rst#create-a-topic-branch-that-you-can-work-in

3) Create a throwaway file on the Desktop (or somewhere outside the repo), based on an example

4) Make changes and use the throwaway file to make sure your packages changes work
   
   - Make small commits after getting a small piece working
   - Push often so your changes are backed up. See `this`_ for more
     
.. _this: https://gist.github.com/blackfalcon/8428401#push-your-branch

5) Early on, create a `pull request`_ such that Levi and team can discuss the changes that you're making. Conversation is good.

.. _pull request: https://yangsu.github.io/pull-request-tutorial/

6) When you're done with the issue you chose, do the following
   
   - Check that the unit tests are passing
   - Check that pyflakes and pylint don't show any issues
   - Merge the master branch into your topic branch (so that you have the latest changes from master)
   
   .. code-block:: git

      git checkout LeviBugFix
      git fetch
      git merge --no-ff origin/master
      
   - Again, check that the unit tests are passing
   
7) Now that your changes are working, communicate that to Levi in the pull request, such that he knows to do the code review associated with the PR. Please *don't* do tons of work and *then* start a PR. Early is good.
