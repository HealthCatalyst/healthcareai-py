# Getting started with healthcare.ai

## What can you do with this package?

- Fill in missing data via imputation
- Create and compare models based on your data
- Save a model to produce daily predictions
- Write predictions back to a database
- Learn what factor drives each prediction

## Installation

### Windows

- If you haven't, install 64-bit Python 3.5 via [the Anaconda distribution](https://www.continuum.io/downloads)
- Open the terminal (i.e., CMD or PowerShell, if using Windows)
- Run `conda install pyodbc`
- Upgrade to latest scipy (note that upgrade command took forever)
- Run `conda remove scipy`
- Run `conda install scipy`
- To install the latest release, run 
    * `pip install healthcareai`
- If you know what you're doing, and instead want the bleeding-edge version direct from our github repo, run
    * `pip install https://github.com/HealthCatalystSLC/healthcareai-py/zipball/master`
 
### Linux

You may need to install the following dependencies:
- `sudo apt-get install python-tk`
- `sudo pip install pyodbc`
    - Note you'll might run into trouble with the `pyodbc` dependency. You may first need to run `sudo apt-get install unixodbc-dev` then retry `sudo pip install pyodbc`. Credit [stackoverflow](http://stackoverflow.com/questions/2960339/unable-to-install-pyodbc-on-linux)

Once you have the dependencies satisfied run `pip install healthcareai` or `sudo pip install healthcareai`

### macOS

- `pip install healthcareai` or `sudo pip install healthcareai`

### Linux and macOS (via docker)
 
- Install [docker](https://docs.docker.com/engine/installation/)
- Clone this repo (look for the green button on the repo main page)
- cd into the cloned directory
- run `docker build -t healthcareai .`
- run the docker instance with `docker run -p 8888:8888 healthcareai` 
- You should then have a jupyter notebook available on `http://localhost:8888`.

### Verify Installation

To verify that *healthcareai* installed correctly, open a terminal and run `python`. This opens an interactive python console (also known as a [REPL](https://en.wikipedia.org/wiki/Read%E2%80%93eval%E2%80%93print_loop)). Then enter this command: `from healthcareai import develop_supervised_model` and hit enter. If no error is thrown, you are ready to rock.

If you did get an error, or run into other installation issues, please [let us know](http://healthcare.ai/contact.html) or better yet post on [Stack Overflow](http://stackoverflow.com/questions/tagged/healthcare-ai)(with the healthcare-ai tag) so we can help others along this process.

## Getting started

- Visit [healthcare.ai](healthcare.ai/py) to read the docs and find examples.
    * Including this [notebook](healthcare.ai/notebooks/Example1.ipynb)
- Open Sphinx (which installed with Anaconda) and copy the examples into a new file
- Modify the queries and parameters to match your data
- If you plan on deploying a model (ie, pushing predictions to SQL Server), run this in SSMS beforehand:
  ```sql
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
  ```
Note that we're currently working on easy connections to other types of databases.

## For Issues

- Double check that the code follows the examples at [healthcare.ai/py](http://healthcare.ai/py/)
- If you're still seeing an error, [create a post in our Google Group](https://groups.google.com/forum/#!forum/healthcareai-users) that contains
    * Details on your environment (OS, database type, R vs Py)
    * Goals (ie, what are you trying to accomplish)
    * Crystal clear steps for reproducing the error
