# Getting started with healthcare.ai

## What can you do with these tools?

- Fill in missing data via imputation
- Train and compare models based on your data
- Save a model to produce daily predictions
- Write predictions back to a database or csv file
- Learn what factor drives each prediction

## Installation

### Windows

- If you haven't, install 64-bit Python 3.5 via [the Anaconda distribution](https://www.continuum.io/downloads)
- Open the terminal (i.e., CMD or PowerShell, if using Windows)
- Run `conda install pyodbc`
- Upgrade to latest scipy (note that upgrade command took forever)
- Run `conda remove scipy`
- Run `conda install scipy`
- Run `conda install scikit-learn`
   Install healthcareai using **one and only one** of these three methods (ordered from easiest to hardest).
     1. **Recommended:** Install the latest release with conda by running `conda install -c catalyst healthcareai`
     2. Install the latest release with pip run `pip install healthcareai`
     3. If you know what you're doing, and instead want the bleeding-edge version direct from our github repo, run `pip install https://github.com/HealthCatalystSLC/healthcareai-py/zipball/master`

#### Why Anaconda?

We recommend using the Anaconda python distribution when working on Windows. There are a number of reasons:
- When running anaconda and installing packages using the `conda` command, you don't need to worry about [dependency hell](https://en.wikipedia.org/wiki/Dependency_hell), particularly because packages aren't compiled on your machine; `conda` installs pre-compiled binaries.
- A great example of the pain the using `conda` saves you is with the python package **scipy**, which, by [their own admission](http://www.scipy.org/scipylib/building/windows.html) *"is difficult"*

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

To verify that *healthcareai* installed correctly, open a terminal and run `python`. This opens an interactive python console (also known as a [REPL](https://en.wikipedia.org/wiki/Read%E2%80%93eval%E2%80%93print_loop)). Then enter this command: `from healthcareai import SupervisedModelTrainer` and hit enter. If no error is thrown, you are ready to rock.

If you did get an error, or run into other installation issues, please [let us know](http://healthcare.ai/contact.html) or better yet post on [Stack Overflow](http://stackoverflow.com/questions/tagged/healthcare-ai)(with the healthcare-ai tag) so we can help others along this process.

## Getting started

- Read through the docs on this site
    * If you like Jupyter notebooks, [see here](https://github.com/HealthCatalystSLC/healthcareai-py/blob/master/notebooks/Example1.ipynb)
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
Note that there are examples that write to other databases (MySQL, SQLite)

## For Issues

- Double check that the code follows the examples in these documents.
- If you're still seeing an error, file an issue on [Stack Overflow](http://stackoverflow.com/) using the healthcare-ai tag. Please provide
  - Details on your environment (OS, database type, R vs Py)
  - Goals (ie, what are you trying to accomplish)
  - Crystal clear steps for reproducing the error
