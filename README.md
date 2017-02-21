# healthcareai

- Windows Build Status: ![Appveyor build status](https://ci.appveyor.com/api/projects/status/17ap55llddwe16wy/branch/master?svg=true)
- PyPI Package: [![PyPI version](https://badge.fury.io/py/healthcareai.svg)](https://badge.fury.io/py/healthcareai)
- MIT Licensed: [![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/HealthCatalystSLC/healthcareai-py/master/LICENSE)

The aim of **healthcareai** is to streamline machine learning in healthcare. The package has two main goals:

-  Allow one to easily create models based on tabular data, and deploy a best model that pushes predictions to SQL Server.
-  Provide tools related to data cleaning, manipulation, and imputation.

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

## Contributing

You want to help? Woohoo! We welcome that and are willing to help newbies get started.

Please see [our contribution guidelines](https://github.com/HealthCatalystSLC/healthcareai-py/blob/master/CONTRIBUTING.md) for instructions on setting up your development environment

### Workflow

1. [Identify an issue that](https://github.com/HealthCatalystSLC/healthcareai-r/issues) suits your skill level
    * Only look for issues in the Backlog category
    * If you're new to open source, please look for issues with the `bug low`, `help wanted`, or `docs` tags
    * Please reach out with questions on details and where to start
2. Create a topic branch to work in; here are [instructions](CONTRIBUTING.md#create-a-topic-branch-that-you-can-work-in)
3. Create a throwaway file on the Desktop (or somewhere outside the repo), based on an example
4. Make changes and use the throwaway file to validate that your packages changes work
    * Make small commits after getting a small piece working
    * Push often so your changes are backed up. See [this](https://gist.github.com/blackfalcon/8428401#push-your-branch) for more details.
5. Early on, create a [pull request](https://yangsu.github.io/pull-request-tutorial/) such that Levi and team can discuss the changes that you're making. Conversation is good.
6. When you have resolved the issue you chose, do the following:
    * Check that the unit tests are passing
    * Check that pyflakes and pylint don't show any issues
    * Merge the master branch into your topic branch (so that you have the latest changes from master)
        ```bash
        git checkout LeviBugFix
        git fetch
        git merge --no-ff origin/master
        ```
    * Again, check that the unit tests are passing
7. Now that your changes are working, communicate that to Levi in the pull request, such that he knows to do the code review associated with the PR. Please *don't* do tons of work and *then* start a PR. Early is good.

## PyPI Package Creation and Updating

**Note these instructions are for maintainers only.**

First, read this [Packaging and Distributing Projects](https://packaging.python.org/distributing/) guide.

It's also worth noting that while this *should* be done on the [pypi test site](https://testpypi.python.org/pypi), I've run into a great deal of trouble with conflicting guides authenticating to the test site. So be smart about this.

1. **Build a source distribution**: from python3 (ran in windows anaconda python 3) run `python setup.py sdist`
2. **Register the package** by using the [form on pypi](https://pypi.python.org/pypi?%3Aaction=pkg_edit&name=healthcareai). Upload your `PKG-INFO` that was generated inside the `.egg` file.
3. **Upload the package** using [twine](https://pypi.python.org/pypi/twine)
    - `twine upload dist/healthcareai-<version>.tar.gz`
    - **NOTE** You can only ever upload a file name **once**. To get around this I was adding a *rc* number to the version in `setup.py`. However, this **will break the appveyor build**, so you'll need to remove the `.rc` before you push to github.
4. Verify install on all three platforms (linux, macOS, windows) by first `pip uninstall healthcareai` and then `pip install healthcareai`, followed by a `from healthcareai import develop_supervised_model` in a python REPL.