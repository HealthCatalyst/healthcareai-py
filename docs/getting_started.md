# Getting Started With Healthcare.ai

## What Can You Do With These Tools?

- Fill in missing data via imputation
- Train and compare models based on your data
- Save a model to produce daily or batch predictions
- Write daily or batch predictions back to a database or csv file
- Learn which factors drive each prediction

## Installation

### Windows

- If you haven't, install 64-bit Python 3.6 via [the Anaconda distribution](https://www.continuum.io/downloads)
- Open the terminal (i.e., CMD or PowerShell, if using Windows)
- **Optional** If you intend to work with MSSQL databases, run `conda install pyodbc`
- Upgrade to latest scipy
- Run `conda remove scipy`
- Run `conda install scipy`
- Run `conda install scikit-learn`
- Install healthcareai using **one and only one** of these three methods (ordered from easiest to hardest).
     1. **Recommended:** Install the latest release with conda by running `conda install -c catalyst healthcareai`
     2. Install the latest release with pip run `pip install healthcareai`
     3. If you know what you're doing, and instead want the bleeding-edge version direct from our github repo, run `pip install https://github.com/HealthCatalyst/healthcareai-py/zipball/master`

#### Why Anaconda?

We recommend using the Anaconda python distribution when working on Windows. There are a number of reasons:

- When running anaconda and installing packages using the `conda` command, you don't need to worry about [dependency hell](https://en.wikipedia.org/wiki/Dependency_hell), particularly because packages aren't compiled on your machine; `conda` installs pre-compiled binaries.
- A great example of the pain the using `conda` saves you is with the python package **scipy**, which, by [their own admission](http://www.scipy.org/scipylib/building/windows.html) *"is difficult"*

### Linux

You may need to install the following dependencies:
- `sudo apt-get install python-tk`
- **Optional** If you intend to work with MSSQL databases, run `sudo pip install pyodbc`
    - Note you'll might run into trouble with the `pyodbc` dependency. You may first need to run `sudo apt-get install unixodbc-dev` then retry `sudo pip install pyodbc`. Credit [stackoverflow](http://stackoverflow.com/questions/2960339/unable-to-install-pyodbc-on-linux)

- Once you have the dependencies satisfied install healthcareai using **one and only one** of these three methods (ordered from easiest to hardest).
     1. **Recommended:** Install the latest release with conda by running `conda install -c catalyst healthcareai`
     2. Install the latest release with pip run `pip install healthcareai` or or `sudo pip install healthcareai`
     3. If you know what you're doing, and instead want the bleeding-edge version direct from our github repo, run `pip install https://github.com/HealthCatalyst/healthcareai-py/zipball/master`

### macOS

- Install healthcareai using **one and only one** of these three methods (ordered from easiest to hardest).
     1. **Recommended:** Install the latest release with conda by running `conda install -c catalyst healthcareai`
     2. Install the latest release with pip run `pip install healthcareai` or or `sudo pip install healthcareai`
     3. If you know what you're doing, and instead want the bleeding-edge version direct from our github repo, run `pip install https://github.com/HealthCatalyst/healthcareai-py/zipball/master`

### Linux and macOS (via docker)

- Install [docker](https://docs.docker.com/engine/installation/)
- Clone this repo (look for the green button on the repo main page)
- cd into the cloned directory
- run `docker build -t healthcareai .`
- run the docker instance with `docker run -p 8888:8888 healthcareai` 
- You should then have a jupyter notebook available on `http://localhost:8888`.

### Verify Installation

To verify that *healthcareai* installed correctly:
1. Open a terminal and run `python` or `ipython`. Either of these opens an interactive python console (also known as a [REPL](https://en.wikipedia.org/wiki/Read%E2%80%93eval%E2%80%93print_loop)).
2. Then enter this command: `from healthcareai import SupervisedModelTrainer` and hit enter. If no error is thrown, you are ready to rock.

If you did get an error, or run into other installation issues, please [let us know](http://healthcare.ai/contact.html) or better yet post on [Stack Overflow](http://stackoverflow.com/questions/tagged/healthcare-ai) (with the healthcare-ai tag) so we can help others along this process.

## Getting Started

1. Read through the docs on this site.
2. Start with either `example_regression_1.py` or `example_classification_1.py`
3. Modify the queries and parameters to match your data.
4. Decide on what kind of prediction output you want.
5. Set up your database tables to match the output schema. See the [prediction types](prediction_types.md) document for details.
    - If you are working in a Health Catalyst EDW ecosystem (primarily MSSQL), please see the [Catalyst EDW Instructions](catalyst_edw_instructions) for SAM setup.
    - Please see the [databases docs](databases.md) for details about writing to different databases (MSSQL, MySQL, SQLite, CSV)

## Where to Get Help

- Double check that the code follows the examples in these documents.
- If you're still seeing an error, file an issue on [Stack Overflow](http://stackoverflow.com/) using the healthcare-ai tag. Please provide
    - Details on your environment (OS, database type, R vs Py)
    - Goals (ie, what are you trying to accomplish)
    - Crystal clear steps to reproduce the error
