# Getting started with healthcare.ai

## What can you do with this package?

- Fill in missing data via imputation
- Create and compare models based on your data
- Save a model to produce daily predictions
- Write predictions back to a database
- Learn what factor drives each prediction

## How to install

### Windows

- If you haven't, install 64-bit Python 3.5 via [the Anaconda distribution](https://www.continuum.io/downloads)
- Open Spyder (which installed with Anaconda)
- Run `conda install pyodbc`
- To install the latest release, run
    - `pip install https://github.com/HealthCatalystSLC/healthcareai-py/zipball/v0.1.7-beta`
- If you know what you're doing, and want the bleeding-edge version, run
    - `pip install https://github.com/HealthCatalystSLC/healthcareai-py/zipball/master`

### Non-Windows

- [Download](https://www.continuum.io/downloads) and install python 3.5 64-bit version
- Download [docker](https://docs.docker.com/engine/installation/)
- To install using docker, run `docker build -t healthcareai .`
- Run the docker instance with `docker run -p 8888:8888 healthcareai`
- You should see a jupyter notebook available at `http://localhost:8888`

## How to help

Check out our [github repo](https://github.com/HealthCatalystSLC/healthcareai-py/)
