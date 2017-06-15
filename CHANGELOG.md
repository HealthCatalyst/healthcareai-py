# Change Log
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/) 
and this project adheres to [Semantic Versioning](http://semver.org/).

## [Unreleased]

### Added

- AzureBlogStorageHelper class to make storing text & pickled objects simple.
- table_archiver() utility function makes it easy to create a timestamped history from any table.

### Changed

### Fixed

## [0.1.8] - 2017-02-16

### Added

- Added changelog
- Changed all docs to markdown
- added `setup.cfg` and some functions in `setup.py` for PyPI
- `mkdocs.yml` for readthedocs hosting
- Conda environment files

### Changed

- Lots of documentation, especially around installation on the various platforms.
- Removed doc templates and css now that docs will live on readthedocs

### Fixed

- example jupyter notebook

## [0.1.7] - 2016-11-01

### Added

This release encompasses basic healthcare ML functionality:

- Model comparison between random forest and logistic regression algorithms
- Model deployment to SQL Server, providing top-three most important features
- Imputation (column mean for numeric and column mode for categorical)
- Hyperparameter tuning, using mtry and number of trees for random forest
- Plots
    - ROC 
    - Random forest feature ranking
- Model performance evaluated via AU_ROC and AU_PR
- First release after setting up the following infrastructure:
    - AppveyorCI
    - Sphinx
    - Nose unit testing
    - Docker

[Unreleased]: https://github.com/HealthCatalyst/healthcareai-py/compare/v0.1.7...HEAD
[0.1.7]: https://github.com/HealthCatalyst/healthcareai-py/releases/tag/v0.1.7-beta