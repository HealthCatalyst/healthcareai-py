# Change Log
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/) 
and this project adheres to [Semantic Versioning](http://semver.org/).

## [Unreleased]

### Added

- Added changelog
- Changed all docs to markdown

### Changed

- Lots of documentation

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

[Unreleased]: https://github.com/HealthCatalystSLC/healthcareai-py/compare/v0.1.7...HEAD
[0.1.7]: https://github.com/HealthCatalystSLC/healthcareai-py/releases/tag/v0.1.7-beta