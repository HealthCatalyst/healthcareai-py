# Change Log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/) 
and this project adheres to [Semantic Versioning](http://semver.org/).

## [Unreleased]

### Added

- Percentage of nulls reported via console.
- new top-level `load_csv()` function makes it easier for users by avoiding any pandas knowledge.
- `SupervisedModelTrainer` now warns users about columns/features with high and low cardinality.
- 9 new sample healthcare data sets from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets.html).

### Changed

- Catalyst db validation test now makes it's own safely named database and table, runs the test then cleans it up. This
eliminated the need for .mdf fixtures on appveyor.
- validate_catalyst_prediction_sam_connection made much more robust with tests.
- Conda environment files cleaned up substantially, speeding up builds.
- Release preparation notes moved out of README and into a separate doc.
- Decorators for `supervised_model_trainer` is now simplified, and debug option is no longer an option.

### Fixed

- Feature importance plots now have a configurable limit to the amount of features they show with a default of 15.
- Dataframe column filter handles None gracefully. For example, if no grain column is specified.
- Getting started section of README vastly improved.


### Deprecated

### Removed

## [1.0] - 2017-08-15

**Note this is a major release and most of the API has changed. Please consult the documentation for full details**

### Added

- SupervisedModelTrainer class: the main simple API
- AdvancedSupervisedModelTrainer class: the main advanced user API
- TrainedSupervisedModel class: the object returned after training that is easy to .save(). It has many convenience functions to get metrics, graphs, and contains the trained model, a feature model and the data preparation pipeline which makes deployment simple.
- SupervisedModelTrainer.ensemble()
- Lasso, KNN
- AzureBlogStorageHelper class to make storing text & pickled objects simple.
- table_archiver() utility function makes it easy to create a timestamped history from any table.
- Five new combinations of prediction dataframes to make deployment easy and flexible.
- Database connection helpers for MSSQL, MySQL, SQLite
- file I/O utilities for pickling objects or JSON
- many new dataframe filters and transformers that are used in the SupervisedModelTrainer or can be used individually.
- feature scaling transformer
- Architecture diagram and document
- Examples split into training and prediction
- Randomized hyperparameter search by default on SupervisedModelTrainer

### Changed

- load_diabetes(): a built in sample dataset rather than manually digging around for the .csv file.
- Better PR/ROC plots w/ ideal cutoffs marked
- Better metrics
- Google style docstrings on most functions
- Much more helpful error messages
- Lots of code simplification and decoupling
- 126 new tests (up from 11)

### Fixed

- [Many, many things](https://github.com/HealthCatalyst/healthcareai-py/issues/163)

### Removed

- DevelopTrainedSupervisedModel **entire class**
- DeploySupervisedModel **entire class**
- model_eval.clfreport()
- model_eval.findtopthreefactors()
- filters.remove_datetime_columns()

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