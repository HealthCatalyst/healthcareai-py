# Training, Comparing and Saving Models

Before you can make predictions, you need to train a model using known data.

## What is `SupervisedModelTrainer`?

- This class lets you train and compare machine learning models on diverse
    datasets.
- You can do both **classification** (for example, predicting Y/N) as well as
    **regression** (for example, predicting a numeric value).

## Am I ready for model creation?

To build a model, please follow these guidelines for setting up your training data:

- Don't use `0` or `1` for the dependent variable when doing binary classification. Use `Y`/`N` instead. The `IIF` function in T-SQL is an easy way to convert your target variables.
- Don't pull in test data in this step. In other words, we just pull in those rows where the target (predicted column) has a value already.
- Feature engineering is always a good idea. There are a few notes in the [hints](hints.md) document. Check out our [blog](http://healthcare.ai/blog) for ideas.

## Step 1: Load training data

### MSSQL

```python
server = 'localhost'
database = 'SAM'
query = """SELECT *
            FROM [SAM].[dbo].[DiabetesClincialSampleData]
            -- In this step, just grab rows that have a target
            WHERE ThirtyDayReadmitFLG is not null"""
engine = hcai_db.build_mssql_engine(server=server, database=database)
dataframe = pd.read_sql(query, engine)

# Handle missing data (if needed)
dataframe.replace(['None'], [None], inplace=True)
```

### CSV

```python
dataframe = pd.read_csv('healthcareai/tests/fixtures/DiabetesClinicalSampleData.csv', na_values=['None'])
```


## Step 2: Set up a Trainer

The `SupervisedModelTrainer` class helps you train models. It cleans and prepares the data before model creation. It also assignes parameters specific to the type of model you eventually want (regression or classification). To set up a trainer you'll need these arguments:

- **dataframe** *(pandas.core.frame.DataFrame)*: The training data in a pandas dataframe
- **predicted_column** *(str)*: The name of the prediction column 
- **model_type** *(str)*: the trainer type - 'classification' or 'regression'
- **impute** *(bool)*: True to impute data (mean of numeric columns and mode of categorical ones). False to drop rows
    that contain any null values.
- **grain_column** *(str)*: The name of the grain column
- **verbose** *(bool)*: Set to true for verbose output. Defaults to False.

### Example code

```python
classification_trainer = SupervisedModelTrainer(
    dataframe=dataframe,
    predicted_column='ThirtyDayReadmitFLG',
    model_type='classification',
    grain_column='PatientEncounterID',
    impute=True,
    verbose=False)
```

## Step 3: Train some models

Now that you have a trainer set up, let's train some models!

### Example code

```python
# Train a KNN model
trained_knn = classification_trainer.knn()

# Train a logistic regression model
trained_lr = classification_trainer.logistic_regression()

# Train a random forest model and view the feature importance plot
trained_random_forest = classification_trainer.random_forest(save_plot=False)
```

## Step 4: Evaluate and compare models

Now that you have trained some models, let's evaluate and compare them.

Each trained model has metrics that can be easily viewed by using the `.metrics` property. Depending on the model type, this can be a large list, so if you just want to see the ROC or PR metrics you can use `.roc()` or `.pr()` methods to print out the ideal cutoff and full list of cutoffs.

### Example code

```python
# Print the entire list of metrics.
print(trained_knn.metrics)

# Print the ROC thresholds
trained_knn.roc()

# Print the PR thresholds
trained_knn.pr()
```

#### Individual plots

```python
# View the ROC and PR plots
trained_knn.roc_plot()
trained_knn.pr_plot()

# View the ROC and PR plots
trained_lr.roc_plot()
trained_lr.pr_plot()

# View the ROC and PR plots
trained_random_forest.roc_plot()
trained_random_forest.pr_plot()
```

#### Comparison plots

The `healthcareai.common.model_eval.tsm_classification_comparison_plots` method plots ROC curves or PR curves of one or more trained models to directly compare models. It's arguments are:

- **plot_type** *(str)*: 'ROC' (default) or 'PR' 
- **trained_supervised_models** *(list | TrainedSupervisedModel)*: a single or list of TrainedSupervisedModels 
- **save** *(bool)*: Save the plot to a file

##### Example code

```python
# Create a list of all the models you just trained that you want to compare
models_to_compare = [trained_knn, trained_lr, trained_random_forest]

# Create a ROC plot that compares them.
tsm_plots.tsm_classification_comparison_plots(
    trained_supervised_models=models_to_compare,
    plot_type='ROC',
    save=False)

# Create a PR plot that compares them.
tsm_plots.tsm_classification_comparison_plots(
    trained_supervised_models=models_to_compare,
    plot_type='PR',
    save=False)
```

## Step 5: Save a Model

After you have trained a model you are happy with, you can save that model for later use.

On any instance of a `TrainedSupervisedModel` (that the trainer returns), use the `.save()` method to save the model.

Models will be saved with a timestamp and algorithm name so it is easy to find which one you want.

```python
# Save the model
trained_random_forest.save()
```


## Full example code

Note: you can run this out-of-the-box from the healthcareai-py folder:

```python
import pandas as pd

import healthcareai.trained_models.trained_supervised_model as tsm_plots
import healthcareai.common.database_connections as hcai_db
from healthcareai.supvervised_model_trainer import SupervisedModelTrainer


def main():
    # ## Load data from a sample .csv file
    dataframe = pd.read_csv('healthcareai/tests/fixtures/DiabetesClinicalSampleData.csv', na_values=['None'])

    # ## Load data from a MSSQL server: Uncomment to pull data from MSSQL server
    # server = 'localhost'
    # database = 'SAM'
    # query = """SELECT *
    #             FROM [SAM].[dbo].[DiabetesClincialSampleData]
    #             -- In this step, just grab rows that have a target
    #             WHERE ThirtyDayReadmitFLG is not null"""
    #
    # engine = hcai_db.build_mssql_engine(server=server, database=database)
    # dataframe = pd.read_sql(query, engine)

    # Drop columns that won't help machine learning
    dataframe.drop(['PatientID'], axis=1, inplace=True)

    # Step 1: Setup a healthcareai classification trainer. This prepares your data for model building
    classification_trainer = SupervisedModelTrainer(
        dataframe=dataframe,
        predicted_column='ThirtyDayReadmitFLG',
        model_type='classification',
        grain_column='PatientEncounterID',
        impute=True,
        verbose=False)

    # Look at the first few rows of your dataframe after loading the data
    print('\n\n-------------------[ Cleaned Dataframe ]--------------------------')
    print(classification_trainer.clean_dataframe.head())

    # Step 2: train some models

    # Train a KNN model
    trained_knn = classification_trainer.knn()

    # View the ROC and PR plots
    trained_knn.roc_plot()
    trained_knn.pr_plot()

    # Uncomment if you want to see all the ROC and/or PR thresholds
    # trained_knn.roc()
    # trained_knn.pr()

    # Train a logistic regression model
    trained_lr = classification_trainer.logistic_regression()

    # View the ROC and PR plots
    trained_lr.roc_plot()
    trained_lr.pr_plot()

    # Uncomment if you want to see all the ROC and/or PR thresholds
    # trained_lr.roc()
    # trained_lr.pr()

    # Train a random forest model and view the feature importance plot
    trained_random_forest = classification_trainer.random_forest(save_plot=False)
    # View the ROC and PR plots
    trained_random_forest.roc_plot()
    trained_random_forest.pr_plot()

    # Uncomment if you want to see all the ROC and/or PR thresholds
    # trained_random_forest.roc()
    # trained_random_forest.pr()

    # Create a list of all the models you just trained that you want to compare
    models_to_compare = [trained_knn, trained_lr, trained_random_forest]

    # Create a ROC plot that compares them.
    tsm_plots.tsm_classification_comparison_plots(
        trained_supervised_models=models_to_compare,
        plot_type='ROC',
        save=False)

    # Create a PR plot that compares them.
    tsm_plots.tsm_classification_comparison_plots(
        trained_supervised_models=models_to_compare,
        plot_type='PR',
        save=False)

    # Once you are happy with the performance of any model, you can save it for use later in predicting new data.
    trained_random_forest.save()


if __name__ == "__main__":
    main()
```
