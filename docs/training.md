# Developing and comparing models

## What is `SupervisedModelTrainer`?

-   This class lets you train and compare machine learning models on diverse
    datasets.
-   You can do both **classification** (for example, predict Y/N) as well as
    **regression** (for example, predict a numeric field).

## Am I ready for model creation?

Maybe. It'll help if you follow these guidelines:

> -   Don't use `0` or `1` for the dependent variable when doing
>     classification. Use `Y`/`N` instead. The IIF function in T-SQL may
>     help here.
> -   Don't pull in test data in this step. In other words, we just pull
>     in those rows where the target (ie, predicted column has a value
>     already).

Feature engineering is always a good idea. Check out our [blog](http://healthcare.ai/blog] for ideas.

## Step 1: Pull in the data

For SQL:

```python
import pyodbc
cnxn = pyodbc.connect("""SERVER=localhost;
                        DRIVER={SQL Server Native Client 11.0};
                        Trusted_Connection=yes;
                        autocommit=True""")

 df = pd.read_sql(
     sql="""SELECT *
            FROM [SAM].[dbo].[HCPyDiabetesClinical]""",
     con=cnxn)


 # Handle missing data (if needed)
 df.replace(['None'],[None],inplace=True)
```

For CSV:

```python
df = pd.read_csv(DiabetesClincialSampleData.csv,
                 na_values=['None'])
```

## Step 2: Set your data-prep parameters

The `SupervisedModelTrainer` helps you train models. It cleans and prepares the data before
model creation. To set up a trainer you'll need these arguments:

-   **Return**: an object.
-   **Arguments**:
    - **modeltype**: a string. This will either be 'classification' or 'regression'.
    - **dataframe**: a data frame. The data your model will be based on.
    - **predictedcol**: a string. Name of variable (or column) that you want to predict.
    - **graincol**: a string, defaults to None. Name of possible GrainID column in your dataset. If specified, this column will be removed, as it won't help the algorithm.
    - **impute**: a boolean. Whether to impute by replacing NULLs with column mean (for numeric columns) or column mode (for categorical columns).
    - **debug**: a boolean, defaults to False. If TRUE, console output when comparing models is verbose for easier debugging.

Example code:

```python
hcai_trainer = SupervisedModelTrainer(
        dataframe=dataframe,
        predicted_column='ThirtyDayReadmitFLG',
        model_type='classification',
        grain_column='PatientEncounterID',
        impute=True,
        verbose=False)
```

## Step 3: Create and compare models

Example code:

```python
# Run the logistic regression model
trained_logistic_regression = hcai_trainer.logistic_regression()
trained_logistic_regression.roc_curve_plot()

# Run the random forest model
trained_random_forest = hcai_trainer.random_forest()
trained_random_forest.roc_curve_plot()

```

### Go further using utility methods

# TODO this needs updating
The `healthcareai.common.model_eval.tsm_classification_comparison_plots` method plots ROC curves or PR curves of one or more trained models to directly comparing models.

-   **Return**: a plot.
-   **Arguments**:
    - **plot_type** (str): 'ROC' (default) or 'PR' 
    - **trained_supervised_model** (list | TrainedSupervisedModel): a single or list of TrainedSupervisedModels
    - **save** (bool): True to save the plot.

Example code:

```python
# Create a comparison ROC plot multiple models
hcaieval.tsm_classification_comparison_plots(
    trained_supervised_model=[trained_random_forest, trained_logistic_regression],
    plot_type='ROC',
    save=False)

# Create a comparison PR plot multiple models
hcaieval.tsm_classification_comparison_plots(
    trained_supervised_model=[trained_random_forest, trained_logistic_regression],
    plot_type='PR',
    save=False)
```

## Full example code

Note: you can run (out-of-the-box) from the healthcareai-py folder:

```python
from healthcareai import SupervisedModelTrainer
import pandas as pd
import time

def main():
    # CSV snippet for reading data into dataframe
    df = pd.read_csv('healthcareai/tests/fixtures/DiabetesClincialSampleData.csv',
                    na_values=['None'])

    # SQL snippet for reading data into dataframe
    import pyodbc
    cnxn = pyodbc.connect("""SERVER=localhost;
                            DRIVER={SQL Server Native Client 11.0};
                            Trusted_Connection=yes;
                            autocommit=True""")

    df = pd.read_sql(
        sql="""SELECT *
            FROM [SAM].[dbo].[HCPyDiabetesClinical]
            -- In this step, just grab rows that have a target
            WHERE ThirtyDayReadmitFLG is not null""",
        con=cnxn)

    # Set None string to be None type
    df.replace(['None'],[None],inplace=True)

    # Look at data that's been pulled in
    print(df.head())
    print(df.dtypes)

    # Drop columns that won't help machine learning
    df.drop(['PatientID'],axis=1,inplace=True)

    # Step 1: compare two models
    o = SupervisedModelTrainer(modeltype='classification',
                            df=df,
                            predictedcol='ThirtyDayReadmitFLG',
                            graincol='PatientEncounterID', #OPTIONAL
                            impute=True,
                            debug=False)

    # Run the linear model
    o.linear(cores=1)

    # Run the random forest model
    o.random_forest(cores=1,
                    tune=True)

    # Look at the RF feature importance rankings
    o.plot_rffeature_importance(save=False)

    # Create ROC plot to compare the two models
    o.plot_roc(debug=False,
            save=False)

    print('\nTime:\n', time.time() - t0)

if __name__ == "__main__":
    main()
```
