Developing and comparing models
-------------------------------

What is ``DevelopSupervisedModel``?
###################################

- This class let's one create and compare custom models on diverse datasets.

- One can do both classification (ie, predict Y/N) as well as regression (ie, predict a numeric field).

- To jump straight to an example notebook, see `here`_

.. _here: https://github.com/HealthCatalystSLC/healthcareai-py/blob/master/notebooks/HCPyToolsExample1.ipynb

Am I ready for model creation?
##############################

Maybe. It'll help if you follow these guidelines:

 - Don't use 0 or 1 for the independent variable when doing classification. Use Y/N instead. The IIF function in T-SQL may help here.

 - Don't pull in test data in this step. In other words, we just pull in those rows where the target (ie, predicted column has a value already).

Of course, feature engineering is always a good idea.


Step 1: Pull in the data
########################

For SQL:

.. code-block:: python

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

For CSV:

.. code-block:: python

    df = pd.read_csv('healthcareai/tests/fixtures/HCPyDiabetesClinical.csv',
                     na_values=['None'])


Step 2: Set your data-prep parameters
#####################################

The ``DevelopSupervisedModel`` class cleans and prepares the data before model creation

- **Return**: an object.
- **Arguments**:
    - **modeltype**: a string. This will either be 'classification' or 'regression'.
    - **df**: a data frame. The data your model will be based on.
    - **predictedcol**: a string. Name of variable (or column) that you want to predict.
    - **graincol**: a string, defaults to None. Name of possible GrainID column in your dataset. If specified, this column will be removed, as it won't help the algorithm.
    - **impute**: a boolean. Whether to impute by replacing NULLs with column mean (for numeric columns) or column mode (for categorical columns).
    - **debug**: a boolean, defaults to False. If TRUE, console output when comparing models is verbose for easier debugging.

Example code:

.. code-block:: python

   o = DevelopSupervisedModel(modeltype='classification',
                              df=df,
                              predictedcol='ThirtyDayReadmitFLG',
                              graincol='PatientEncounterID', #OPTIONAL
                              impute=True,
                              debug=False)


Step 3: Create and compare models
#################################

Example code:

.. code-block:: python

   # Run the linear model
   o.linear(cores=1)

   # Run the random forest model
   o.random_forest(cores=1)


Go further using utility methods
################################

The ``plot_rffeature_importance`` method plots the input columns in order of importance to the model.  

- **Return**: a plot.
- **Arguments**:
    - **save**: a boolean, defaults to False. If True, the plot is saved to the location displayed in the console.

Example code:

.. code-block:: sql

   # Look at the feature importance rankings
   o.plot_rffeature_importance(save=False)

The ``plot_roc`` method plots the AU_ROC chart, for easier model comparison.

- **Return**: a plot.
- **Arguments**:
    - **save**: a boolean, defaults to False. If True, the plot is saved to the location displayed in the console.
    - **debug**: a boolean. If True, console output is verbose for easier debugging.

Example code:

.. code-block:: python

   # Create ROC plot to compare the two models
   o.plot_roc(debug=False,
              save=False)

Full example code
#################

Note: you can run (out-of-the-box) from the healthcareai-py folder:

.. code-block:: python

  from healthcareai import DevelopSupervisedModel
  import pandas as pd
  import time

  def main():

      t0 = time.time()

      # CSV snippet for reading data into dataframe
      df = pd.read_csv('healthcareai/tests/fixtures/HCPyDiabetesClinical.csv',
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
      df.drop(['PatientID','InTestWindowFLG'],axis=1,inplace=True)

      # Step 1: compare two models
      o = DevelopSupervisedModel(modeltype='classification',
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