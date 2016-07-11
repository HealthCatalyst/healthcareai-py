from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np
import pandas as pd
from hcpytools.impute_custom import DataFrameImputer
from hcpytools import modelutilities
import ceODBC
import datetime
import logging
import sys


class DeploySupervisedModel(object):
    """ Describe what this does




    """
    def __init__(self,
                 modeltype,
                 df,
                 graincol,
                 windowcol,
                 predictedcol,
                 impute,
                 debug=False):

        if debug:
            print('\nTypes of original dataframe:')
            print(df.dtypes)
            print('\nShape and top 5 rows of original dataframe:')
            print(df.shape)
            print(df.head())

        self.modeltype = modeltype
        self.graincol = graincol
        self.impute = impute

        # Remove DTS columns
        # TODO: make this work with col names shorter than three letters (ie when headers aren't read in)
        cols = [c for c in df.columns if c[-3:] != 'DTS']
        df = df[cols]

        if debug:
            print('\nDataframe after removing DTS columns:')
            print(df.head())
            print('\nNow either doing imputation or dropping rows with NULLs...')
            print('\nTotal df shape before impute/remove')
            print(df.shape)

        pd.options.mode.chained_assignment = None  # default='warn'
        # TODO: put try/catch here when type = class and numeric predictor is selected
        df.replace('NULL', np.nan, inplace=True)

        if debug:
            print('\nTotal df after replacing NULL with nan')
            print(df.shape)
            print(df.head())

        if self.impute:

            # This class comes from here: http://stackoverflow.com/a/25562948/5636012
            df = DataFrameImputer().fit_transform(df)

            if debug:
                print('\nTotal df after doing imputation:')
                print(df.shape)
                print(df.head())

        else:
            # TODO switch similar statements to work inplace (w/o making a copy)
            # If not impute, only remove rows in train that contain NULLS

            if debug:
                print('\nTrain portion before impute/remove')
                print(df.ix[df[windowcol] == 'N'].shape)

            df_train_temp = df.ix[df[windowcol] == 'N']
            df_test_temp = df.ix[df[windowcol] == 'Y']
            df_train_temp.dropna(axis=0, how='any', inplace=True)

            # Recombine same (unaltered) test rows with modified training set
            df = pd.concat([df_test_temp, df_train_temp])

            if debug:
                print('\nTotal df after dropping rows with NULLS:')
                print(df.shape)
                print(df.head())
                print('\nTrain portion after removing training rows with NULLs')
                print(df.ix[df[windowcol] == 'N'].shape)

        # Convert predicted col to 0/1 (otherwise won't work with GridSearchCV)
        # Note that this makes it such that hcpytools only handles N/Y in pred column
        if modeltype == 'classification':
            # Turning off warning around replace
            #pd.options.mode.chained_assignment = None  # default='warn'
            # TODO: put try/catch here when type = class and numeric predictor is selected
            df[predictedcol].replace(['Y','N'],[1,0], inplace=True)

            if debug:
                print('\nDataframe after converting to 1/0 instead of Y/N for classification:')
                print(df.head())

        # Split grain col into test portion (for use in SQL output) and drop from model work
        self.graincol_test = df[graincol].loc[df[windowcol] == 'Y']
        pd.options.mode.chained_assignment = None  # default='warn' # This is ignoring this message:
        # TODO: Fix this SettingWithCopyWarning: value trying to be set on a copy of a slice from a DataFrame"

        df.drop(graincol, axis=1, inplace = True)

        if debug:
            print('\nGrain_test col after splitting it from main df')
            print(np.shape(self.graincol_test))
            print(self.graincol_test.head())

            print('\nDataframe after splitting out grain col and before creating dummy vars:')
            print(df.head())

        # Create dummy vars for all cols but predictedcol
        # First switch predicted col to numeric (so it's not dummified) and then switch back
        df[predictedcol] = pd.to_numeric(arg=df[predictedcol],errors='raise')
        df = pd.get_dummies(df, drop_first=True, prefix_sep='.')

        if debug:
            print('\nDataframe after creating dummy vars:')
            print(df.head())

        # Split train off of maindf
        df_train = df.ix[df[windowcol + '.Y'] == 0]

        if debug:
            print('\nTraining dataframe right after splitting it off:')
            print(df.head())

        # TODO: make sure this works with Impute=True (ie remove pred col from imputation)
        # Always remove rows where predicted col is NULL in train
        df_train.dropna(subset=[predictedcol], how='all', inplace=True)
        if debug:
            print('\nTraining df after removing rows where predicted col is NULL:')
            print(df_train.head())
            print('\nSplitting off test set from main df...')

        # Split test off of main df
        df_test = df.ix[df[windowcol + '.Y'] == 1]

        if debug:
            print('\nTest set after splitting off of main df:')
            print(df_test.head())

        # Drop window col from train and test
        df_train = df_train.drop(windowcol + '.Y', axis=1)

        if debug:
            print('\nTrain set after dropping window col (from training set):')
            print(df_train.head())

        df_test = df_test.drop(windowcol + '.Y', axis=1)

        if debug:
            print('\nTest set after dropping window col (from test set):')
            print(df_test.head())

        # Check user input and remove rows with NULLS if not doing imputation
        self.impute = impute

        self.X_train = df_train.drop([predictedcol], axis=1)
        self.y_train = np.squeeze(df_train[[predictedcol]])

        self.X_test = df_test.drop([predictedcol], axis=1)
        self.y_test = np.squeeze(df_test[[predictedcol]])

        if debug:
            print('\nShape of X_train, y_train, X_test, and y_test:')
            print(self.X_train.shape)
            print(self.y_train.shape)
            print(self.X_test.shape)
            print(self.y_test.shape)

    def deploy(self,
               method,
               cores,
               server,
               dest_db_schema_table,
               trees=200,
               use_saved_model=False,
               debug=False,
               sqlwrite=False):

        if debug:
            print('\ngraincol test shape and cell type before db prelim-insert check')
            print(np.shape(self.graincol_test))
            print(type(self.graincol_test.iloc[0]))

        # First, check the connection by inserting test data (and rolling back)
        cecnxn = ceODBC.connect("DRIVER={SQL Server Native Client 11.0};SERVER=" + server + ";Trusted_Connection=yes;")
        cursor = cecnxn.cursor()
        # TODO: add try catch to improve message about what parameters to change in case of error
        if self.modeltype == 'classification': predictedvalcol = 'PredictedProbNBR'
        else: predictedvalcol = 'PredictedValueNBR'
        dt = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] # good for datetime/datetime2

        try:

            cursor.execute("""insert into """ + dest_db_schema_table + """
                                            (BindingID, BindingNM, LastLoadDTS, """ + self.graincol + """,""" + predictedvalcol + """,
                                            Factor1TXT, Factor2TXT, Factor3TXT)
                                            values (0, 'PyTest', ?, ?, 0.98, 'FirstCol', 'SecondCol', 'ThirdCol')
                                            """, (dt, int(self.graincol_test.iloc[0])))
            cecnxn.rollback()
            # See here for logging (to file) options: http://stackoverflow.com/a/9014540/5636012
            print("\nSuccessfully inserted a test row into {}.".format(dest_db_schema_table))
            print("SQL insert successfuly rolled back (since it was just a test).")

        except ceODBC.DatabaseError:
            print("\nFailed to insert values into {}.".format(dest_db_schema_table))
            print("Check that the table exists and that the column structure is correct.")
            print("Example column structure can be found in the docs--most likely")
            print("your GrainID col doesn't match that from your input table")

        finally:
            try:
                cecnxn.close()
            except ceODBC.DatabaseError:
                print('\nAn attempt to complete a transaction has failed. No corresponding transaction found.')
                print('Perhaps you don''t have permission to write to this server.')


        if self.modeltype == 'classification' and method == 'linear':

            algorithm = LogisticRegression(n_jobs=cores)

            y_pred = modelutilities.clfreport(modeltype=self.modeltype,
                                              debug=debug,
                                              devcheck='notdev',
                                              algo=algorithm,
                                              X_train=self.X_train,
                                              y_train=self.y_train,
                                              X_test=self.X_test,
                                              use_saved_model = use_saved_model
                                              )

        elif self.modeltype == 'regression' and method == 'linear':

            algorithm = LinearRegression(n_jobs=cores)

            y_pred = modelutilities.clfreport(modeltype=self.modeltype,
                                                     debug=debug,
                                                     devcheck='notdev',
                                                     algo=algorithm,
                                                     X_train=self.X_train,
                                                     y_train=self.y_train,
                                                     X_test=self.X_test,
                                                     use_saved_model = use_saved_model
                                                     )

        if self.modeltype == 'classification' and method == 'rf':

            algorithm = RandomForestClassifier(n_estimators=trees,
                                               n_jobs=cores,
                                               verbose=(2 if debug is True else 0))

            y_pred = modelutilities.clfreport(modeltype=self.modeltype,
                                              debug=debug,
                                              devcheck='notdev',
                                              algo=algorithm,
                                              X_train=self.X_train,
                                              y_train=self.y_train,
                                              X_test=self.X_test,
                                              use_saved_model = use_saved_model
                                              )

        elif self.modeltype == 'regression' and method == 'rf':

            algorithm = RandomForestRegressor(n_estimators=trees,
                                              n_jobs=cores,
                                              verbose=(2 if debug is True else 0))

            y_pred = modelutilities.clfreport(modeltype=self.modeltype,
                                                     debug=debug,
                                                     devcheck='notdev',
                                                     algo=algorithm,
                                                     X_train=self.X_train,
                                                     y_train=self.y_train,
                                                     X_test=self.X_test,
                                                     use_saved_model = use_saved_model
                                                     )

        # Calculate three imp columns
        first_fact, second_fact, third_fact = modelutilities.findtopthreefactors(debug,
                                                                                 self.X_train,
                                                                                 self.y_train,
                                                                                 self.X_test,
                                                                                 self.impute,
                                                                                 self.modeltype,
                                                                                 use_saved_model)

        # Convert to base int instead of numpy data type
        graincol_baseint = [int(self.graincol_test.iloc[i]) for i in range(0,len(self.graincol_test))]
        y_pred_baseint = [float(y_pred[i]) for i in range(0,len(y_pred))]

        # Create columns for export to SQL Server
        X_test_length = len(self.X_test.iloc[:, 0])
        bindingid = [0] * X_test_length
        bindingnm = ['Python'] * X_test_length
        lastloaddts = [datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]] * X_test_length

        # Put everything into 2-d list for export
        output_2dlist = list(zip(bindingid,
                                 bindingnm,
                                 lastloaddts,
                                 graincol_baseint,
                                 y_pred_baseint,
                                 first_fact,
                                 second_fact,
                                 third_fact))

        if debug:
            print('\nTop rows of 2-d list immediately before insert into db')
            print(pd.DataFrame(output_2dlist[0:3]).head())

        cecnxn = ceODBC.connect("DRIVER={SQL Server Native Client 11.0};SERVER=" + server + ";Trusted_Connection=yes;")
        cursor = cecnxn.cursor()

        try:
            cursor.executemany("""insert into """ + dest_db_schema_table + """
                            (BindingID, BindingNM, LastLoadDTS, """ + self.graincol + """,""" + predictedvalcol + """,
                            Factor1TXT, Factor2TXT, Factor3TXT)
                            values (?,?,?,?,?,?,?,?)""", output_2dlist)
            affected_count = cursor.rowcount
            cecnxn.commit()
            print("\nSuccessfully inserted {} rows into {}.".format(affected_count, dest_db_schema_table))

        except:
            print("\nFailed to insert values into {}.".format(dest_db_schema_table))
            print("Was your test insert successful earlier? If so, what might have changed")
            print("with your entity since then?")

        finally:
            try:
                cecnxn.close()
            except ceODBC.DatabaseError:
                print('\nAn attempt to complete a transaction has failed. No corresponding transaction found.')
                print('Perhaps you don''t have permission to write to this server.')