from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from healthcareai.common.transformers import DataFrameImputer
from healthcareai.common import model_eval
from healthcareai.common import filters
from healthcareai.common.healthcareai_error import HealthcareAIError

import numpy as np
import pandas as pd
import pyodbc
import datetime
import math


class DeploySupervisedModel(object):
    """
    A likely incomplete list of functionality
    - [ ] refactor out data preprocessing pipeline
    - [ ] retrain on full dataset
    - [ ] tests db connection
    - [ ] writes to db
    - [ ] loads or saves pickle via model mess
    - [ ] performance on top n factors
    - [ ] refactor top n factors
    """

    def __init__(self,
                 model_type,
                 dataframe,
                 grain_column,
                 window_column,
                 predicted_column,
                 impute,
                 debug=False):
        self.modeltype = model_type
        if model_type == 'classification':
            self.predicted_column_name = 'PredictedProbNBR'
        else:
            self.predicted_column_name = 'PredictedValueNBR'
        self.df = dataframe
        self.grain_column = grain_column
        self.impute = impute
        self.y_pred = None  # Only used for unit test

        if debug:
            print('\nTypes of original dataframe:')
            print(self.df.dtypes)
            print('\nShape and top 5 rows of original dataframe:')
            print(self.df.shape)
            print(self.df.head())

        self.df = filters.remove_DTS_postfix_columns(self.df)

        if debug:
            print('\nDataframe after removing DTS columns:')
            print(self.df.head())
            print('\nNow either doing imputation or dropping rows with NULLs')
            print('\nTotal df shape before impute/remove')
            print(self.df.shape)

        pd.options.mode.chained_assignment = None  # default='warn'


        # TODO: put try/catch here when type = class and pred col is numer
        self.df.replace('NULL', np.nan, inplace=True)

        if debug:
            print('\nTotal df after replacing NULL with nan')
            print(self.df.shape)
            print(self.df.head())

        if self.impute:

            self.df = DataFrameImputer().fit_transform(self.df)

            if debug:
                print('\nTotal df after doing imputation:')
                print(self.df.shape)
                print(self.df.head())

        else:
            # TODO switch similar statements to work inplace
            # If not impute, only remove rows in train that contain NULLS

            if debug:
                print('\nTrain portion before impute/remove')
                print(self.df.ix[self.df[window_column] == 'N'].shape)

            df_train_temp = self.df.ix[self.df[window_column] == 'N']
            df_test_temp = self.df.ix[self.df[window_column] == 'Y']
            df_train_temp.dropna(axis=0, how='any', inplace=True)

            # Recombine same (unaltered) test rows with modified training set
            self.df = pd.concat([df_test_temp, df_train_temp])

            if debug:
                print('\nTotal df after dropping rows with NULLS:')
                print(self.df.shape)
                # noinspection PyUnresolvedReferences
                print(self.df.head())
                print('\nTrain portion after removing training rows w/ NULLs')
                # noinspection PyUnresolvedReferences,PyUnresolvedReferences
                print(self.df.ix[self.df[window_column] == 'N'].shape)

        #Call new function!!
        # Convert predicted col to 0/1 (otherwise won't work with GridSearchCV)
        # Makes it so that healthcareai only handles N/Y in pred column
        if model_type == 'classification':
            # Turning off warning around replace
            # pd.options.mode.chained_assignment = None  # default='warn'
            # TODO: put try/catch here when type = class and pred col is numer
            self.df[predicted_column].replace(['Y', 'N'], [1, 0], inplace=True)

            if debug:
                print("""\nDataframe after converting to 1/0 instead of Y/N
                for classification:""")
                print(self.df.head())

        # Split grain column into test piece; is taken out of ML and used in dfout
        self.grain_column_test = self.df[grain_column].loc[self.df[window_column] == 'Y']
        pd.options.mode.chained_assignment = None  # default='warn'
        # TODO: Fix this copy of a slice SettingWithCopyWarning:

        self.df.drop(grain_column, axis=1, inplace=True)

        if debug:
            print('\nGrain_test col after splitting it from main df')
            print(np.shape(self.grain_column_test))
            print(self.grain_column_test.head())

            print('\ndf after splitting out grain_column and before creating dummies')
            print(self.df.head())

        # Create dummy vars for all cols but predictedcol
        # First switch (temporarily) pred col to numeric (so it's not dummy)
        self.df[predicted_column] = pd.to_numeric(arg=self.df[predicted_column], errors='raise')
        self.df = pd.get_dummies(self.df, drop_first=True, prefix_sep='.')

        if debug:
            print('\nDataframe after creating dummy vars:')
            print(self.df.head())

        # Split train off of maindf
        # noinspection PyUnresolvedReferences
        df_train = self.df.ix[self.df[window_column + '.Y'] == 0]

        if debug:
            print('\nTraining dataframe right after splitting it off:')
            print(self.df.head())

        # Always remove rows where predicted col is NULL in train
        df_train.dropna(subset=[predicted_column], how='all', inplace=True)
        if debug:
            print('\nTraining df after removing rows where pred col is NULL:')
            print(df_train.head())
            print('\nSplitting off test set from main df...')

        # Split test off of main df
        # noinspection PyUnresolvedReferences
        df_test = self.df.ix[self.df[window_column + '.Y'] == 1]

        if debug:
            print('\nTest set after splitting off of main df:')
            print(df_test.head())

        # Drop window col from train and test
        df_train = df_train.drop(window_column + '.Y', axis=1)

        if debug:
            print('\nTrain set after dropping window col (from training set):')
            print(df_train.head())

        df_test = df_test.drop(window_column + '.Y', axis=1)

        if debug:
            print('\nTest set after dropping window col (from test set):')
            print(df_test.head())

        # Check user input and remove rows with NULLS if not doing imputation
        self.impute = impute

        self.X_train = df_train.drop([predicted_column], axis=1)
        self.y_train = np.squeeze(df_train[[predicted_column]])

        self.X_test = df_test.drop([predicted_column], axis=1)
        self.y_test = np.squeeze(df_test[[predicted_column]])

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
               mtry=None,
               use_saved_model=False,
               debug=False):

        if debug:
            print("""\ngrain_column test shape and cell type before db
            prelim-insert check""")
            print(np.shape(self.grain_column_test))
            print(type(self.grain_column_test.iloc[0]))

        validate_destination_table_connection(server,
                                              dest_db_schema_table,
                                              self.grain_column,
                                              self.predicted_column_name)

        if self.modeltype == 'classification' and method == 'linear':

            algorithm = LogisticRegression(n_jobs=cores)

            self.y_pred = model_eval.clfreport(
                model_type=self.modeltype,
                debug=debug,
                develop_model_mode=False,
                algo=algorithm,
                X_train=self.X_train,
                y_train=self.y_train,
                X_test=self.X_test,
                use_saved_model=use_saved_model)

        elif self.modeltype == 'regression' and method == 'linear':

            algorithm = LinearRegression(n_jobs=cores)

            self.y_pred = model_eval.clfreport(
                model_type=self.modeltype,
                debug=debug,
                develop_model_mode=False,
                algo=algorithm,
                X_train=self.X_train,
                y_train=self.y_train,
                X_test=self.X_test,
                use_saved_model=use_saved_model)

        if self.modeltype == 'classification' and method == 'rf':

            # TODO: think about moving this to model_eval mtry function
            if not mtry:
                mtry = math.floor(math.sqrt(len(self.X_train.columns.values)))

            algorithm = RandomForestClassifier(n_estimators=trees,
                                               max_features=mtry,
                                               n_jobs=cores,
                                               verbose=(
                                                   2 if debug is True else 0)
                                               )

            self.y_pred = model_eval.clfreport(
                model_type=self.modeltype,
                debug=debug,
                develop_model_mode=False,
                algo=algorithm,
                X_train=self.X_train,
                y_train=self.y_train,
                X_test=self.X_test,
                use_saved_model=use_saved_model)

        elif self.modeltype == 'regression' and method == 'rf':

            # TODO: think about moving this to model_eval mtry function
            if not mtry:
                mtry = math.floor(len(self.X_train.columns.values)/3)

            algorithm = RandomForestRegressor(n_estimators=trees,
                                              max_features=mtry,
                                              n_jobs=cores,
                                              verbose=(
                                                  2 if debug is True else 0)
                                              )

            self.y_pred = model_eval.clfreport(
                model_type=self.modeltype,
                debug=debug,
                develop_model_mode=False,
                algo=algorithm,
                X_train=self.X_train,
                y_train=self.y_train,
                X_test=self.X_test,
                use_saved_model=use_saved_model)

        # Calculate three imp columns
        first_fact, second_fact, third_fact = model_eval.find_top_three_factors(debug,
                                                                                self.X_train,
                                                                                self.y_train,
                                                                                self.X_test,
                                                                                self.modeltype,
                                                                                use_saved_model)

        # Convert to base int instead of numpy data type for SQL insert
        grain_column_baseint = [int(self.grain_column_test.iloc[i])
                                for i in range(0, len(self.grain_column_test))]
        y_pred_baseint = [float(self.y_pred[i])
                          for i in range(0, len(self.y_pred))]

        # Create columns for export to SQL Server
        X_test_length = len(self.X_test.iloc[:, 0])
        bindingid = [0] * X_test_length
        bindingnm = ['Python'] * X_test_length

        # Create vector with time to the millisecond
        lastloaddts = [datetime.datetime.utcnow().
                       strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]] * \
            X_test_length

        # Put everything into 2-d list for export
        output_2dlist = list(zip(bindingid,
                                 bindingnm,
                                 lastloaddts,
                                 grain_column_baseint,
                                 y_pred_baseint,
                                 first_fact,
                                 second_fact,
                                 third_fact))

        if debug:
            print('\nTop rows of 2-d list immediately before insert into db')
            print(pd.DataFrame(output_2dlist[0:3]).head())

        cecnxn = pyodbc.connect("""DRIVER={SQL Server Native Client 11.0};
                                   SERVER=""" + server + """;
                                   Trusted_Connection=yes;""")
        cursor = cecnxn.cursor()

        try:
            cursor.executemany("""insert into """ + dest_db_schema_table + """
                               (BindingID, BindingNM, LastLoadDTS, """ +
                               self.grain_column + """,""" + self.predicted_column_name + """,
                               Factor1TXT, Factor2TXT, Factor3TXT)
                               values (?,?,?,?,?,?,?,?)""", output_2dlist)
            cecnxn.commit()

            # Todo: count and display (via pyodbc) how many rows inserted
            print("\nSuccessfully inserted rows into {}.".
                  format(dest_db_schema_table))

        except pyodbc.DatabaseError:
            print("\nFailed to insert values into {}.".
                  format(dest_db_schema_table))
            print("Was your test insert successful earlier?")
            print("If so, what has changed with your entity since then?")

        finally:
            try:
                cecnxn.close()
            except pyodbc.DatabaseError:
                print("""\nAn attempt to complete a transaction has failed.
                      No corresponding transaction found. \nPerhaps you don't
                      have permission to write to this server.""")


def validate_destination_table_connection(server, destination_table, grain_column, predicted_column_name):
    # First, check the connection by inserting test data (and rolling back)
    db_connection = pyodbc.connect("""DRIVER={SQL Server Native Client 11.0};
                               SERVER=""" + server + """;
                               Trusted_Connection=yes;""")

    # The following allows output to work with datetime/datetime2
    temp_date = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    try:
        cursor = db_connection.cursor()
        cursor.execute("""INSERT INTO """ + destination_table + """
                       (BindingID, BindingNM, LastLoadDTS, """ +
                       grain_column + """,""" + predicted_column_name + """,
                       Factor1TXT, Factor2TXT, Factor3TXT)
                       VALUES (0, 'PyTest', ?, 33, 0.98,
                       'FirstCol', 'SecondCol', 'ThirdCol')""",
                       temp_date)

        print("Successfully inserted a test row into {}.".format(destination_table))
        db_connection.rollback()
        print("SQL insert successfully rolled back (since it was a test).")
        write_result = True
    except pyodbc.DatabaseError:
        write_result = False
        error_message = """Failed to insert values into {}. Check that the table exists with right column structure.
        Your Grain ID column might not match that in your input table.""".format(destination_table)
        raise HealthcareAIError(error_message)

    finally:
        try:
            db_connection.close()
            result = write_result
        except pyodbc.DatabaseError:
            error_message = """An attempt to complete a transaction has failed. No corresponding transaction found.
            \nPerhaps you don\'t have permission to write to this server."""
            raise HealthcareAIError(error_message)

    return result
