from sklearn import cross_validation
from sklearn.linear_model import LinearRegression, LogisticRegressionCV, RandomizedLogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
import numpy as np
import pandas as pd
from hcpytools import modelutilities
from hcpytools.impute_custom import DataFrameImputer


class DevelopSupervisedModel(object):
    """ This class helps create a model using several common classifiers (reporting AUC)
    and regressors (reporting MAE/MSE);

    Parameters
    ----------
    df : pandas dataframe

    predicted : y column who's values are being predicted

    modeltype : whether the model will be 'classification' or 'regression'

    impute : whether imputation is done on the data; if not, rows with nulls are removed

    Returns
    -------
    self : object
    """
    def __init__(self,
                 modeltype,
                 df,
                 predictedcol,
                 impute,
                 graincol='',
                 debug = False):

        if debug:
            print('Shape and top 5 rows of original dataframe:')
            print(df.shape)
            print(df.head())

        # Remove DTS columns
        # TODO: make this work with col names shorter than three letters (ie when headers aren't read in)
        cols = [c for c in df.columns if c[-3:] != 'DTS']
        df = df[cols]

        # Remove graincol (if specified)
        if graincol:
            pd.options.mode.chained_assignment = None  # default='warn' # This is ignoring this message:
            # TODO: Fix this SettingWithCopyWarning: value trying to be set on a copy of a slice from a DataFrame"
            df.drop(graincol, axis=1, inplace=True)

        if debug:
            print('\nDataframe after removing DTS columns:')
            print(df.head())
            print('\nNow either doing imputation or dropping rows with NULLs...')

        if impute:
            df = DataFrameImputer().fit_transform(df)
            # This class comes from here: http://stackoverflow.com/a/25562948/5636012
            if debug:
                print('\ndf after doing imputation:')
                print(df.shape)
                print(df.head())
        else:
            #TODO switch similar statements to work inplace (w/o making a copy)
            df = df.dropna(axis=0, how='any', inplace=True)
            print('\ndf after dropping rows with NULLS:')
            print(df.shape)
            print(df.head())

        # Convert predicted col to 0/1 (otherwise won't work with GridSearchCV)
        # Note that this makes it such that hcpytools only handles N/Y in pred column
        if modeltype == 'classification':
            # Turning off warning around replace
            pd.options.mode.chained_assignment = None  # default='warn'
            # TODO: put try/catch here when type = class and numeric predictor is selected
            df[predictedcol].replace(['Y','N'],[1,0], inplace=True)

            if debug:
                print('\nDataframe after converting to 1/0 instead of Y/N for classification:')
                print(df.head())

        # Remove rows with null values in predicted col
        df = df[pd.notnull(df[predictedcol])]

        if debug:
            print('\ndf after removing rows where predicted col is NULL:')
            print(df.shape)
            print(df.head())

        y = np.squeeze(df[[predictedcol]])
        X = df.drop([predictedcol], axis=1)
        X = pd.get_dummies(X, drop_first=True, prefix_sep='.')

        # Split the dataset in two equal parts
        self.X_train, self.X_test, self.y_train, self.y_test = cross_validation.train_test_split(
            X, y, test_size=.20, random_state=0)

        if debug:
            print('\nShape of X_train, y_train, X_test, and y_test:')
            print(self.X_train.shape)
            print(self.y_train.shape)
            print(self.X_test.shape)
            print(self.y_test.shape)

        self.df = df
        self.predictedcol = predictedcol
        self.modeltype = modeltype
        self.impute = impute

    def linear(self, cores, debug=False, tune=False):

        if self.modeltype == 'classification':
            algo = LogisticRegressionCV(cv=5)

            # TODO: Get RandomizedLogistic (auto feature selection) working in place of linear
            # pipe = Pipeline(steps=[('RLR', RandomizedLogisticRegression),
            #                        ('logistic', LogisticRegressionCV)])
            # params = dict(RLR__scaling=[0.3, 0.5, 0.7])
            # algo = GridSearchCV(estimator=pipe, param_grid=params, cv=5)

        # TODO: see if CV splits needed for linear regress (if we stay with it instead of lasso)
        elif self.modeltype == 'regression':
            algo = LinearRegression()

        modelutilities.clfreport(modeltype=self.modeltype,
                                 debug=debug,
                                 devcheck='yesdev',
                                 algo=algo,
                                 X_train=self.X_train,
                                 y_train=self.y_train,
                                 X_test=self.X_test,
                                 y_test=self.y_test,
                                 cores=cores)

    def randomforest(self, cores, trees=200, tune=False, debug=False):

        if self.modeltype == 'classification':
            algo = RandomForestClassifier(n_estimators=trees, verbose=(2 if debug is True else 0))

        elif self.modeltype == 'regression':
            algo = RandomForestRegressor(n_estimators=trees, verbose=(2 if debug is True else 0))

        params = {'n_estimators': [10,50,100,250,500]}

        col_list = self.X_train.columns.values

        modelutilities.clfreport(modeltype=self.modeltype,
                                 debug=debug,
                                 devcheck='yesdev',
                                 algo=algo,
                                 X_train=self.X_train,
                                 y_train=self.y_train,
                                 X_test=self.X_test,
                                 y_test=self.y_test,
                                 param=params,
                                 cores=cores,
                                 tune=tune,
                                 col_list=col_list)





