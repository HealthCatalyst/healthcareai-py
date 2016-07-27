from sklearn import cross_validation
from sklearn.linear_model import LinearRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hcpytools import modelutilities
from hcpytools.impute_custom import DataFrameImputer
import os

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

        self.y_probab_linear = modelutilities.clfreport(modeltype=self.modeltype,
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

        self.col_list = self.X_train.columns.values

        self.y_probab_rf, self.rfclf = modelutilities.clfreport(modeltype=self.modeltype,
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
                                                                             col_list=self.col_list)

    def plotROC(self, debug=False, save=False):
        fpr_linear, tpr_linear, _ = roc_curve(self.y_test, self.y_probab_linear)
        roc_auc_linear = auc(fpr_linear, tpr_linear)

        fpr_rf, tpr_rf, _ = roc_curve(self.y_test, self.y_probab_rf)
        roc_auc_rf = auc(fpr_rf, tpr_rf)

        if debug:
            print('Linear model:')
            print('FPR, and TRP')
            print(pd.DataFrame({'FPR': fpr_linear, 'TPR': tpr_linear})) #Add cutoff

            print('Random forest model:')
            print('FPR, and TRP')
            print(pd.DataFrame({'FPR': fpr_rf, 'TPR': tpr_rf})) #Add cutoff

        plt.figure()
        plt.plot(fpr_linear, tpr_linear, color='b', label='Logistic (area = %0.2f)' % roc_auc_linear)
        plt.plot(fpr_rf, tpr_rf, color='g', label='RandomForest (area = %0.2f)' % roc_auc_rf)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        if save:
            plt.savefig('ROC.png')
            source_path = os.path.dirname(os.path.abspath(__file__))
            print('\nROC file saved in: {}'.format(source_path))
            plt.show()
        else:
            plt.show()

    def rfFeatureImportance(self, save=False):
        importances = self.rfclf.feature_importances_
        std = np.std([tree.feature_importances_ for tree in self.rfclf.estimators_],
                     axis=0)
        indices = np.argsort(importances)[::-1]
        namelist = [self.col_list[i] for i in indices]
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(self.X_train.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
        plt.xticks(range(self.X_train.shape[1]), namelist, rotation=90)
        plt.xlim([-1, self.X_train.shape[1]])
        plt.gca().set_ylim(bottom=0)
        plt.tight_layout()
        if save:
            plt.savefig('FeatureImportances.png')
            source_path = os.path.dirname(os.path.abspath(__file__))
            print('\nFeature importances file saved in: {}'.format(source_path))
            plt.show()
        else:
            plt.show()
