from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.externals import joblib
import math
import numpy as np
import pandas as pd


def clfreport(modeltype,
              debug,
              devcheck,
              algo,
              X_train,
              y_train,
              X_test,
              y_test=None,
              param=None,
              cores=4,
              tune=False,
              use_saved_model=False,
              col_list=None):
    """

    Parameters
    ----------
    modeltype
    debug
    devcheck
    algo
    X_train
    y_train
    X_test
    y_test
    param
    cores
    tune
    use_saved_model
    col_list

    Returns
    -------

    """

    # Initialize conditional vars that depend on ifelse to avoid PC warnng
    y_pred_class = None
    y_pred = None

    if devcheck == 'yesdev':

        if tune:
            clf = GridSearchCV(algo,
                               param,
                               cv=5,
                               scoring='roc_auc',
                               n_jobs=cores)
        else:
            clf = algo

        if debug:
            print('\nclf object right before fitting main model:')
            print(clf)

        if modeltype == 'classification':
            y_pred = np.squeeze(clf.fit(X_train, y_train).predict_proba(
                X_test)[:, 1])
            #y_pred_class = clf.fit(X_train, y_train).predict(X_test)
        elif modeltype == 'regression':
            y_pred = clf.fit(X_train, y_train).predict(X_test)

        print('\n', algo)
        print("Best hyper-parameters found after tuning:")

        if hasattr(clf, 'best_params_') and tune:
            print(clf.best_params_)
        else:
            print("No hyper-parameter tuning was done.")

        if modeltype == 'classification':
            print('\nMetrics:')
            roc_auc = roc_auc_score(y_test, y_pred)
            print('AU_ROC Score:', roc_auc)

            precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
            pr_auc = auc(recall, precision)
            print('\nAU_PR Score:', pr_auc)

        elif modeltype == 'regression':
            print('##########################################################')
            print('Model accuracy:')
            print('\nRMSE error:', math.sqrt(mean_squared_error(y_test,
                                                                y_pred_class)))
            print('\nMean absolute error:', mean_absolute_error(y_test,
                                                                y_pred), '\n')
            print('##########################################################')

        # TODO: refactor this logic to be simpler
        # Return without printing variable importance for linear case
        if (not hasattr(clf, 'feature_importances_')) and (not
            hasattr(clf, 'best_estimator_')):

            return y_pred, roc_auc

        # Print variable importance if rf and not tuning
        elif hasattr(clf, 'feature_importances_'):
            write_feature_importances(clf.feature_importances_,
                                          col_list)

            return y_pred, roc_auc, clf

        # Print variable importance if rf and tuning
        elif hasattr(clf.best_estimator_, 'feature_importances_'):
            write_feature_importances(
                clf.best_estimator_.feature_importances_,
                col_list)

            return y_pred, roc_auc, clf

    elif devcheck == 'notdev':

        clf = algo

        if use_saved_model is True:

            clf = joblib.load('probability.pkl')

        else:

            if debug:
                print('\nclf object right before fitting main model:')

            clf.fit(X_train, y_train)

            joblib.dump(clf, 'probability.pkl', compress=1)

        if modeltype == 'classification':
            y_pred = np.squeeze(clf.predict_proba(X_test)[:, 1])
        elif modeltype == 'regression':
            y_pred = clf.predict(X_test)

    return y_pred

def findtopthreefactors(debug,
                        X_train,
                        y_train,
                        X_test,
                        modeltype,
                        use_saved_model):

    # Initialize conditional vars that depend on ifelse to avoid PC warnng
    clf = None

    if modeltype == 'classification':
        clf = LogisticRegression()

    elif modeltype == 'regression':
        clf = LinearRegression()

    if use_saved_model is True:

        clf = joblib.load('factorlogit.pkl')

    elif use_saved_model is False:

        if modeltype == 'classification':

            if debug:
                print('\nclf object right before fitting factor ranking model')
                print(clf)

            clf.fit(X_train, y_train).predict_proba(X_test)

        elif modeltype == 'regression':

            if debug:
                print('\nclf object right before fitting factor ranking model')
                print(clf)

            clf.fit(X_train, y_train).predict(X_test)

        joblib.dump(clf, 'factorlogit.pkl', compress=1)

    if debug:
        print('\nCoeffs right before multiplic. to determine top 3 factors')
        print(clf.coef_)
        print('\nX_test right before this multiplication')
        print(X_test.loc[:3, :])

    # Populate X_test array of ordered col importance;
    # Start by multiplying X_test vals by coeffs
    res = X_test.values * clf.coef_

    if debug:
        print('\nResult of coef * Xtest row by row multiplication')
        for i in range(0, 3):
            print(res[i, :])

    col_list = X_test.columns.values

    first_fact = []
    second_fact = []
    third_fact = []

    if debug:
        print('\nSorting column importance rankings for each row in X_test...')

    # TODO: switch 2-d lists to numpy array
    # (although would always convert back to list for ceODBC
    for i in range(0, len(res[:, 1])):
        list_of_indexrankings = np.array((-res[i]).argsort().ravel())
        first_fact.append(col_list[list_of_indexrankings[0]])
        second_fact.append(col_list[list_of_indexrankings[1]])
        third_fact.append(col_list[list_of_indexrankings[2]])

    if debug:
        print('\nTop three factors for top five rows:')  # pretty-print w/ df
        print(pd.DataFrame({'first': first_fact[:3],
                            'second': second_fact[:3],
                            'third': third_fact[:3]}))

    return first_fact, second_fact, third_fact

def write_feature_importances(importance_attr, col_list):
    """
    This function prints an ordered list of rf-related feature importance.

    Parameters
    ----------
    importance_attr (attribute) : This is the feature importance attribute
    from a scikit-learn method that represents feature importances
    col_list (list) : Vector holding list of column names

    Returns
    -------
    Nothing. Simply prints feature importance list to console.
    """
    indices = np.argsort(importance_attr)[::-1]
    print('\nVariable importance:')
    for f in range(0, len(col_list)):
        print("%d. %s (%f)" % (f + 1, col_list[indices[f]],
                               importance_attr[indices[f]]))

def calculate_rfmtry(number_of_columns, type):
    if number_of_columns < 3:
        message = "You need more than two columns to tune hyperparameters."
        raise ValueError(message)

    if type == 'classification':
        # Default to grid of 1,2,3 for start of less than 2
        start_temp = math.floor(math.sqrt(number_of_columns))
        start = start_temp if start_temp >= 2 else 2
        grid_mtry = [start-1,start,start+1]

    if type == 'regression':
        # Default to grid of 1,2,3 for start of less than 2
        start_temp = math.floor(number_of_columns/3)
        start = start_temp if start_temp >= 2 else 2
        grid_mtry = [start-1,start,start+1]

    return grid_mtry

if __name__ == "__main__":
    pass
