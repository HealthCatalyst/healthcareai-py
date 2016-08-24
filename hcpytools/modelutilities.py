from sklearn.grid_search import GridSearchCV
from sklearn.metrics import roc_auc_score, cohen_kappa_score
from sklearn.metrics import precision_score, recall_score
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
            y_pred_class = clf.fit(X_train, y_train).predict(X_test)
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
            print('AUC Score:', roc_auc_score(y_test, y_pred))
            print('The following metrics are very sensitive to cut point:')
            print('Kappa:', cohen_kappa_score(y_test, y_pred_class))
            print('Recall/Sensitivity:', recall_score(y_test, y_pred_class))
            print('Precision/PPV:', precision_score(y_test,
                                                    y_pred_class),
                  '\n')

        elif modeltype == 'regression':
            print('##########################################################')
            print('Model accuracy:')
            print('\nRMSE error:', math.sqrt(mean_squared_error(y_test,
                                                                y_pred_class)))
            print('\nMean absolute error:', mean_absolute_error(y_test,
                                                                y_pred), '\n')
            print('##########################################################')

        # Print variable importance if it's an attribute
        if hasattr(clf, 'feature_importances_'):

            importances = clf.feature_importances_
            indices = np.argsort(importances)[::-1]
            print('Variable importance:')
            for f in range(0, X_train.shape[1]):
                print("%d. %s (%f)" % (f + 1, col_list[indices[f]],
                                       importances[indices[f]]))

            return y_pred, clf

        else:  # return dev case without feature importances

            return y_pred

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

if __name__ == "__main__":
    pass
