import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression

from healthcareai.common.output_utilities import load_pickle_file, save_object_as_pickle


def find_top_three_factors(debug,
                           X_train,
                           y_train,
                           X_test,
                           model_type,
                           use_saved_model):

    # Initialize conditional vars that depend on ifelse to avoid PC warnng
    clf = None

    if model_type == 'classification':
        clf = LogisticRegression()
    elif model_type == 'regression':
        clf = LinearRegression()

    if use_saved_model is True:
        clf = load_pickle_file('factorlogit.pkl')
    elif use_saved_model is False:
        if debug:
            print('\nclf object right before fitting factor ranking model')
            print(clf)

        if model_type == 'classification':
            clf.fit(X_train, y_train).predict_proba(X_test)
        elif model_type == 'regression':
            clf.fit(X_train, y_train).predict(X_test)
            save_object_as_pickle('factorlogit.pkl', clf)

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