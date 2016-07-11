from sklearn.grid_search import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.externals import joblib
from scipy import linalg, optimize
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
              y_test='',
              param='',
              cores=4,
              tune=False,
              use_saved_model=False,
              col_list=''):

    if devcheck == 'yesdev':

        if tune: clf = GridSearchCV(algo, param, cv=5, scoring='roc_auc', n_jobs=cores)
        else: clf = algo

        if debug:
            print('\nclf object right before fitting main model:')
            print(clf)

        if modeltype == 'classification':
            y_pred = np.squeeze(clf.fit(X_train, y_train).predict_proba(X_test)[:,1])
        elif modeltype == 'regression':
            y_pred = clf.fit(X_train, y_train).predict(X_test)

        print('\n',algo)
        print("Best hyper-parameters found after tuning:")

        if hasattr(clf, 'best_params_') and tune:
            print(clf.best_params_)
        else:
            print("No hyper-parameter tuning was done.")

        if modeltype == 'classification':
            print('\nAUC Score:', roc_auc_score(y_test, y_pred), '\n')
        elif modeltype == 'regression':
            print('##########################################################')
            print('Model accuracy:')
            print('\nRMSE error:', math.sqrt(mean_squared_error(y_test, y_pred)))
            print('\nMean absolute error:', mean_absolute_error(y_test, y_pred), '\n')
            print('##########################################################')


        # Print variable importance if it's an attribute
        if hasattr(clf, 'feature_importances_'):

            importances = clf.feature_importances_
            indices = np.argsort(importances)[::-1]
            print('Variable importance:')
            for f in range(0,X_train.shape[1]):
                print("%d. %s (%f)" % (f + 1, col_list[indices[f]], importances[indices[f]]))

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
            y_pred = np.squeeze(clf.predict_proba(X_test)[:,1])
        elif modeltype == 'regression':
            y_pred = clf.predict(X_test)

    return y_pred

def findtopthreefactors(debug,
                        X_train,
                        y_train,
                        X_test,
                        impute,
                        modeltype,
                        use_saved_model):

    if modeltype == 'classification': clf = LogisticRegression()

    elif modeltype == 'regression': clf = LinearRegression()

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
        print('\nCoeffs right before multiplication to determine top 3 factors')
        print(clf.coef_)
        print('\nX_test right before this multiplication')
        print(X_test.loc[:3,:])

    # Populate X_test array of ordered col importance; start by multiplying X_test vals by coeffs
    res = X_test.values * clf.coef_

    if debug:
        print('\nResult of coef * Xtest row by row multiplication')
        for i in range(0,3):
            print(res[i,:])

    col_list = X_test.columns.values

    first_fact=[]
    second_fact = []
    third_fact = []

    if debug:
        print('\nSorting column importance rankings for each row in X_test...')

    # TODO: switch 2-d lists to numpy array (although would always convert back to list for ceODBC
    for i in range(0,len(res[:,1])):
        list_of_indexrankings = np.array((-res[i]).argsort().ravel())
        first_fact.append(col_list[list_of_indexrankings[0]])
        second_fact.append(col_list[list_of_indexrankings[1]])
        third_fact.append(col_list[list_of_indexrankings[2]])

    if debug:
        print('\nTop three factors for top five rows:') # pretty-printing via pandas dataframe
        print(pd.DataFrame({'first': first_fact[:3], 'second': second_fact[:3], 'third': third_fact[:3]}))

    return first_fact, second_fact, third_fact

def group_lasso(X, y, alpha, groups, max_iter=1000, rtol=1e-15,
                verbose=False):
    """
    Linear least-squares with l2/l1 regularization solver.
    Solves problem of the form:
               .5 * |Xb - y| + n_samples * alpha * Sum(w_j * |b_j|)
    where |.| is the l2-norm and b_j is the coefficients of b in the
    j-th group. This is commonly known as the `group lasso`.
    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        Design Matrix.
    y : array of shape (n_samples,)
    alpha : float or array
        Amount of penalization to use.
    groups : array of shape (n_features,)
        Group label. For each column, it indicates
        its group apertenance.
    rtol : float
        Relative tolerance. ensures ||(x - x_) / x_|| < rtol,
        where x_ is the approximate solution and x is the
        true solution.
    Returns
    -------
    x : array
        vector of coefficients
    References
    ----------
    "Efficient Block-coordinate Descent Algorithms for the Group Lasso",
    Qin, Scheninberg, Goldfarb
    """

    # .. local variables ..
    X, y, groups, alpha = list(map(np.asanyarray, (X, y, groups, alpha)))
    if len(groups) != X.shape[1]:
        raise ValueError("Incorrect shape for groups")
    w_new = np.zeros(X.shape[1], dtype=X.dtype)
    alpha = alpha * X.shape[0]

    # .. use integer indices for groups ..
    group_labels = [np.where(groups == i)[0] for i in np.unique(groups)]
    H_groups = [np.dot(X[:, g].T, X[:, g]) for g in group_labels]
    eig = list(map(linalg.eigh, H_groups))
    Xy = np.dot(X.T, y)
    initial_guess = np.zeros(len(group_labels))

    def f(x, qp2, eigvals, alpha):
        return 1 - np.sum(qp2 / ((x * eigvals + alpha) ** 2))

    def df(x, qp2, eigvals, penalty):
        # .. first derivative ..
        return np.sum((2 * qp2 * eigvals) / ((penalty + x * eigvals) ** 3))

    if X.shape[0] > X.shape[1]:
        H = np.dot(X.T, X)
    else:
        H = None

    for n_iter in range(max_iter):
        w_old = w_new.copy()
        for i, g in enumerate(group_labels):
            # .. shrinkage operator ..
            eigvals, eigvects = eig[i]
            w_i = w_new.copy()
            w_i[g] = 0.
            if H is not None:
                X_residual = np.dot(H[g], w_i) - Xy[g]
            else:
                X_residual = np.dot(X.T, np.dot(X[:, g], w_i)) - Xy[g]
            qp = np.dot(eigvects.T, X_residual)
            if len(g) < 2:
                # for single groups we know a closed form solution
                w_new[g] = - np.sign(X_residual) * max(abs(X_residual) - alpha, 0)
            else:
                if alpha < linalg.norm(X_residual, 2):
                    initial_guess[i] = optimize.newton(f, initial_guess[i], df, tol=.5,
                                                       args=(qp ** 2, eigvals, alpha))
                    w_new[g] = - initial_guess[i] * np.dot(eigvects / (eigvals * initial_guess[i] + alpha), qp)
                else:
                    w_new[g] = 0.

        # .. dual gap ..
        max_inc = linalg.norm(w_old - w_new, np.inf)
        if True:  # max_inc < rtol * np.amax(w_new):
            residual = np.dot(X, w_new) - y
            group_norm = alpha * np.sum([linalg.norm(w_new[g], 2)
                                         for g in group_labels])
            if H is not None:
                norm_Anu = [linalg.norm(np.dot(H[g], w_new) - Xy[g]) \
                            for g in group_labels]
            else:
                norm_Anu = [linalg.norm(np.dot(H[g], residual)) \
                            for g in group_labels]
            if np.any(norm_Anu > alpha):
                nnu = residual * np.min(alpha / norm_Anu)
            else:
                nnu = residual
            primal_obj = .5 * np.dot(residual, residual) + group_norm
            dual_obj = -.5 * np.dot(nnu, nnu) - np.dot(nnu, y)
            dual_gap = primal_obj - dual_obj
            if verbose:
                print
                'Relative error: %s' % (dual_gap / dual_obj)
            if np.abs(dual_gap / dual_obj) < rtol:
                break

    return w_new

def check_kkt(A, b, x, penalty, groups):
    """Check KKT conditions for the group lasso
    Returns True if conditions are satisfied, False otherwise
    """
    group_labels = [groups == i for i in np.unique(groups)]
    penalty = penalty * A.shape[0]
    z = np.dot(A.T, np.dot(A, x) - b)
    safety_net = 1e-1  # sort of tolerance
    for g in group_labels:
        if linalg.norm(x[g]) == 0:
            if not linalg.norm(z[g]) < penalty + safety_net:
                return False
        else:
            w = - penalty * x[g] / linalg.norm(x[g], 2)
            if not np.allclose(z[g], w, safety_net):
                return False
    return True

def gl_predict(coeffs):

    pass

if __name__ == "__main__":
    pass

