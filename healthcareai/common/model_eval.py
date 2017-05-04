import math
import os

import numpy as np
import pandas as pd
import sklearn
from matplotlib import pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import GridSearchCV

from healthcareai.common.top_factors import write_feature_importances
from healthcareai.common.file_io_utilities import save_object_as_pickle, load_pickle_file


def clfreport(model_type,
              debug,
              develop_model_mode,
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
    Given a model type, algorithm and test data, do/return/save/side effect the following in no particular order:
    - [x] runs grid search
    - [x] save/load a pickled model
    - [ ] print out debug messages
    - [x] train the classifier
    - [ ] print out grid params
    - [ ] calculate metrics
    - [ ] feature importances
    - [ ] logging
    - [x] production predictions from pickle file
    - do some numpy manipulation
        - lines ~50?
    - possible returns:
        - a single prediction
        - a prediction and an roc_auc score
        - spits out feature importances (if they exist)
        - saves a pickled model

    Note this serves at least 3 uses
    """

    # Initialize conditional vars that depend on ifelse to avoid PC warning
    y_pred_class = None
    y_pred = None
    algorithm = algo

    # compare algorithms
    if develop_model_mode is True:
        if tune:
            # Set up grid search
            algorithm = GridSearchCV(algo, param, cv=5, scoring='roc_auc', n_jobs=cores)

        if debug:
            print('\nalgorithm object right before fitting main model:')
            print(algorithm)

        print('\n', algo)

        if model_type == 'classification':
            y_pred = np.squeeze(algorithm.fit(X_train, y_train).predict_proba(X_test)[:, 1])

            roc_auc = roc_auc_score(y_test, y_pred)
            precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
            pr_auc = auc(recall, precision)

            print_classification_metrics(pr_auc, roc_auc)
        elif model_type == 'regression':
            y_pred = algorithm.fit(X_train, y_train).predict(X_test)

            print_regression_metrics(y_pred, y_pred_class, y_test)

        if hasattr(algorithm, 'best_params_') and tune:
            print("Best hyper-parameters found after tuning:")
            print(algorithm.best_params_)
        else:
            print("No hyper-parameter tuning was done.")

        # TODO: refactor this logic to be simpler
        # These returns are TIGHTLY coupled with their uses in develop and deploy. Both will have to be unwound together
        has_importances = hasattr(algorithm, 'feature_importances_')
        has_best_estimator = hasattr(algorithm, 'best_estimator_')

        if not has_importances and not has_best_estimator:
            # Return without printing variable importance for linear case
            return y_pred, roc_auc
        elif has_importances:
            # Print variable importance if rf and not tuning
            write_feature_importances(algorithm.feature_importances_, col_list)
            return y_pred, roc_auc, algorithm
        elif hasattr(algorithm.best_estimator_, 'feature_importances_'):
            # Print variable importance if rf and tuning
            write_feature_importances(algorithm.best_estimator_.feature_importances_, col_list)
            return y_pred, roc_auc, algorithm

    elif develop_model_mode is False:
        y_pred = do_deploy_mode_stuff(X_test, X_train, algorithm, debug, model_type, use_saved_model, y_pred, y_train)

    # TODO is it possible to get to this return if you are in develop_model_mode?
    return y_pred


def do_deploy_mode_stuff(X_test, X_train, algorithm, debug, model_type, use_saved_model, y_pred, y_train):
    if use_saved_model is True:
        algorithm = load_pickle_file('probability.pkl')
    else:
        if debug:
            print('\nclf object right before fitting main model:')

        algorithm.fit(X_train, y_train)
        save_object_as_pickle('probability.pkl', algorithm)

    if model_type == 'classification':
        y_pred = np.squeeze(algorithm.predict_proba(X_test)[:, 1])
    elif model_type == 'regression':
        y_pred = algorithm.predict(X_test)
    return y_pred


def print_regression_metrics(y_pred, y_pred_class, y_test):
    print('##########################################################')
    print('Model accuracy:')
    print('\nRMSE error:', math.sqrt(mean_squared_error(y_test, y_pred_class)))
    print('\nMean absolute error:', mean_absolute_error(y_test, y_pred), '\n')
    print('##########################################################')


def print_classification_metrics(pr_auc, roc_auc):
    print('\nMetrics:')
    print('AU_ROC ScoreX:', roc_auc)
    print('\nAU_PR Score:', pr_auc)


def GenerateAUC(predictions, labels, aucType='SS', plotFlg=False, allCutoffsFlg=False):
    # TODO refactor this
    """
    This function creates an ROC or PR curve and calculates the area under it.

    Parameters
    ----------
    predictions (list) : predictions coming from an ML algorithm of length n.
    labels (list) : true label values corresponding to the predictions. Also length n.
    aucType (str) : either 'SS' for ROC curve or 'PR' for precision recall curve. Defaults to 'SS'
    plotFlg (bol) : True will return plots. Defaults to False.
    allCutoffsFlg (bol) : True will return plots. Defaults to False.

    Returns
    -------
    AUC (float) : either AU_ROC or AU_PR
    """
    # Error check for uneven length predictions and labels
    if len(predictions) != len(labels):
        raise Exception('Data vectors are not equal length!')

    # make AUC type upper case.
    aucType = aucType.upper()

    # check to see if AUC is SS or PR. If not, default to SS
    if aucType not in ['SS', 'PR']:
        print('Drawing ROC curve with Sensitivity/Specificity')
        aucType = 'SS'

    # Compute ROC curve and ROC area
    if aucType == 'SS':
        fpr, tpr, thresh = roc_curve(labels, predictions)
        area = auc(fpr, tpr)
        print('Area under ROC curve (AUC): %0.2f' % area)
        # get ideal cutoffs for suggestions
        d = (fpr - 0) ** 2 + (tpr - 1) ** 2
        ind = np.where(d == np.min(d))
        bestTpr = tpr[ind]
        bestFpr = fpr[ind]
        cutoff = thresh[ind]
        print("Ideal cutoff is %0.2f, yielding TPR of %0.2f and FPR of %0.2f" % (cutoff, bestTpr, bestFpr))
        if allCutoffsFlg is True:
            print('%-7s %-6s %-5s' % ('Thresh', 'TPR', 'FPR'))
            for i in range(len(thresh)):
                print('%-7.2f %-6.2f %-6.2f' % (thresh[i], tpr[i], fpr[i]))

        # plot ROC curve
        if plotFlg is True:
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange',
                     lw=2, label='ROC curve (area = %0.2f)' % area)
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic curve')
            plt.legend(loc="lower right")
            plt.show()
        return ({'AU_ROC': area,
                 'BestCutoff': cutoff[0],
                 'BestTpr': bestTpr[0],
                 'BestFpr': bestFpr[0]})
    # Compute PR curve and PR area
    else:  # must be PR
        # Compute Precision-Recall and plot curve
        precision, recall, thresh = precision_recall_curve(labels, predictions)
        area = average_precision_score(labels, predictions)
        print('Area under PR curve (AU_PR): %0.2f' % area)
        # get ideal cutoffs for suggestions
        d = (precision - 1) ** 2 + (recall - 1) ** 2
        ind = np.where(d == np.min(d))
        bestPre = precision[ind]
        bestRec = recall[ind]
        cutoff = thresh[ind]
        print("Ideal cutoff is %0.2f, yielding TPR of %0.2f and FPR of %0.2f"
              % (cutoff, bestPre, bestRec))
        if allCutoffsFlg is True:
            print('%-7s %-10s %-10s' % ('Thresh', 'Precision', 'Recall'))
            for i in range(len(thresh)):
                print('%5.2f %6.2f %10.2f' % (thresh[i], precision[i], recall[i]))

        # plot PR curve
        if plotFlg is True:
            # Plot Precision-Recall curve
            plt.figure()
            plt.plot(recall, precision, lw=2, color='darkred',
                     label='Precision-Recall curve' % area)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.title('Precision-Recall AUC={0:0.2f}'.format(
                area))
            plt.legend(loc="lower right")
            plt.show()
        return ({'AU_PR': area,
                 'BestCutoff': cutoff[0],
                 'BestPrecision': bestPre[0],
                 'BestRecall': bestRec[0]})


def calculate_regression_metrics(trained_model, x_test, y_test):
    """
    Given a trained model, calculate metrics

    Args:
        trained_model (sklearn.base.BaseEstimator): a scikit-learn estimator that has been `.fit()`
        y_test (numpy.ndarray): A 1d numpy array of the y_test set (predictions)
        x_test (numpy.ndarray): A 2d numpy array of the x_test set (features)

    Returns:
        dict: A dictionary of metrics objects
    """
    # Get predictions
    predictions = trained_model.predict(x_test)

    # Calculate individual metrics
    mean_squared_error = sklearn.metrics.mean_squared_error(y_test, predictions)
    mean_absolute_error = sklearn.metrics.mean_absolute_error(y_test, predictions)

    result = {'mean_squared_error': mean_squared_error, 'mean_absolute_error': mean_absolute_error}

    return result


def calculate_classification_metrics(trained_model, x_test, y_test):
    """
    Given a trained model, calculate metrics

    Args:
        trained_model (sklearn.base.BaseEstimator): a scikit-learn estimator that has been `.fit()`
        x_test (numpy.ndarray): A 2d numpy array of the x_test set (features)
        y_test (numpy.ndarray): A 1d numpy array of the y_test set (predictions)

    Returns:
        dict: A dictionary of metrics objects
    """
    # Get binary classification predictions
    # TODO make predict_proba for user and predict for metric calculations
    predictions = np.squeeze(trained_model.predict(x_test))
    y_test = np.squeeze(y_test)

    # Build a dictionary of calculated metrics
    roc_auc = sklearn.metrics.roc_auc_score(y_test, predictions)
    accuracy = sklearn.metrics.accuracy_score(y_test, predictions)

    return {'roc_auc': roc_auc, 'accuracy': accuracy}


def display_roc_plot(y_test, y_probab_linear, y_probab_rf, save=False, debug=False):
    """
    Generates a ROC plot for linear and random forest models

    Args:
        y_test (list): A 1d list of predictions
        y_probab_linear: 
        y_probab_rf: 
        save: Whether to save the plot
        debug: Verbosity of output. If True, shows list of FPR/TPR for each point in the plot (default False)

    Returns:
        matplotlib.figure.Figure: The matplot figure
    """

    # TODO refactor this to take an arbitrary number of models rather than just a linear and random forest

    # Linear model calculations
    fpr_linear, tpr_linear, _ = sklearn.metrics.roc_curve(y_test, y_probab_linear)
    roc_auc_linear = sklearn.metrics.auc(fpr_linear, tpr_linear)

    # Random forest model calculations
    fpr_rf, tpr_rf, _ = sklearn.metrics.roc_curve(y_test, y_probab_rf)
    roc_auc_rf = sklearn.metrics.auc(fpr_rf, tpr_rf)

    # TODO: add cutoff associated with FPR/TPR

    if debug:
        print('Linear model:')
        print('FPR, and TRP')
        print(pd.DataFrame({'FPR': fpr_linear, 'TPR': tpr_linear}))
        print('Random forest model:')
        print('FPR, and TRP')
        print(pd.DataFrame({'FPR': fpr_rf, 'TPR': tpr_rf}))

    plt.figure()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")

    plt.plot(fpr_linear, tpr_linear, color='b', label='Logistic (area = %0.2f)' % roc_auc_linear)
    plt.plot(fpr_rf, tpr_rf, color='g', label='RandomForest (area = %0.2f)' % roc_auc_rf)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    if save:
        plt.savefig('ROC.png')
        source_path = os.path.dirname(os.path.abspath(__file__))
        print('\nROC file saved in: {}'.format(source_path))
        plt.show()
    else:
        plt.show()

    # return figure if anyone wants to save or manipulate it in another way
    # return figure


if __name__ == "__main__":
    pass
