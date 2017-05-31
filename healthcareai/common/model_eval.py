import os
import sklearn

import numpy as np
import pandas as pd
import sklearn.metrics as skmetrics

from matplotlib import pyplot as plt

from healthcareai.common.healthcareai_error import HealthcareAIError

DIAGONAL_LINE_COLOR = '#bbbbbb'
DIAGONAL_LINE_STYLE = 'dotted'


def compute_roc(y_test, probability_predictions):
    """
    Compute TPRs, FPRs, best cutoff, ROC auc, and raw thresholds

    Args:
        y_test (list) : true label values corresponding to the predictions. Also length n.
        probability_predictions (list) : predictions coming from an ML algorithm of length n.

    Returns:
        dict: 

    """
    _validate_predictions_and_labels_are_equal_length(probability_predictions, y_test)

    # Calculate ROC
    false_positive_rates, true_positive_rates, roc_thresholds = skmetrics.roc_curve(y_test, probability_predictions)
    roc_auc = skmetrics.roc_auc_score(y_test, probability_predictions)

    # get ROC ideal cutoffs (upper left, or 0,1)
    roc_distances = (false_positive_rates - 0) ** 2 + (true_positive_rates - 1) ** 2

    # To prevent the case where there are two points with the same minimum distance, return only the first
    # np.where returns a tuple (we want the first element in the first array)
    roc_index = np.where(roc_distances == np.min(roc_distances))[0][0]
    best_tpr = true_positive_rates[roc_index]
    best_fpr = false_positive_rates[roc_index]
    ideal_roc_cutoff = roc_thresholds[roc_index]

    return {'roc_auc': roc_auc,
            'best_roc_cutoff': ideal_roc_cutoff,
            'best_true_positive_rate': best_tpr,
            'best_false_positive_rate': best_fpr,
            'true_positive_rates': true_positive_rates,
            'false_positive_rates': false_positive_rates,
            'roc_thresholds': roc_thresholds}


def compute_pr(y_test, probability_predictions):
    """ 
    Compute Precision-Recall, thresholds and PR AUC

    Args:
        y_test (list) : true label values corresponding to the predictions. Also length n.
        probability_predictions (list) : predictions coming from an ML algorithm of length n.

    Returns:
        dict: 

    """
    _validate_predictions_and_labels_are_equal_length(probability_predictions, y_test)

    # Calculate PR
    precisions, recalls, pr_thresholds = skmetrics.precision_recall_curve(y_test, probability_predictions)
    pr_auc = skmetrics.average_precision_score(y_test, probability_predictions)

    # get ideal cutoffs for suggestions (upper right or 1,1)
    pr_distances = (precisions - 1) ** 2 + (recalls - 1) ** 2

    # To prevent the case where there are two points with the same minimum distance, return only the first
    # np.where returns a tuple (we want the first element in the first array)
    pr_index = np.where(pr_distances == np.min(pr_distances))[0][0]
    best_precision = precisions[pr_index]
    best_recall = recalls[pr_index]
    ideal_pr_cutoff = pr_thresholds[pr_index]

    return {'pr_auc': pr_auc,
            'best_pr_cutoff': ideal_pr_cutoff,
            'best_precision': best_precision,
            'best_recall': best_recall,
            'precisions': precisions,
            'recalls': recalls,
            'pr_thresholds': pr_thresholds}


def calculate_regression_metrics(trained_sklearn_estimator, x_test, y_test):
    """
    Given a trained estimator, calculate metrics

    Args:
        trained_sklearn_estimator (sklearn.base.BaseEstimator): a scikit-learn estimator that has been `.fit()`
        y_test (numpy.ndarray): A 1d numpy array of the y_test set (predictions)
        x_test (numpy.ndarray): A 2d numpy array of the x_test set (features)

    Returns:
        dict: A dictionary of metrics objects
    """
    # Get predictions
    predictions = trained_sklearn_estimator.predict(x_test)

    # Calculate individual metrics
    mean_squared_error = skmetrics.mean_squared_error(y_test, predictions)
    mean_absolute_error = skmetrics.mean_absolute_error(y_test, predictions)

    result = {'mean_squared_error': mean_squared_error, 'mean_absolute_error': mean_absolute_error}

    return result


def calculate_binary_classification_metrics(trained_sklearn_estimator, x_test, y_test):
    """
    Given a trained estimator, calculate metrics

    Args:
        trained_sklearn_estimator (sklearn.base.BaseEstimator): a scikit-learn estimator that has been `.fit()`
        x_test (numpy.ndarray): A 2d numpy array of the x_test set (features)
        y_test (numpy.ndarray): A 1d numpy array of the y_test set (predictions)

    Returns:
        dict: A dictionary of metrics objects
    """
    # Squeeze down y_test to 1D
    y_test = np.squeeze(y_test)

    _validate_predictions_and_labels_are_equal_length(x_test, y_test)

    # Get binary and probability classification predictions
    binary_predictions = np.squeeze(trained_sklearn_estimator.predict(x_test))
    probability_predictions = np.squeeze(trained_sklearn_estimator.predict_proba(x_test)[:, 1])

    # Calculate accuracy
    accuracy = skmetrics.accuracy_score(y_test, binary_predictions)
    roc = compute_roc(y_test, probability_predictions)
    pr = compute_pr(y_test, probability_predictions)

    # Unpack the roc and pr dictionaries so the metric lookup is easier for plot and ensemble methods
    return {'accuracy': accuracy, **roc, **pr}


def roc_plot_from_thresholds(roc_thresholds_by_model, save=False, debug=False):
    # TODO consolidate this and PR plotter into 1 function
    # TODO make the colors randomly generated from rgb values
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    # Initialize plot
    plt.figure()
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TRP)')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.plot([0, 1], [0, 1], linestyle=DIAGONAL_LINE_STYLE, color=DIAGONAL_LINE_COLOR)

    # TODO hack to convert to array if it is a single dictionary
    if isinstance(roc_thresholds_by_model, dict):
        roc_thresholds_by_model = [roc_thresholds_by_model]

    # Calculate and plot for each model
    for i, model in enumerate(roc_thresholds_by_model):
        # Extract model name and metrics from dictionary
        model_name, metrics = model.popitem()
        roc_auc = metrics['roc_auc']
        tpr = metrics['true_positive_rates']
        fpr = metrics['false_positive_rates']
        best_true_positive_rate = metrics['best_true_positive_rate']
        best_false_positive_rate = metrics['best_false_positive_rate']

        if debug:
            print('{} model:'.format(model_name))
            print(pd.DataFrame({'FPR': fpr, 'TPR': tpr}))

        # TODO deal with colors ...
        # plot the line
        temp_color = colors[i]
        label = '{} (ROC AUC = {})'.format(model_name, round(roc_auc, 2))
        plt.plot(fpr, tpr, color=temp_color, label=label)
        plt.plot([best_false_positive_rate], [best_true_positive_rate], marker='*', markersize=10, color=temp_color)

    plt.legend(loc="lower right")
    # TODO: add cutoff associated with FPR/TPR

    if save:
        plt.savefig('ROC.png')
        source_path = os.path.dirname(os.path.abspath(__file__))
        print('\nROC plot saved in: {}'.format(source_path))

    plt.show()


def pr_plot_from_thresholds(pr_thresholds_by_model, save=False, debug=False):
    # TODO consolidate this and PR plotter into 1 function
    # TODO make the colors randomly generated from rgb values
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    # Initialize plot
    plt.figure()
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision Recall (PR)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.plot([0, 1], [1, 0], linestyle=DIAGONAL_LINE_STYLE, color=DIAGONAL_LINE_COLOR)

    # TODO hack to convert to array if it is a single dictionary
    if isinstance(pr_thresholds_by_model, dict):
        pr_thresholds_by_model = [pr_thresholds_by_model]

    # Calculate and plot for each model
    for i, model in enumerate(pr_thresholds_by_model):
        # Extract model name and metrics from dictionary
        model_name, metrics = model.popitem()
        pr_auc = metrics['pr_auc']
        precision = metrics['precisions']
        recall = metrics['recalls']
        best_recall = metrics['best_recall']
        best_precision = metrics['best_precision']

        if debug:
            print('{} model:'.format(model_name))
            print(pd.DataFrame({'Recall': recall, 'Precision': precision}))

        # plot the line
        temp_color = colors[i]
        label = '{} (PR AUC = {})'.format(model_name, round(pr_auc, 2))
        plt.plot(recall, precision, color=temp_color, label=label)
        plt.plot([best_recall], [best_precision], marker='*', markersize=10, color=temp_color)

    plt.legend(loc="lower left")
    # TODO: add cutoff associated with P/R

    if save:
        plt.savefig('PR.png')
        source_path = os.path.dirname(os.path.abspath(__file__))
        print('\nPR plot saved in: {}'.format(source_path))

    plt.show()


def plot_random_forest_feature_importance(trained_rf_classifier, x_train, feature_names, save=False):
    """
    Given a scikit learn random forest classifier, an x_train array, the feature names save or display a feature
    importance plot.
    
    Args:
        trained_rf_classifier (sklearn.ensemble.RandomForestClassifier): 
        x_train (numpy.array): A 2D numpy array that was used for training 
        feature_names (list): Column names in the x_train set
        save (bool): True to save the plot, false to display it in a blocking thread
    """
    # Unwrap estimator if it is a sklearn randomized search estimator
    # best_rf = get_estimator_from_trained_supervised_model(trained_rf_classifier)
    best_rf = trained_rf_classifier
    # Validate estimator is a random forest classifier and raise error if it is not
    if not isinstance(best_rf, sklearn.ensemble.RandomForestClassifier):
        print(type(trained_rf_classifier))
        raise HealthcareAIError('Feature plotting only works with a scikit learn RandomForestClassifier.')

    # Arrange columns in order of importance
    # TODO this portion could probably be extracted and tested, since the plot is difficult to test
    importances = best_rf.feature_importances_
    feature_importances = [tree.feature_importances_ for tree in best_rf.estimators_]
    standard_deviations = np.std(feature_importances, axis=0)
    indices = np.argsort(importances)[::-1]
    namelist = [feature_names[i] for i in indices]

    # Turn off matplotlib interactive mode
    plt.ioff()

    # Set up the plot
    figure = plt.figure()
    plt.title("Feature Importance")
    plt.ylabel('Relative Feature Importance')

    # Plot each feature
    x_train_shape = x_train.shape[1]
    x_train_range = range(x_train_shape)

    plt.bar(x_train_range, importances[indices], color="g", yerr=standard_deviations[indices], align="center")
    plt.xticks(x_train_range, namelist, rotation=90)
    plt.xlim([-1, x_train_shape])
    plt.gca().set_ylim(bottom=0)
    plt.tight_layout()

    # Save or display the plot
    if save:
        plt.savefig('FeatureImportances.png')
        source_path = os.path.dirname(os.path.abspath(__file__))
        print('\nFeature importance plot saved in: {}'.format(source_path))

        # Close the figure so it does not get displayed
        plt.close(figure)
    else:
        plt.show()


def _validate_predictions_and_labels_are_equal_length(predictions, true_values):
    if len(predictions) == len(true_values):
        return True
    else:
        raise HealthcareAIError('The number of predictions is not equal to the number of true_values.')


if __name__ == '__main__':
    pass
