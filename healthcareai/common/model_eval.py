import os
import sklearn

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import sklearn.metrics as skmetrics

from healthcareai.common.healthcareai_error import HealthcareAIError


def compute_roc(true_values, predictions):
    """
    Compute TPRs, FPRs, best cutoff and AUC
    
    Args:
        predictions (list) : predictions coming from an ML algorithm of length n.
        true_values (list) : true label values corresponding to the predictions. Also length n.

    Returns:
        dict: 

    """
    validate_predictions_and_labels_are_equal_length(predictions, true_values)

    false_positive_rates, true_postitive_rates, thresholds = skmetrics.roc_curve(true_values, predictions)
    area = skmetrics.roc_auc_score(true_values, predictions)

    # get ideal cutoffs for suggestions (upper left, or 0,1)
    d = (false_positive_rates - 0) ** 2 + (true_postitive_rates - 1) ** 2

    # TODO this might have the same bug the r package had
    ind = np.where(d == np.min(d))[0]
    best_tpr = true_postitive_rates[ind]
    best_fpr = false_positive_rates[ind]
    cutoff = thresholds[ind]

    return {'ROC_AUC': area,
            'best_cutoff': cutoff[0],
            'best_true_positive_rate': best_tpr[0],
            'best_false_positive_rate': best_fpr[0],
            'tpr': true_postitive_rates,
            'fpr': false_positive_rates,
            'thresholds': thresholds
            }


def compute_pr(predictions, true_values):
    """ 
    Compute Precision-Recall, thresholds and AUC

    Args:
        predictions (list) : predictions coming from an ML algorithm of length n.
        true_values (list) : true label values corresponding to the predictions. Also length n.

    Returns:
        dict: 

    """

    validate_predictions_and_labels_are_equal_length(predictions, true_values)

    precision, recall, thresholds = skmetrics.precision_recall_curve(true_values, predictions)
    area = skmetrics.average_precision_score(true_values, predictions)

    # get ideal cutoffs for suggestions (upper right or 1,1)
    d = (precision - 1) ** 2 + (recall - 1) ** 2

    # TODO this might have the same bug the r package had
    ind = np.where(d == np.min(d))[0]
    best_precision = precision[ind]
    best_recall = recall[ind]
    ideal_cutoff = thresholds[ind]

    return {'PR_AUC': area,
            'best_cutoff': ideal_cutoff[0],
            'best_precision': best_precision[0],
            'best_recall': best_recall[0],
            'precisions': precision,
            'recalls': recall,
            'thresholds': thresholds
            }


def validate_predictions_and_labels_are_equal_length(predictions, true_values):
    if len(predictions) != len(true_values):
        raise HealthcareAIError('The number of predictions is not equal to the number of true_values.')


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


def calculate_classification_metrics(trained_sklearn_estimator, x_test, y_test):
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

    # Get binary classification predictions
    binary_predictions = np.squeeze(trained_sklearn_estimator.predict(x_test))

    # Get probability classification predictions
    probability_predictions = np.squeeze(trained_sklearn_estimator.predict_proba(x_test)[:, 1])

    # Calculate some metrics
    precision, recall, thresholds = skmetrics.precision_recall_curve(y_test, probability_predictions)
    pr_auc = skmetrics.auc(recall, precision)
    roc_auc = skmetrics.roc_auc_score(y_test, probability_predictions)
    accuracy = skmetrics.accuracy_score(y_test, binary_predictions)

    return {
        'roc_auc': roc_auc,
        'accuracy': accuracy,
        'pr_auc': pr_auc,
    }


def tsm_classification_comparison_plots(trained_supervised_model, plot_type='ROC'):
    """
    Given a single or list of trained supervised models, plot a ROC or PR curve for each one
    
    Args:
        plot_type (str): 'ROC' (default) or 'PR' 
        trained_supervised_model (list | TrainedSupervisedModel): a single or list of TrainedSupervisedModels 
    """
    # Input validation plus switching
    if plot_type == 'ROC':
        plotter = roc_plot_from_predictions
    elif plot_type == 'PR':
        plotter = pr_plot_from_predictions
    else:
        raise HealthcareAIError('Please choose either plot_type=\'ROC\' or plot_type=\'PR\'')

    predictions_by_model = []
    # TODO doing this properly leads to a circular dependency so dirty hack string matching was needed
    # if isinstance(trained_supervised_model, TrainedSupervisedModel):
    if type(trained_supervised_model).__name__ == 'TrainedSupervisedModel':
        entry = build_classification_model_prediction_dictionary(trained_supervised_model)
        predictions_by_model.append(entry)
        test_set_actual = trained_supervised_model.test_set_actual
    elif isinstance(trained_supervised_model, list):
        for model in trained_supervised_model:
            # TODO doing this properly leads to a circular dependency so dirty hack string matching was needed
            # if isinstance(trained_supervised_model, TrainedSupervisedModel):
            if type(model).__name__ != 'TrainedSupervisedModel':
                raise HealthcareAIError('One of the objects in the list is not a TrainedSupervisedModel')

            entry = build_classification_model_prediction_dictionary(model)
            predictions_by_model.append(entry)

            # TODO so, you could check for different GUIDs that could be saved in each TSM!
            # The assumption here is that each TSM was trained on the same train test split,
            # which happens when instantiating SupervisedModelTrainer
            test_set_actual = model.test_set_actual
    else:
        raise HealthcareAIError('This requires either a single TrainedSupervisedModel or a list of them')

    # Plot with the selected plotter
    plotter(test_set_actual, predictions_by_model, save=False, debug=False)


def build_classification_model_prediction_dictionary(trained_supervised_model):
    # TODO low priority, but test this
    """
    Given a single trained supervised model build a simple dictionary containing the model name and predictions from the
    test set. Raises an error if 

    Args:
        trained_supervised_model (TrainedSupervisedModel): 

    Returns:
        dict: predictions by model name
    """
    if trained_supervised_model.is_regression:
        raise HealthcareAIError('ROC/PR plots are not used to evaluate regression models.')

    name = trained_supervised_model.model_name
    predictions = np.squeeze(trained_supervised_model.test_set_predictions[:, 1])

    return {name: predictions}


def roc_plot_from_predictions(y_test, y_predictions_by_model, save=False, debug=False):
    # TODO consolidate this and PR plotter into 1 function
    # TODO make the colors randomly generated from rgb values
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    # Initialize plot
    plt.figure()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.plot([0, 1], [0, 1], 'k--')

    # TODO hack to convert to array if it is a single dictionary
    if isinstance(y_predictions_by_model, dict):
        y_predictions_by_model = [y_predictions_by_model]

    # Calculate and plot for each model
    for i, model in enumerate(y_predictions_by_model):
        model_name, y_predictions = model.popitem()
        # calculate metrics
        fpr, tpr, _ = skmetrics.roc_curve(y_test, y_predictions)
        roc_auc_linear = skmetrics.auc(fpr, tpr)

        if debug:
            print('{} model:'.format(model_name))
            print(pd.DataFrame({'FPR': fpr, 'TPR': tpr}))

        # TODO deal with colors ...
        # plot the line
        temp_color = colors[i]
        label = '{} (AUC = {})'.format(model_name, round(roc_auc_linear, 2))
        plt.plot(fpr, tpr, color=temp_color, label=label)

    plt.legend(loc="lower right")
    # TODO: add cutoff associated with FPR/TPR

    if save:
        plt.savefig('ROC.png')
        source_path = os.path.dirname(os.path.abspath(__file__))
        print('\nROC plot saved in: {}'.format(source_path))

    plt.show()


def pr_plot_from_predictions(y_test, y_predictions_by_model, save=False, debug=False):
    # TODO consolidate this and PR plotter into 1 function
    # TODO make the colors randomly generated from rgb values
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    # Initialize plot
    plt.figure()
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.plot([0, 1], [1, 0], 'k--')

    # TODO hack to convert to array if it is a single dictionary
    if isinstance(y_predictions_by_model, dict):
        y_predictions_by_model = [y_predictions_by_model]

    # Calculate and plot for each model
    for i, model in enumerate(y_predictions_by_model):
        model_name, y_predictions = model.popitem()
        # calculate metrics
        precision, recall, _thresholds = skmetrics.precision_recall_curve(y_test, y_predictions)
        area = skmetrics.average_precision_score(y_test, y_predictions)

        if debug:
            print('{} model:'.format(model_name))
            print(pd.DataFrame({'Recall': recall, 'Precision': precision}))

        # plot the line
        temp_color = colors[i]
        label = '{} (AUC = {})'.format(model_name, round(area, 2))
        plt.plot(recall, precision, color=temp_color, label=label)

    plt.legend(loc="lower left")

    if save:
        plt.savefig('PR.png')
        source_path = os.path.dirname(os.path.abspath(__file__))
        print('\nPR plot saved in: {}'.format(source_path))

    plt.show()


def plot_rf_from_tsm(trained_supervised_model, x_train, save=False):
    """
    Given an instance of a TrainedSupervisedModel, the x_train data, display or save a feature importance graph
    Args:
        trained_supervised_model (TrainedSupervisedModel): 
        x_train (numpy.array): A 2D numpy array that was used for training 
        save (bool): True to save the plot, false to display it in a blocking thread
    """
    model = get_estimator_from_trained_supervised_model(trained_supervised_model)
    column_names = trained_supervised_model.column_names
    plot_random_forest_feature_importance(model, x_train, column_names, save=save)


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

    # Turn off interactive mode
    plt.ioff()

    # Set up the plot
    figure = plt.figure()
    plt.title("Feature importances")

    # Plot each feature
    x_train_shape = x_train.shape[1]
    x_train_range = range(x_train_shape)

    plt.bar(x_train_range, importances[indices], color="r", yerr=standard_deviations[indices], align="center")
    plt.xticks(x_train_range, namelist, rotation=90)
    plt.xlim([-1, x_train_shape])
    plt.gca().set_ylim(bottom=0)
    plt.tight_layout()

    # Save or display the plot
    if save:
        plt.savefig('FeatureImportances.png')
        source_path = os.path.dirname(os.path.abspath(__file__))
        print('\nFeature importances saved in: {}'.format(source_path))

        # Close the figure so it does not get displayed
        plt.close(figure)
    else:
        plt.show()


def get_estimator_from_trained_supervised_model(trained_supervised_model):
    """
    Given an instance of a TrainedSupervisedModel, return the main estimator, regardless of random search
    Args:
        trained_supervised_model (TrainedSupervisedModel): 

    Returns:
        sklearn.base.BaseEstimator: 

    """
    # Validate input is a TSM
    if type(trained_supervised_model).__name__ != 'TrainedSupervisedModel':
        raise HealthcareAIError('This requires an instance of a TrainedSupervisedModel')
    """
    1. check if it is a TSM
        Y: proceed
        N: raise error?
    2. check if tsm.model is a meta estimator
        Y: extract best_estimator_
        N: return tsm.model
    """
    # Check if tsm.model is a meta estimator
    result = get_estimator_from_meta_estimator(trained_supervised_model.model)

    return result


def get_estimator_from_meta_estimator(model):
    """
    Given an instance of a trained sklearn estimator, return the main estimator, regardless of random search
    Args:
        model (sklearn.base.BaseEstimator): 

    Returns:
        sklearn.base.BaseEstimator: 
    """
    if not issubclass(type(model), sklearn.base.BaseEstimator):
        raise HealthcareAIError('This requires an instance of sklearn.base.BaseEstimator')

    if issubclass(type(model), sklearn.base.MetaEstimatorMixin):
        result = model.best_estimator_
    else:
        result = model
        
    return result


def get_hyperparameters_from_meta_estimator(model):
    """
    Given an instance of a trained sklearn estimator, return the best hyperparameters if it is a meta estimator
    Args:
        model (sklearn.base.BaseEstimator): 

    Returns:
        dict: The best hyperparameters 
    """
    if not issubclass(type(model), sklearn.base.BaseEstimator):
        raise HealthcareAIError('This requires an instance of sklearn.base.BaseEstimator')

    if issubclass(type(model), sklearn.base.MetaEstimatorMixin):
        result = model.best_params_
    else:
        result = None

    return result

if __name__ == '__main__':
    pass