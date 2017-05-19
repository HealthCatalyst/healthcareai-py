import os
import math
import sklearn

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import GridSearchCV

from healthcareai.common.healthcareai_error import HealthcareAIError
from healthcareai.common.top_factors import write_feature_importances
from healthcareai.common.file_io_utilities import save_object_as_pickle, load_pickle_file


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
    # Squeeze down y_test to 1D
    y_test = np.squeeze(y_test)

    # Get binary classification predictions
    binary_predictions = np.squeeze(trained_model.predict(x_test))

    # Get probability classification predictions
    probability_predictions = np.squeeze(trained_model.predict_proba(x_test)[:, 1])

    # Calculate some metrics
    precision, recall, thresholds = precision_recall_curve(y_test, probability_predictions)
    pr_auc = auc(recall, precision)
    roc_auc = sklearn.metrics.roc_auc_score(y_test, binary_predictions)
    accuracy = sklearn.metrics.accuracy_score(y_test, binary_predictions)

    return {
        'roc_auc': roc_auc,
        'accuracy': accuracy,
        'pr_auc': pr_auc,
    }


def tsm_comparison_roc_plot(trained_supervised_model):
    """
    Given a single or list of trained supervised models, plot a roc curve for each one
    
    Args:
        trained_supervised_model (list | TrainedSupervisedModel): 
    """
    predictions_by_model = []
    # TODO doing this properly leads to a circular dependency so dirty hack string matching was needed
    # if isinstance(trained_supervised_model, TrainedSupervisedModel):
    if type(trained_supervised_model).__name__ == 'TrainedSupervisedModel':
        entry = build_model_prediction_dictionary(trained_supervised_model)
        predictions_by_model.append(entry)
        test_set_actual = trained_supervised_model.test_set_actual
    elif isinstance(trained_supervised_model, list):
        for model in trained_supervised_model:
            entry = build_model_prediction_dictionary(model)
            predictions_by_model.append(entry)

            # TODO so, you could check for different GUIDs that could be saved in each TSM!
            # The assumption here is that each TSM was trained on the same train test split,
            # which happens when instantiating SupervisedModelTrainer
            test_set_actual = model.test_set_actual
    else:
        # TODO test this
        raise HealthcareAIError('This requires either a single TrainedSupervisedModel or a list of them')

    roc_plot_from_predictions(test_set_actual, predictions_by_model, save=False, debug=False)


def build_model_prediction_dictionary(trained_supervised_model):
    # TODO low priority, but test this
    """
    Given a single trained supervised model build a simple dictionary containing the model name and predictions from the
    test set. Raises an error if 

    Args:
        trained_supervised_model (TrainedSupervisedModel): 

    Returns:
        dict: 
    """
    if trained_supervised_model.model_type == 'regression':
        raise HealthcareAIError('ROC plots are not used to evaluate regression models.')

    name = trained_supervised_model.model_name
    # predictions = first_class_prediction_from_binary_probabilities(trained_supervised_model.test_set_predictions)
    predictions = np.squeeze(trained_supervised_model.test_set_predictions[:, 1])

    return {name: predictions}


def roc_plot_from_predictions(y_test, y_predictions_by_model, save=False, debug=False):
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
        fpr, tpr, _ = sklearn.metrics.roc_curve(y_test, y_predictions)
        roc_auc_linear = sklearn.metrics.auc(fpr, tpr)

        if debug:
            print('{} model:'.format(model_name))
            print(pd.DataFrame({'FPR': fpr, 'TPR': tpr}))

        # TODO deal with colors ...
        # plot the line
        temp_color = colors[i]
        label = '{} (area = {})'.format(model_name, round(roc_auc_linear, 2))
        plt.plot(fpr, tpr, color=temp_color, label=label)

    plt.legend(loc="lower right")
    # TODO: add cutoff associated with FPR/TPR

    if save:
        plt.savefig('ROC.png')
        source_path = os.path.dirname(os.path.abspath(__file__))
        print('\nROC plot saved in: {}'.format(source_path))

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
    # TODO name these sanely
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