import json
import os
import sklearn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

import healthcareai.common.model_eval as model_evaluation
import healthcareai.common.top_factors as factors
from healthcareai.common import helpers
from healthcareai.common import model_eval
from healthcareai.common.healthcareai_error import HealthcareAIError
from healthcareai.common.helpers import count_unique_elements_in_column
from healthcareai.common.randomized_search import prepare_randomized_search
from healthcareai.trained_models.trained_supervised_model import TrainedSupervisedModel


class DevelopSupervisedModel(object):
    """
    This class helps create a model using several common classifiers
    (reporting AUC) and regressors (reporting MAE/MSE). When instantiating,
    the data is prepped and one-fifth is held out so model accuracy can be
    assessed.

    Parameters
    ----------
    modeltype (str) : whether the model will be 'classification' or 'regression'
    df (dataframe) : data that your model is based on
    predictedcol (str) : y column (in ticks) who's values are being predicted
    impute (boolean) : whether imputation is done on the data; if not, rows with nulls are removed
    graincol (str) : OPTIONAL | column (in ticks) that represents the data's grain
    debug (boolean) : OPTIONAL | verbosity of the output

    Returns
    -------
    Object representing the cleaned data, against which methods are run
    """

    def __init__(self, dataframe, model_type, predicted_column, grain_column=None, verbose=False):
        self.dataframe = dataframe
        self.model_type = model_type
        self.predicted_column = predicted_column
        self.grain_column = grain_column
        self.verbose = verbose
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        # TODO implement (or avoid) these attributes, which really might be methods
        self.y_probab_linear = None
        self.y_probab_rf = None
        self.col_list = None
        self.rfclf = None
        self.au_roc = None
        self.ensemble_results = None
        self.pipeline = None

        self.console_log(
            'Shape and top 5 rows of original dataframe:\n{}\n{}'.format(self.dataframe.shape, self.dataframe.head()))

    def under_sampling(self, random_state=0):
        # TODO convert to fit transform
        # NB: Must be done BEFORE train/test split
        #     so that when we split the under/over sampled
        #     dataset. We do under/over sampling on
        #     the entire dataframe.
        #     Must be done after imputation, since
        #     under/over sampling will not work with
        #     missing values.
        #     Must be done after target column is converted to
        #     numerical value (so under/over sampling from
        #     imblearn works).
        y = np.squeeze(self.dataframe[[self.predicted_column]])
        X = self.dataframe.drop([self.predicted_column], axis=1)

        under_sampler = RandomUnderSampler(random_state=random_state)
        X_under_sampled, y_under_sampled = under_sampler.fit_sample(X, y)

        X_under_sampled = pd.DataFrame(X_under_sampled)
        X_under_sampled.columns = X.columns
        y_under_sampled = pd.Series(y_under_sampled)

        dataframe_under_sampled = X_under_sampled
        dataframe_under_sampled[self.predicted_column] = y_under_sampled
        self.dataframe = dataframe_under_sampled

    def over_sampling(self, random_state=0):
        # TODO convert to fit transform
        # NB: Must be done BEFORE train/test split
        #     so that when we split the under/over sampled
        #     dataset. We do under/over sampling on
        #     the entire dataframe.
        #     Must be done after imputation, since
        #     under/over sampling will not work with
        #     missing values.
        #     Must be done after target column is converted to
        #     numerical value (so under/over sampling from
        #     imblearn works).
        y = np.squeeze(self.dataframe[[self.predicted_column]])
        X = self.dataframe.drop([self.predicted_column], axis=1)

        over_sampler = RandomOverSampler(random_state=random_state)
        X_over_sampled, y_over_sampled = over_sampler.fit_sample(X, y)

        X_over_sampled = pd.DataFrame(X_over_sampled)
        X_over_sampled.columns = X.columns
        y_over_sampled = pd.Series(y_over_sampled)

        dataframe_over_sampled = X_over_sampled
        dataframe_over_sampled[self.predicted_column] = y_over_sampled
        self.dataframe = dataframe_over_sampled

    def feature_scaling(self, columns_to_scale):
        # TODO convert to fit transform
        # NB: Must happen AFTER self.X_train, self.X_test,
        #     self.y_train, self.y_test are defined.
        #     Must happen AFTER imputation is done so there
        #     are no missing values.
        #     Must happen AFTER under/over sampling is done
        #     so that we scale the under/over sampled dataset.
        # TODO: How to warn the user if they call this method at the wrong time?
        X_train_scaled_subset = self.X_train[columns_to_scale]
        X_test_scaled_subset = self.X_test[columns_to_scale]
        scaler = StandardScaler()

        scaler.fit(X_train_scaled_subset)

        X_train_scaled_subset_dataframe = pd.DataFrame(scaler.transform(X_train_scaled_subset))
        X_train_scaled_subset_dataframe.index = X_train_scaled_subset.index
        X_train_scaled_subset_dataframe.columns = X_train_scaled_subset.columns
        self.X_train[columns_to_scale] = X_train_scaled_subset_dataframe

        X_test_scaled_subset_dataframe = pd.DataFrame(scaler.transform(X_test_scaled_subset))
        X_test_scaled_subset_dataframe.index = X_test_scaled_subset.index
        X_test_scaled_subset_dataframe.columns = X_test_scaled_subset.columns
        self.X_test[columns_to_scale] = X_test_scaled_subset_dataframe

    def print_out_dataframe_shape_and_head(self, message):
        self.console_log(message)
        self.console_log(self.dataframe.shape)
        self.console_log(self.dataframe.head())

    def train_test_split(self):
        y = np.squeeze(self.dataframe[[self.predicted_column]])
        X = self.dataframe.drop([self.predicted_column], axis=1)

        self.X_train, self.X_test, self.y_train, self.y_test = model_selection.train_test_split(
            X, y, test_size=.20, random_state=0)

        self.console_log('\nShape of X_train: {}\ny_train: {}\nX_test: {}\ny_test: {}'.format(
            self.X_train.shape,
            self.y_train.shape,
            self.X_test.shape,
            self.y_test.shape))

    def save_output_to_csv(self, filename, output):
        # TODO timeRan is borked
        output_dataframe = pd.DataFrame([(timeRan, self.model_type, output['modelLabels'],
                                          output['gridSearch_BestScore'],
                                          output['gridSearch_ScoreMetric'],) \
                                         + x for x in list(output.items())], \
                                        columns=['TimeStamp', 'ModelType',
                                                 'ModelLabels', 'BestScore',
                                                 'BestScoreMetric', 'Metric',
                                                 'MetricValue']).set_index('TimeStamp')
        # save files locally #
        output_dataframe.to_csv(filename + '.txt', header=False)

    def ensemble_regression(self, scoring_metric='roc_auc', model_by_name=None):
        # TODO stub
        pass

    def ensemble_classification(self, scoring_metric='roc_auc', model_by_name=None):
        """
        This provides a simple way to put data in and have healthcare.ai train a few models and pick the best one for
        your data.
        """
        # TODO enumerate, document and validate scoring options
        # http://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values
        # TODO Does one of those options make the most sense to pick a default?
        # TODO Can we algorithmically determine the best choice?

        self.validate_score_metric_for_number_of_classes(scoring_metric)
        score_by_name = {}

        # Here is the default list of algorithms to try for the ensemble
        # Adding an ensemble method is as easy as adding a new key:value pair in the `model_by_name` dictionary
        if model_by_name is None:
            model_by_name = {}
            model_by_name['KNN'] = self.knn(randomized_search=True, scoring_metric=scoring_metric)
            model_by_name['Logistic Regression'] = self.logistic_regression()
            model_by_name['Random Forest Classifier'] = self.random_forest_classifier(
                randomized_search=True,
                scoring_metric=scoring_metric).best_estimator_

        for name, model in model_by_name.items():
            # TODO this may need to ferret out each classification score separately
            score = self.metrics(model)
            score_by_name[name] = score[scoring_metric]

            self.console_log('{} algorithm: score = {}'.format(name, score))

        sorted_names_and_scores = sorted(score_by_name.items(), key=lambda x: x[1])
        best_algorithm_name, best_score = sorted_names_and_scores[-1]
        best_model = model_by_name[best_algorithm_name]

        results = {
            'best_score': best_score,
            'best_algorithm_name': best_algorithm_name,
            'model_scores': score_by_name,
            'best_model': best_model
        }

        print('Based on the scoring metric {}, the best algorithm found is: {}'.format(scoring_metric,
                                                                                       best_algorithm_name))
        print('{} {} = {}'.format(best_algorithm_name, scoring_metric, best_score))

        self.ensemble_results = results
        return results

    def write_classification_metrics_to_json(self):
        # TODO a similar method should be created for regression metrics
        # TODO this is really not in the right place
        if self.ensemble_results is None:
            raise HealthcareAIError('Ensemble must be run before metrics can be written')
        output = {}
        y_pred = self.ensemble_results['best_model'].predict(self.X_test)
        accuracy = sklearn.metrics.accuracy_score(self.y_test, y_pred)
        confusion_matrix = sklearn.metrics.confusion_matrix(self.y_test, y_pred)
        output['accuracy'] = accuracy
        output['confusion_matrix'] = confusion_matrix.tolist()
        output['auc_roc'] = self.ensemble_results['best_score']
        output['algorithm_name'] = self.ensemble_results['best_algorithm_name']
        with open('classification_metrics.json', 'w') as fp:
            json.dump(output, fp, indent=4, sort_keys=True)

    def validate_score_metric_for_number_of_classes(self, metric):
        """
        Check that a user's choice of scoring metric makes sense with the number of prediction classes

        Args:
            metric (str): a string of the scoring metric
        """

        # TODO make this more robust for other scoring metrics
        classes = count_unique_elements_in_column(self.dataframe, self.predicted_column)
        if classes is 2:
            pass
        elif classes > 2 and metric is 'roc_auc':
            raise (HealthcareAIError(
                'AUC (aka roc_auc) cannot be used for more than two classes. Please choose another metric such as \'accuracy\''))

    def metrics(self, trained_model):
        """
        Given a trained model, calculate the appropriate performance metrics.
        
        This is intended to be a thin wrapper around the toolbox metrics.

        Args:
            trained_model (BaseEstimator): A scikit-learn trained algorithm
        """
        performance_metrics = None

        if self.model_type is 'classification':
            performance_metrics = model_evaluation.calculate_classification_metrics(trained_model,
                                                                                    self.X_test,
                                                                                    self.y_test)
        elif self.model_type is 'regression':
            performance_metrics = model_evaluation.calculate_regression_metrics(trained_model, self.X_test, self.y_test)

        return performance_metrics

    def logistic_regression(self, scoring_metric='roc_auc', hyperparameter_grid=None, randomized_search=True):
        """
        A light wrapper for Sklearn's logistic regression that performs randomized search over a default (and
        overideable) hyperparameter grid.
        """
        if hyperparameter_grid is None:
            # TODO sensible default hyperparameter grid
            pass
            # hyperparameter_grid = {'n_neighbors': neighbor_list, 'weights': ['uniform', 'distance']}

        algorithm = prepare_randomized_search(
            LogisticRegressionCV,
            scoring_metric,
            hyperparameter_grid,
            randomized_search,
            # 5 cross validation folds
            cv=5)

        trained_supervised_model = self.trainer(algorithm)

        return trained_supervised_model

    def linear_regression(self, scoring_metric='roc_auc', hyperparameter_grid=None, randomized_search=True):
        """
        A light wrapper for Sklearn's linear regression that performs randomized search over a default (and
        overideable) hyperparameter grid.
        """
        if hyperparameter_grid is None:
            # TODO sensible default hyperparameter grid
            pass

        algorithm = prepare_randomized_search(
            LinearRegression,
            scoring_metric,
            hyperparameter_grid,
            randomized_search)

        trained_supervised_model = self.trainer(algorithm)

        return trained_supervised_model

    def knn(self, scoring_metric='roc_auc', hyperparameter_grid=None, randomized_search=True):
        if hyperparameter_grid is None:
            # TODO add sensible KNN hyperparameter grid
            neighbor_list = list(range(10, 26))
            hyperparameter_grid = {'n_neighbors': neighbor_list, 'weights': ['uniform', 'distance']}

        algorithm = prepare_randomized_search(
            KNeighborsClassifier,
            scoring_metric,
            hyperparameter_grid,
            randomized_search,
            n_neighbors=5)

        trained_supervised_model = self.trainer(algorithm)

        return trained_supervised_model

    def linear(self, cores=4, debug=False):
        # TODO deprecate
        """
        This method creates and assesses the accuracy of a logistic regression
        model.

        Parameters
        ----------
        cores (num) : Number of cores to use (default 4)
        debug (boolean) : Verbosity of output (default False)

        Returns
        -------
        Nothing. Output to console describes model accuracy.
        """

        if self.model_type == 'classification':
            algo = LogisticRegressionCV(cv=5)
        elif self.model_type == 'regression':
            algo = LinearRegression()
        else:
            algo = None

        self.y_probab_linear, self.au_roc = model_eval.clfreport(
            model_type=self.model_type,
            debug=debug,
            develop_model_mode=True,
            algo=algo,
            X_train=self.X_train,
            y_train=self.y_train,
            X_test=self.X_test,
            y_test=self.y_test,
            cores=cores)

    def random_forest_2(self,
                        trees=200,
                        scoring_metric='roc_auc',
                        hyperparameter_grid=None,
                        randomized_search=True):
        """A convenience method that allows a user to simply call .random_forest() and get the right one."""
        # TODO rename to random_forest after the other is deprecated
        if self.model_type == 'classification':
            self.random_forest_classifier(trees=trees,
                                          scoring_metric=scoring_metric,
                                          hyperparameter_grid=hyperparameter_grid,
                                          randomized_search=randomized_search)
        elif self.model_type == 'regression':
            self.random_forest_regressor(trees=200,
                                         scoring_metric=scoring_metric,
                                         hyperparameter_grid=hyperparameter_grid,
                                         randomized_search=randomized_search)

    def random_forest_classifier(self, trees=200, scoring_metric='roc_auc', hyperparameter_grid=None,
                                 randomized_search=True):
        if hyperparameter_grid is None:
            # TODO add sensible hyperparameter grid
            max_features = helpers.calculate_random_forest_mtry_hyperparameter(len(self.X_test.columns),
                                                                               self.model_type)
            hyperparameter_grid = {'n_estimators': [10, 50, 200], 'max_features': max_features}

        algorithm = prepare_randomized_search(
            RandomForestClassifier,
            scoring_metric,
            hyperparameter_grid,
            randomized_search,
            trees=trees)

        trained_supervised_model = self.trainer(algorithm)

        return trained_supervised_model

    def random_forest_regressor(self,
                                trees=200,
                                scoring_metric='neg_mean_squared_error',
                                hyperparameter_grid=None,
                                randomized_search=True):
        if hyperparameter_grid is None:
            # TODO add sensible hyperparameter grid
            max_features = helpers.calculate_random_forest_mtry_hyperparameter(len(self.X_test.columns),
                                                                               self.model_type)
            hyperparameter_grid = {'n_estimators': [10, 50, 200], 'max_features': max_features}

        algorithm = prepare_randomized_search(
            RandomForestRegressor,
            scoring_metric,
            hyperparameter_grid,
            randomized_search,
            trees=trees)

        trained_supervised_model = self.trainer(algorithm)

        return trained_supervised_model

    def random_forest(self, cores=4, trees=200, tune=False, debug=False):
        # TODO deprecate after replacements are implemented.
        """
        This method creates and assesses the accuracy of a logistic regression
        model.

        Parameters
        ----------
        cores (num) : Number of cores to use (default 4)
        trees (num) : Number of trees in the random forest (default 200)
        tune (boolean) : Whether to tune hyperparameters. This iterates number
        of trees from 100, 250, and 500.
        debug (boolean) : Verbosity of output (default False)

        Returns
        -------
        Nothing. Output to console describes model accuracy.
        """

        # TODO: refactor, such that each algo doesn't need an if/else tree
        if self.model_type == 'classification':
            algo = RandomForestClassifier(n_estimators=trees,
                                          verbose=(2 if debug is True else 0))

        elif self.model_type == 'regression':
            algo = RandomForestRegressor(n_estimators=trees,
                                         verbose=(2 if debug is True else 0))

        else:  # Here to appease pep8
            algo = None

        params = {'max_features': helpers.calculate_random_forest_mtry_hyperparameter(len(self.X_test.columns),
                                                                                      self.model_type)}

        self.col_list = self.X_train.columns.values

        self.y_probab_rf, self.au_roc, self.rfclf = model_eval.clfreport(
            model_type=self.model_type,
            debug=debug,
            develop_model_mode=True,
            algo=algo,
            X_train=self.X_train,
            y_train=self.y_train,
            X_test=self.X_test,
            y_test=self.y_test,
            param=params,
            cores=cores,
            tune=tune,
            col_list=self.col_list)

        return self.rfclf

    def trainer(self, algorithm):
        algorithm.fit(self.X_train, self.y_train)
        trained_factor_model = factors.prepare_fit_model_for_factors(self.model_type,
                                                                     self.X_train,
                                                                     self.y_train)
        trained_supervised_model = TrainedSupervisedModel(
            algorithm,
            trained_factor_model,
            self.pipeline,
            self.model_type,
            self.X_test.columns.values,
            self.grain_column,
            self.predicted_column,
            None,
            None,
            self.metrics(algorithm))
        return trained_supervised_model

    def plot_rffeature_importance(self, save=False):
        # TODO refactor this as a tool + advanced/simple wrapper
        """
        Plots feature importances for random forest models

        Parameters
        ----------
        save (boolean) : Whether to save the plot

        Returns
        -------
        Nothing. A plot is created and displayed.
        """

        # Arrange columns in order of importance
        if hasattr(self.rfclf, 'best_estimator_'):
            importances = self.rfclf.best_estimator_.feature_importances_
            std = np.std(
                [tree.feature_importances_ for tree in
                 self.rfclf.best_estimator_.estimators_],
                axis=0)
        else:
            importances = self.rfclf.feature_importances_
            std = np.std(
                [tree.feature_importances_ for tree in
                 self.rfclf.estimators_],
                axis=0)

        indices = np.argsort(importances)[::-1]
        namelist = [self.col_list[i] for i in indices]

        # Plot these columns
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(self.X_train.shape[1]),
                importances[indices], color="r",
                yerr=std[indices], align="center")
        plt.xticks(range(self.X_train.shape[1]), namelist, rotation=90)
        plt.xlim([-1, self.X_train.shape[1]])
        plt.gca().set_ylim(bottom=0)
        plt.tight_layout()
        if save:
            plt.savefig('FeatureImportances.png')
            source_path = os.path.dirname(os.path.abspath(__file__))
            print('\nFeature importances saved in: {}'.format(source_path))
            plt.show()
        else:
            plt.show()

    def console_log(self, message):
        if self.verbose:
            print('DSM: {}'.format(message))

    def plot_roc(self, save=False, debug=True):
        """ Show the ROC plot """
        # TODO refactor this to take an arbitrary number of models rather than just a linear and random forest
        model_evaluation.display_roc_plot(self.ytest, self.y_probab_linear, self.y_probab_rf, save=save, debug=debug)
