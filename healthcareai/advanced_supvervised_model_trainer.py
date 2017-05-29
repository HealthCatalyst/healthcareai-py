import sklearn
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

import healthcareai.common.model_eval as hcai_model_evaluation
import healthcareai.common.top_factors as hcai_factors
import healthcareai.trained_models.trained_supervised_model as hcai_tsm
import healthcareai.common.helpers as hcai_helpers

from healthcareai.common.randomized_search import prepare_randomized_search
from healthcareai.common.healthcareai_error import HealthcareAIError

SUPPORTED_MODEL_TYPES = ['classification', 'regression']


class AdvancedSupervisedModelTrainer(object):
    """
    This class helps create a model using several common classifiers and regressors, both of which report appropiate
    metrics.
    """

    def __init__(self, dataframe, model_type, predicted_column, grain_column=None, verbose=False):
        """
        Creates an instance of AdvancedSupervisedModelTrainer.
        
        Args:
            dataframe (pandas.DataFrame): The training data
            model_type (str): 'classification' or 'regression'
            predicted_column (str): The name of the predicted/target/label column
            grain_column (str): The grain column
            verbose (bool): Verbose output
        """
        # Validate model type is sane
        if model_type not in SUPPORTED_MODEL_TYPES:
            raise HealthcareAIError('A trainer must be one of these types: {}'.format(SUPPORTED_MODEL_TYPES))

        self.dataframe = dataframe
        self.model_type = model_type
        self.predicted_column = predicted_column
        self.grain_column = grain_column
        self.verbose = verbose
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.pipeline = None

        self._console_log(
            'Shape and top 5 rows of original dataframe:\n{}\n{}'.format(self.dataframe.shape, self.dataframe.head()))

    @property
    def is_classification(self):
        """
        Returns True if trainer is set up for classification 
        
        Easy check to consolidate magic strings in all the model type switches.
        """
        return self.model_type == 'classification'

    @property
    def is_regression(self):
        """
        Returns True if trainer is set up for regression 
        
        Easy check to consolidate magic strings in all the model type switches.
        """
        return self.model_type == 'regression'

    def feature_scaling(self, columns_to_scale):
        # TODO convert to TransformerMixin
        # TODO document
        # TODO test
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

    def train_test_split(self, random_seed=None):
        """
        Splits the dataframe into train and test sets.
        
        Args:
            random_seed (int): An optional random seed for the random number generator. It is best to leave at the None
                deafault. Useful for unit tests when reproducibility is required.
        """
        y = np.squeeze(self.dataframe[[self.predicted_column]])
        X = self.dataframe.drop([self.predicted_column], axis=1)

        self.X_train, self.X_test, self.y_train, self.y_test = sklearn.model_selection.train_test_split(
            X, y, test_size=.20, random_state=random_seed)

        self._console_log('\nShape of X_train: {}\ny_train: {}\nX_test: {}\ny_test: {}'.format(
            self.X_train.shape,
            self.y_train.shape,
            self.X_test.shape,
            self.y_test.shape))

    def ensemble_regression(self, scoring_metric='neg_mean_squared_error', model_by_name=None):
        # TODO stub
        self.validate_regression('Ensemble Regression')
        raise HealthcareAIError('We apologize. An ensemble linear regression has not yet been implemented.')

    def ensemble_classification(self, scoring_metric='roc_auc', trained_model_by_name=None):
        """
        This provides a simple way to put data in and have healthcare.ai train a few models and pick the best one for
        your data.

        Args:
            scoring_metric (str): The metric used to rank the models. Defaults to 'roc_auc'
            trained_model_by_name (dict): A dictionary of trained models to compare for a custom ensemble

        Returns:
            TrainedSupervisedModel: The best TrainedSupervisedModel found.
        """
        self.validate_classification('Ensemble Classification')
        self.validate_score_metric_for_number_of_classes(scoring_metric)
        score_by_name = {}

        # Here is the default list of algorithms to try for the ensemble
        # Adding an ensemble method is as easy as adding a new key:value pair in the `model_by_name` dictionary
        if trained_model_by_name is None:
            # TODO because these now all return TSMs it will be additionally slow by all the factor models.
            # TODO Could these be trained separately then after the best is found, train the factor model and add to TSM?
            trained_model_by_name = {
                'KNN': self.knn(randomized_search=True, scoring_metric=scoring_metric),
                'Logistic Regression': self.logistic_regression(randomized_search=True),
                'Random Forest Classifier': self.random_forest_classifier(
                    trees=200,
                    randomized_search=True,
                    scoring_metric=scoring_metric)}

        for name, model in trained_model_by_name.items():
            # Unroll estimator from trained supervised model
            estimator = hcai_tsm.get_estimator_from_trained_supervised_model(model)

            # Get the score objects for the estimator
            score = self.metrics(estimator)
            self._console_log('{} algorithm: score = {}'.format(name, score))

            # TODO this may need to ferret out each classification score separately
            score_by_name[name] = score[scoring_metric]

        sorted_names_and_scores = sorted(score_by_name.items(), key=lambda x: x[1])
        best_algorithm_name, best_score = sorted_names_and_scores[-1]
        best_model = trained_model_by_name[best_algorithm_name]

        self._console_log('Based on the scoring metric {}, the best algorithm found is: {}'.format(scoring_metric,
                                                                                                   best_algorithm_name))
        self._console_log('{} {} = {}'.format(best_algorithm_name, scoring_metric, best_score))

        return best_model

    def validate_score_metric_for_number_of_classes(self, metric):
        """
        Check that a user's choice of scoring metric makes sense with the number of prediction classes

        Args:
            metric (str): a string of the scoring metric
        """

        # TODO test this for errors and multiclass
        # TODO make this more robust for other scoring metrics
        classes = hcai_helpers.count_unique_elements_in_column(self.dataframe, self.predicted_column)
        if classes is 2:
            # return True for testing
            return True
        elif classes > 2 and metric in ['pr_auc', 'roc_auc']:
            raise (HealthcareAIError(
                'AUC (aka roc_auc) cannot be used for more than two classes. Please choose another metric such as \'accuracy\''))

    def metrics(self, trained_sklearn_estimator):
        """
        Given a trained estimator, calculate the appropriate performance metrics.
        
        This is intended to be a thin wrapper around the toolbox metrics.

        Args:
            trained_sklearn_estimator (sklearn.base.BaseEstimator): A scikit-learn trained algorithm
        """
        performance_metrics = None

        if self.model_type is 'classification':
            performance_metrics = hcai_model_evaluation.calculate_binary_classification_metrics(
                trained_sklearn_estimator,
                self.X_test,
                self.y_test,

            )
        elif self.model_type is 'regression':
            performance_metrics = hcai_model_evaluation.calculate_regression_metrics(trained_sklearn_estimator,
                                                                                     self.X_test, self.y_test)

        return performance_metrics

    def logistic_regression(self, scoring_metric='roc_auc', hyperparameter_grid=None, randomized_search=True):
        """
        A light wrapper for Sklearn's logistic regression that performs randomized search over an overideable default 
        hyperparameter grid.
        """
        self.validate_classification('Logistic Regression')
        if hyperparameter_grid is None:
            hyperparameter_grid = {'C': [0.01, 0.1, 1, 10, 100], 'class_weight': [None, 'balanced']}

        algorithm = prepare_randomized_search(
            LogisticRegression,
            scoring_metric,
            hyperparameter_grid,
            randomized_search,
            class_weight='balanced')

        trained_supervised_model = self._trainer(algorithm)

        return trained_supervised_model

    def linear_regression(self, scoring_metric='neg_mean_squared_error', hyperparameter_grid=None,
                          randomized_search=True):
        """
        A light wrapper for Sklearn's linear regression that performs randomized search over an overridable default
        hyperparameter grid.
        """
        self.validate_regression('Linear Regression')
        if hyperparameter_grid is None:
            hyperparameter_grid = {"fit_intercept": [True, False]}
            pass

        algorithm = prepare_randomized_search(
            LinearRegression,
            scoring_metric,
            hyperparameter_grid,
            randomized_search)

        trained_supervised_model = self._trainer(algorithm)

        return trained_supervised_model

    def knn(self, scoring_metric='roc_auc', hyperparameter_grid=None, randomized_search=True):
        """
        A light wrapper for Sklearn's knn classifier that performs randomized search over an overridable default
        hyperparameter grid.
        """
        self.validate_classification('KNN')
        if hyperparameter_grid is None:
            hyperparameter_grid = {'n_neighbors': list(range(5, 26)), 'weights': ['uniform', 'distance']}

        algorithm = prepare_randomized_search(
            KNeighborsClassifier,
            scoring_metric,
            hyperparameter_grid,
            randomized_search,
            n_neighbors=5)

        trained_supervised_model = self._trainer(algorithm)

        return trained_supervised_model

    def random_forest(self,
                      trees=200,
                      scoring_metric='roc_auc',
                      hyperparameter_grid=None,
                      randomized_search=True):
        """ A convenience method that allows a user to simply call .random_forest() and get the right one. """
        if self.is_classification:
            return self.random_forest_classifier(trees=trees,
                                                 scoring_metric=scoring_metric,
                                                 hyperparameter_grid=hyperparameter_grid,
                                                 randomized_search=randomized_search)
        elif self.is_regression:
            return self.random_forest_regressor(trees=trees,
                                                scoring_metric=scoring_metric,
                                                hyperparameter_grid=hyperparameter_grid,
                                                randomized_search=randomized_search)

    def random_forest_classifier(self, trees, scoring_metric='roc_auc', hyperparameter_grid=None,
                                 randomized_search=True):
        """
        A light wrapper for Sklearn's random forest classifier that performs randomized search over an overridable
        default hyperparameter grid.
        """
        self.validate_classification('Random Forest Classifier')
        if hyperparameter_grid is None:
            max_features = hcai_helpers.calculate_random_forest_mtry_hyperparameter(len(self.X_test.columns),
                                                                                    self.model_type)
            hyperparameter_grid = {'n_estimators': [10, 50, 200], 'max_features': max_features}

        algorithm = prepare_randomized_search(
            RandomForestClassifier,
            scoring_metric,
            hyperparameter_grid,
            randomized_search,
            n_estimators=trees)

        trained_supervised_model = self._trainer(algorithm)

        return trained_supervised_model

    def random_forest_regressor(self,
                                trees=200,
                                scoring_metric='neg_mean_squared_error',
                                hyperparameter_grid=None,
                                randomized_search=True):
        """
        A light wrapper for Sklearn's random forest regressor that performs randomized search over an overridable
        default hyperparameter grid.
        """
        self.validate_regression('Random Forest Regressor')
        if hyperparameter_grid is None:
            max_features = hcai_helpers.calculate_random_forest_mtry_hyperparameter(len(self.X_test.columns),
                                                                                    self.model_type)
            hyperparameter_grid = {'n_estimators': [10, 50, 200], 'max_features': max_features}

        algorithm = prepare_randomized_search(
            RandomForestRegressor,
            scoring_metric,
            hyperparameter_grid,
            randomized_search,
            n_estimators=trees)

        trained_supervised_model = self._trainer(algorithm)

        return trained_supervised_model

    def _trainer(self, algorithm, include_factor_model=True):
        """
        Trains an algorithm, prepares metrics, builds and returns a TrainedSupervisedModel

        Args:
            algorithm (sklearn.base.BaseEstimator): The scikit learn algorithm, ready to fit.
            include_factor_model (bool): Trains a model for top factors. Defaults to True

        Returns:
            TrainedSupervisedModel: a TrainedSupervisedModel
        """
        algorithm.fit(self.X_train, self.y_train)

        # Build prediction sets for ROC/PR curve generation. Note this does increase the size of the TSM because the
        # test set is saved inside the object as well as the calculated thresholds.
        # See https://github.com/HealthCatalyst/healthcareai-py/issues/264 for a discussion on pros/cons
        # PEP 8
        test_set_predictions = None
        test_set_class_labels = None
        if self.is_classification:
            # Save both the probabilities and labels
            test_set_predictions = algorithm.predict_proba(self.X_test)
            test_set_class_labels = algorithm.predict(self.X_test)
        elif self.is_regression:
            test_set_predictions = algorithm.predict(self.X_test)

        if include_factor_model:
            factor_model = hcai_factors.prepare_fit_model_for_factors(self.model_type, self.X_train, self.y_train)
        else:
            factor_model = None

        trained_supervised_model = hcai_tsm.TrainedSupervisedModel(
            model=algorithm,
            feature_model=factor_model,
            fit_pipeline=self.pipeline,
            model_type=self.model_type,
            column_names=self.X_test.columns.values,
            grain_column=self.grain_column,
            prediction_column=self.predicted_column,
            test_set_predictions=test_set_predictions,
            test_set_class_labels=test_set_class_labels,
            test_set_actual=self.y_test,
            metric_by_name=self.metrics(algorithm))

        return trained_supervised_model

    def validate_regression(self, model_name=None):
        if not self.is_regression:
            raise HealthcareAIError('A {} model can only be trained with a regression trainer.'.format(model_name))

    def validate_classification(self, model_name=None):
        if not self.is_classification:
            raise HealthcareAIError('A {} model can only be trained with a classification trainer.'.format(model_name))

    def _console_log(self, message):
        if self.verbose:
            print('DSM: {}'.format(message))


if __name__ == "__main__":
    pass
