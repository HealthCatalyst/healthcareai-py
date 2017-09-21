import sklearn
import numpy as np
import time

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
from sklearn.neighbors import KNeighborsClassifier

import healthcareai.common.model_eval as hcai_model_evaluation
import healthcareai.common.top_factors as hcai_factors
import healthcareai.trained_models.trained_supervised_model as hcai_tsm
import healthcareai.common.helpers as hcai_helpers

from healthcareai.common.randomized_search import get_algorithm, get_algorithm_neural_network
from healthcareai.common.healthcareai_error import HealthcareAIError

SUPPORTED_MODEL_TYPES = ['classification', 'regression']


class AdvancedSupervisedModelTrainer(object):
    """
    This class helps create a model using several common classifiers and regressors, both of which report appropiate
    metrics.
    """

    # TODO check constructor usages
    def __init__(
        self,
        dataframe,
        model_type,
        predicted_column,
        grain_column=None,
        data_scaling=False,
        original_column_names=None,
        verbose=False):
        """
        Creates an instance of AdvancedSupervisedModelTrainer.
        
        Args:
            dataframe (pandas.DataFrame): The training data
            model_type (str): 'classification' or 'regression'
            predicted_column (str): The name of the predicted/target/label column
            grain_column (str): The grain column
            original_column_names (list): The original column names of the dataframe before going through the pipeline
                (before dummification). These are used to check that the data contains all the necessary columns if
                pre-pipeline data is going to be fed to the trained model.
            data_scaling (bool): if True, perform feature scaling on numeric variables; if False, not.
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
        # TODO this is probably redundant now that original_column_names exists
        self.columns = None
        self.x_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.pipeline = None
        self.original_column_names = original_column_names
        self.categorical_column_info = None
        self.data_scaling = data_scaling

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

    @property
    def class_labels(self):
        """Return the class labels sorted"""
        self.validate_classification()
        return np.sort(self.y_test.unique())

    @property
    def number_of_classes(self):
        """Number of classes"""
        self.validate_classification()
        return len(self.class_labels)

    @property
    def is_binary_classification(self):
        """Check if a classification is binary."""
        return self.number_of_classes == 2
    def train_test_split(self, random_seed=None):
        """
        Splits the dataframe into train and test sets.
        
        Args:
            random_seed (int): An optional random seed for the random number generator. It is best to leave at the None
                deafault. Useful for unit tests when reproducibility is required.
        """
        y = np.squeeze(self.dataframe[[self.predicted_column]])
        X = self.dataframe.drop([self.predicted_column], axis=1)

        # TODO this is probably redundant now that original_column_names exists
        # Save off a copy of the column names before converting to a numpy array
        self.columns = X.columns.values

        self.x_train, self.X_test, self.y_train, self.y_test = sklearn.model_selection.train_test_split(
            X, y, test_size=.20, random_state=random_seed)

        # Scale the x variables, turn this on when using neural network
        if self.data_scaling:
            names = list(self.columns)
            is_numeric = np.vectorize(lambda x: np.issubdtype(x, np.number))
            numeric_col_bool = list(is_numeric(X.dtypes))
            numeric_col = [i for (i, v) in zip(names, numeric_col_bool) if v]
            self.feature_scaling(numeric_col)

        self.x_train = np.array(self.x_train)
        self.X_test = np.array(self.X_test)

        self._console_log('\nShape of X_train: {}\ny_train: {}\nX_test: {}\ny_test: {}'.format(
            self.x_train.shape,
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
        if self.number_of_classes == 2:
            # return True for testing
            return True
        elif self.number_of_classes > 2 and metric in ['pr_auc', 'roc_auc']:
            raise (HealthcareAIError('ROC and PR cannot be used for more than two classes. Please choose another'
                                     ' metric such as \'accuracy\''))

    def metrics(self, trained_sklearn_estimator):
        """
        Given a trained estimator, calculate the appropriate performance metrics.
        
        This is intended to be a thin wrapper around the toolbox metrics.

        Args:
            trained_sklearn_estimator (sklearn.base.BaseEstimator): A scikit-learn trained algorithm

        Returns:
            dict: The metrics appropriate for the model type
        """
        results = None

        if self.is_classification:
            results = hcai_model_evaluation.compute_classification_metrics(
                trained_sklearn_estimator,
                self.X_test,
                self.y_test)
        elif self.is_regression:
            results = hcai_model_evaluation.compute_regression_metrics(
                trained_sklearn_estimator,
                self.X_test,
                self.y_test)

        return results

    def logistic_regression(self,
                            scoring_metric='roc_auc',
                            hyperparameter_grid=None,
                            randomized_search=True,
                            number_iteration_samples=10):
        """
        A light wrapper for Sklearn's logistic regression that performs randomized search over an overideable default 
        hyperparameter grid.

        Args:
            scoring_metric (str): Any sklearn scoring metric appropriate for regression
            hyperparameter_grid (dict): hyperparameters by name
            randomized_search (bool): True for randomized search (default)
            number_iteration_samples (int): Number of models to train during the randomized search for exploring the
                hyperparameter space. More may lead to a better model, but will take longer.

        Returns:
            TrainedSupervisedModel: 
        """
        self.validate_classification('Logistic Regression')
        if hyperparameter_grid is None:
            hyperparameter_grid = {'C': [0.01, 0.1, 1, 10, 100], 'class_weight': [None, 'balanced']}
            number_iteration_samples = 10

        algorithm = get_algorithm(LogisticRegression,
                                  scoring_metric,
                                  hyperparameter_grid,
                                  randomized_search,
                                  number_iteration_samples=number_iteration_samples)

        trained_supervised_model = self._create_trained_supervised_model(algorithm)

        return trained_supervised_model

    def linear_regression(self,
                          scoring_metric='neg_mean_squared_error',
                          hyperparameter_grid=None,
                          randomized_search=True,
                          number_iteration_samples=2):
        """
        A light wrapper for Sklearn's linear regression that performs randomized search over an overridable default
        hyperparameter grid.
        
        Args:
            scoring_metric (str): Any sklearn scoring metric appropriate for regression
            hyperparameter_grid (dict): hyperparameters by name
            randomized_search (bool): True for randomized search (default)
            number_iteration_samples (int): Number of models to train during the randomized search for exploring the
                hyperparameter space. More may lead to a better model, but will take longer.

        Returns:
            TrainedSupervisedModel:
        """
        self.validate_regression('Linear Regression')
        if hyperparameter_grid is None:
            hyperparameter_grid = {"fit_intercept": [True, False]}
            number_iteration_samples = 2

        algorithm = get_algorithm(LinearRegression,
                                  scoring_metric,
                                  hyperparameter_grid,
                                  randomized_search,
                                  number_iteration_samples=number_iteration_samples)

        trained_supervised_model = self._create_trained_supervised_model(algorithm)

        return trained_supervised_model

    def lasso_regression(self, scoring_metric='neg_mean_squared_error',
                         hyperparameter_grid=None,
                         randomized_search=True,
                         number_iteration_samples=2):
        """
        A light wrapper for Sklearn's lasso regression that performs randomized search over an overridable default
        hyperparameter grid.

        Args:
            scoring_metric (str): Any sklearn scoring metric appropriate for regression
            hyperparameter_grid (dict): hyperparameters by name
            randomized_search (bool): True for randomized search (default)
            number_iteration_samples (int): Number of models to train during the randomized search for exploring the
                hyperparameter space. More may lead to a better model, but will take longer.

        Returns:
            TrainedSupervisedModel:
        """
        self.validate_regression('Lasso Regression')
        if hyperparameter_grid is None:
            hyperparameter_grid = {"fit_intercept": [True, False]}
            number_iteration_samples = 2

        algorithm = get_algorithm(Lasso,
                                  scoring_metric,
                                  hyperparameter_grid,
                                  randomized_search,
                                  number_iteration_samples=number_iteration_samples)

        trained_supervised_model = self._create_trained_supervised_model(algorithm)

        return trained_supervised_model

    def knn(self,
            scoring_metric='accuracy',
            hyperparameter_grid=None,
            randomized_search=True,
            number_iteration_samples=10):
        """
        A light wrapper for Sklearn's knn classifier that performs randomized search over an overridable default
        hyperparameter grid.
        
        Args:
            scoring_metric (str): Any sklearn scoring metric appropriate for classification
            hyperparameter_grid (dict): hyperparameters by name
            randomized_search (bool): True for randomized search (default)
            number_iteration_samples (int): Number of models to train during the randomized search for exploring the
                hyperparameter space. More may lead to a better model, but will take longer.

        Returns:
            TrainedSupervisedModel: 
        """
        self.validate_classification('KNN')
        if hyperparameter_grid is None:
            neighbors = list(range(5, 26))
            hyperparameter_grid = {'n_neighbors': neighbors, 'weights': ['uniform', 'distance']}
            number_iteration_samples = 10

            print('KNN Grid: {}'.format(hyperparameter_grid))
        algorithm = get_algorithm(KNeighborsClassifier,
                                  scoring_metric,
                                  hyperparameter_grid,
                                  randomized_search,
                                  number_iteration_samples=number_iteration_samples)

        trained_supervised_model = self._create_trained_supervised_model(algorithm)

        return trained_supervised_model

    def random_forest_classifier(self,
                                 trees=200,
                                 scoring_metric='accuracy',
                                 hyperparameter_grid=None,
                                 randomized_search=True,
                                 number_iteration_samples=5):
        """
        A light wrapper for Sklearn's random forest classifier that performs randomized search over an overridable
        default hyperparameter grid.
        
        Args:
            trees (int): number of trees to use if not performing a randomized grid search
            scoring_metric (str): Any sklearn scoring metric appropriate for classification
            hyperparameter_grid (dict): hyperparameters by name
            randomized_search (bool): True for randomized search (default)
            number_iteration_samples (int): Number of models to train during the randomized search for exploring the
                hyperparameter space. More may lead to a better model, but will take longer.

        Returns:
            TrainedSupervisedModel: 
        """
        self.validate_classification('Random Forest Classifier')
        if hyperparameter_grid is None:
            # TODO this is probably redundant now that original_column_names exists
            max_features = hcai_helpers.calculate_random_forest_mtry_hyperparameter(len(self.columns),
                                                                                    self.model_type)
            hyperparameter_grid = {'n_estimators': [100, 200, 300], 'max_features': max_features}
            number_iteration_samples = 5

        algorithm = get_algorithm(RandomForestClassifier,
                                  scoring_metric,
                                  hyperparameter_grid,
                                  randomized_search,
                                  number_iteration_samples=number_iteration_samples,
                                  n_estimators=trees)

        trained_supervised_model = self._create_trained_supervised_model(algorithm)

        return trained_supervised_model

    def random_forest_regressor(self,
                                trees=200,
                                scoring_metric='neg_mean_squared_error',
                                hyperparameter_grid=None,
                                randomized_search=True,
                                number_iteration_samples=5):
        """
        A light wrapper for Sklearn's random forest regressor that performs randomized search over an overridable
        default hyperparameter grid.
        
        Args:
            trees (int): number of trees to use if not performing a randomized grid search
            scoring_metric (str): Any sklearn scoring metric appropriate for regression
            hyperparameter_grid (dict): hyperparameters by name
            randomized_search (bool): True for randomized search (default)
            number_iteration_samples (int): Number of models to train during the randomized search for exploring the
                hyperparameter space. More may lead to a better model, but will take longer.

        Returns:
            TrainedSupervisedModel: 
        """
        self.validate_regression('Random Forest Regressor')
        if hyperparameter_grid is None:
            # TODO this is probably redundant now that original_column_names exists
            max_features = hcai_helpers.calculate_random_forest_mtry_hyperparameter(len(self.columns),
                                                                                    self.model_type)
            hyperparameter_grid = {'n_estimators': [10, 50, 200], 'max_features': max_features}
            number_iteration_samples = 5

        algorithm = get_algorithm(RandomForestRegressor,
                                  scoring_metric,
                                  hyperparameter_grid,
                                  randomized_search,
                                  number_iteration_samples=number_iteration_samples,
                                  n_estimators=trees)

        trained_supervised_model = self._create_trained_supervised_model(algorithm)

        return trained_supervised_model

    def create_nn(self,
                  neurons_num=None,
                  activation='relu',
                  optimizer='adam',
                  scoring_metric='accuracy'):
        """
        Define one type of neural network architecture based on Keras.
        The neural network contains one input layer, one hidden layer, and one output layer.

        Args:
            neurons_num (int): number of neurons in the hidden layer.
            activation (str): the type of activation functions. See 'Activations' from the Keras docs.
            optimizer (str): the type of optimizers. See 'Optimizers' from the Keras docs.
            scoring_metric (str): See 'Metrics' from the Keras docs. Note the scoring_metric in Keras is different from sklearn.

        Returns:
            neuralnet: a neural network architecture

        """
        # Calculate input dimension
        input_dim = self.x_train.shape[1]

        # Calculate output dimension
        out_dim = len(list(set(self.y_train)))

        # Calculate number of neurons
        if neurons_num is None:
            neurons_num = (input_dim + 2) // 2

        # Create a neural network architecture
        neuralnet = Sequential()
        neuralnet.add(Dense(input_dim, input_dim=input_dim, activation=activation))
        neuralnet.add(Dense(neurons_num, activation=activation))
        neuralnet.add(Dense(out_dim, activation='softmax'))
        neuralnet.compile(loss='sparse_categorical_crossentropy',
                          # can also use one hot encoding with categorical_crossentropy
                          # for binary: use sigmoid with binary_crossentropy or settings for multiclass
                          # for multiclass: use softmax with sparse_categorical_crossentropy
                          optimizer=optimizer,
                          metrics=[scoring_metric])
        return neuralnet

    def neural_network_classifier(self,
                                  scoring_metric='accuracy',
                                  hyperparameter_grid=None,
                                  randomized_search=True,
                                  number_iteration_samples=1):
        """
        Tune the neural network using grid search over the parameters.
        Tuning is performed by cross validation.

        Args:
            scoring_metric (str): Keras metrics
            hyperparameter_grid (dict): hyperparameters by name
            randomized_search (bool): True for randomized search (default)
            number_iteration_samples (int): Number of models to train during the randomized search for exploring the
                hyperparameter space. More may lead to a better model, but will take longer.

        Returns:
            TrainedSupervisedModel:
        """
        self.validate_classification('Neural Network Classifier')

        # Create a wrapper for sklearn
        neuralnet = KerasClassifier(build_fn=self.create_nn, verbose=0)

        # Initiate a hyperparameter grid if not specified
        if hyperparameter_grid is None:
            activation = ['relu']
            batch_size = [5]
            epochs = [20]
            optimizer = ['adam']

            hyperparameter_grid = dict(activation=activation,
                                       batch_size=batch_size,
                                       epochs=epochs,
                                       optimizer=optimizer)

        algorithm = get_algorithm_neural_network(neuralnet,
                                                 scoring_metric,
                                                 hyperparameter_grid,
                                                 randomized_search,
                                                 number_iteration_samples)

        trained_supervised_model = self._create_trained_supervised_model(algorithm)

        return trained_supervised_model

    def _create_trained_supervised_model(self, algorithm, include_factor_model=True):
        """
        Trains an algorithm, prepares metrics, builds and returns a TrainedSupervisedModel

        Args:
            algorithm (sklearn.base.BaseEstimator): The scikit learn algorithm, ready to fit.
            include_factor_model (bool): Trains a model for top factors. Defaults to True

        Returns:
            TrainedSupervisedModel: a TrainedSupervisedModel
        """
        # Get time before model training
        t0 = time.time()

        algorithm.fit(self.x_train, self.y_train)

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
            factor_model = hcai_factors.prepare_fit_model_for_factors(self.model_type, self.x_train, self.y_train)
        else:
            factor_model = None

        trained_supervised_model = hcai_tsm.TrainedSupervisedModel(
            model=algorithm,
            feature_model=factor_model,
            fit_pipeline=self.pipeline,
            model_type=self.model_type,
            # TODO this is probably redundant now that original_column_names exists
            column_names=self.columns,
            grain_column=self.grain_column,
            prediction_column=self.predicted_column,
            test_set_predictions=test_set_predictions,
            test_set_class_labels=test_set_class_labels,
            test_set_actual=self.y_test,
            metric_by_name=self.metrics(algorithm),
            original_column_names=self.original_column_names,
            categorical_column_info=self.categorical_column_info,
            training_time=time.time() - t0)

        return trained_supervised_model

    def validate_regression(self, model_name=None):
        """
        Raises error if not a regression trainer.
        
        Args:
            model_name (str): Nice string for error messages
        """
        if not self.is_regression:
            raise HealthcareAIError('A {} model can only be trained with a regression trainer.'.format(model_name))

    def validate_classification(self, model_name=None):
        """
        Raises error if not a classification trainer.

        Args:
            model_name (str): Nice string for error messages
        """
        if not self.is_classification:
            raise HealthcareAIError('A {} model can only be trained with a classification trainer.'.format(model_name))


    def _console_log(self, message):
        if self.verbose:
            print('AdvancedSupervisedModelTrainer :: {}'.format(message))
