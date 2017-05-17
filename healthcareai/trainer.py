import time

import healthcareai.pipelines.data_preparation as pipelines
import healthcareai.common.model_eval as hcaieval
from healthcareai.advanced_trainer import AdvancedSupervisedModelTrainer


class SupervisedModelTrainer(object):
    def __init__(self, dataframe, predicted_column, model_type, impute=True, grain_column=None, verbose=False):
        self.grain_column = grain_column,
        self.predicted_column = predicted_column,
        self.grain_column = grain_column,
        self.grain_column = grain_column,

        # Build the pipeline
        pipeline = pipelines.full_pipeline(model_type, predicted_column, grain_column, impute=impute)

        # Run the raw data through the data preparation pipeline
        clean_dataframe = pipeline.fit_transform(dataframe)

        # Instantiate the advanced class
        self._dsm = AdvancedSupervisedModelTrainer(clean_dataframe, model_type, predicted_column, grain_column, verbose)

        # Save the pipeline to the parent class
        self._dsm.pipeline = pipeline

        # Split the data into train and test
        self._dsm.train_test_split()

    def random_forest(self, save_plot=False):
        """ Train a random forest model and print out the model performance metrics. """
        # TODO Convenience method. Probably not needed?
        if self._dsm.model_type is 'classification':
            return self.random_forest_classification(save_plot=save_plot)
        elif self._dsm.model_type is 'regression':
            return self.random_forest_regression()

    def knn(self):
        """ Train a knn model and print out the model performance metrics. """
        model_name = 'KNN'
        print('Training {}'.format(model_name))
        t0 = time.time()

        # Train the model and display the model metrics
        trained_model = self._dsm.knn(scoring_metric='roc_auc', hyperparameter_grid=None, randomized_search=True)
        print_training_timer(model_name, t0)
        print(trained_model.metrics)

        return trained_model

    def random_forest_regression(self):
        """ Train a random forest regression model and print out the model performance metrics. """
        model_name = 'Random Forest Regression'
        print('Training {}'.format(model_name))
        t0 = time.time()

        # Train the model and display the model metrics
        trained_model = self._dsm.random_forest_regressor(trees=200, scoring_metric='neg_mean_squared_error',
                                                          randomized_search=True)
        print_training_timer(model_name, t0)
        print(trained_model.metrics)

        return trained_model

    def random_forest_classification(self, save_plot=False):
        """ Train a random forest classification model and print out the model performance metrics. """
        model_name = 'Random Forest Classification'
        print('Training {}'.format(model_name))
        t0 = time.time()

        # Train the model and display the model metrics
        trained_model = self._dsm.random_forest_classifier(trees=200, scoring_metric='roc_auc', randomized_search=True)
        print_training_timer(model_name, t0)
        print(trained_model.metrics)

        # Save or show the feature importance graph
        hcaieval.plot_rf_from_tsm(trained_model, self._dsm.X_train, save=save_plot)

        return trained_model

    def logistic_regression(self):
        """ Train a logistic regression model and print out the model performance metrics. """
        model_name = 'Logistic Regression'
        print('Training {}'.format(model_name))
        t0 = time.time()

        # Train the model and display the model metrics
        trained_model = self._dsm.logistic_regression(randomized_search=False)
        print_training_timer(model_name, t0)
        print(trained_model.metrics)

        return trained_model

    def linear_regression(self):
        """ Train a linear regression model and print out the model performance metrics. """
        model_name = 'Linear Regression'
        print('Training {}'.format(model_name))
        t0 = time.time()

        # Train the model and display the model metrics
        trained_model = self._dsm.linear_regression(randomized_search=False)
        print_training_timer(model_name, t0)
        print(trained_model.metrics)

        return trained_model

    def ensemble(self):
        """ Train a ensemble model and print out the model performance metrics. """
        model_name = 'ensemble {}'.format(self._dsm.model_type)
        print('Training {}'.format(model_name))
        t0 = time.time()

        # Train the appropriate ensemble of models and display the model metrics
        if self._dsm.model_type is 'classification':
            metric = 'roc_auc'
            trained_model = self._dsm.ensemble_classification(scoring_metric=metric)
        elif self._dsm.model_type is 'regression':
            # TODO stub
            metric = 'neg_mean_squared_error'
            trained_model = self._dsm.ensemble_regression(scoring_metric=metric)

        print(
            'Based on the scoring metric {}, the best algorithm found is: {}'.format(metric, trained_model.model_name))

        print_training_timer(model_name, t0)
        print(trained_model.metrics)

        return trained_model

    def print_metrics(self, trained_model):
        """
        Given a trained model, calculate and print the appropriate performance metrics.

        Args:
            trained_model (BaseEstimator): A scikit-learn trained algorithm
        """
        print(self.metrics(trained_model))

    def metrics(self, trained_model):
        """
        Given a trained model, get the appropriate performance metrics.

        Args:
            trained_model (BaseEstimator): A scikit-learn trained algorithm
        """
        return self._dsm.metrics(trained_model)

    def get_advanced_features(self):
        return self._dsm


def print_training_timer(model_name, start_timestamp):
    """ Given an original timestamp, prints the amount of time that has passed. 

    Args:
        start_timestamp (float): Start time 
        model_name (str): model name
    """
    stop_time = time.time()
    delta_time = round(stop_time - start_timestamp, 2)
    print('Trained a {} model in {} seconds'.format(model_name, delta_time))
