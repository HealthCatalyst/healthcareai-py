import time

import healthcareai.pipelines.data_preparation as hcai_pipelines
import healthcareai.trained_models.trained_supervised_model as hcai_tsm
from healthcareai.advanced_supvervised_model_trainer import AdvancedSupervisedModelTrainer


class SupervisedModelTrainer(object):
    """
    This class helps create a model using several common classifiers and regressors, both of which report appropiate
    metrics.
    """

    def __init__(self, dataframe, predicted_column, model_type, impute=True, grain_column=None, verbose=False):
        self.grain_column = grain_column,
        self.predicted_column = predicted_column,
        self.grain_column = grain_column,
        self.grain_column = grain_column,

        # Build the pipeline
        # TODO This pipeline may drop nulls in prediction rows if impute=False
        # TODO See https://github.com/HealthCatalyst/healthcareai-py/issues/276
        pipeline = hcai_pipelines.full_pipeline(model_type, predicted_column, grain_column, impute=impute)

        # Run the raw data through the data preparation pipeline
        clean_dataframe = pipeline.fit_transform(dataframe)

        # Instantiate the advanced class
        self._advanced_trainer = AdvancedSupervisedModelTrainer(clean_dataframe, model_type, predicted_column,
                                                                grain_column, verbose)

        # Save the pipeline to the parent class
        self._advanced_trainer.pipeline = pipeline

        # Split the data into train and test
        self._advanced_trainer.train_test_split()

    @property
    def clean_dataframe(self):
        """ Returns the dataframe after the preparation pipeline (imputation and such) """
        return self._advanced_trainer.dataframe

    def random_forest(self, save_plot=False):
        """ Train a random forest model and print out the model performance metrics. """
        # TODO Convenience method. Probably not needed?
        if self._advanced_trainer.model_type is 'classification':
            return self.random_forest_classification(save_plot=save_plot)
        elif self._advanced_trainer.model_type is 'regression':
            return self.random_forest_regression()

    def knn(self):
        """ Train a knn model and print out the model performance metrics. """
        model_name = 'KNN'
        print('\nTraining {}'.format(model_name))
        t0 = time.time()

        # Train the model and display the model metrics
        trained_model = self._advanced_trainer.knn(scoring_metric='roc_auc', hyperparameter_grid=None,
                                                   randomized_search=True)
        print_training_results(model_name, t0, trained_model)

        return trained_model

    def random_forest_regression(self):
        """ Train a random forest regression model and print out the model performance metrics. """
        model_name = 'Random Forest Regression'
        print('\nTraining {}'.format(model_name))
        t0 = time.time()

        # Train the model and display the model metrics
        trained_model = self._advanced_trainer.random_forest_regressor(trees=200,
                                                                       scoring_metric='neg_mean_squared_error',
                                                                       randomized_search=True)
        print_training_results(model_name, t0, trained_model)

        return trained_model

    def random_forest_classification(self, save_plot=False):
        """ Train a random forest classification model, print out performance metrics and show a ROC plot. """
        model_name = 'Random Forest Classification'
        print('\nTraining {}'.format(model_name))
        t0 = time.time()

        # Train the model and display the model metrics
        trained_model = self._advanced_trainer.random_forest_classifier(trees=200, scoring_metric='roc_auc',
                                                                        randomized_search=True)
        print_training_results(model_name, t0, trained_model)

        # Save or show the feature importance graph
        hcai_tsm.plot_rf_features_from_tsm(trained_model, self._advanced_trainer.X_train, save=save_plot)

        return trained_model

    def logistic_regression(self):
        """ Train a logistic regression model and print out the model performance metrics. """
        model_name = 'Logistic Regression'
        print('\nTraining {}'.format(model_name))
        t0 = time.time()

        # Train the model and display the model metrics
        trained_model = self._advanced_trainer.logistic_regression(randomized_search=False)
        print_training_results(model_name, t0, trained_model)

        return trained_model

    def linear_regression(self):
        """ Train a linear regression model and print out the model performance metrics. """
        model_name = 'Linear Regression'
        print('\nTraining {}'.format(model_name))
        t0 = time.time()

        # Train the model and display the model metrics
        trained_model = self._advanced_trainer.linear_regression(randomized_search=False)
        print_training_results(model_name, t0, trained_model)

        return trained_model

    def ensemble(self):
        """ Train a ensemble model and print out the model performance metrics. """
        # TODO consider making a scoring parameter (which will necessitate some more logic
        model_name = 'ensemble {}'.format(self._advanced_trainer.model_type)
        print('\nTraining {}'.format(model_name))
        t0 = time.time()

        # Train the appropriate ensemble of models and display the model metrics
        if self._advanced_trainer.model_type is 'classification':
            metric = 'roc_auc'
            trained_model = self._advanced_trainer.ensemble_classification(scoring_metric=metric)
        elif self._advanced_trainer.model_type is 'regression':
            # TODO stub
            metric = 'neg_mean_squared_error'
            trained_model = self._advanced_trainer.ensemble_regression(scoring_metric=metric)

        print(
            'Based on the scoring metric {}, the best algorithm found is: {}'.format(metric,
                                                                                     trained_model.algorithm_name))

        print_training_results(model_name, t0, trained_model)

        return trained_model

    @property
    def advanced_features(self):
        """ Returns the underlying AdvancedSupervisedModelTrainer instance. For advanced users only. """
        return self._advanced_trainer


def print_training_timer(model_name, start_timestamp):
    """ Given an original timestamp, prints the amount of time that has passed. 

    Args:
        start_timestamp (float): Start time 
        model_name (str): model name
    """
    stop_time = time.time()
    delta_time = round(stop_time - start_timestamp, 2)
    print('    Trained a {} model in {} seconds'.format(model_name, delta_time))


def print_training_results(model_name, t0, trained_model):
    """
    Print metrics, stats and hyperparameters of a training.
    Args:
        model_name (str): Name of the model 
        t0 (float): Training start time
        trained_model (TrainedSupervisedModel): The trained supervised model
    """
    print_training_timer(model_name, t0)

    hyperparameters = trained_model.best_hyperparameters
    if hyperparameters is None:
        hyperparameters = 'N/A: No hyperparameter search was performed'
    print('Best hyperparameters found are:\n    {}'.format(hyperparameters))

    if trained_model.is_classification:
        print('{} performance metrics:\n    Accuracy: {:03.2f}\n    ROC AUC: {:03.2f}\n    PR AUC: {:03.2f}'.format(
            model_name,
            trained_model.metrics['accuracy'],
            trained_model.metrics['roc_auc'],
            trained_model.metrics['pr_auc']))
    elif trained_model.is_regression:
        print('{} performance metrics:\n    Mean Squared Error (MSE): {}\n    Mean Absolute Error (MAE): {}'.format(
            model_name,
            trained_model.metrics['mean_squared_error'],
            trained_model.metrics['mean_absolute_error']))
