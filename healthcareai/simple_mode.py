from healthcareai.develop_supervised_model import DevelopSupervisedModel
import healthcareai.pipelines.data_preparation as pipelines


class SimpleDevelopSupervisedModel(object):
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
        self._dsm = DevelopSupervisedModel(clean_dataframe, model_type, predicted_column, grain_column, verbose)

        # Save the pipeline to the parent class
        self._dsm.pipeline = pipeline

        # Split the data into train and test
        self._dsm.train_test_split()

    def random_forest(self):
        # TODO Convenience method. Probably not needed?
        if self._dsm.model_type is 'classification':
            return self.random_forest_classification()
        elif self._dsm.model_type is 'regression':
            return self.random_forest_regression()

    def knn(self):
        print('Training knn')

        # Train the model and display the model metrics
        trained_model = self._dsm.knn(scoring_metric='roc_auc', hyperparameter_grid=None, randomized_search=True)
        print(trained_model.metrics())

        return trained_model

    def random_forest_regression(self):
        print('Training random_forest_regression')

        # Train the model and display the model metrics
        trained_model = self._dsm.random_forest_regressor(trees=200, scoring_metric='neg_mean_squared_error',
                                                          randomized_search=True)
        print(trained_model.metrics())

        return trained_model

    def random_forest_classification(self):
        print('Training random_forest_classification')

        # Train the model and display the model metrics
        trained_model = self._dsm.random_forest_classifier(trees=200, scoring_metric='roc_auc', randomized_search=True)
        print(trained_model.metrics())

        return trained_model

    def logistic_regression(self):
        print('Training logistic_regression')

        # Train the model and display the model metrics
        trained_model = self._dsm.logistic_regression(randomized_search=False)
        print(trained_model.metrics())

        return trained_model

    def linear_regression(self):
        print('Training linear_regression')

        # Train the model and display the model metrics
        trained_model = self._dsm.linear_regression(randomized_search=False)
        print(trained_model.metrics())

        return trained_model

    def ensemble(self):
        print('Running ensemble training')

        # Train the appropriate ensemble of models and display the model metrics
        if self._dsm.model_type is 'classification':
            metric = 'roc_auc'
            trained_model = self._dsm.ensemble_classification(scoring_metric=metric)
        elif self._dsm.model_type is 'regression':
            # TODO stub
            metric = 'neg_mean_squared_error'
            trained_model = self._dsm.ensemble_regression(scoring_metric=metric)

        print('Based on the scoring metric {}, the best algorithm found is: {}'.format(
            metric,
            type(trained_model.model.estimator).__name__))

        print(trained_model.metrics())
        return trained_model

    def plot_roc(self):
        """ Plot ROC curve """
        # TODO This will not work without a linear and random forest model for now until the base function is refactored
        self._dsm.plot_roc(save=False, debug=False)

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
