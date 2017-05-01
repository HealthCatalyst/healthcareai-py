from healthcareai.develop_supervised_model import DevelopSupervisedModel
import healthcareai.pipelines.data_preparation as pipelines
from healthcareai.trained_models.trained_supervised_model import TrainedSupervisedModel
import healthcareai.common.top_factors as factors


class SimpleDevelopSupervisedModel(object):
    def __init__(self, dataframe, predicted_column, model_type, impute=True, grain_column=None, verbose=False):
        self.grain_column = grain_column,
        self.predicted_column = predicted_column,
        self.grain_column = grain_column,
        self.grain_column = grain_column,

        # Build the pipeline
        self.pipeline = pipelines.full_pipeline(model_type, predicted_column, grain_column, impute=impute)

        # Run the raw data through the data preparation pipeline
        clean_dataframe = self.pipeline.fit_transform(dataframe)

        # Instantiate the advanced class
        self._dsm = DevelopSupervisedModel(clean_dataframe, model_type, predicted_column, grain_column, verbose)

        # Split the data into train and test
        self._dsm.train_test_split()

    def random_forest(self):
        # TODO Convenience method. Probably not needed?
        if self._dsm.model_type is 'classification':
            self.random_forest_classification()
        elif self._dsm.model_type is 'regression':
            self.random_forest_regression()

    def knn(self):
        print('Training knn')
        # Train the model
        trained_model = self._dsm.knn(
            scoring_metric='roc_auc',
            hyperparameter_grid=None,
            randomized_search=True)

        # Display the model metrics
        self.print_performance_metrics(trained_model)

    def random_forest_regression(self):
        print('Training random_forest_regression')
        # Train the model
        trained_model = self._dsm.random_forest_regressor(trees=200, scoring_metric='roc_auc', randomized_search=True)
        # Display the model metrics
        self.print_performance_metrics(trained_model)

    def random_forest_classification(self):
        print('Training random_forest_classification')
        # Train the model
        trained_model = self._dsm.random_forest_classifier(trees=200, scoring_metric='roc_auc', randomized_search=True)
        # Display the model metrics
        self.print_performance_metrics(trained_model)

    def logistic_regression(self):
        print('Training logistic_regression')
        # Train the model
        trained_model = self._dsm.logistic_regression()
        # Display the model metrics
        self.print_performance_metrics(trained_model)

    def linear_regression(self):
        print('Training linear_regression')
        # Train the model
        trained_model = self._dsm.linear_regression(randomized_search=False)
        # Display the model metrics
        self.print_performance_metrics(trained_model)

        trained_factor_model = factors.prepare_fit_model_for_factors(self._dsm.model_type, self._dsm.X_train, self._dsm.y_train)

        foo = TrainedSupervisedModel(trained_model,
                                     trained_factor_model,
                                     self.pipeline,
                                     self._dsm.model_type,
                                     self._dsm.X_test.columns.values,
                                     self._dsm.grain_column,
                                     self._dsm.predicted_column,
                                     None,
                                     None)

        return foo

    def ensemble(self):
        if self._dsm.model_type is 'classification':
            self._dsm.ensemble_classification(scoring_metric='roc_auc')
        elif self._dsm.model_type is 'regression':
            # TODO stub
            # self._dsm.ensemble_regression(scoring_metric='roc_auc')
            pass

    def plot_roc(self):
        self._dsm.plot_roc(save=False, debug=False)

    def print_performance_metrics(self, trained_model):
        """
        Given a trained model, calculate and print the appropriate performance metrics.

        Args:
            trained_model (BaseEstimator): A scikit-learn trained algorithm
        """
        performance_metrics = None

        if self._dsm.model_type is 'classification':
            performance_metrics = self._dsm.calculate_classification_metric(trained_model)
        elif self._dsm.model_type is 'regression':
            performance_metrics = self._dsm.calculate_regression_metric(trained_model)

        print(performance_metrics)

    def get_advanced_features(self):
        return self._dsm
