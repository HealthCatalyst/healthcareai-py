from .develop_supervised_model import DevelopSupervisedModel


class SimpleDevelopSupervisedModel(object):
    def __init__(self, dataframe, predicted_column, model_type, impute, grain_column=None, verbose=False):
        self._dsm = DevelopSupervisedModel(dataframe, model_type, predicted_column, grain_column, verbose)
        self._dsm.data_preparation(impute=impute)

    def knn(self):
        self._dsm.knn(
            scoring_metric='roc_auc',
            hyperparameter_grid=None,
            randomized_search=True)

    def random_forest(self):
        if self._dsm.model_type is 'classification':
            self._dsm.random_forest_classifier(
                trees=200,
                scoring_metric='roc_auc',
                hyperparameter_grid=None,
                randomized_search=True)
        elif self._dsm.model_type is 'regression':
            # TODO STUB
            pass

    def ensemble(self):
        if self._dsm.model_type is 'classification':
            self._dsm.ensemble_classification(
                scoring_metric='roc_auc'
            )
        elif self._dsm.model_type is 'regression':
            # TODO STUB
            pass

    def plot_roc(self):
        self._dsm.plot_roc(save=False, debug=False)

    def logistic_regression(self):
        self._dsm.logistic_regression()

    def linear_regression(self):
        # Train the model
        trained_model = self._dsm.linear_regression(randomized_search=False)

        # Calculate the model metrics
        performance_metrics = self._dsm.calculate_regression_metric(trained_model)

        print(performance_metrics)

    def get_advanced_features(self):
        return self._dsm