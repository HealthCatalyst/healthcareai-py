from .develop_supervised_model import DevelopSupervisedModel

class SimpleDevelopSupervisedModel(object):
    def __init__(self, dataframe, predictedcol, modeltype, graincol=None, verbose=False):
        self._dsm = DevelopSupervisedModel(dataframe, predictedcol, modeltype, graincol, verbose)

    def imputation(self):
        pass

    def knn(self):
        self._dsm.knn(
            scoring_metric='roc_auc',
            hyperparameter_grid=None,
            randomized_search=True)

    def random_forest(self):
        if self._dsm.modeltype is 'classification':
            self._dsm.advanced_random_forest_classifier(
                trees=200,
                scoring_metric='roc_auc',
                hyperparameter_grid=None,
                randomized_search=True)
        elif self._dsm.modeltype is 'regression':
            # TODO STUB
            pass

    def ensemble(self):
        if self._dsm.modeltype is 'classification':
            self._dsm.ensemble_classification(
                scoring_metric='roc_auc'
            )
        elif self._dsm.modeltype is 'regression':
            # TODO STUB
            pass

    def plot_roc(self):
        self._dsm.plot_roc(save=False, debug=False)

    def logistic_regression(self):
        self._dsm.logistic_regression()

    def get_advanced_features(self):
        return self._dsm