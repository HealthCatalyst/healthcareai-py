import unittest
from healthcareai.supervised_model_trainer import SupervisedModelTrainer
import healthcareai.datasets as hcai_datasets


class TestTrainerDecorator(unittest.TestCase):
    """

    """
    def setUp(self):
        df = hcai_datasets.load_diabetes()

        # Drop columns that won't help machine learning
        columns_to_remove = ['PatientID']
        df.drop(columns_to_remove, axis=1, inplace=True)

        self.classification_trainer = SupervisedModelTrainer(dataframe=df,
                                                             predicted_column='ThirtyDayReadmitFLG',
                                                             model_type='classification',
                                                             impute=True,
                                                             grain_column='PatientEncounterID',
                                                             verbose=False)

        self.classification_trainer.undecorated_lasso_regression = \
            lambda myself: myself._advanced_trainer.lasso_regression(randomized_search=False)

    def test_decorator(self):
        model = self.classification_trainer.lasso_regression()
