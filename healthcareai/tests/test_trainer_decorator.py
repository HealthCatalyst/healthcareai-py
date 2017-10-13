import unittest
from healthcareai.supervised_model_trainer import SupervisedModelTrainer
import healthcareai.datasets as hcai_datasets
import sys
from io import StringIO


class TestTrainerDecorator(unittest.TestCase):

    """Tests for the training decorator.

    We will compare a decorated linear regression and a undecorated linear regression output. They should not be
    the same.

    """

    def setUp(self):
        df = hcai_datasets.load_diabetes()

        # Drop columns that won't help machine learning
        columns_to_remove = ['PatientID']
        df.drop(columns_to_remove, axis=1, inplace=True)

        self.regression_trainer = SupervisedModelTrainer(df,
                                                         'SystolicBPNBR',
                                                         'regression',
                                                         grain_column='PatientEncounterID',
                                                         impute=True,
                                                         verbose=False)

        def undecorated_lr(self):
            return self._advanced_trainer.linear_regression(randomized_search=False)

        self.regression_trainer.undecorated_lr = undecorated_lr.__get__(self.regression_trainer,
                                                                        self.regression_trainer.__class__)

    def test_decorator(self):
        out = StringIO()
        sys.stdout = out
        self.regression_trainer.linear_regression()
        decorated_output = out.getvalue().strip()

        out = StringIO()
        sys.stdout = out
        self.regression_trainer.undecorated_lr()
        undecorated_output = out.getvalue().strip()

        assert decorated_output != undecorated_output
