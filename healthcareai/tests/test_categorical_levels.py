import unittest
import numpy as np
import pandas as pd

from healthcareai.common.healthcareai_error import HealthcareAIError
from healthcareai.supervised_model_trainer import SupervisedModelTrainer
from healthcareai.common.get_categorical_levels import get_categorical_levels


class TestTopFactors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """ Load a dataframe, train a linear model and prepare prediction data frames for assertions """
        rows = 200
        np.random.seed(112358)
        train_df = pd.DataFrame({'id': range(rows),
                                 'x': np.random.uniform(low=-5, high=5, size=rows),
                                 'y': np.random.normal(loc=0, scale=1, size=rows),
                                 'color': np.random.choice(['red', 'blue', 'green'], size=rows),
                                 'gender': np.random.choice(['male', 'female'], size=rows)},
                                columns=['id', 'x', 'y', 'color', 'gender'])
        # Assign labels
        # build true decision boundary using temp variable
        train_df['temp'] = 2 * train_df['x'] - train_df['y']
        train_df.loc[train_df['color'] == 'red', 'temp'] += 1
        train_df.loc[train_df['color'] == 'blue', 'temp'] += -1
        train_df.loc[train_df['gender'] == 'male', 'temp'] += 2
        # Add noise to avoid perfect separation
        train_df['temp'] += np.random.normal(scale=1, size=rows)
        # Add label
        train_df['response'] = np.where(train_df['temp'] > 0, 'Y', 'N')
        # drop temp column
        train_df.drop('temp', axis=1, inplace=True)

        hcai = SupervisedModelTrainer(
            dataframe=train_df,
            predicted_column='response',
            model_type='classification',
            impute=True,
            grain_column='id')

        # Train the logistic regression model
        cls.trained_lr = hcai.logistic_regression()

        # single row prediction dataframe
        cls.one_row1 = pd.DataFrame({'id': [2017],
                                     'x': [1.2],
                                     'y': [0.7],
                                     'color': ['red'],
                                     'gender': ['female']},
                                    columns=['id', 'x', 'y', 'color', 'gender'])

        # single row prediction dataframe with different values
        cls.one_row2 = pd.DataFrame({'id': [1066],
                                     'x': [0],
                                     'y': [-1],
                                     'color': ['green'],
                                     'gender': ['male']},
                                    columns=['id', 'x', 'y', 'color', 'gender'])

        # put these rows in a dataframe with all of the training data
        cls.large = pd.concat([cls.one_row1, cls.one_row2, train_df.drop('response', axis=1)])
        # prediction dataframe missing a numeric column
        cls.missing_x = pd.DataFrame({'id': range(50),
                                      'y': np.random.normal(loc=0, scale=1, size=50),
                                      'color': np.random.choice(['red', 'blue', 'green'], size=50),
                                      'gender': np.random.choice(['male', 'female'], size=50)},
                                     columns=['id', 'y', 'color', 'gender'])

        # prediction dataframe missing a categorical column
        cls.missing_color = pd.DataFrame({'id': range(50),
                                          'x': np.random.uniform(low=-5, high=5, size=50),
                                          'y': np.random.normal(loc=0, scale=1, size=50),
                                          'gender': np.random.choice(['male', 'female'], size=50)},
                                         columns=['id', 'x', 'y', 'gender'])

        # dataframe with new category level in one column
        cls.new_color = pd.DataFrame({'id': [1728, 1729],
                                      'x': [1.2, 1.2],
                                      'y': [-0.3, -0.3],
                                      'color': ['purple', np.NaN],
                                      'gender': ['female', 'female']},
                                     columns=['id', 'x', 'y', 'color', 'gender'])

        # dataframe with new category levels in two columns
        cls.new_color_and_gender = pd.DataFrame({'id': [1728, 1729],
                                                 'x': [1.2, 1.2],
                                                 'y': [-0.3, -0.3],
                                                 'color': ['purple', np.NaN],
                                                 'gender': ['other', np.NaN]},
                                                columns=['id', 'x', 'y', 'color', 'gender'])

        # dataframe with known distribution of cagegory levels
        cls.get_levels_df = pd.DataFrame({'grain': range(10),
                                          'letters': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'D'],
                                          'numeric': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                          'numbers_mod_3': ['1', '2', '0', '1', '2', '0', '1', '2', '0', '1'],
                                          'float': [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
                                          'mathematicians': ['Gauss', 'Euler', 'Gauss', 'Galois', 'Gauss',
                                                             'Euler', 'Grothendiek', 'Wiles', 'Hilbert', 'Hilbert'],
                                          'predicted': ['Y', 'Y', 'N', 'Y', 'Y', 'N', 'N', 'N', 'Y', 'Y']},
                                         columns=['grain', 'letters', 'numeric', 'numbers_mod_3', 'float',
                                                  'mathematicians', 'predicted'])
        # Set mathematician column to category and choose the order in which the levels are listed (default is
        # alphabetical)
        cls.get_levels_df['mathematicians'] = cls.get_levels_df['mathematicians'].astype('category',
                                                                                         categories=['Wiles',
                                                                                                     'Euler',
                                                                                                     'Grotheniek',
                                                                                                     'Hilbert',
                                                                                                     'Gauss'],
                                                                                         ordered=False)

        # Reset random seed
        np.random.seed()

    def test_single_row_prediction(self):
        # This test checks that we can make predictions on single row dataframes
        row1_predictions = self.trained_lr.make_predictions(self.one_row1)
        row2_predictions = self.trained_lr.make_predictions(self.one_row2)
        # Compare predictions to fixed values
        self.assertEqual(np.round(row1_predictions.iloc[0, 1], decimals=6), 0.921645)
        self.assertEqual(np.round(row2_predictions.iloc[0, 1], decimals=6), 0.935244)
        # As a futher sanity check, note that using the "true" decision boundary, we would have
        # sigmoid(2.4 - 0.7 + 1) = sigmoid(2.7) ~ 0.937 in the first case and
        # sigmoid(0 + 1 + 0 + 2) = sigmoid(3) ~ 0.953 in the second case

    def test_predictions_are_independent(self):
        # This test checks that the prediction for a single row is independent of the other rows
        row1_predictions = self.trained_lr.make_predictions(self.one_row1)
        row2_predictions = self.trained_lr.make_predictions(self.one_row2)
        full_predictions = self.trained_lr.make_predictions(self.large)
        # Compare predictions when the row is by itself or in a dataframe with all the training data
        self.assertEqual(row1_predictions.iloc[0, 1], full_predictions.iloc[0, 1])
        self.assertEqual(row2_predictions.iloc[0, 1], full_predictions.iloc[1, 1])

    def test_raises_error_on_missing_column(self):
        # This test checks that an error is raised when a column (numeric or categorical) is missing completely
        self.assertRaises(HealthcareAIError, self.trained_lr.make_predictions, dataframe=self.missing_x)
        self.assertRaises(HealthcareAIError, self.trained_lr.make_predictions, dataframe=self.missing_color)

    def test_impute_new_categorical_levels(self):
        # This test checks that new factor levels are being imputed correctly by comparing the prediction on a row with
        # new factor levels to a row with the same data and the new levels replaced with NaNs
        new_factor_predictions1 = self.trained_lr.make_predictions(self.new_color)
        new_factor_predictions2 = self.trained_lr.make_predictions(self.new_color_and_gender)
        self.assertEqual(new_factor_predictions1.iloc[0, 1], new_factor_predictions1.iloc[1, 1])
        self.assertEqual(new_factor_predictions2.iloc[0, 1], new_factor_predictions2.iloc[1, 1])

    def test_get_categorical_levels(self):
        # This test checkst that get_categorical_levels() behaves as desired
        categorical_level_info = get_categorical_levels(dataframe=self.get_levels_df,
                                                        columns_to_ignore=['grain', 'predicted'])
        # Check that numeric columns are not included
        self.assertFalse('float' in categorical_level_info)
        # Check that specified columns are not included
        self.assertFalse('predicted' in categorical_level_info)
        # Check that the right number of columns are included
        self.assertEqual(len(categorical_level_info), 3)
        # Check that the distributions are correctly calculated
        self.assertEqual(np.round(categorical_level_info['letters']['A'], 6), 0.4)
        self.assertEqual(np.round(categorical_level_info['letters']['C'], 6), 0.2)
        self.assertEqual(np.round(categorical_level_info['numbers_mod_3']['2'], 6), 0.3)
        # Check that the desired ordering of category levels is preserved
        self.assertEqual(categorical_level_info['numbers_mod_3'].index[1], '1')
        self.assertEqual(categorical_level_info['mathematicians'].index[0], 'Wiles')
        self.assertEqual(categorical_level_info['mathematicians'].index[4], 'Gauss')
