import unittest

import numpy as np

from healthcareai.common.healthcareai_error import HealthcareAIError
from healthcareai.trained_models.trained_supervised_model import \
    _max_probability_extractor, _probabilities_by_label


class TestProbabilitesByLabel(unittest.TestCase):
    def test_returns_list(self):
        self.assertIsInstance(_probabilities_by_label(None, None), list)

    def test_raise_error_if_column_number_unequal_to_labels(self):
        labels = ['a', 'b']
        matrix = np.array([
            [1, 2, 3]
        ])

        self.assertRaises(
            HealthcareAIError,
            _probabilities_by_label,
            labels,
            matrix
        )

    def test_one_for_one(self):
        expected = [{'a': 1, 'b': 7}]

        labels = ['a', 'b']
        matrix = np.array([
            [1, 7]
        ])
        observed = _probabilities_by_label(labels, matrix)

        self.assertEqual(expected, observed)

    def test_three_by_four(self):
        expected = [
            {'a': 3, 'b': 5, 'c': 4, 'd': 1},
            {'a': 5, 'b': 4, 'c': 1, 'd': 0},
            {'a': 3, 'b': 5, 'c': 4, 'd': 1},
            {'a': 3, 'b': 5, 'c': 4, 'd': 1},
        ]

        labels = ['a', 'b', 'c', 'd']
        matrix = np.array([
            [3, 5, 4, 1],
            [5, 4, 1, 0],
            [3, 5, 4, 1],
            [3, 5, 4, 1]
        ])
        observed = _probabilities_by_label(labels, matrix)

        self.assertEqual(expected, observed)


class TestMaxProbabilityExtractor(unittest.TestCase):
    def test_max_of_one(self):
        stuff = {'a': 33}
        observed = _max_probability_extractor(stuff)

        self.assertIsInstance(observed, (int, float))
        self.assertEqual(33, observed)

    def test_max_of_many(self):
        stuff = {'a': 33, 'b': 99, 'c': 0, 'd': 98.99}
        observed = _max_probability_extractor(stuff)

        self.assertIsInstance(observed, (int, float))
        self.assertEqual(99, observed)
