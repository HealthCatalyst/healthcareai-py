import unittest
import healthcareai.datasets as ds

class TestDatasets(unittest.TestCase):
    def test_load_acute_inflammations(self):
        df = ds.load_acute_inflammations()
        self.assertEqual(120, df.shape[0])
        self.assertEqual(9, df.shape[1])

    def test_load_cervical_cancer(self):
        df = ds.load_cervical_cancer()
        self.assertEqual(858, df.shape[0])
        self.assertEqual(37, df.shape[1])

    def test_load_diabetes(self):
        df = ds.load_diabetes()
        self.assertEqual(1000, df.shape[0])
        self.assertEqual(7, df.shape[1])

    def test_load_diagnostic_breast_cancer(self):
        df = ds.load_diagnostic_breast_cancer()
        self.assertEqual(569, df.shape[0])
        self.assertEqual(32, df.shape[1])

    def test_load_fertility(self):
        df = ds.load_fertility()
        self.assertEqual(100, df.shape[0])
        self.assertEqual(11, df.shape[1])

    def test_load_heart_disease(self):
        df = ds.load_heart_disease()
        self.assertEqual(270, df.shape[0])
        self.assertEqual(15, df.shape[1])

    def test_load_mammographic_masses(self):
        df = ds.load_mammographic_masses()
        self.assertEqual(961, df.shape[0])
        self.assertEqual(7, df.shape[1])

    def test_load_pima_indians_diabetes(self):
        df = ds.load_pima_indians_diabetes()
        self.assertEqual(768, df.shape[0])
        self.assertEqual(10, df.shape[1])

    def test_load_prognostic_breast_cancer(self):
        df = ds.load_prognostic_breast_cancer()
        self.assertEqual(198, df.shape[0])
        self.assertEqual(35, df.shape[1])

    def test_load_thoracic_surgery(self):
        df = ds.load_thoracic_surgery()
        self.assertEqual(470, df.shape[0])
        self.assertEqual(18, df.shape[1])
