from healthcareai.common.model_eval import GenerateAUC
import pandas as pd
import numpy as np
import unittest

class Test_AUC(unittest.TestCase):
    # TODO: check that col types remain the same after imputation
    def setUp(self):
        self.df = pd.DataFrame({'a':np.repeat(np.arange(.1,1.1,.1),10)})
        b = np.repeat(0,100)
        b[[56,62,63,68,74,75,76,81,82,84,85,87,88] + list(range(90,100))] = 1
        self.df['b'] = b

    def runTestROC(self):
        auc = GenerateAUC(self['a'],self['b'],'SS',False)
        self.assertAlmostEqual(auc, 0.94)

    def runTestPR(self):
        auc = GenerateAUC(self['a'],self['b'],'PR',False)
        self.assertAlmostEqual(auc, 0.86)

    def tearDown(self):
        del self.df