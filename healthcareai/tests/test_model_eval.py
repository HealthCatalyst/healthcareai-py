from healthcareai.common.model_eval import GenerateAUC
import pandas as pd
import numpy as np
import unittest

class Test_AUC(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({'a':np.repeat(np.arange(.1,1.1,.1),10)})
        b = np.repeat(0,100)
        b[[56,62,63,68,74,75,76,81,82,84,85,87,88] + list(range(90,100))] = 1
        self.df['b'] = b

    def runTestROC_AUC(self):
        out = GenerateAUC(self['a'],self['b'],'SS',False,False)
        self.assertAlmostEqual(out['AU_ROC'], 0.94)

    def runTestPR_AUC(self):
        out = GenerateAUC(self['a'],self['b'],'PR',False,False)
        self.assertAlmostEqual(out['AU_PR'], 0.86)

    def runTestROC_tpr(self):
        out = GenerateAUC(self['a'],self['b'],'SS',False,False)
        self.assertAlmostEqual(out['BestTpr'], 0.9565)

    def runTestROC_fpr(self):
        out = GenerateAUC(self['a'],self['b'],'SS',False,False)
        self.assertAlmostEqual(out['BestFpr'], 0.2338)

    def runTestPR_precision(self):
        out = GenerateAUC(self['a'],self['b'],'PR',False,False)
        self.assertAlmostEqual(out['BestPrecision'], 0.8)

    def runTestPR_recall(self):
        out = GenerateAUC(self['a'], self['b'], 'PR', False, False)
        self.assertAlmostEqual(out['BestRecall'], 0.6956)

    def tearDown(self):
        del self.df