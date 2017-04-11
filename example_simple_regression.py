"""This file is used to create and compare two models on a particular dataset.
It provides examples of reading from both csv and SQL Server. Note that this
example can be run as-is after installing healthcare.ai. After you have
found that one of the models works well on your data, move to Example2
"""
from healthcareai.simple_mode import SimpleDevelopSupervisedModel
import pandas as pd
import time

# Start a timer
t0 = time.time()

# CSV snippet for reading data into dataframe
dataframe = pd.read_csv('healthcareai/tests/fixtures/HCPyDiabetesClinical.csv', na_values=['None'])

# Drop columns that won't help machine learning
dataframe.drop(['PatientID', 'InTestWindowFLG'], axis=1, inplace=True)

# Look at the first few rows of your dataframe after the data preparation
print(dataframe.head())

# Step 1: Setup healthcareai for developing a regression model.
hcai = SimpleDevelopSupervisedModel(dataframe, 'SystolicBPNBR', 'regression', impute=True, grain_column='PatientEncounterID')

# Run the linear regression model
hcai.linear_regression()

print('\nTime:\n', time.time() - t0)
