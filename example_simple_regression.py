"""This file is used to create and compare two models on a particular dataset.
It provides examples of reading from both csv and SQL Server. Note that this
example can be run as-is after installing healthcare.ai. After you have
found that one of the models works well on your data, move to Example2
"""
import pandas as pd
import time

import healthcareai.common.file_io_utilities as io
import healthcareai.common.predict as predict

from healthcareai.simple_mode import SimpleDevelopSupervisedModel
from healthcareai.simple_deploy_supervised_model import SimpleDeploySupervisedModel

# Start a timer
t0 = time.time()

# CSV snippet for reading data into dataframe
dataframe = pd.read_csv('healthcareai/tests/fixtures/DiabetesClincialSampleData.csv', na_values=['None'])

# Drop columns that won't help machine learning
dataframe.drop(['PatientID', 'InTestWindowFLG'], axis=1, inplace=True)

# Look at the first few rows of your dataframe after the data preparation
print(dataframe.head())

# Step 1: Setup healthcareai for developing a regression model.
hcai = SimpleDevelopSupervisedModel(dataframe,
                                    'SystolicBPNBR',
                                    'regression',
                                    impute=True,
                                    grain_column='PatientEncounterID')

# Train the linear regression model
trained_linear_model = hcai.linear_regression()
print('Model trained in {} seconds'.format(time.time() - t0))

# Once you are happy with the result of the trained model, it is time to save the model.
saved_model_filename = 'linear_regression_2017-04-18.pkl'
io.save_object_as_pickle(saved_model_filename, trained_linear_model)
print('model saved as {}'.format(saved_model_filename))


#
# # Deploy the model
# deployment = SimpleDeploySupervisedModel(
#     model_type='regression',
#     impute=True,
#     saved_model_filename='linear_regression_2017-04-18.pkl',
#     destination_server='localhost',
#     destination_db_schema_table='[SAM].[foo].bar',
#     predicted_column='foo',
#     grain_column='PatientID'
# )


# Now that you have a saved model, run a prediction
loaded_model = io.load_pickle_file(saved_model_filename)
print('Model loaded. Type: {}'.format(type(loaded_model)))

# TODO swap out fake data for real databaes sql
prediction_dataframe = pd.read_csv('healthcareai/tests/fixtures/DiabetesClincialSampleData.csv', na_values=['None'])

# Set None string to be None type
prediction_dataframe.replace(['None'], [None], inplace=True)

# Drop columns that won't help machine learning
prediction_dataframe.drop(['PatientID', 'SystolicBPNBR'], axis=1, inplace=True)

print(prediction_dataframe.head())

# TODO Now make this predict something

# TODO 2017-04-21 This is where I left off.
#   - [ ] convert dataframe to numpy array (is this needed?)
#   - [ ] run dataframe through the pipeline so that the columns are ready to go
#   - [ ] drop the target column
#   - [ ] once this works, you've completed the bare bones chain.
#       - [ ] assess whether the entire Deploy class is needed at all! Maybe the functions can be torn out to make the toolbox then rebuilt into a simple class interface

predictions = predict.predict_regression_from_pickle(prediction_dataframe, saved_model_filename)
print(predictions)
# loaded_model.predict_to_db(prediction_dataframe)

## API I'd love to see

df = pd.read_csv('healthcareai/tests/fixtures/HCPyDiabetesClinical.csv', na_values=['None'])
df_to_score = df[df['InTestWindowFLG'] == 'Y']
rf_model = load_saved_model('rf_model.pkl')

# Score records and save scores to .csv.
# note this does all the pipeline and data prep so you don't have to think about it.
df_scored = rf_model.score(df_to_score, saveto='scores.csv')

