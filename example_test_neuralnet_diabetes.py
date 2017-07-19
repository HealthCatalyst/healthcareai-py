import pandas as pd
from sklearn.pipeline import Pipeline

from healthcareai import AdvancedSupervisedModelTrainer
import healthcareai.common.filters as hcai_filters
import healthcareai.common.transformers as hcai_transformers
import healthcareai.datasets as hcai_datasets
import healthcareai.trained_models.trained_supervised_model as hcai_tsm
import healthcareai.pipelines.data_preparation as hcai_pipelines


if __name__ == "__main__":
    # Load the diabetes sample data
    dataframe = hcai_datasets.load_diabetes()

    # Drop columns that won't help machine learning
    dataframe.drop(['PatientID'], axis=1, inplace=True)

    # Step 1: Prepare the data using optional imputation. There are two options for this:

    # ## Option 1: Use built in data prep pipeline that does enocding, imputation, null filtering, dummification
    clean_training_dataframe = hcai_pipelines.full_pipeline(
        'classification',
        'ThirtyDayReadmitFLG',
        'PatientEncounterID',
        impute=True).fit_transform(dataframe)

    # Step 2: Instantiate an Advanced Trainer class with your clean and prepared training data
    classification_trainer = AdvancedSupervisedModelTrainer(
        dataframe=clean_training_dataframe,
        model_type='classification',
        predicted_column='ThirtyDayReadmitFLG',
        grain_column='PatientEncounterID',
        data_scaling=True,
        verbose=False)

    # Step 3: split the data into train and test
    classification_trainer.train_test_split()

    # Train the neural network classifier with randomized search
    trained_nn = classification_trainer.neural_network_classifier(randomized_search=True)

    # Create ROC/PR curve
    trained_nn.roc_plot()
    trained_nn.pr_plot()

    # Train the neural network classifier without randomized search
    trained_nn = classification_trainer.neural_network_classifier(randomized_search=False)

    # Create ROC/PR curve
    trained_nn.roc_plot()
    trained_nn.pr_plot()