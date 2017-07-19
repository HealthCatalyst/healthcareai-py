#TODO this ctg data is used for testing neural network. This example can be deleted after the test is done.

import healthcareai.datasets as hcai_datasets
import healthcareai.pipelines.data_preparation as hcai_pipelines
from healthcareai import AdvancedSupervisedModelTrainer

if __name__ == "__main__":
    # Load the ctg dataset
    dataframe = hcai_datasets.base.load_ctg()

    # Drop the CLASS variable
    dataframe.drop(['CLASS'], axis=1, inplace=True)

    # Recode the NSP variable for binary classification
    index = dataframe.NSP == 1
    dataframe.loc[index, 'NSP'] = 0
    index = dataframe.NSP != 0
    dataframe.loc[index, 'NSP'] = 1

    # Initiate an Advanced Trainer class with your clean and prepared training data
    classification_trainer = AdvancedSupervisedModelTrainer(
        dataframe=dataframe,
        model_type='classification',
        predicted_column='NSP',
        grain_column='id',
        data_scaling=True,
        verbose=False)

    # Split the data into training and testing sets
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
