import healthcareai.datasets as hcai_datasets
import healthcareai.pipelines.data_preparation as hcai_pipelines
from healthcareai import AdvancedSupervisedModelTrainer

if __name__ == "__main__":
    # Load the dataset
    dataframe = hcai_datasets.base.load_multiclass()

    # Drop the original string labeled target variable
    dataframe.drop(['target_str'], axis=1, inplace=True)

    # ## Use built in data prep pipeline that does enocding, imputation, null filtering, dummification
    clean_training_dataframe = hcai_pipelines.full_pipeline(
        'classification',
        'target_num',
        'PatientID',
        impute=True).fit_transform(dataframe)

    # ## Build a neural network
    # Initiate an Advanced Trainer class with your clean and prepared training data
    classification_trainer = AdvancedSupervisedModelTrainer(
        dataframe=clean_training_dataframe,
        model_type='classification',
        predicted_column='target_num',
        grain_column='PatientID',
        data_scaling=True,
        verbose=False)

    # Split the data into training and testing sets
    classification_trainer.train_test_split()

    # Train the neural network classifier with randomized search
    trained_nn = classification_trainer.neural_network_classifier(randomized_search=True)
    trained_nn.confusion_matrix_plot()
    trained_nn.print_confusion_matrix()
