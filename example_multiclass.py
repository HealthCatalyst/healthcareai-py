"""Creates and compares classification models using sample clinical data.

Please use this example to learn about healthcareai before moving on to the next example.

If you have not installed healthcare.ai, refer to the instructions here:
  http://healthcareai-py.readthedocs.io

To run this example:
  python3 example_classification_1.py

This code uses the diabetes sample data in datasets/data/diabetes.csv.
"""

import healthcareai
import healthcareai.trained_models.trained_supervised_model as tsm_plots


def main():
    """Template script for using healthcareai to train a classification model."""
    # Load the included dermatology sample data that has 6 classes
    dataframe = healthcareai.load_dermatology()

    # Peek at the first 5 rows of data
    print(dataframe.head(5))

    # Drop columns that won't help machine learning
    dataframe.drop(['target_str'], axis=1, inplace=True)

    # Step 1: Setup a healthcareai classification trainer. This prepares your data for model building
    classification_trainer = healthcareai.SupervisedModelTrainer(
        dataframe,
        predicted_column='target_num',
        model_type='classification',
        grain_column='PatientID',
        impute=True)

    # Look at the first few rows of your dataframe after loading the data
    print('\n\n-------------------[ Cleaned Dataframe ]--------------------------')
    print(classification_trainer.clean_dataframe.head())

    # Step 2: train some models

    # Train a KNN model
    trained_knn = classification_trainer.knn()

    # Train a logistic regression model
    trained_lr = classification_trainer.logistic_regression()

    # Train a random forest model and view the feature importance plot
    trained_random_forest = classification_trainer.random_forest(save_plot=False)

    # Once you are happy with the performance of any model, you can save it for use later in predicting new data.
    # File names are timestamped and look like '2017-05-31T12-36-21_classification_RandomForestClassifier.pkl')
    # Note the file you saved and that will be used in example_classification_2.py
    # trained_random_forest.save()


if __name__ == "__main__":
    main()
