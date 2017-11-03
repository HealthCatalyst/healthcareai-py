"""
Trains Supervised Models.

Provides users a simple interface for machine learning.

More advanced users may use `AdvancedSupervisedModelTrainer`
"""

import healthcareai.pipelines.data_preparation as hcai_pipelines
import healthcareai.trained_models.trained_supervised_model as hcai_tsm
import healthcareai.common.cardinality_checks as hcai_ordinality
import healthcareai.common.missing_target_check as hcai_target_check
from healthcareai.common.categorical_levels import calculate_categorical_frequencies
from healthcareai.advanced_supvervised_model_trainer import AdvancedSupervisedModelTrainer
from healthcareai.common.trainer_output import trainer_output


class SupervisedModelTrainer(object):
    """Train supervised models.

    This class trains models using several common classifiers and regressors
    and reports appropriate metrics.
    """

    def __init__(
            self,
            dataframe,
            predicted_column,
            model_type,
            impute=True,
            grain_column=None,
            binary_positive_label=None,
            verbose=True):
        """
        Set up a SupervisedModelTrainer.

        Helps the user by checking for high cardinality features (such as IDs or
        other unique identifiers) and low cardinality features (a column where
        all values are equal.

        If you have a binary classification task (one with two categories of
        predictions), there are many common ways to encode your prediction
        categories. healthcareai helps you by making these assumptions about
        which is the positive class label. healthcareai assumes the following
        are 'positive labels':

        | Labels | Positive Label |
        | ------ | -------------- |
        | `True` | `True`/`False` |
        | `1`    | `1`/`0`        |
        | `1`    | `1`/`-1`       |
        | `Y`    | `Y`/`N`        |
        | `Yes`  | `Yes`/`No`     |

        If you have another encoding you prefer to use you may specify the
        `binary_positive_label` argument. For example, if you want to
        identify `high_utilizers` vs `low_utilizers`) you would add the
        `binary_positive_label='high_utilizers` argument when creating your
        `SupervisedModelTrainer`.

        Args:
            dataframe (pandas.core.frame.DataFrame): The training data in a pandas dataframe
            predicted_column (str): The name of the prediction column
            model_type (str): trainer type ('classification' or 'regression')
            impute (bool): True to impute data (mean of numeric columns and mode of categorical ones). False to drop rows
                that contain any null values.
            grain_column (str): The name of the grain column
            binary_positive_label (str|int): Optional positive class label for binary classification tasks.
            verbose (bool): Set to true for verbose output. Defaults to True.

        Raises:
            HealthcareAIError: Target column contains missing data.
        """
        self.predicted_column = predicted_column
        self.grain_column = grain_column
        self.binary_positive_label = binary_positive_label

        hcai_target_check.missing_target_check(dataframe, predicted_column)

        # Low/high cardinality checks. Warn the user and allow them to proceed.
        hcai_ordinality.check_high_cardinality(dataframe, self.grain_column)
        hcai_ordinality.check_one_cardinality(dataframe)

        # Build the pipelines
        # Note: Missing numeric values are imputed in prediction. If we didnt't
        # impute, those rows would be dropped, resulting in missing predictions.
        train_pipeline = hcai_pipelines.training_pipeline(
            predicted_column,
            grain_column,
            impute=impute)
        prediction_pipeline = hcai_pipelines.prediction_pipeline(
            predicted_column,
            grain_column)

        # Fit both pipelines with the same raw data.
        clean_dataframe = train_pipeline.fit_transform(dataframe)
        _ = prediction_pipeline.fit(dataframe)

        # Instantiate the advanced class
        self._advanced_trainer = AdvancedSupervisedModelTrainer(
            dataframe=clean_dataframe,
            model_type=model_type,
            predicted_column=predicted_column,
            grain_column=grain_column,
            original_column_names=dataframe.columns.values,
            binary_positive_label=self.binary_positive_label,
            verbose=verbose)

        # Save the pipeline to the underlying advanced trainer
        self._advanced_trainer.pipeline = prediction_pipeline

        # Split the data into train and test
        self._advanced_trainer.train_test_split()

        self._advanced_trainer.categorical_column_info = calculate_categorical_frequencies(
            dataframe=dataframe,
            columns_to_ignore=[grain_column, predicted_column])

    @property
    def clean_dataframe(self):
        """Return the dataframe after the preparation pipeline."""
        return self._advanced_trainer.dataframe

    @property
    def class_labels(self):
        """Return class labels."""
        return self._advanced_trainer.class_labels

    @property
    def number_of_classes(self):
        """Return number of classes."""
        return self._advanced_trainer.number_of_classes

    def random_forest(self, feature_importance_limit=15, save_plot=False):
        """
        Train a random forest model and print model performance metrics.

        Args:
            feature_importance_limit (int): The maximum number of features to
            show in the feature importance plot
            save_plot (bool): For the feature importance plot, True to save
            plot (will not display). False by default to
                display.

        Returns:
            TrainedSupervisedModel: A trained supervised model.
        """
        if self._advanced_trainer.model_type is 'classification':
            return self.random_forest_classification(
                feature_importance_limit=feature_importance_limit,
                save_plot=save_plot)
        elif self._advanced_trainer.model_type is 'regression':
            return self.random_forest_regression()

    @trainer_output
    def knn(self):
        """Train a knn model and print model performance metrics.

        Returns:
            TrainedSupervisedModel: A trained supervised model.
        """
        return self._advanced_trainer.knn(
            scoring_metric='accuracy',
            hyperparameter_grid=None,
            randomized_search=True)

    @trainer_output
    def random_forest_regression(self):
        """Train a random forest regression model and print performance metrics.

        Returns:
            TrainedSupervisedModel: A trained supervised model.
        """
        return self._advanced_trainer.random_forest_regressor(
            trees=200,
            scoring_metric='neg_mean_squared_error',
            randomized_search=True)

    @trainer_output
    def random_forest_classification(self, feature_importance_limit=15,
                                     save_plot=False):
        """Train random forest classification and show feature importance plot.

        Args:
            feature_importance_limit (int): The maximum number of features to
                show in the feature importance plot
            save_plot (bool): For the feature importance plot, True to save
                plot (will not display). False by default to display.

        Returns:
            TrainedSupervisedModel: A trained supervised model.
        """
        model = self._advanced_trainer.random_forest_classifier(
            trees=200,
            scoring_metric='accuracy',
            randomized_search=True)

        # Save or show the feature importance graph
        hcai_tsm.plot_rf_features_from_tsm(
            model,
            self._advanced_trainer.x_train,
            feature_limit=feature_importance_limit,
            save=save_plot)

        return model

    @trainer_output
    def logistic_regression(self):
        """Train a logistic regression model and print performance metrics.

        Returns:
            TrainedSupervisedModel: A trained supervised model.
        """
        return self._advanced_trainer.logistic_regression(
            randomized_search=False)

    @trainer_output
    def linear_regression(self):
        """Train a linear regression model and print performance metrics.

        Returns:
            TrainedSupervisedModel: A trained supervised model.
        """
        return self._advanced_trainer.linear_regression(randomized_search=False)

    @trainer_output
    def lasso_regression(self):
        """Train a lasso regression model and print performance metrics.

        Returns:
            TrainedSupervisedModel: A trained supervised model.
        """
        return self._advanced_trainer.lasso_regression(randomized_search=False)

    @trainer_output
    def ensemble(self):
        """Train a ensemble model and print performance metrics.

        Returns:
            TrainedSupervisedModel: A trained supervised model.
        """
        # TODO consider making a scoring parameter (which will need more logic
        if self._advanced_trainer.model_type is 'classification':
            metric = 'accuracy'
            model = self._advanced_trainer.ensemble_classification(
                scoring_metric=metric)
        elif self._advanced_trainer.model_type is 'regression':
            metric = 'neg_mean_squared_error'
            model = self._advanced_trainer.ensemble_regression(
                scoring_metric=metric)

            print('Based on the scoring metric {}, the best algorithm found '
                  'is: {}'.format(metric, model.algorithm_name))

        return model

    @property
    def advanced_features(self):
        """
        Return the underlying AdvancedSupervisedModelTrainer instance.

        For advanced users only.
        """
        return self._advanced_trainer
