import pandas as pd
import numpy as np
from datetime import datetime

import healthcareai.common.file_io_utilities as io_utilities
import healthcareai.common.model_eval as model_evaluation
import healthcareai.common.top_factors as factors
import healthcareai.common.write_predictions_to_database as hcaidb
import healthcareai.common.database_connection_validation as hcaidbval
from healthcareai.common.healthcareai_error import HealthcareAIError


class TrainedSupervisedModel(object):
    """
    The meta-object that is created when training supervised models. 
    
    This object contains:
        - trained estimator
        - trained linear estimator used for row level factor analysis
        - column metadata including transformed feature columns, grain & predicted column
        - the fit data preparation pipeline used for transforming new data for prediction
        - calculated metrics
        - test set actuals, predicted values/probabilities, predicted classes
    """

    def __init__(self,
                 model,
                 feature_model,
                 fit_pipeline,
                 model_type,
                 column_names,
                 grain_column,
                 prediction_column,
                 test_set_predictions,
                 test_set_class_labels,
                 test_set_actual,
                 metric_by_name):
        self.model = model
        self.feature_model = feature_model
        self.fit_pipeline = fit_pipeline
        self.column_names = column_names
        self._model_type = model_type
        self.grain_column = grain_column
        self.prediction_column = prediction_column
        self.test_set_predictions = test_set_predictions
        self.test_set_class_labels = test_set_class_labels
        self.test_set_actual = test_set_actual
        self._metric_by_name = metric_by_name

    @property
    def algorithm_name(self):
        """ Model name extracted from the class type """
        model = model_evaluation.get_estimator_from_meta_estimator(self.model)
        name = type(model).__name__

        return name

    @property
    def is_classification(self):
        """ easy check to consolidate magic strings in all the model type switches """
        return self.model_type == 'classification'

    @property
    def is_regression(self):
        """ easy check to consolidate magic strings in all the model type switches """
        return self.model_type == 'regression'

    @property
    def best_hyperparameters(self):
        """ Best hyperparameters found if model is a meta estimator """
        return model_evaluation.get_hyperparameters_from_meta_estimator(self.model)

    @property
    def model_type(self):
        """ Model type (regression or classification) """
        return self._model_type

    @property
    def binary_classification_scores(self):
        # TODO low priority, but test this
        """ Returns the probability scores of the first class for a binary classification model. """
        if self.is_regression:
            raise HealthcareAIError('ROC/PR plots are not used to evaluate regression models.')

        predictions = np.squeeze(self.test_set_predictions[:, 1])

        return predictions

    @property
    def metrics(self):
        """ Return the metrics that were calculated when the model was trained. """
        return self._metric_by_name

    def save(self, filename):
        """
        Save this object to a pickle file with the given file name
        
        Args:
            filename (str): Name of the file
        """

        # TODO should this timestamp a model name automatically? (for example 2017-04-26_01.33.55_random_forest.pkl)
        io_utilities.save_object_as_pickle(filename, self)
        print('Model saved as {}'.format(filename))

    def make_predictions(self, dataframe):
        """
        Given a new dataframe, apply data transformations and return a dataframe of predictions 

        Args:
            dataframe (pandas.core.frame.DataFrame): Raw prediction dataframe

        Returns:
            pandas.core.frame.DataFrame: A dataframe containing the grain id and predicted values
        """

        # Run the raw dataframe through the preparation process
        prepared_dataframe = self.prepare_and_subset(dataframe)

        # make predictions returning probabity of a class or value of regression
        if self.is_classification:
            # Only save the prediction of one of the two classes
            y_predictions = self.model.predict_proba(prepared_dataframe)[:, 1]
        elif self.is_regression:
            y_predictions = self.model.predict(prepared_dataframe)
        else:
            raise HealthcareAIError('Model type appears to be neither regression or classification.')

        # Create a new dataframe with the grain column from the original dataframe
        results = pd.DataFrame()
        results[self.grain_column] = dataframe[[self.grain_column]]
        results['Prediction'] = y_predictions

        return results

    def prepare_and_subset(self, dataframe):
        """
        Run the raw dataframe through the saved pipeline and return a dataframe that contains only the columns that were
         in the original model.
        
        This prevents any unexpected changes to incoming columns from interfering with the predictions.

        Args:
            dataframe (pandas.core.frame.DataFrame): Raw prediction dataframe

        Returns:
            pandas.core.frame.DataFrame: A dataframe that has been run through the pipeline and subsetted to only the
             columns the model expects.
        """

        try:
            # Raise an error here if any of the columns the model expects are not in the prediction dataframe

            # Run the saved data preparation pipeline
            # TODO dummies will not run if a prediction dataframe only has: 1 row or all the same value in a categorical
            # TODO column
            prepared_dataframe = self.fit_pipeline.transform(dataframe)

            # Subset the dataframe to only columns that were saved from the original model training
            prepared_dataframe = prepared_dataframe[self.column_names]
        except KeyError as ke:
            error_message = """One or more of the columns that the saved trained model needs is not in the dataframe.\n
            Please compare these lists to see which field(s) is/are missing. Note that you can pass in extra fields,\n
            which will be ignored, but you must pass in all the required fields.\n
            
            Required fields: {}
            
            Given fields: {}
            
            Likely missing field(s): {}
            """.format(self.column_names, list(dataframe.columns), ke)
            raise HealthcareAIError(error_message)

        return prepared_dataframe

    def make_factors(self, dataframe, number_top_features=3):
        """
        Given a prediction dataframe, build and return a list of the top k feautures in dataframe format
        
        Args:
            dataframe (pandas.core.frame.DataFrame): Raw prediction dataframe
            number_top_features (int): Number of top features per row

        Returns:
            pandas.core.frame.DataFrame:  
        """

        # Run the raw dataframe through the preparation process
        prepared_dataframe = self.prepare_and_subset(dataframe)

        # Create a new dataframe with the grain column from the original dataframe
        results = dataframe[[self.grain_column]]

        # Create a list of column names
        reason_col_names = ['Factor{}TXT'.format(i) for i in range(1, number_top_features + 1)]

        # Get a 2 dimensional list of all the factors
        top_features = factors.top_k_features(prepared_dataframe, self.feature_model, k=number_top_features)

        # Verify that the number of factors matches the number of rows in the original dataframe.
        if len(top_features) != len(dataframe):
            raise HealthcareAIError('Warning! The number of predictions does not match the number of rows.')

        # Create a dataframe from the column names and top features
        reasons_df = pd.DataFrame(top_features, columns=reason_col_names, index=dataframe.index)

        # Join the top features and results dataframes
        results = pd.concat([results, reasons_df], axis=1, join_axes=[dataframe.index])
        # results.set_index(keys=self.grain_column, inplace=True)

        return results

    def make_predictions_with_k_factors(self, dataframe, number_top_features=3):
        """
        Given a prediction dataframe, build and return a dataframe with the grain column, the predictions and the top k
        feautures.

        Args:
            dataframe (pandas.core.frame.DataFrame): Raw prediction dataframe
            number_top_features (int): Number of top features per row

        Returns:
            pandas.core.frame.DataFrame:  
        """

        # TODO Note this is inefficient since we are running the raw dataframe through the pipeline twice.
        # Get the factors and predictions
        results = self.make_factors(dataframe, number_top_features=number_top_features)
        predictions = self.make_predictions(dataframe)

        # Verify that the number of predictions matches the number of rows in the original dataframe.
        if len(predictions) != len(dataframe):
            raise HealthcareAIError('Warning! The number of predictions does not match the number of rows.')

        # Add predictions column to dataframe
        results['Prediction'] = predictions['Prediction']

        return results

    def make_original_with_predictions_and_features(self, dataframe, number_top_features=3):
        """
        Given a prediction dataframe, build and return a dataframe with the all the original columns, the predictions, 
        and the top k feautures.

        Args:
            dataframe (pandas.core.frame.DataFrame): Raw prediction dataframe
            number_top_features (int): Number of top features per row

        Returns:
            pandas.core.frame.DataFrame:  
        """

        # TODO Note this is inefficient since we are running the raw dataframe through the pipeline twice.
        # Get the factors and predictions
        results = self.make_predictions_with_k_factors(dataframe, number_top_features=number_top_features)

        # replace the original prediction column
        original_dataframe = dataframe.drop([self.prediction_column], axis=1)

        # Join the two dataframes together
        results = pd.concat([original_dataframe, results], axis=1)

        return results

    def create_catalyst_dataframe(self, dataframe):
        """
        Given a prediction dataframe, build and return a dataframe with the health catalyst specific column names, the
        predictions, and the top 3 feautures.

        Args:
            dataframe (pandas.core.frame.DataFrame): Raw prediction dataframe

        Returns:
            pandas.core.frame.DataFrame:  
        """
        # Get predictions and on the top 3 features (catalyst SAMs expect 3 factors)
        factors_and_predictions_df = self.make_predictions_with_k_factors(dataframe, number_top_features=3)

        # Add all the catalyst-specific columns to back into the SAM
        factors_and_predictions_df['BindingID'] = 0
        factors_and_predictions_df['BindingNM'] = 'Python'
        factors_and_predictions_df['LastLoadDTS'] = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

        return factors_and_predictions_df

    def predict_to_catalyst_sam(self, dataframe, server, database, table, schema=None, predicted_column_name=None):
        """
        Given a dataframe you want predictions on, make predictions and save them to a catalyst-specific EDW table.
        Args:
            dataframe (pandas.core.frame.DataFrame): Raw prediction dataframe
            server (str): the target server name
            database (str): the database name
            table (str): the destination table name
            schema (str): the optional schema
            predicted_column_name (str): optional predicted column name (defaults to PredictedProbNBR or PredictedValueNBR)
        """

        # Make predictions in specific format
        sam_df = self.create_catalyst_dataframe(dataframe)

        # Rename prediction column to default based on model type or given one
        if predicted_column_name is None:
            if self.is_classification:
                predicted_column_name = 'PredictedProbNBR'
            elif self.is_regression:
                predicted_column_name = 'PredictedValueNBR'
        sam_df.rename(columns={'Prediction': predicted_column_name}, inplace=True)

        try:
            engine = hcaidb.build_mssql_engine(server, database)
            hcaidb.write_to_db_agnostic(engine, table, sam_df, schema=schema)
        except HealthcareAIError as hcaie:
            # Run validation and alert user
            hcaidbval.validate_destination_table_connection(server, table, self.grain_column, self.prediction_column)
            raise HealthcareAIError(hcaie.message)

    def predict_to_sqlite(self,
                          prediction_dataframe,
                          database,
                          table,
                          prediction_generator,
                          predicted_column_name=None):
        """
        Given a dataframe you want predictions on, make predictions and save them to an sqlite table

        Args:
            prediction_dataframe (pandas.core.frame.DataFrame): Raw prediction dataframe
            database (str): database file name
            table (str): table name
            prediction_generator (method): one of the trained supervised model prediction methods
            predicted_column_name (str): optional predicted column name (defaults to PredictedProbNBR or PredictedValueNBR)
        """
        # validate inputs
        if type(prediction_generator).__name__ != 'method':
            raise HealthcareAIError(
                'Use of this method requires a prediction generator from a trained supervised model')

        # Get predictions from given generator
        sam_df = prediction_generator(prediction_dataframe)

        # Rename prediction column to default based on model type or given one
        if predicted_column_name is None:
            if self.is_classification:
                predicted_column_name = 'PredictedProbNBR'
            elif self.is_regression:
                predicted_column_name = 'PredictedValueNBR'

        sam_df.rename(columns={'Prediction': predicted_column_name}, inplace=True)
        engine = hcaidb.build_sqlite_engine(database)
        hcaidb.write_to_db_agnostic(engine, table, sam_df)

    def roc_curve_plot(self):
        """ Returns a plot of the ROC curve of the holdout set from model training. """
        self.validate_classification()
        model_evaluation.tsm_classification_comparison_plots(trained_supervised_model=self, plot_type='ROC')

    def roc(self, print_output=True):
        """ Prints out ROC details and returns them with cutoffs. """
        self.validate_classification()
        roc = self._metric_by_name

        if print_output:
            print("""\nReceiver Operating Characteristic (ROC):
            Ideal ROC cutoff is {:0.2f}, yielding TPR of {:0.2f} and FPR of {:0.2f}""".format(
                roc['best_roc_cutoff'], roc['best_true_positive_rate'], roc['best_false_positive_rate']))

            print('|--------------------------------|')
            print('|               ROC              |')
            print('|  Threshhold  |  TPR   |  FPR   |')
            print('|--------------|--------|--------|')
            for i in range(len(roc['roc_thresholds'])):
                print('|        {:03.2f}  |  {:03.2f}  |  {:03.2f}  |'.format(roc['roc_thresholds'][i], roc['true_positive_rates'][i], roc['false_positive_rates'][i]))
            print('|--------------------------------|')

        return roc

    def pr_curve_plot(self):
        """ Returns a plot of the PR curve of the holdout set from model training. """
        self.validate_classification()
        model_evaluation.tsm_classification_comparison_plots(trained_supervised_model=self, plot_type='PR')

    def pr(self, print_output=True):
        """ Prints out PR details and returns them with cutoffs. """
        self.validate_classification()
        pr = self._metric_by_name

        if print_output:
            print("""\nPrecision-Recall:
        Area under Precision Recall curve (PR AUC): {:0.2f}
        Ideal PR cutoff is {:0.2f}, yielding precision of {:04.3f} and recall of {:04.3f}""".format(
        pr['pr_auc'], pr['best_pr_cutoff'], pr['best_precision'], pr['best_recall']))

            print('|---------------------------------|')
            print('|   Precision-Recall Thresholds   |')
            print('| Threshhold | Precision | Recall |')
            print('|------------|-----------|--------|')
            for i in range(len(pr['pr_thresholds'])):
                print('|    {:03.2f}    |    {:03.2f}   |  {:03.2f}  |'.format(pr['pr_thresholds'][i], pr['precisions'][i], pr['recalls'][i]))
            print('|---------------------------------|')

        return pr

    def validate_classification(self):
        """
        Checks that a model is classification and raises an error if it is not. Run this on any method that only makes
        sense for classification.
        """
        # TODO add binary check and rename to validate_binary_classification
        if self.model_type != 'classification':
            raise HealthcareAIError('This function only runs on a binary classification model.')