import pandas as pd
from datetime import datetime
import healthcareai.common.file_io_utilities as io
import healthcareai.common.top_factors as factors
from healthcareai.common.healthcareai_error import HealthcareAIError


class TrainedSupervisedModel(object):
    """
    The meta-object that is created when training supervised models. 
    
    This object contains:
        - trained estimator
        - trained linear estimator used for row level factor analysis
        - the fit data preparation pipeline used for transforming new data for prediction
        - calculated metrics
    """

    def __init__(self,
                 model,
                 feature_model,
                 fit_pipeline,
                 model_type,
                 column_names,
                 grain_column,
                 prediction_column,
                 y_pred,
                 y_actual,
                 metric_by_name):
        self.model = model
        self.feature_model = feature_model
        self.fit_pipeline = fit_pipeline
        self.column_names = column_names
        self.model_type = model_type
        self.grain_column = grain_column
        self.prediction_column = prediction_column
        self.y_pred = y_pred
        self.y_actual = y_actual
        self._metric_by_name = metric_by_name

    def save(self, filename):
        """
        Save this object to a pickle file with the given file name
        
        Args:
            filename (str): Name of the file
        """

        # TODO should this timestamp a model name automatically? (for example 2017-04-26_01.33.55_random_forest.pkl)
        io.save_object_as_pickle(filename, self)
        print('Model saved as {}'.format(filename))

    def make_predictions(self, dataframe):
        """
        Given a new dataframe, apply data transformations and return a list of predictions 

        Args:
            dataframe (pandas.core.frame.DataFrame): Raw prediction dataframe

        Returns:
            list: A list of predicted values that represents a column
        """

        # Run the raw dataframe through the preparation process
        prepared_dataframe = self.prepare_and_subset(dataframe)

        # make predictions
        # TODO this will have to be classification or regression aware by using either .predict() or .predictproba()
        # y_predictions = self.model.predict_proba(dataframe)[:, 1]
        y_predictions = self.model.predict(prepared_dataframe)

        return y_predictions

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
        predictions_list = self.make_predictions(dataframe)

        # Verify that the number of predictions matches the number of rows in the original dataframe.
        if len(predictions_list) != len(dataframe):
            raise HealthcareAIError('Warning! The number of predictions does not match the number of rows.')

        # Add predictions column to dataframe
        results[self.prediction_column] = predictions_list

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

    def metrics(self):
        """ Return the metrics that were calculated when the model was trained. """
        return self._metric_by_name

    def roc_curve_plot(self):
        # TODO stubs - may be implemented elsewhere and needs to be moved here.
        """
        Returns a plot of the roc curve of the holdout set from model training.
        """
        pass
