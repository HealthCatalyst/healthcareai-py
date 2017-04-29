import pandas as pd
import healthcareai.common.file_io_utilities as io
import healthcareai.common.top_factors as factors
from healthcareai.common import filters
from healthcareai.common.healthcareai_error import HealthcareAIError


class TrainedSupervisedModel(object):
    def __init__(self,
                 model,
                 feature_model,
                 fit_pipeline,
                 model_type,
                 column_names,
                 grain_column,
                 prediction_column,
                 y_pred,
                 y_actual):
        self.model = model
        self.feature_model = feature_model
        self.fit_pipeline = fit_pipeline
        self.column_names = column_names
        self.model_type = model_type
        self.grain_column = grain_column
        self.prediction_column = prediction_column
        self.y_pred = y_pred
        self.y_actual = y_actual

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
        """ Given a new dataframe, apply data transformations and return a list of predictions """
        prepared_dataframe = self.prepare_and_subset(dataframe)

        # make predictions
        # TODO this will have to be classification or regression aware by using either .predict() or .predictproba()
        # y_predictions = self.model.predict_proba(dataframe)[:, 1]
        y_predictions = self.model.predict(prepared_dataframe)

        return y_predictions

    def make_factors(self, dataframe, number_top_features=3):
        prepared_dataframe = self.prepare_and_subset(dataframe)

        # Create a list of column names
        reason_col_names = ['Factor%iTXT' % i for i in range(1, number_top_features + 1)]
        top_features = factors.top_k_features(prepared_dataframe, self.feature_model, k=number_top_features)

        # join top features columns to results dataframe
        reasons_df = pd.DataFrame(top_features, columns=reason_col_names, index=dataframe.index)
        results = pd.concat([dataframe, reasons_df], axis=1, join_axes=[dataframe.index])

        return results

    def prepare_and_subset(self, dataframe):
        """
        Run the raw dataframe through the saved pipeline and return a dataframe that contains only the columns
        that were in the original model.
        
        This prevents any unexpected changes to incoming columns from interfering with the predictions.
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

        # ID, predictions, factors

        factors = self.create_factors(original_df)
        predictions = self.create_predctions(original_df)

        # join top features columns to results dataframe
        results = pd.concat([predictions, factors], axis=1, join_axes=[original_df.index])

        return results

    def create_all(self, original_df):
        # ID, x1, x2, ..., predictions, factors

        # Get predictions and factors
        predictions_and_factors = self.create_predictions_factors(original_df)

        # join top features columns to results dataframe
        results = pd.concat([original_df, predictions_and_factors], axis=1, join_axes=[original_df.index])

        return results

    def create_catalyst(self, original_df):
        # ID, bindings, metadata, otherstuff, predictions, factors
        # TODO stub
        pass

    def prep_and_predict(self, original_df):
        # Run the saved data preparation pipeline
        print('prep and prepare:')
        print(original_df.dtypes)

        original_df = self.fit_pipeline.transform(original_df)

        # Drop the predicted column
        original_df = filters.DataframeColumnRemover(self.prediction_column).fit_transform(original_df)

        # TODO think about an exclusions list or something so that you don't have to explicitly drop the predicted column
        # TODO this may make it so that
        # dataframe = dataframe[[c for c in dataframe.columns if c in self.column_names]]

        # make predictions
        # TODO this may have to be classification or regression aware by using either .predict() or .predictproba()
        # y_predictions = self.model.predict_proba(dataframe)[:, 1]
        y_predictions = self.model.predict(original_df)
        print('prep and prepare after pipeline:')
        print(original_df.dtypes)

        return y_predictions

    def get_roc_auc(self):
        """
        Returns the roc_auc of the holdout set from model training.
        """
        pass
        # return roc_auc_score(self.y_actual, self.y_pred)

    def roc_curve_plot(self):
        # TODO stubs - may be implemented elsewhere and needs to be moved here.
        """
        Returns a plot of the roc curve of the holdout set from model training.
        """
        pass