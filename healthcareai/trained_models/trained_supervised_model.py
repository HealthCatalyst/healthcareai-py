import pandas as pd
import healthcareai.common.file_io_utilities as io
import healthcareai.common.top_factors as top_factors
from healthcareai.common import filters


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
        # Run the saved data preparation pipeline
        prepared_dataframe = self.fit_pipeline.transform(dataframe)

        # Subset the dataframe to only columns that were saved from the original model training
        # This prevents any unexpected changes to incoming columns from interfering with the predictions.
        prepared_dataframe = prepared_dataframe[self.column_names]

        # make predictions
        # TODO this will have to be classification or regression aware by using either .predict() or .predictproba()
        # y_predictions = self.model.predict_proba(dataframe)[:, 1]
        y_predictions = self.model.predict(prepared_dataframe)

        return y_predictions

    def make_factors(self, prepared_dataframe):
        # ID, predictions, factors
        factors = pd.DataFrame({
            'id': [1, 2, 3],
            'factor1': ['F', 'M', 'F'],
            'factor2': ['F', 'M', 'F'],
            'factor3': ['F', 'M', 'F'],
        })
        return factors

    def create_predictions_factors(self, original_df):
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