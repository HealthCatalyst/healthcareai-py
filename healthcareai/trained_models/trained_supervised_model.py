import pandas as pd
import healthcareai.common.file_io_utilities as io
import healthcareai.common.top_factors as top_factors
from healthcareai.common import filters


class TrainedSupervisedModel(object):
    def __init__(self,
                 model,
                 feature_model,
                 fit_pipeline,
                 # column_names,
                 prediction_type,
                 grain_column,
                 prediction_column,
                 y_pred,
                 y_actual):
        self.model = model
        self.feature_model = feature_model
        self.fit_pipeline = fit_pipeline
        # self.column_names = column_names
        self.predictiontype = prediction_type
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

    def predict(self, dataframe):
        """
        Given a new dataframe, apply data transformations and return a dataframe of predictions
        Args:
            dataframe (Pandas.DataFrame): a pandas dataframe

        Returns:
            Pandas.DataFrame:
        """
        # Copy the incoming dataframe so we can rebuild it later
        df = dataframe.copy()

        # join prediction and top features columns to dataframe
        # TODO should this return a dataframe with the same target column name as the training set?
        df[self.prediction_column] = self.prep_and_predict(df)

        # # bring back the grain column and reset the df index
        # df.insert(0, self.grain_column, dataframe[self.grain_column])
        # df.reset_index(drop=True, inplace=True)

        return df

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