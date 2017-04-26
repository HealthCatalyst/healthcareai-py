import pandas as pd
from healthcareai.common.file_io_utilities import save_object_as_pickle
import healthcareai.common.top_factors as top_factors

class TrainedSupervisedModel(object):
    def __init__(self,
                 model,
                 feature_model,
                 pipeline,
                 column_names,
                 prediction_type,
                 graincol,
                 y_pred,
                 y_actual):
        self.model = model
        self.feature_model = feature_model
        self.pipeline = pipeline
        self.column_names = column_names
        self.predictiontype = prediction_type
        self.graincol = graincol
        self.y_pred = y_pred
        self.y_actual = y_actual

    def save(self, filepath):
        save_object_as_pickle(filepath, self)

    def predict(self, dataframe):
        # Get predictive scores
        df = dataframe.copy()
        df = self.pipeline.transform(df)
        df = df[[c for c in df.columns if c in self.column_names]]

        y_pred = self.model.predict_proba(df)[:, 1]

        # join prediction and top features columns to dataframe
        df['y_pred'] = y_pred

        # bring back the grain column and reset the df index
        df.insert(0, self.graincol, dataframe[self.graincol])
        df.reset_index(drop=True, inplace=True)

    def predict_with_factors(self, dataframe, number_top_features=3):
        """ Returns model with predicted probability scores and top n features """

        # Get predictive scores
        df = dataframe.copy()
        df = self.pipeline.transform(df)
        df = df[[c for c in df.columns if c in self.column_names]]

        y_pred = self.model.predict_proba(df)[:, 1]

        # Get top 3 reasons
        reason_col_names = ['Factor%iTXT' % i for i in range(1, number_top_features + 1)]
        top_feats_lists = top_factors.get_top_k_features(df, self.feature_model, k=number_top_features)

        # join prediction and top features columns to dataframe
        df['y_pred'] = y_pred
        reasons_df = pd.DataFrame(top_feats_lists, columns=reason_col_names,
                                  index=df.index)
        df = pd.concat([df, reasons_df], axis=1, join_axes=[df.index])

        # bring back the grain column and reset the df index
        df.insert(0, self.graincol, dataframe[self.graincol])
        df.reset_index(drop=True, inplace=True)

        return df

    def get_roc_auc(self):
        """
        Returns the roc_auc of the holdout set from model training.
        """
        pass
        # return roc_auc_score(self.y_actual, self.y_pred)

    def roc_curve_plot(self):
        """
        Returns a plot of the roc curve of the holdout set from model training.
        """
        pass