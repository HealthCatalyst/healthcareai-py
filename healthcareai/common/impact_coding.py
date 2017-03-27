import pandas as pd
from sklearn import model_selection


def impact_coding_on_a_single_column(dataframe, predicted_column, impact_column):
    train, test = model_selection.train_test_split(dataframe, test_size=0.8, random_state=0)
    x_bar = train[predicted_column].mean()
    impact = pd.DataFrame(
        train.groupby([impact_column])[predicted_column].mean().rename(impact_column + "_impact_coded"))
    impact.reset_index(level=0, inplace=True)
    impact[impact_column + "_impact_coded"] = impact[impact_column + "_impact_coded"] - x_bar
    post_df = test.merge(impact, how='left', on=impact_column)
    post_df.drop(impact_column, axis=1, inplace=True)
    post_df[impact_column + "_impact_coded"].fillna(value=x_bar, inplace=True)

    return post_df
