import numpy as np


def write_feature_importances(importance_attr, col_list):
    """
    This function prints an ordered list of rf-related feature importance.

    Parameters
    ----------
    importance_attr (attribute) : This is the feature importance attribute
    from a scikit-learn method that represents feature importances
    col_list (list) : Vector holding list of column names

    Returns
    -------
    Nothing. Simply prints feature importance list to console.
    """
    indices = np.argsort(importance_attr)[::-1]
    print('\nVariable importance:')

    for f in range(0, len(col_list)):
        print("%d. %s (%f)" % (f + 1, col_list[indices[f]], importance_attr[indices[f]]))