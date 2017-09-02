import pandas as pd

from healthcareai.common.healthcareai_error import HealthcareAIError


def load_csv(file_path):
    """
    Loads a csv file into a pandas dataframe. Checks for common null/missing values.
    Args:
        file_path (str): Full or relative path to file.

    Returns:
        (pandas.core.frame.DataFrame): The csv file in a dataframe
    """
    try:
        df = pd.read_csv(file_path, na_values=['None', 'null'])
        df = df.rename(columns=lambda x: x.strip())
        return df
    except FileNotFoundError:
        raise HealthcareAIError(
            """No csv file was found at: {}.\nPlease check your path and try again.""".format(file_path))
