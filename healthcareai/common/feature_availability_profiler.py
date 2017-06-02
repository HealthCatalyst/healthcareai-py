import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt

from healthcareai.common.healthcareai_error import HealthcareAIError


def feature_availability_profiler(
        data_frame,
        admit_col_name='AdmitDTS',
        last_load_col_name='LastLoadDTS',
        plot_flag=True,
        list_flag=False):
    """
    This function counts the number of populated data values over time for a
    given dataframe.

    Parameters
    ----------
    data_frame (dataframe) : dataframe of features to count populated data in. This
      table must have a 2 date columns: one for patient admission date, one for
      today's date (or when the table was last loaded). Minimum 3 columns.
    admit_col_name (str) : name of column containing patient admission date
    last_load_col_name (str) : name of column containing today's date or when the
      table was last loaded.
    plotFlag (bol) : True will return a plot of the data availability.
    list_flag (bol) : True will return a matrix of populated fields vs. time.

    Returns
    -------
    num_data (df) : a DF of populated fields vs. time.
    :param data_frame: Your dataframe
    :param admit_col_name: The name of the column containing the admit dates
    :param last_load_col_name: The name of the column containing the last load date
    :param plot_flag: Shows or hides plot
    :param list_flag: Shows or hides listS
    """

    df = data_frame

    # Error checks
    if df[admit_col_name].dtype != 'datetime64[ns]':
        raise HealthcareAIError('Admit Date column is not a date type')
    if df[last_load_col_name].dtype != 'datetime64[ns]':
        raise HealthcareAIError('Last Load Date column is not a date type')
    if df.shape[1] < 3:
        raise HealthcareAIError('Dataframe must be at least 3 columns')

    # Look at data that's been pulled in
    print(df.head())
    a, b = df.shape
    print('Loaded {} rows and {} columns'.format(str(a), str(b)))

    # Get most recent date
    last_load = max(df[last_load_col_name])
    print('Data was last loaded on {} (from {})'.format(str(last_load), str(last_load_col_name)))
    oldest_admit = min(df[admit_col_name])
    print('Oldest data is from {} (from {})'.format(str(oldest_admit), str(admit_col_name)))

    # get key list to count
    key_list = [col for col in df.columns if col not in ['index', last_load_col_name, admit_col_name]]
    print('Column names to count:')
    for col in key_list:
        print(col)

    # create a container for final null counts
    num_data = {'Age': []}
    for key in key_list:
        num_data[key] = []

    # get date range to count over
    date_spread = last_load - oldest_admit
    if date_spread.days < 90:
        date_range = [1 / 24, 2 / 24, 4 / 24, 8 / 24, 12 / 24] + list(range(1, date_spread.days))
    else:
        date_range = [1 / 24, 2 / 24, 4 / 24, 8 / 24, 12 / 24] + list(range(1, 91))

    # count null percentage over date range
    for i in date_range:
        start = last_load - timedelta(days=i)
        num_nulls_temp = count_nulls_in_date_range(df, start, last_load, admit_col_name)
        num_data['Age'].append(i)
        for key in key_list:
            num_data[key].append(num_nulls_temp[key])

    # print nulls if desired
    num_data = pd.DataFrame(num_data)
    num_data['Age'] = num_data['Age'].round(decimals=1)
    num_data.set_index('Age', inplace=True)
    print('Age is the number of days since patient admission.')

    if list_flag is True:
        print(num_data)

    if plot_flag is True:
        # plot nulls vs time.
        plt.plot(num_data)
        plt.plot(lw=2, linestyle='--')
        plt.xlabel('Days since Admission')
        plt.ylabel('Populated Values (%)')
        plt.title('Feature Availability Over Time')
        plt.legend(labels=key_list, loc="lower right")
        plt.show()

    return num_data


def count_nulls_in_date_range(df, start, end, admit_col_name):
    """Counts nulls for a given dataframe column within a date range."""
    mask = (df[admit_col_name] > start) & (df[admit_col_name] <= end)
    df = df.loc[mask]
    num_data = 100 - np.round(100 * df.isnull().sum() / df.shape[0])

    return num_data


if __name__ == "__main__":
    pass
