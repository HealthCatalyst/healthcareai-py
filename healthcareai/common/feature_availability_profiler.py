import pandas as pd
import time
import numpy as np
from datetime import datetime, timedelta, date
import matplotlib.pyplot as plt
from healthcareai.common.healthcareai_error import HealthcareAIError

def feature_availability_profiler(dataFrame, admitColName='AdmitDTS',
                                  lastLoadColName='LastLoadDTS',
                                  plotFlag=True,
                                  listFlag=False):

    """
    This function counts the number of populated data values over time for a
    given dataframe.

    Parameters
    ----------
    dataFrame (dataframe) : dataframe of features to count populated data in. This
      table must have a 2 date columns: one for patient admission date, one for
      today's date (or when the table was last loaded). Minimum 3 columns.
    admitColName (str) : name of column containing patient admission date
    lastLoadColName (str) : name of column containing today's date or when the
      table was last loaded.
    plotFlag (bol) : True will return a plot of the data availability.
    listFlag (bol) : True will return a matrix of populated fields vs. time.

    Returns
    -------
    numData (df) : a DF of populated fields vs. time.
    :param plotFlag:
    """

    df = dataFrame

    # Error checks
    if df[admitColName].dtype != 'datetime64[ns]':
        raise HealthcareAIError('Admit Date column is not a date type')
    if df[lastLoadColName].dtype != 'datetime64[ns]':
        raise HealthcareAIError('Last Load Date column is not a date type')
    if df.shape[1] < 3:
        raise HealthcareAIError('Dataframe must be at least 3 columns')

    # Look at data that's been pulled in
    print(df.head())
    a,b = df.shape
    print('Loaded ' + str(a) + ' rows and ' + str(b) + ' columns')

    # Get most recent date
    lastLoad = max(df[lastLoadColName])
    print('Data was last loaded on ' + str(lastLoad) + ' (from LastLoadDTS)')
    oldestAdmit = min(df['AdmitDTS'])
    print('Oldest data is from ' + str(oldestAdmit) + ' (from AdmitDTS)')

    # get key list to count
    keyList = [col for col in df.columns if col not in ['index', lastLoadColName, admitColName]]
    print('Column names to count:')
    for col in keyList:
        print(col)

    # create a container for final null counts
    numData = {'Age': []}
    for key in keyList:
        numData[key] = []

    # get date range to count over
    dateSpread = lastLoad - oldestAdmit
    if dateSpread.days < 90:
        dateRange = [1/24, 2/24, 4/24, 8/24 ,12/24] + list(range(1,dateSpread.days))
    else:
        dateRange = [1/24, 2/24, 4/24, 8/24, 12/24] + list(range(1, 91))

    # count null percentage over date range
    for i in dateRange:
        start = lastLoad - timedelta(days=i)
        numNullsTemp = count_nulls_in_date_range(df, start, lastLoad, admitColName)
        numData['Age'].append(i)
        for key in keyList:
            numData[key].append(numNullsTemp[key])

    # print nulls if desired
    numData = pd.DataFrame(numData)
    numData['Age'] = numData['Age'].round(decimals=1)
    numData.set_index('Age', inplace=True)
    print('Age is the number of days since patient admission.')
    if listFlag is True:
        print(numData)

    if plotFlag is True:
        # plot nulls vs time.
        plt.plot(numData)
        plt.plot(lw=2, linestyle='--')
        plt.xlabel('Days since Admission')
        plt.ylabel('Populated Values (%)')
        plt.title('Feature Availability Over Time')
        plt.legend(labels=keyList, loc="lower right")
        plt.show()

    return(numData)

def count_nulls_in_date_range(df, start, end, admitColName):

    # called by feature_availability_profiler to count nulls within a date range.
    mask = (df[admitColName] > start) & (df[admitColName] <= end)
    df = df.loc[mask]
    numData = 100 - np.round(100*df.isnull().sum()/df.shape[0])
    return(numData)
