from healthcareai.common.healthcareai_error import HealthcareAIError
import time
import datetime
import pandas as pd


# import records


def table_archiver(server, database, source_table, destination_table, timestamp_column_name='ArchivedDTS'):
    """
    Takes a table and archives a complete copy of it with the addition of a timestamp of when the archive occurred to a
    given destination table on the same database.

    This should build a new table if the table doesn't exist.

    :param server: server name
    :param database: database name
    :param source_table: source table name
    :param destination_table: destination table name
    :param timestamp_column_name: new timestamp column name
    :rtype: str: basic stats about how many records were archived
    
    Example usage:
    
    table_archiver('localhost', 'SAM_123', 'RiskScores', 'RiskScoreArchive', 'ArchiveDTS')
    """
    # Basic input validation
    if type(server) is not str:
        raise HealthcareAIError('Please specify a server address')
    if type(database) is not str:
        raise HealthcareAIError('Please specify a database name')
    if type(source_table) is not str:
        raise HealthcareAIError('Please specify a source table name')
    if type(destination_table) is not str:
        raise HealthcareAIError('Please specify a destination table name')

    start_time = time.time()

    connection_string = 'mssql+pyodbc://{}/{}?driver=SQL+Server+Native+Client+11.0'.format(server, database)

    # check if destination exists
    # db = records.Database(connection_string)
    # table_names = db.get_table_names()

    # if destination_table in table_names:
    #     print('destination table exists')
    #     before = count_records_in_table(connection_string, destination_table)
    #     print(before)
    # else:
    #     print('The destination table {} does not exist. Creating now.'.format(destination_table))

    # Load the table to be archived
    df = pd.read_sql_table(source_table, connection_string)
    number_records_to_add = len(df)

    # Add timestamp to dataframe
    df[timestamp_column_name] = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')

    # Save the new dataframe out to the db without the index, appending values
    df.to_sql(destination_table, connection_string, index=False, if_exists='append')

    # after = count_records_in_table(connection_string, destination_table)
    # if before is not None:
    #     number_records_added = after - before
    #     if number_records_to_add is not number_records_to_add:
    #         print('There is a discrepancy between the number added and the number that we pulled from the source table')
    #
    #     print('To add: {} Added: {}'.format(number_records_to_add, number_records_added))

    end_time = time.time()
    delta_time = end_time - start_time
    result = 'Archived {} records from {}/{}/{} to {} in {} seconds'.format(number_records_to_add, server, database,
                                                                            source_table, destination_table,
                                                                            delta_time)
    
    return result

# def count_records_in_table(connection_string, table_name):
#     if connection_string is None:
#         raise HealthcareAIError('No connection string specified')
#     if table_name is None:
#         raise HealthcareAIError('No table name specified')
#
#     db = records.Database(connection_string)
#     count_raw = db.query('SELECT COUNT(*) as before_count FROM :table', table=table_name)
#     count = count_raw.as_dict()[0]['before_count']
#
#     return count
