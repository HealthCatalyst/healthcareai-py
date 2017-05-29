"""
This file creates catalyst-EDW specific tables
"""
import sqlite3

from healthcareai.common.healthcareai_error import HealthcareAIError


def drop_table(db_name, table_name):
    """ Given a sqlite db filename, drops a given table if it exists. """
    db = sqlite3.connect(db_name)
    cursor = db.cursor()

    query = 'DROP TABLE IF EXISTS {};'.format(table_name)
    cursor.execute(query)


def is_table_empty(db_name, table_name):
    """ Checks if a table on a given sqlite db file is empty. """
    db = sqlite3.connect(db_name)
    cursor = db.cursor()

    query = 'SELECT COUNT(*) FROM {};'.format(table_name)
    cursor.execute(query)
    x = cursor.fetchone()[0]

    return x == 0


def setup_deploy_tables(db_name):
    """ Delete and recreate Health Catalyst specific destination tables. WARNING: DATA LOSS WILL OCCUR. """
    # Setup db connection
    db = sqlite3.connect(db_name)
    cursor = db.cursor()

    # Drop tables
    drop_table(db_name, 'PredictionClassificationBASE')
    drop_table(db_name, 'PredictionRegressionBASE')

    # Set up tables
    classification_table_setup = """
       CREATE TABLE IF NOT EXISTS PredictionClassificationBASE (
            BindingID [int] ,
            BindingNM [varchar] (255),
            LastLoadDTS [datetime2] (7),
            PatientEncounterID [decimal] (38, 0),
            PredictedProbNBR [decimal] (38, 2),
            Factor1TXT [varchar] (255),
            Factor2TXT [varchar] (255),
            Factor3TXT [varchar] (255));
            """
    cursor.execute(classification_table_setup)

    regression_table_setup = """
        CREATE TABLE IF NOT EXISTS PredictionRegressionBASE (
            BindingID [int],
            BindingNM [varchar] (255),
            LastLoadDTS [datetime2] (7),
            PatientEncounterID [decimal] (38, 0),
            PredictedValueNBR [decimal] (38, 2),
            Factor1TXT [varchar] (255),
            Factor2TXT [varchar] (255),
            Factor3TXT [varchar] (255));
            """
    cursor.execute(regression_table_setup)

    # Verify both are empty
    a = is_table_empty(db_name, 'PredictionClassificationBASE')
    b = is_table_empty(db_name, 'PredictionRegressionBASE')

    if a and b:
        return True
    else:
        raise HealthcareAIError('There was a problem setting up test tables')


if __name__ == "__main__":
    pass
