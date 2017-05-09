import sqlite3

from healthcareai.common.healthcareai_error import HealthcareAIError


def drop_table(db_name, table_name):
    db = sqlite3.connect(db_name)
    cursor = db.cursor()

    query = 'DROP TABLE IF EXISTS {}};'.format(table_name)
    cursor.execute(query)


def is_table_empty(db_name, table_name):
    db = sqlite3.connect(db_name)
    cursor = db.cursor()

    query = 'SELECT COUNT(*) FROM {};'.format(table_name)
    cursor.execute(query)
    x = cursor.fetchone()[0]

    return x == 0


def setup_deploy_tables(db_name):
    # Setup db connection
    db = sqlite3.connect(db_name)
    cursor = db.cursor()

    # Drop tables
    drop_table(db_name, 'HCPyDeployClassificationBASE')
    drop_table(db_name, 'HCPyDeployRegressionBASE')

    # Set up tables
    classification_table_setup = """
       CREATE TABLE IF NOT EXISTS HCPyDeployClassificationBASE (
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
        CREATE TABLE IF NOT EXISTS HCPyDeployRegressionBASE (
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
    a = is_table_empty('foo2.db', 'HCPyDeployClassificationBASE')
    b = is_table_empty('foo2.db', 'HCPyDeployRegressionBASE')

    if a and b:
        return True
    else:
        raise HealthcareAIError('There was a problem setting up test tables')
